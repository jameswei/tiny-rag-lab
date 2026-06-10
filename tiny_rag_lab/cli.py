import argparse
import re
import time
from pathlib import Path


def _make_embedder(model_name: str | None = None):
    """Create the default SentenceTransformerEmbedder.

    Isolated into a helper so tests can patch it with FakeEmbedder without
    changing the CLI interface.
    """
    from tiny_rag_lab.embeddings import SentenceTransformerEmbedder
    if model_name is None:
        return SentenceTransformerEmbedder()
    return SentenceTransformerEmbedder(model_name)


def _make_generator(args):
    """Create an OpenAIGenerator from CLI args or environment variables.

    Isolated so tests can patch it with FakeGenerator.
    Priority: CLI flag > environment variable > SDK default.
    """
    import os
    from tiny_rag_lab.generation import OpenAIGenerator
    api_key = getattr(args, "api_key", None) or os.environ.get("OPENAI_API_KEY")
    base_url = getattr(args, "base_url", None) or os.environ.get("OPENAI_BASE_URL")
    model = getattr(args, "model", None) or os.environ.get("OPENAI_MODEL")
    return OpenAIGenerator(model=model, api_key=api_key, base_url=base_url)


_CITATION_RE = re.compile(r"\[Source: ([^\]]+)\]")


def cmd_index(args):
    from tiny_rag_lab.chunking import chunk_documents
    from tiny_rag_lab.documents import load_documents
    from tiny_rag_lab.index_writer import write_index

    corpus_root = Path(args.corpus).resolve()
    index_dir = Path(args.index_dir)

    print(f"Loading corpus from {corpus_root} ...")
    docs = load_documents(corpus_root)
    print(f"Loaded {len(docs)} document(s)")

    chunks = chunk_documents(
        docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    print(
        f"Chunked into {len(chunks)} chunk(s)"
        f" (size={args.chunk_size}, overlap={args.chunk_overlap})"
    )

    embedder = _make_embedder()
    backend = type(embedder).__name__
    model = getattr(embedder, "model_name", backend)
    print(f"Embedding with {model} (dim={embedder.dim}) ...")
    embeddings = embedder.embed([c.text for c in chunks])

    print(f"Writing index to {index_dir} ...")
    write_index(
        index_dir,
        docs=docs,
        chunks=chunks,
        embeddings=embeddings,
        corpus_root=corpus_root,
        embedding_backend=backend,
        embedding_model=model,
        embedding_dim=embedder.dim,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(
        f"\nDone.\n"
        f"  Documents : {len(docs)}\n"
        f"  Chunks    : {len(chunks)}\n"
        f"  Model     : {model}\n"
        f"  Index dir : {index_dir}"
    )


def cmd_retrieve(args):
    from tiny_rag_lab.bm25 import BM25Retriever
    from tiny_rag_lab.hybrid import retrieve_hybrid
    from tiny_rag_lab.index_loader import load_index
    from tiny_rag_lab.retrieval import retrieve

    index = load_index(Path(args.index_dir))
    retriever = getattr(args, "retriever", "dense")

    if retriever == "bm25":
        embedder = None
        results = BM25Retriever(index.chunks).retrieve(args.query, top_k=args.top_k)
    elif retriever == "hybrid":
        model_name = index.manifest.get("embedding_model")
        embedder = _make_embedder(model_name)
        results = retrieve_hybrid(args.query, index, embedder, top_k=args.top_k)
    else:
        model_name = index.manifest.get("embedding_model")
        embedder = _make_embedder(model_name)
        results = retrieve(args.query, index, embedder, top_k=args.top_k)

    if not results:
        print("No results found.")
        return

    print(f"Top {len(results)} result(s) for: {args.query!r}\n")
    for r in results:
        title = r.chunk.metadata.get("title", "")
        path = r.chunk.metadata.get("path", r.chunk.doc_id)
        preview = r.chunk.text[:200].replace("\n", " ").strip()
        if len(r.chunk.text) > 200:
            preview += " ..."
        print(f"Rank {r.rank}  score={r.score:.4f}  chunk_id={r.chunk.chunk_id}")
        print(f"  Title : {title}")
        print(f"  Path  : {path}")
        print(f"  {preview}")
        print()


def cmd_ask(args):
    from tiny_rag_lab.index_loader import load_index
    from tiny_rag_lab.models import RagTrace
    from tiny_rag_lab.prompting import assemble_prompt, format_source_table
    from tiny_rag_lab.retrieval import retrieve_by_vector

    index = load_index(Path(args.index_dir))
    model_name = index.manifest.get("embedding_model")
    embedder = _make_embedder(model_name)
    generator = _make_generator(args)

    # Stage: embed query
    t0 = time.perf_counter()
    query_vec = embedder.embed([args.query])[0]
    t_embed = time.perf_counter() - t0

    # Stage: retrieve
    t1 = time.perf_counter()
    results = retrieve_by_vector(query_vec, index, top_k=args.top_k)
    t_retrieve = time.perf_counter() - t1

    # Assemble prompt and generate
    prompt = assemble_prompt(args.query, results)

    t2 = time.perf_counter()
    answer = generator.generate(prompt)
    t_generate = time.perf_counter() - t2

    citations = _CITATION_RE.findall(answer)

    trace = RagTrace(
        query=args.query,
        retrieved_chunks=results,
        prompt=prompt,
        answer=answer,
        citations=citations,
        latency_by_stage={
            "embed": t_embed,
            "retrieve": t_retrieve,
            "generate": t_generate,
        },
    )

    print(trace.answer)
    print()
    print(format_source_table(results))
    print()
    print(
        f"Timings:"
        f"  embed={trace.latency_by_stage['embed']:.3f}s"
        f"  retrieve={trace.latency_by_stage['retrieve']:.3f}s"
        f"  generate={trace.latency_by_stage['generate']:.3f}s"
    )


def cmd_eval(args):
    from pathlib import Path

    from tiny_rag_lab.eval import format_eval_report, load_eval_samples, run_retrieval_eval
    from tiny_rag_lab.index_loader import load_index

    index = load_index(Path(args.index_dir))
    model_name = index.manifest.get("embedding_model")
    embedder = _make_embedder(model_name)

    samples = load_eval_samples(Path(args.qa_file))
    report = run_retrieval_eval(samples, index, embedder, top_k=args.top_k)
    print(format_eval_report(report))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="rag",
        description="tiny-rag-lab: a learning-first RAG engine",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # rag index
    p_index = subparsers.add_parser(
        "index",
        help="build a local index from a corpus directory",
    )
    p_index.add_argument(
        "--corpus", required=True, metavar="PATH",
        help="path to corpus directory",
    )
    p_index.add_argument(
        "--index-dir", default=".tiny-rag/index", metavar="PATH",
        help="where to write index artifacts (default: .tiny-rag/index)",
    )
    p_index.add_argument(
        "--chunk-size", type=int, default=800, metavar="INT",
        help="chunk size in characters (default: 800)",
    )
    p_index.add_argument(
        "--chunk-overlap", type=int, default=120, metavar="INT",
        help="overlap between consecutive chunks in characters (default: 120)",
    )
    p_index.set_defaults(func=cmd_index)

    # rag retrieve
    p_retrieve = subparsers.add_parser(
        "retrieve",
        help="show ranked chunks for a query",
    )
    p_retrieve.add_argument("query", help="query text")
    p_retrieve.add_argument(
        "--index-dir", default=".tiny-rag/index", metavar="PATH",
        help="path to index directory (default: .tiny-rag/index)",
    )
    p_retrieve.add_argument(
        "--top-k", type=int, default=5, metavar="INT",
        help="number of chunks to retrieve (default: 5)",
    )
    p_retrieve.add_argument(
        "--retriever", choices=["dense", "bm25", "hybrid"], default="dense",
        help="retrieval strategy (default: dense)",
    )
    p_retrieve.set_defaults(func=cmd_retrieve)

    # rag ask
    p_ask = subparsers.add_parser(
        "ask",
        help="run the full RAG pipeline and return a grounded answer",
    )
    p_ask.add_argument("query", help="query text")
    p_ask.add_argument(
        "--index-dir", default=".tiny-rag/index", metavar="PATH",
        help="path to index directory (default: .tiny-rag/index)",
    )
    p_ask.add_argument(
        "--top-k", type=int, default=5, metavar="INT",
        help="number of chunks to retrieve (default: 5)",
    )
    p_ask.add_argument(
        "--model", default=None, metavar="NAME",
        help="generation model name (default: env OPENAI_MODEL or gpt-4o-mini)",
    )
    p_ask.add_argument(
        "--api-key", default=None, metavar="KEY",
        help="OpenAI API key (default: env OPENAI_API_KEY)",
    )
    p_ask.add_argument(
        "--base-url", default=None, metavar="URL",
        help="OpenAI-compatible base URL (default: env OPENAI_BASE_URL)",
    )
    p_ask.set_defaults(func=cmd_ask)

    # rag eval
    p_eval = subparsers.add_parser(
        "eval",
        help="evaluate retrieval quality against a qa.jsonl file",
    )
    p_eval.add_argument(
        "--qa-file", required=True, metavar="PATH",
        help="path to qa.jsonl evaluation file",
    )
    p_eval.add_argument(
        "--index-dir", default=".tiny-rag/index", metavar="PATH",
        help="path to index directory (default: .tiny-rag/index)",
    )
    p_eval.add_argument(
        "--top-k", type=int, default=5, metavar="INT",
        help="number of chunks to retrieve per question (default: 5)",
    )
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
