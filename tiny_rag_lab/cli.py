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


def _make_reranker(name: str, model: str | None):
    """Create a Reranker from CLI name and optional model.

    Returns None for name="none" so the existing flow runs unchanged.
    For "cross-encoder", returns a CrossEncoderReranker (lazy — no I/O
    here). Isolated so tests can patch it with FakeReranker.
    """
    if name == "none":
        return None
    if name == "cross-encoder":
        from tiny_rag_lab.reranker import CrossEncoderReranker
        return CrossEncoderReranker(model_name=model or None)
    raise ValueError(f"unknown reranker: {name!r}")


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
    from tiny_rag_lab.retrieval import retrieve_by_vector
    from tiny_rag_lab.reranker import chunk_traces_from_rerank
    from tiny_rag_lab.trace import RetrieveTrace, format_retrieve_trace, write_trace_json

    t0 = time.perf_counter()
    index = load_index(Path(args.index_dir))
    latency: dict[str, float] = {"load": time.perf_counter() - t0}

    retriever = getattr(args, "retriever", "dense")
    reranker_name = getattr(args, "reranker", "none")
    rerank_top_n = getattr(args, "rerank_top_n", 20)
    reranker_model = getattr(args, "reranker_model", None)

    # Validate reranker flags.
    if reranker_name == "none" and reranker_model is not None:
        raise ValueError(
            f"--reranker-model is only valid with --reranker cross-encoder, "
            f"got --reranker {reranker_name}"
        )
    if reranker_name != "none" and rerank_top_n < 1:
        raise ValueError(f"--rerank-top-n must be >= 1, got {rerank_top_n}")
    if reranker_name != "none" and rerank_top_n < args.top_k:
        raise ValueError(
            f"--rerank-top-n ({rerank_top_n}) must be >= --top-k ({args.top_k}) "
            f"when --reranker is active"
        )

    # Build reranker (None when "none" → existing flow).
    reranker = _make_reranker(reranker_name, reranker_model)

    # Base retriever fetches rerank_top_n when reranker is active, else top_k.
    retrieval_k = rerank_top_n if reranker is not None else args.top_k

    if retriever == "bm25":
        t0 = time.perf_counter()
        results = BM25Retriever(index.chunks).retrieve(args.query, top_k=retrieval_k)
        latency["retrieve"] = time.perf_counter() - t0
    elif retriever == "hybrid":
        # Time embed and retrieve separately for hybrid by calling the two
        # stages directly instead of retrieve_hybrid (which folds them together).
        from tiny_rag_lab.hybrid import reciprocal_rank_fusion
        model_name = index.manifest.get("embedding_model")
        embedder = _make_embedder(model_name)
        t0 = time.perf_counter()
        query_vec = embedder.embed([args.query])[0]
        latency["embed"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        bm25_retriever = BM25Retriever(index.chunks)
        dense_results = retrieve_by_vector(query_vec, index, top_k=retrieval_k)
        bm25_results = bm25_retriever.retrieve(args.query, top_k=retrieval_k)
        results = reciprocal_rank_fusion(
            [dense_results, bm25_results], top_k=retrieval_k
        )
        latency["retrieve"] = time.perf_counter() - t0
    else:  # dense
        model_name = index.manifest.get("embedding_model")
        embedder = _make_embedder(model_name)
        t0 = time.perf_counter()
        query_vec = embedder.embed([args.query])[0]
        latency["embed"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        results = retrieve_by_vector(query_vec, index, top_k=retrieval_k)
        latency["retrieve"] = time.perf_counter() - t0

    # Phase 1.9: rerank when a reranker is active.
    rerank_audit = None
    if reranker is not None:
        from tiny_rag_lab.reranker import apply_reranker
        t0 = time.perf_counter()
        results, rerank_audit = apply_reranker(
            args.query, results, reranker, args.top_k,
        )
        latency["rerank"] = time.perf_counter() - t0

    chunks = chunk_traces_from_rerank(results, rerank_audit)
    trace = RetrieveTrace(
        query=args.query,
        retriever=retriever,
        top_k=args.top_k,
        chunks=chunks,
        latency_by_stage=latency,
        reranker=reranker.name if reranker else "none",
        rerank_top_n=rerank_top_n if reranker else None,
    )

    print(format_retrieve_trace(trace))

    trace_out = getattr(args, "trace_out", None)
    if trace_out:
        write_trace_json(trace, Path(trace_out))


def cmd_ask(args):
    from tiny_rag_lab.index_loader import load_index
    from tiny_rag_lab.prompting import assemble_prompt
    from tiny_rag_lab.retrieval import retrieve_by_vector
    from tiny_rag_lab.reranker import chunk_traces_from_rerank
    from tiny_rag_lab.trace import AskTrace, ChunkTrace, format_ask_trace, write_trace_json

    t0 = time.perf_counter()
    index = load_index(Path(args.index_dir))
    t_load = time.perf_counter() - t0

    reranker_name = getattr(args, "reranker", "none")
    rerank_top_n = getattr(args, "rerank_top_n", 20)
    reranker_model = getattr(args, "reranker_model", None)

    # Validate reranker flags.
    if reranker_name == "none" and reranker_model is not None:
        raise ValueError(
            f"--reranker-model is only valid with --reranker cross-encoder, "
            f"got --reranker {reranker_name}"
        )
    if reranker_name != "none" and rerank_top_n < 1:
        raise ValueError(f"--rerank-top-n must be >= 1, got {rerank_top_n}")
    if reranker_name != "none" and rerank_top_n < args.top_k:
        raise ValueError(
            f"--rerank-top-n ({rerank_top_n}) must be >= --top-k ({args.top_k}) "
            f"when --reranker is active"
        )

    reranker = _make_reranker(reranker_name, reranker_model)
    retrieval_k = rerank_top_n if reranker is not None else args.top_k

    model_name = index.manifest.get("embedding_model")
    embedder = _make_embedder(model_name)
    generator = _make_generator(args)

    t0 = time.perf_counter()
    query_vec = embedder.embed([args.query])[0]
    t_embed = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = retrieve_by_vector(query_vec, index, top_k=retrieval_k)
    t_retrieve = time.perf_counter() - t0

    # Phase 1.9: rerank when a reranker is active.
    rerank_audit = None
    t_rerank = 0.0
    if reranker is not None:
        from tiny_rag_lab.reranker import apply_reranker
        t0 = time.perf_counter()
        results, rerank_audit = apply_reranker(
            args.query, results, reranker, args.top_k,
        )
        t_rerank = time.perf_counter() - t0

    t0 = time.perf_counter()
    prompt = assemble_prompt(args.query, results)
    t_prompt_assembly = time.perf_counter() - t0

    t0 = time.perf_counter()
    answer = generator.generate(prompt)
    t_generate = time.perf_counter() - t0

    citations = _CITATION_RE.findall(answer)

    chunks = chunk_traces_from_rerank(results, rerank_audit)

    latency = {
        "load": t_load,
        "embed": t_embed,
        "retrieve": t_retrieve,
        "prompt_assembly": t_prompt_assembly,
        "generate": t_generate,
    }
    if reranker is not None:
        latency["rerank"] = t_rerank

    trace = AskTrace(
        query=args.query,
        retriever="dense",
        top_k=args.top_k,
        chunks=chunks,
        prompt=prompt,
        answer=answer,
        citations=citations,
        latency_by_stage=latency,
        reranker=reranker.name if reranker else "none",
        rerank_top_n=rerank_top_n if reranker else None,
    )

    print(format_ask_trace(trace))

    trace_out = getattr(args, "trace_out", None)
    if trace_out:
        write_trace_json(trace, Path(trace_out))


def cmd_eval(args):
    from pathlib import Path

    from tiny_rag_lab.eval import format_eval_report, load_eval_samples, run_retrieval_eval
    from tiny_rag_lab.index_loader import load_index

    index = load_index(Path(args.index_dir))
    retriever = getattr(args, "retriever", "dense")
    reranker_name = getattr(args, "reranker", "none")
    rerank_top_n = getattr(args, "rerank_top_n", 20)
    reranker_model = getattr(args, "reranker_model", None)

    # Validate reranker flags.
    if reranker_name == "none" and reranker_model is not None:
        raise ValueError(
            f"--reranker-model is only valid with --reranker cross-encoder, "
            f"got --reranker {reranker_name}"
        )
    if reranker_name != "none" and rerank_top_n < 1:
        raise ValueError(f"--rerank-top-n must be >= 1, got {rerank_top_n}")
    if reranker_name != "none" and rerank_top_n < args.top_k:
        raise ValueError(
            f"--rerank-top-n ({rerank_top_n}) must be >= --top-k ({args.top_k}) "
            f"when --reranker is active"
        )

    # Build reranker (None when "none" → existing flow).
    reranker = _make_reranker(reranker_name, reranker_model)

    # Base retriever fetches rerank_top_n when reranker is active, else top_k.
    retrieval_k = rerank_top_n if reranker is not None else args.top_k

    if retriever == "bm25":
        embedder = None
    else:
        model_name = index.manifest.get("embedding_model")
        embedder = _make_embedder(model_name)

    samples = load_eval_samples(Path(args.qa_file))
    report = run_retrieval_eval(
        samples, index, embedder, top_k=args.top_k, retriever=retriever,
        reranker=reranker,
        rerank_top_n=rerank_top_n if reranker else None,
    )
    print(format_eval_report(report))


def cmd_diagnose(args):
    from tiny_rag_lab.failure import (
        format_diagnosis_report,
        load_failure_cases,
        run_diagnosis,
    )
    from tiny_rag_lab.index_loader import load_index

    index = load_index(Path(args.index_dir))
    cases = load_failure_cases(Path(args.cases_file))

    needs_embedder = any(
        c.baseline.retriever in ("dense", "hybrid") or
        c.intervention.retriever in ("dense", "hybrid")
        for c in cases
    )
    embedder = _make_embedder(index.manifest.get("embedding_model")) if needs_embedder else None

    # Build reranker only when at least one case uses a non-none reranker.
    needs_reranker = any(
        config.reranker != "none"
        for c in cases
        for config in (c.baseline, c.intervention)
    )
    reranker = None
    if needs_reranker:
        # Cross-encoder is the only implemented reranker; use default model.
        reranker = _make_reranker("cross-encoder", None)

    report = run_diagnosis(cases, index, embedder, reranker=reranker)
    print(format_diagnosis_report(report))


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
    p_retrieve.add_argument(
        "--trace-out", default=None, metavar="PATH",
        help="write JSON trace to PATH (optional)",
    )
    p_retrieve.add_argument(
        "--reranker", choices=["none", "cross-encoder"], default="none",
        help="second-pass reranker (default: none)",
    )
    p_retrieve.add_argument(
        "--rerank-top-n", type=int, default=20, metavar="INT",
        help=(
            "candidates to feed the reranker; must be >= top_k "
            "(default: 20; ignored when --reranker none)"
        ),
    )
    p_retrieve.add_argument(
        "--reranker-model", default=None, metavar="NAME",
        help="cross-encoder model name (default: ms-marco-MiniLM-L-6-v2)",
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
    p_ask.add_argument(
        "--trace-out", default=None, metavar="PATH",
        help="write JSON trace to PATH (optional)",
    )
    p_ask.add_argument(
        "--reranker", choices=["none", "cross-encoder"], default="none",
        help="second-pass reranker (default: none)",
    )
    p_ask.add_argument(
        "--rerank-top-n", type=int, default=20, metavar="INT",
        help=(
            "candidates to feed the reranker; must be >= top_k "
            "(default: 20; ignored when --reranker none)"
        ),
    )
    p_ask.add_argument(
        "--reranker-model", default=None, metavar="NAME",
        help="cross-encoder model name (default: ms-marco-MiniLM-L-6-v2)",
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
    p_eval.add_argument(
        "--retriever", choices=["dense", "bm25", "hybrid"], default="dense",
        help="retrieval strategy (default: dense)",
    )
    p_eval.add_argument(
        "--reranker", choices=["none", "cross-encoder"], default="none",
        help="second-pass reranker (default: none)",
    )
    p_eval.add_argument(
        "--rerank-top-n", type=int, default=20, metavar="INT",
        help=(
            "candidates to feed the reranker; must be >= top_k "
            "(default: 20; ignored when --reranker none)"
        ),
    )
    p_eval.add_argument(
        "--reranker-model", default=None, metavar="NAME",
        help="cross-encoder model name (default: ms-marco-MiniLM-L-6-v2)",
    )
    p_eval.set_defaults(func=cmd_eval)

    # rag diagnose
    p_diagnose = subparsers.add_parser(
        "diagnose",
        help="run baseline vs. intervention retrieval for curated failure cases",
    )
    p_diagnose.add_argument(
        "--cases-file", required=True, metavar="PATH",
        help="path to failure cases JSONL file",
    )
    p_diagnose.add_argument(
        "--index-dir", default=".tiny-rag/index", metavar="PATH",
        help="path to index directory (default: .tiny-rag/index)",
    )
    p_diagnose.set_defaults(func=cmd_diagnose)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()