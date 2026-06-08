import argparse
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
    from tiny_rag_lab.index_loader import load_index
    from tiny_rag_lab.retrieval import retrieve

    index = load_index(Path(args.index_dir))
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
    raise NotImplementedError("rag ask: not yet implemented (P1-T18)")


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
    p_ask.set_defaults(func=cmd_ask)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
