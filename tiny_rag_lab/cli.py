import argparse


def cmd_index(args):
    raise NotImplementedError("rag index: not yet implemented (P1-T13)")


def cmd_retrieve(args):
    raise NotImplementedError("rag retrieve: not yet implemented (P1-T14)")


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
