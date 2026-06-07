import pytest

from tiny_rag_lab.cli import build_parser


def _help_exits_zero(args):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(args)
    assert exc.value.code == 0


def test_top_level_help():
    _help_exits_zero(["--help"])


def test_index_help():
    _help_exits_zero(["index", "--help"])


def test_retrieve_help():
    _help_exits_zero(["retrieve", "--help"])


def test_ask_help():
    _help_exits_zero(["ask", "--help"])


def test_index_parses_args():
    parser = build_parser()
    args = parser.parse_args(
        ["index", "--corpus", "corpus/watsonx-docsqa", "--chunk-size", "400"]
    )
    assert args.corpus == "corpus/watsonx-docsqa"
    assert args.chunk_size == 400
    assert args.chunk_overlap == 120  # default
    assert args.index_dir == ".tiny-rag/index"  # default


def test_retrieve_parses_args():
    parser = build_parser()
    args = parser.parse_args(["retrieve", "what is watsonx?", "--top-k", "3"])
    assert args.query == "what is watsonx?"
    assert args.top_k == 3


def test_ask_parses_args():
    parser = build_parser()
    args = parser.parse_args(["ask", "what is watsonx?"])
    assert args.query == "what is watsonx?"
    assert args.top_k == 5  # default
