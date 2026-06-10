"""Tests for T06 — rag eval CLI command.

Patches _make_embedder with FakeEmbedder so no model download is needed.
"""
from pathlib import Path
from unittest.mock import patch

import pytest

from tiny_rag_lab.cli import build_parser, cmd_eval, cmd_index
from tiny_rag_lab.embeddings import FakeEmbedder

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"
FIXTURE_QA = Path(__file__).parent / "fixtures" / "eval" / "qa.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embedder_factory(dim: int = 8):
    def _make(model_name=None):
        return FakeEmbedder(dim=dim)
    return _make


def _index_args(corpus, index_dir):
    return build_parser().parse_args([
        "index", "--corpus", str(corpus),
        "--index-dir", str(index_dir),
        "--chunk-size", "500", "--chunk-overlap", "50",
    ])


def _eval_args(qa_file, index_dir, top_k=3):
    return build_parser().parse_args([
        "eval",
        "--qa-file", str(qa_file),
        "--index-dir", str(index_dir),
        "--top-k", str(top_k),
    ])


@pytest.fixture()
def eval_setup(tmp_path):
    """Build an index with fake embedder, return index_dir."""
    index_dir = tmp_path / "index"
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_index(_index_args(FIXTURE_CORPUS, index_dir))
    return index_dir


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

def test_eval_help():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["eval", "--help"])
    assert exc.value.code == 0


def test_eval_parser_qa_file_required():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["eval", "--index-dir", ".tiny-rag/index"])
    assert exc.value.code != 0


def test_eval_parser_qa_file_flag():
    args = build_parser().parse_args(["eval", "--qa-file", "qa.jsonl"])
    assert args.qa_file == "qa.jsonl"


def test_eval_parser_index_dir_default():
    args = build_parser().parse_args(["eval", "--qa-file", "qa.jsonl"])
    assert args.index_dir == ".tiny-rag/index"


def test_eval_parser_top_k_default():
    args = build_parser().parse_args(["eval", "--qa-file", "qa.jsonl"])
    assert args.top_k == 5


def test_eval_parser_top_k_flag():
    args = build_parser().parse_args(["eval", "--qa-file", "qa.jsonl", "--top-k", "10"])
    assert args.top_k == 10


# ---------------------------------------------------------------------------
# End-to-end CLI tests
# ---------------------------------------------------------------------------

def test_eval_prints_hit_rate_label(eval_setup, capsys):
    args = _eval_args(FIXTURE_QA, eval_setup)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_eval(args)
    assert "Hit Rate" in capsys.readouterr().out


def test_eval_prints_mrr_label(eval_setup, capsys):
    args = _eval_args(FIXTURE_QA, eval_setup)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_eval(args)
    assert "MRR" in capsys.readouterr().out


def test_eval_prints_context_precision_label(eval_setup, capsys):
    args = _eval_args(FIXTURE_QA, eval_setup)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_eval(args)
    assert "Context Precision" in capsys.readouterr().out


def test_eval_prints_context_recall_label(eval_setup, capsys):
    args = _eval_args(FIXTURE_QA, eval_setup)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_eval(args)
    assert "Context Recall" in capsys.readouterr().out


def test_eval_output_is_nonempty(eval_setup, capsys):
    args = _eval_args(FIXTURE_QA, eval_setup)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_eval(args)
    assert len(capsys.readouterr().out.strip()) > 0
