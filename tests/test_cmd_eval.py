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


def _eval_args(qa_file, index_dir, top_k=3, retriever="dense"):
    return build_parser().parse_args([
        "eval",
        "--qa-file", str(qa_file),
        "--index-dir", str(index_dir),
        "--top-k", str(top_k),
        "--retriever", retriever,
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


# ---------------------------------------------------------------------------
# T04 (Phase 1.5) — --retriever flag
# ---------------------------------------------------------------------------

def test_eval_help_shows_retriever_flag(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["eval", "--help"])
    assert "--retriever" in capsys.readouterr().out


def test_eval_parser_retriever_default_is_dense():
    args = build_parser().parse_args(["eval", "--qa-file", "qa.jsonl"])
    assert args.retriever == "dense"


def test_eval_parser_invalid_retriever_exits_nonzero():
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["eval", "--qa-file", "qa.jsonl", "--retriever", "bad"])
    assert exc_info.value.code != 0


def test_eval_bm25_prints_all_metric_labels(eval_setup, capsys):
    args = _eval_args(FIXTURE_QA, eval_setup, retriever="bm25")
    # BM25 path must not call _make_embedder
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=RuntimeError("no embedder for bm25")):
        cmd_eval(args)
    out = capsys.readouterr().out
    assert "Hit Rate" in out
    assert "MRR" in out
    assert "Context Precision" in out
    assert "Context Recall" in out


def test_eval_hybrid_prints_all_metric_labels(eval_setup, capsys):
    args = _eval_args(FIXTURE_QA, eval_setup, retriever="hybrid")
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_eval(args)
    out = capsys.readouterr().out
    assert "Hit Rate" in out
    assert "MRR" in out
    assert "Context Precision" in out
    assert "Context Recall" in out


def test_eval_bm25_report_header_contains_retriever_name(eval_setup, capsys):
    args = _eval_args(FIXTURE_QA, eval_setup, retriever="bm25")
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=RuntimeError("no embedder for bm25")):
        cmd_eval(args)
    assert "retriever=bm25" in capsys.readouterr().out


def test_eval_hybrid_report_header_contains_retriever_name(eval_setup, capsys):
    args = _eval_args(FIXTURE_QA, eval_setup, retriever="hybrid")
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_eval(args)
    assert "retriever=hybrid" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# P2.0-T03 — --judge flag on cmd_eval
# ---------------------------------------------------------------------------

from tiny_rag_lab.judge import FakeJudge
from tiny_rag_lab.generation import FakeGenerator


def _eval_args_with_judge(qa_file, index_dir, judge="fake", generator="fake", top_k=3):
    return build_parser().parse_args([
        "eval",
        "--qa-file", str(qa_file),
        "--index-dir", str(index_dir),
        "--top-k", str(top_k),
        "--judge", judge,
        "--generator", generator,
    ])


def test_eval_parser_judge_flag_default_is_none():
    args = build_parser().parse_args([
        "eval", "--qa-file", "f.jsonl", "--index-dir", "d",
    ])
    assert args.judge == "none"


def test_eval_parser_generator_flag_default_is_openai():
    args = build_parser().parse_args([
        "eval", "--qa-file", "f.jsonl", "--index-dir", "d",
    ])
    assert args.generator == "openai"


def test_eval_parser_judge_choices():
    for choice in ("none", "fake", "openai"):
        args = build_parser().parse_args([
            "eval", "--qa-file", "f.jsonl", "--index-dir", "d", "--judge", choice,
        ])
        assert args.judge == choice


def test_eval_help_shows_judge_flag(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["eval", "--help"])
    assert "--judge" in capsys.readouterr().out


def test_eval_judge_none_output_has_no_answer_section(eval_setup, capsys):
    """--judge none produces only the retrieval report (identical to Phase 1.9)."""
    args = _eval_args(FIXTURE_QA, eval_setup)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_eval(args)
    out = capsys.readouterr().out
    assert "Evaluation report" in out
    assert "Answer quality report" not in out


def test_eval_judge_fake_prints_both_sections(eval_setup, capsys):
    """--judge fake --generator fake exits 0 and stdout contains both sections."""
    args = _eval_args_with_judge(FIXTURE_QA, eval_setup)
    with (
        patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()),
        patch("tiny_rag_lab.cli._make_judge", return_value=FakeJudge()),
        patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=FakeGenerator()),
    ):
        cmd_eval(args)
    out = capsys.readouterr().out
    assert "Evaluation report" in out
    assert "Answer quality report" in out


def test_eval_judge_fake_answer_section_contains_metrics(eval_setup, capsys):
    args = _eval_args_with_judge(FIXTURE_QA, eval_setup)
    with (
        patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()),
        patch("tiny_rag_lab.cli._make_judge", return_value=FakeJudge()),
        patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=FakeGenerator()),
    ):
        cmd_eval(args)
    out = capsys.readouterr().out
    assert "Faithfulness" in out
    assert "Answer Relevance" in out
    assert "Citation Support" in out


# ---------------------------------------------------------------------------
# P2.1-T04 — --context-budget flag on rag eval
# ---------------------------------------------------------------------------

def test_eval_parser_context_budget_default_is_zero():
    args = build_parser().parse_args(["eval", "--qa-file", "q.jsonl", "--index-dir", "d"])
    assert args.context_budget == 0


def test_eval_help_shows_context_budget_flag(capsys):
    try:
        build_parser().parse_args(["eval", "--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "--context-budget" in out


def test_eval_context_budget_8192_with_judge_exits_zero(eval_setup, capsys):
    """--context-budget 8192 --judge fake --generator fake exits 0."""
    args = build_parser().parse_args([
        "eval",
        "--qa-file", str(FIXTURE_QA),
        "--index-dir", str(eval_setup),
        "--judge", "fake",
        "--generator", "fake",
        "--context-budget", "8192",
    ])
    with (
        patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()),
        patch("tiny_rag_lab.cli._make_judge", return_value=FakeJudge()),
        patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=FakeGenerator()),
    ):
        cmd_eval(args)
    out = capsys.readouterr().out
    assert "Answer quality report" in out


def test_eval_context_budget_zero_output_identical_to_no_budget(eval_setup, capsys):
    """--context-budget 0 output identical to Phase 2.0 (no packing)."""
    def _run(budget_args):
        args = build_parser().parse_args([
            "eval",
            "--qa-file", str(FIXTURE_QA),
            "--index-dir", str(eval_setup),
            "--judge", "fake",
            "--generator", "fake",
        ] + budget_args)
        with (
            patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()),
            patch("tiny_rag_lab.cli._make_judge", return_value=FakeJudge()),
            patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=FakeGenerator()),
        ):
            cmd_eval(args)
        return capsys.readouterr().out

    out_default = _run([])
    out_zero = _run(["--context-budget", "0"])
    assert out_default == out_zero


def test_eval_negative_context_budget_raises(eval_setup):
    """--context-budget -1 raises ValueError (with --judge fake)."""
    args = build_parser().parse_args([
        "eval",
        "--qa-file", str(FIXTURE_QA),
        "--index-dir", str(eval_setup),
        "--judge", "fake",
        "--generator", "fake",
        "--context-budget", "-1",
    ])
    with (
        patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()),
        patch("tiny_rag_lab.cli._make_judge", return_value=FakeJudge()),
        patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=FakeGenerator()),
        pytest.raises(ValueError, match="context-budget"),
    ):
        cmd_eval(args)


def test_eval_negative_context_budget_raises_without_judge(eval_setup):
    """--context-budget -1 raises ValueError even with default --judge none."""
    args = build_parser().parse_args([
        "eval",
        "--qa-file", str(FIXTURE_QA),
        "--index-dir", str(eval_setup),
        "--context-budget", "-1",
    ])
    with (
        patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()),
        pytest.raises(ValueError, match="context-budget"),
    ):
        cmd_eval(args)
