"""Tests for T06 — rag diagnose CLI command.

Patches _make_embedder with FakeEmbedder so no model download is needed.
"""
import io
from pathlib import Path
from unittest.mock import patch

import pytest

from tiny_rag_lab.cli import build_parser, cmd_diagnose, cmd_index
from tiny_rag_lab.embeddings import FakeEmbedder

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"
FIXTURE_CASES = Path(__file__).parent / "fixtures" / "failure" / "cases.jsonl"
from tiny_rag_lab.reranker import FakeReranker


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


def _diagnose_args(cases_file, index_dir):
    return build_parser().parse_args([
        "diagnose",
        "--cases-file", str(cases_file),
        "--index-dir", str(index_dir),
    ])


@pytest.fixture()
def diagnose_setup(tmp_path):
    """Build an index with FakeEmbedder; return index_dir."""
    index_dir = tmp_path / "index"
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_index(_index_args(FIXTURE_CORPUS, index_dir))
    return index_dir


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

def test_diagnose_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["diagnose", "--help"])
    assert exc.value.code == 0


def test_diagnose_help_shows_cases_file_flag(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["diagnose", "--help"])
    out = capsys.readouterr().out
    assert "--cases-file" in out


def test_diagnose_missing_cases_file_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["diagnose", "--index-dir", ".tiny-rag/index"])
    assert exc.value.code != 0


def test_diagnose_parser_cases_file_flag():
    args = build_parser().parse_args([
        "diagnose", "--cases-file", "cases.jsonl",
    ])
    assert args.cases_file == "cases.jsonl"


def test_diagnose_parser_index_dir_default():
    args = build_parser().parse_args([
        "diagnose", "--cases-file", "cases.jsonl",
    ])
    assert args.index_dir == ".tiny-rag/index"


def test_diagnose_parser_index_dir_custom():
    args = build_parser().parse_args([
        "diagnose", "--cases-file", "cases.jsonl",
        "--index-dir", "/tmp/myindex",
    ])
    assert args.index_dir == "/tmp/myindex"


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------

def test_diagnose_prints_diagnosis_report(diagnose_setup, capsys):
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_reranker", return_value=FakeReranker()):
        cmd_diagnose(_diagnose_args(FIXTURE_CASES, diagnose_setup))
    out = capsys.readouterr().out
    assert "Diagnosis report" in out


def test_diagnose_output_contains_n_cases(diagnose_setup, capsys):
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_reranker", return_value=FakeReranker()):
        cmd_diagnose(_diagnose_args(FIXTURE_CASES, diagnose_setup))
    out = capsys.readouterr().out
    assert "n=9" in out


def test_diagnose_output_contains_case_id(diagnose_setup, capsys):
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_reranker", return_value=FakeReranker()):
        cmd_diagnose(_diagnose_args(FIXTURE_CASES, diagnose_setup))
    out = capsys.readouterr().out
    assert "fc001" in out


def test_diagnose_output_contains_outcome_word(diagnose_setup, capsys):
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_reranker", return_value=FakeReranker()):
        cmd_diagnose(_diagnose_args(FIXTURE_CASES, diagnose_setup))
    out = capsys.readouterr().out
    assert any(word in out for word in ("FIXED", "MOVED", "CONFIRMED", "UNCHANGED"))


def test_diagnose_output_no_ansi_codes(diagnose_setup, capsys):
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_reranker", return_value=FakeReranker()):
        cmd_diagnose(_diagnose_args(FIXTURE_CASES, diagnose_setup))
    out = capsys.readouterr().out
    assert "\x1b" not in out


def test_diagnose_shows_in_rag_help(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    out = capsys.readouterr().out
    assert "diagnose" in out


# ---------------------------------------------------------------------------
# P1.9-T06 — reranker in cmd_diagnose
# ---------------------------------------------------------------------------

def test_diagnose_output_contains_fc007(diagnose_setup, capsys):
    """Diagnose output includes fc007 case."""
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_reranker", return_value=FakeReranker()):
        cmd_diagnose(_diagnose_args(FIXTURE_CASES, diagnose_setup))
    out = capsys.readouterr().out
    assert "fc007" in out


# ---------------------------------------------------------------------------
# P2.0-T05 — --judge flag on rag diagnose
# ---------------------------------------------------------------------------

def _diagnose_judge_args(cases_file, index_dir, judge="none", generator="fake"):
    return build_parser().parse_args([
        "diagnose",
        "--cases-file", str(cases_file),
        "--index-dir", str(index_dir),
        "--judge", judge,
        "--generator", generator,
    ])


def test_diagnose_parser_judge_default_is_none():
    args = build_parser().parse_args([
        "diagnose", "--cases-file", "cases.jsonl",
    ])
    assert args.judge == "none"


def test_diagnose_parser_judge_fake():
    args = build_parser().parse_args([
        "diagnose", "--cases-file", "cases.jsonl", "--judge", "fake",
    ])
    assert args.judge == "fake"


def test_diagnose_parser_generator_default_is_openai():
    args = build_parser().parse_args([
        "diagnose", "--cases-file", "cases.jsonl",
    ])
    assert args.generator == "openai"


def test_diagnose_parser_generator_fake():
    args = build_parser().parse_args([
        "diagnose", "--cases-file", "cases.jsonl",
        "--judge", "fake", "--generator", "fake",
    ])
    assert args.generator == "fake"


def test_diagnose_judge_none_output_identical_retrieval_only(diagnose_setup, capsys):
    """--judge none output contains only retrieval diagnosis (Phase 1.9 compatible)."""
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_reranker", return_value=FakeReranker()):
        cmd_diagnose(_diagnose_judge_args(FIXTURE_CASES, diagnose_setup, judge="none"))
    out = capsys.readouterr().out
    assert "Diagnosis report" in out
    assert "Answer diagnosis report" not in out


def test_diagnose_judge_fake_prints_retrieval_report(diagnose_setup, capsys):
    """--judge fake still prints the retrieval diagnosis report."""
    from tiny_rag_lab.judge import JudgeVerdict, FakeJudge
    fake_judge = FakeJudge()
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_reranker", return_value=FakeReranker()), \
         patch("tiny_rag_lab.cli._make_judge", return_value=fake_judge), \
         patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=None):
        cmd_diagnose(_diagnose_judge_args(FIXTURE_CASES, diagnose_setup, judge="fake"))
    out = capsys.readouterr().out
    assert "Diagnosis report" in out


def test_diagnose_judge_fake_prints_answer_diagnosis_section(diagnose_setup, capsys):
    """--judge fake appends answer diagnosis section after retrieval report."""
    from tiny_rag_lab.judge import JudgeVerdict, FakeJudge
    fc008_baseline = "The document covers Roman history and medieval castles."
    fc008_intervention = "The document covers the topics described in the sample corpus."
    fc009_baseline = "The nested document lives in the root directory [Source: with_h1.md]."
    fc009_intervention = "The nested document lives in a subdirectory [Source: subdir/nested.md]."
    fake_judge = FakeJudge(verdict_map={
        fc008_baseline: JudgeVerdict(faithfulness=0.1, answer_relevance=0.8, citation_support=0.9, answer_correctness=None, judge_name="fake", latency=0.0),
        fc008_intervention: JudgeVerdict(faithfulness=0.9, answer_relevance=0.8, citation_support=0.9, answer_correctness=None, judge_name="fake", latency=0.0),
        fc009_baseline: JudgeVerdict(faithfulness=0.9, answer_relevance=0.8, citation_support=0.2, answer_correctness=None, judge_name="fake", latency=0.0),
        fc009_intervention: JudgeVerdict(faithfulness=0.9, answer_relevance=0.8, citation_support=0.9, answer_correctness=None, judge_name="fake", latency=0.0),
    })
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_reranker", return_value=FakeReranker()), \
         patch("tiny_rag_lab.cli._make_judge", return_value=fake_judge), \
         patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=None):
        cmd_diagnose(_diagnose_judge_args(FIXTURE_CASES, diagnose_setup, judge="fake"))
    out = capsys.readouterr().out
    assert "Answer diagnosis report" in out
    assert "fc008" in out
    assert "fc009" in out


def test_diagnose_judge_fake_answer_report_n2(diagnose_setup, capsys):
    """--judge fake answer section shows n=2 (only fc008+fc009 are answer cases)."""
    from tiny_rag_lab.judge import FakeJudge
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_reranker", return_value=FakeReranker()), \
         patch("tiny_rag_lab.cli._make_judge", return_value=FakeJudge()), \
         patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=None):
        cmd_diagnose(_diagnose_judge_args(FIXTURE_CASES, diagnose_setup, judge="fake"))
    out = capsys.readouterr().out
    assert "Answer diagnosis report  (n=2)" in out