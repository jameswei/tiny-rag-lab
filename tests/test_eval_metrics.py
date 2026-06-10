"""Tests for tiny_rag_lab/eval.py — dataclasses, metric functions, formatter.

T01: dataclass tests (marked with 'dataclass' in name)
T03: metric function tests (marked with 'metric' in name)
T05: formatter tests (marked with 'format' in name)
"""
import dataclasses

import pytest

from tiny_rag_lab.eval import (
    EvalReport,
    EvalResult,
    EvalSample,
    context_precision_at_k,
    context_recall_at_k,
    format_eval_report,
    hit_at_k,
    reciprocal_rank,
)


# ---------------------------------------------------------------------------
# T01 — EvalSample dataclass
# ---------------------------------------------------------------------------

def test_evalsample_dataclass_fields():
    s = EvalSample(
        question_id="q1",
        question="What is RAG?",
        answer="Retrieval-Augmented Generation.",
        gold_doc_ids=["docs/rag.md"],
    )
    assert s.question_id == "q1"
    assert s.question == "What is RAG?"
    assert s.answer == "Retrieval-Augmented Generation."
    assert s.gold_doc_ids == ["docs/rag.md"]


def test_evalsample_dataclass_asdict_roundtrip():
    s = EvalSample(
        question_id="q1",
        question="What is RAG?",
        answer="RAG stands for Retrieval-Augmented Generation.",
        gold_doc_ids=["docs/rag.md", "docs/intro.md"],
    )
    d = dataclasses.asdict(s)
    assert d["question_id"] == "q1"
    assert d["question"] == "What is RAG?"
    assert d["answer"] == "RAG stands for Retrieval-Augmented Generation."
    assert d["gold_doc_ids"] == ["docs/rag.md", "docs/intro.md"]


def test_evalsample_dataclass_gold_doc_ids_is_list():
    s = EvalSample(question_id="q1", question="Q?", answer="A.", gold_doc_ids=[])
    assert isinstance(s.gold_doc_ids, list)


def test_evalsample_dataclass_gold_doc_ids_defaults_to_empty():
    s = EvalSample(question_id="q1", question="Q?", answer="A.")
    assert s.gold_doc_ids == []


# ---------------------------------------------------------------------------
# T01 — EvalResult dataclass
# ---------------------------------------------------------------------------

def test_evalresult_dataclass_defaults():
    r = EvalResult(
        question_id="q1",
        question="Q?",
        gold_doc_ids=["docs/a.md"],
        retrieved_doc_ids=["docs/b.md"],
    )
    assert r.hit is False
    assert r.reciprocal_rank == 0.0
    assert r.context_precision == 0.0
    assert r.context_recall == 0.0


def test_evalresult_dataclass_fields():
    r = EvalResult(
        question_id="q1",
        question="Q?",
        gold_doc_ids=["docs/a.md"],
        retrieved_doc_ids=["docs/a.md", "docs/b.md"],
        hit=True,
        reciprocal_rank=1.0,
        context_precision=0.5,
        context_recall=1.0,
    )
    assert r.hit is True
    assert r.reciprocal_rank == 1.0
    assert r.context_precision == 0.5
    assert r.context_recall == 1.0


def test_evalresult_dataclass_asdict_roundtrip():
    r = EvalResult(
        question_id="q2",
        question="Q?",
        gold_doc_ids=["docs/a.md"],
        retrieved_doc_ids=["docs/a.md"],
        hit=True,
        reciprocal_rank=1.0,
        context_precision=1.0,
        context_recall=1.0,
    )
    d = dataclasses.asdict(r)
    assert d["question_id"] == "q2"
    assert d["hit"] is True
    assert d["reciprocal_rank"] == 1.0


def test_evalresult_dataclass_retrieved_doc_ids_is_list():
    r = EvalResult(
        question_id="q1", question="Q?",
        gold_doc_ids=[], retrieved_doc_ids=[],
    )
    assert isinstance(r.retrieved_doc_ids, list)


def test_evalresult_dataclass_list_fields_default_to_empty():
    r = EvalResult(question_id="q1", question="Q?")
    assert r.gold_doc_ids == []
    assert r.retrieved_doc_ids == []


# ---------------------------------------------------------------------------
# T01 — EvalReport dataclass
# ---------------------------------------------------------------------------

def test_evalreport_dataclass_defaults():
    report = EvalReport(n_questions=10, top_k=5)
    assert report.hit_rate == 0.0
    assert report.mrr == 0.0
    assert report.mean_context_precision == 0.0
    assert report.mean_context_recall == 0.0
    assert report.per_question == []


def test_evalreport_dataclass_per_question_is_list():
    report = EvalReport(n_questions=0, top_k=5)
    assert isinstance(report.per_question, list)


def test_evalreport_dataclass_asdict_roundtrip():
    report = EvalReport(
        n_questions=2,
        top_k=3,
        hit_rate=0.5,
        mrr=0.333,
        mean_context_precision=0.4,
        mean_context_recall=0.6,
        per_question=[],
    )
    d = dataclasses.asdict(report)
    assert d["n_questions"] == 2
    assert d["top_k"] == 3
    assert d["hit_rate"] == 0.5
    assert d["mrr"] == 0.333


def test_evalreport_dataclass_per_question_stores_evalresult():
    result = EvalResult(
        question_id="q1", question="Q?",
        gold_doc_ids=["a.md"], retrieved_doc_ids=["a.md"],
        hit=True, reciprocal_rank=1.0,
    )
    report = EvalReport(n_questions=1, top_k=5, per_question=[result])
    assert len(report.per_question) == 1
    assert report.per_question[0].question_id == "q1"


# ---------------------------------------------------------------------------
# T03 — hit_at_k
# ---------------------------------------------------------------------------

def test_metric_hit_at_k_true_when_gold_in_retrieved():
    assert hit_at_k(["a", "b"], ["b"]) is True


def test_metric_hit_at_k_false_when_no_gold_in_retrieved():
    assert hit_at_k(["a", "b"], ["c"]) is False


def test_metric_hit_at_k_false_for_empty_retrieved():
    assert hit_at_k([], ["a"]) is False


def test_metric_hit_at_k_true_for_first_position():
    assert hit_at_k(["a", "b"], ["a"]) is True


def test_metric_hit_at_k_multiple_gold_docs():
    assert hit_at_k(["x", "a"], ["a", "b"]) is True


# ---------------------------------------------------------------------------
# T03 — reciprocal_rank
# ---------------------------------------------------------------------------

def test_metric_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0


def test_metric_reciprocal_rank_second_position():
    assert reciprocal_rank(["a", "b", "c"], ["b"]) == pytest.approx(0.5)


def test_metric_reciprocal_rank_third_position():
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)


def test_metric_reciprocal_rank_no_hit():
    assert reciprocal_rank(["a", "b"], ["c"]) == 0.0


def test_metric_reciprocal_rank_empty_retrieved():
    assert reciprocal_rank([], ["a"]) == 0.0


def test_metric_reciprocal_rank_uses_first_hit():
    # gold has two docs; first hit is at position 2
    assert reciprocal_rank(["x", "a", "b"], ["a", "b"]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# T03 — context_precision_at_k
# ---------------------------------------------------------------------------

def test_metric_context_precision_all_relevant():
    assert context_precision_at_k(["a", "b"], ["a", "b", "c"]) == pytest.approx(1.0)


def test_metric_context_precision_half_relevant():
    assert context_precision_at_k(["a", "b"], ["a"]) == pytest.approx(0.5)


def test_metric_context_precision_none_relevant():
    assert context_precision_at_k(["a", "b"], ["c"]) == pytest.approx(0.0)


def test_metric_context_precision_empty_retrieved():
    assert context_precision_at_k([], ["a"]) == 0.0


def test_metric_context_precision_counts_duplicate_positions():
    # same doc at two positions: 2 hits out of 3
    assert context_precision_at_k(["a", "a", "b"], ["a"]) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# T03 — context_recall_at_k
# ---------------------------------------------------------------------------

def test_metric_context_recall_all_gold_covered():
    assert context_recall_at_k(["a", "b"], ["a", "b"]) == pytest.approx(1.0)


def test_metric_context_recall_half_gold_covered():
    assert context_recall_at_k(["a", "b"], ["a", "c"]) == pytest.approx(0.5)


def test_metric_context_recall_none_covered():
    assert context_recall_at_k(["x", "y"], ["a", "b"]) == pytest.approx(0.0)


def test_metric_context_recall_empty_gold():
    assert context_recall_at_k(["a"], []) == 0.0


def test_metric_context_recall_deduplicates_retrieved():
    # "a" appears twice in retrieved but counts as one unique coverage
    assert context_recall_at_k(["a", "a"], ["a", "b"]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# T05 — format_eval_report
# ---------------------------------------------------------------------------

def _make_report(**kwargs):
    defaults = dict(n_questions=10, top_k=5, hit_rate=0.723, mrr=0.581,
                    mean_context_precision=0.312, mean_context_recall=0.651)
    defaults.update(kwargs)
    return EvalReport(**defaults)


def test_format_eval_report_contains_hit_rate_label():
    assert "Hit Rate" in format_eval_report(_make_report())


def test_format_eval_report_contains_mrr_label():
    assert "MRR" in format_eval_report(_make_report())


def test_format_eval_report_contains_context_precision_label():
    assert "Context Precision" in format_eval_report(_make_report())


def test_format_eval_report_contains_context_recall_label():
    assert "Context Recall" in format_eval_report(_make_report())


def test_format_eval_report_contains_n_questions():
    out = format_eval_report(_make_report(n_questions=42))
    assert "42" in out


def test_format_eval_report_contains_top_k():
    out = format_eval_report(_make_report(top_k=7))
    assert "7" in out


def test_format_eval_report_values_rounded_to_3_decimal_places():
    out = format_eval_report(_make_report(hit_rate=0.7234567))
    assert "0.723" in out


def test_format_eval_report_no_ansi_escape_codes():
    out = format_eval_report(_make_report())
    assert "\033[" not in out
    assert "\x1b[" not in out


def test_format_eval_report_returns_string():
    assert isinstance(format_eval_report(_make_report()), str)


def test_format_eval_report_contains_retriever_name():
    out = format_eval_report(_make_report(retriever="bm25"))
    assert "retriever=bm25" in out


def test_format_eval_report_retriever_defaults_to_dense():
    out = format_eval_report(_make_report())
    assert "retriever=dense" in out
