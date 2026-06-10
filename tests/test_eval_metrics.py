"""Tests for tiny_rag_lab/eval.py — dataclasses, metric functions, formatter.

T01: dataclass tests (marked with 'dataclass' in name)
T03: metric function tests (marked with 'metric' in name)  — added in T03
T05: formatter tests (marked with 'format' in name)        — added in T05
"""
import dataclasses

import pytest

from tiny_rag_lab.eval import EvalReport, EvalResult, EvalSample


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
