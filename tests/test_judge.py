"""Tests for tiny_rag_lab/judge.py — T01 contracts.

Covers: JudgeVerdict serialization, FakeJudge default and map behavior,
detect_answer_failure_label label assignment, and AnswerDetectionThresholds.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from tiny_rag_lab.judge import (
    AnswerDetectionThresholds,
    FakeJudge,
    JudgeVerdict,
    _default_verdict,
    detect_answer_failure_label,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verdict(
    faithfulness: float = 1.0,
    answer_relevance: float = 1.0,
    citation_support: float = 1.0,
    answer_correctness: float | None = None,
) -> JudgeVerdict:
    return JudgeVerdict(
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        citation_support=citation_support,
        answer_correctness=answer_correctness,
        judge_name="fake",
        latency=0.0,
    )


# ---------------------------------------------------------------------------
# JudgeVerdict serialization
# ---------------------------------------------------------------------------

def test_judge_verdict_asdict_roundtrip():
    v = _verdict(faithfulness=0.8, answer_correctness=0.6)
    d = dataclasses.asdict(v)
    assert d["faithfulness"] == 0.8
    assert d["answer_correctness"] == 0.6
    assert d["judge_name"] == "fake"
    assert d["latency"] == 0.0
    assert d["notes"] == ""


def test_judge_verdict_answer_correctness_none_serializes_as_null():
    v = _verdict(answer_correctness=None)
    d = dataclasses.asdict(v)
    serialized = json.dumps(d)
    parsed = json.loads(serialized)
    assert parsed["answer_correctness"] is None


def test_judge_verdict_answer_correctness_float_serializes():
    v = _verdict(answer_correctness=0.75)
    d = dataclasses.asdict(v)
    serialized = json.dumps(d)
    parsed = json.loads(serialized)
    assert parsed["answer_correctness"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# FakeJudge — default verdict (no map)
# ---------------------------------------------------------------------------

def test_fake_judge_default_no_map():
    judge = FakeJudge()
    result = judge.judge(query="q", context=["ctx"], answer="any answer")
    expected = _default_verdict()
    assert result.faithfulness == expected.faithfulness
    assert result.answer_relevance == expected.answer_relevance
    assert result.citation_support == expected.citation_support
    assert result.answer_correctness == expected.answer_correctness
    assert result.judge_name == "fake"


def test_fake_judge_returns_default_for_unrecognized_answer():
    custom = _verdict(faithfulness=0.1)
    judge = FakeJudge(verdict_map={"specific answer": custom})
    result = judge.judge(query="q", context=["ctx"], answer="some other answer")
    assert result.faithfulness == _default_verdict().faithfulness


# ---------------------------------------------------------------------------
# FakeJudge — verdict_map keyed by answer string
# ---------------------------------------------------------------------------

def test_fake_judge_verdict_map_hit():
    low_faith = _verdict(faithfulness=0.2)
    judge = FakeJudge(verdict_map={"bad answer": low_faith})
    result = judge.judge(query="q", context=["ctx"], answer="bad answer")
    assert result.faithfulness == pytest.approx(0.2)


def test_fake_judge_verdict_map_miss_returns_default():
    low_faith = _verdict(faithfulness=0.2)
    judge = FakeJudge(verdict_map={"bad answer": low_faith})
    result = judge.judge(query="q", context=["ctx"], answer="good answer")
    assert result.faithfulness == pytest.approx(_default_verdict().faithfulness)


def test_fake_judge_verdict_map_keyed_by_answer_not_query():
    """Two different queries with the same answer string return same verdict."""
    custom = _verdict(faithfulness=0.3)
    judge = FakeJudge(verdict_map={"the answer": custom})
    r1 = judge.judge(query="question A", context=[], answer="the answer")
    r2 = judge.judge(query="question B", context=[], answer="the answer")
    assert r1.faithfulness == pytest.approx(0.3)
    assert r2.faithfulness == pytest.approx(0.3)


def test_fake_judge_is_deterministic():
    judge = FakeJudge()
    r1 = judge.judge(query="q", context=["c"], answer="a")
    r2 = judge.judge(query="q", context=["c"], answer="a")
    assert dataclasses.asdict(r1) == dataclasses.asdict(r2)


# ---------------------------------------------------------------------------
# detect_answer_failure_label
# ---------------------------------------------------------------------------

def test_detect_low_faithfulness_returns_unsupported_answer():
    from tiny_rag_lab.failure import LABEL_UNSUPPORTED_ANSWER
    v = _verdict(faithfulness=0.3, citation_support=0.9)
    label = detect_answer_failure_label(v)
    assert label == LABEL_UNSUPPORTED_ANSWER


def test_detect_low_citation_support_returns_citation_mismatch():
    from tiny_rag_lab.failure import LABEL_CITATION_MISMATCH
    v = _verdict(faithfulness=0.8, citation_support=0.3)
    label = detect_answer_failure_label(v)
    assert label == LABEL_CITATION_MISMATCH


def test_detect_all_good_returns_no_failure():
    from tiny_rag_lab.failure import LABEL_NO_FAILURE
    v = _verdict(faithfulness=0.9, citation_support=0.9)
    label = detect_answer_failure_label(v)
    assert label == LABEL_NO_FAILURE


def test_detect_faithfulness_at_threshold_boundary_is_passing():
    """faithfulness == threshold is not below threshold, so no failure."""
    from tiny_rag_lab.failure import LABEL_NO_FAILURE
    thresholds = AnswerDetectionThresholds(faithfulness_threshold=0.5)
    v = _verdict(faithfulness=0.5, citation_support=0.9)
    label = detect_answer_failure_label(v, thresholds)
    assert label == LABEL_NO_FAILURE


def test_detect_faithfulness_below_threshold_fires_first():
    """faithfulness failure takes priority over citation_support failure."""
    from tiny_rag_lab.failure import LABEL_UNSUPPORTED_ANSWER
    v = _verdict(faithfulness=0.1, citation_support=0.1)
    label = detect_answer_failure_label(v)
    assert label == LABEL_UNSUPPORTED_ANSWER


def test_detect_custom_thresholds():
    from tiny_rag_lab.failure import LABEL_UNSUPPORTED_ANSWER
    thresholds = AnswerDetectionThresholds(faithfulness_threshold=0.9)
    v = _verdict(faithfulness=0.8, citation_support=0.9)
    label = detect_answer_failure_label(v, thresholds)
    assert label == LABEL_UNSUPPORTED_ANSWER


def test_detect_default_thresholds_used_when_none():
    """Calling with thresholds=None uses AnswerDetectionThresholds defaults."""
    from tiny_rag_lab.failure import LABEL_UNSUPPORTED_ANSWER
    v = _verdict(faithfulness=0.3)
    assert detect_answer_failure_label(v, None) == LABEL_UNSUPPORTED_ANSWER


# ---------------------------------------------------------------------------
# EvalSample back-compat (tested here for convenience; also covered in test_eval_runner.py)
# ---------------------------------------------------------------------------

def test_eval_sample_new_fields_have_defaults():
    from tiny_rag_lab.eval import EvalSample
    s = EvalSample(question_id="q1", question="Q?", answer="A", gold_doc_ids=["d1"])
    assert s.reference_answer is None
    assert s.expected_facts == []


def test_eval_sample_accepts_reference_answer_and_expected_facts():
    from tiny_rag_lab.eval import EvalSample
    s = EvalSample(
        question_id="q1",
        question="Q?",
        answer="A",
        gold_doc_ids=["d1"],
        reference_answer="The correct answer.",
        expected_facts=["fact 1", "fact 2"],
    )
    assert s.reference_answer == "The correct answer."
    assert s.expected_facts == ["fact 1", "fact 2"]


# ---------------------------------------------------------------------------
# AnswerEvalReport / AnswerEvalResult existence check (dataclass sanity)
# ---------------------------------------------------------------------------

def test_answer_eval_report_default_fields():
    from tiny_rag_lab.eval import AnswerEvalReport
    r = AnswerEvalReport(n_questions=0)
    assert r.judge == "none"
    assert r.mean_faithfulness == 0.0
    assert r.mean_answer_correctness is None
    assert r.per_question == []


def test_answer_eval_result_default_fields():
    from tiny_rag_lab.eval import AnswerEvalResult
    r = AnswerEvalResult(question_id="q1", question="Q?")
    assert r.verdict is None
