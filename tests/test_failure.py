"""Tests for tiny_rag_lab/failure.py.

Test naming convention:
  *_dataclass_* — dataclass round-trip and JSON serialization (T01)
  *_load_*      — load_failure_cases loader (T02)
  *_detect_*    — detect_failure_label function (T03)
  *_runner_*    — run_diagnosis runner (T04)
  *_format_*    — format_diagnosis_report formatter (T05)
"""
import dataclasses
import json
from pathlib import Path

import pytest

from tiny_rag_lab.failure import (
    load_failure_cases,
    detect_failure_label,
    LABEL_DISTRACTOR_EVIDENCE,
    LABEL_MISSING_EVIDENCE,
    LABEL_LOW_RANK_EVIDENCE,
    LABEL_NO_FAILURE,
    LABEL_UNANSWERABLE,
    DetectionThresholds,
    DiagnosisReport,
    DiagnosisResult,
    FailureCase,
    RetrieverConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_diagnosis_result(**overrides) -> DiagnosisResult:
    defaults = dict(
        case_id="fc001",
        question="What is X?",
        expected_label=LABEL_MISSING_EVIDENCE,
        baseline_label=LABEL_MISSING_EVIDENCE,
        intervention_label=LABEL_NO_FAILURE,
        baseline_retrieved_doc_ids=["other.md"],
        intervention_retrieved_doc_ids=["gold.md", "other.md"],
        baseline_metrics={"hit": 0.0, "reciprocal_rank": 0.0, "context_precision": 0.0, "context_recall": 0.0},
        intervention_metrics={"hit": 1.0, "reciprocal_rank": 1.0, "context_precision": 0.5, "context_recall": 1.0},
        fixed=True,
        moved=False,
    )
    defaults.update(overrides)
    return DiagnosisResult(**defaults)


# ---------------------------------------------------------------------------
# T01 — dataclass round-trips and JSON serialization
# ---------------------------------------------------------------------------

def test_dataclass_retriever_config_defaults():
    rc = RetrieverConfig()
    assert rc.retriever == "dense"
    assert rc.top_k == 5


def test_dataclass_retriever_config_custom():
    rc = RetrieverConfig(retriever="bm25", top_k=10)
    assert rc.retriever == "bm25"
    assert rc.top_k == 10


def test_dataclass_retriever_config_json_serializable():
    rc = RetrieverConfig(retriever="hybrid", top_k=3)
    d = dataclasses.asdict(rc)
    text = json.dumps(d)
    assert json.loads(text) == {"retriever": "hybrid", "top_k": 3}


def test_dataclass_failure_case_defaults():
    fc = FailureCase(case_id="fc001", question="What is X?")
    assert fc.gold_doc_ids == []
    assert fc.expected_label == LABEL_NO_FAILURE
    assert fc.baseline == RetrieverConfig()
    assert fc.intervention == RetrieverConfig()
    assert fc.notes == ""


def test_dataclass_failure_case_empty_gold_doc_ids_round_trips():
    fc = FailureCase(
        case_id="fc005",
        question="What is the capital of France?",
        gold_doc_ids=[],
        expected_label=LABEL_UNANSWERABLE,
        baseline=RetrieverConfig(retriever="dense", top_k=3),
        intervention=RetrieverConfig(retriever="dense", top_k=3),
        notes="Unanswerable from corpus.",
    )
    d = dataclasses.asdict(fc)
    assert d["gold_doc_ids"] == []
    assert d["expected_label"] == LABEL_UNANSWERABLE
    text = json.dumps(d)
    restored = json.loads(text)
    assert restored["gold_doc_ids"] == []


def test_dataclass_failure_case_json_serializable():
    fc = FailureCase(
        case_id="fc001",
        question="What is X?",
        gold_doc_ids=["doc_a.md"],
        expected_label=LABEL_MISSING_EVIDENCE,
        baseline=RetrieverConfig(retriever="dense", top_k=1),
        intervention=RetrieverConfig(retriever="dense", top_k=4),
        notes="top_k=1 misses the gold doc.",
    )
    text = json.dumps(dataclasses.asdict(fc))
    parsed = json.loads(text)
    assert parsed["case_id"] == "fc001"
    assert parsed["baseline"]["retriever"] == "dense"
    assert parsed["intervention"]["top_k"] == 4


def test_dataclass_failure_case_nested_retriever_config_is_json_native():
    fc = FailureCase(case_id="x", question="q")
    d = dataclasses.asdict(fc)
    # nested RetrieverConfig must produce a plain dict, not a dataclass object
    assert isinstance(d["baseline"], dict)
    assert isinstance(d["intervention"], dict)


def test_dataclass_detection_thresholds_defaults():
    dt = DetectionThresholds()
    assert dt.low_rank_threshold == 3
    assert dt.distractor_precision_threshold == 0.5


def test_dataclass_detection_thresholds_custom():
    dt = DetectionThresholds(low_rank_threshold=5, distractor_precision_threshold=0.3)
    assert dt.low_rank_threshold == 5
    assert dt.distractor_precision_threshold == 0.3


def test_dataclass_detection_thresholds_json_serializable():
    dt = DetectionThresholds(low_rank_threshold=2, distractor_precision_threshold=0.4)
    text = json.dumps(dataclasses.asdict(dt))
    parsed = json.loads(text)
    assert parsed["low_rank_threshold"] == 2
    assert parsed["distractor_precision_threshold"] == 0.4


def test_dataclass_diagnosis_result_fields_present():
    dr = _make_diagnosis_result()
    assert dr.case_id == "fc001"
    assert dr.baseline_retrieved_doc_ids == ["other.md"]
    assert dr.intervention_retrieved_doc_ids == ["gold.md", "other.md"]
    assert isinstance(dr.baseline_metrics, dict)
    assert isinstance(dr.intervention_metrics, dict)
    assert dr.fixed is True
    assert dr.moved is False


def test_dataclass_diagnosis_result_has_retrieved_doc_id_fields():
    # Verify the trace-backed fields exist and are list[str]
    dr = _make_diagnosis_result(
        baseline_retrieved_doc_ids=["a.md", "b.md"],
        intervention_retrieved_doc_ids=["c.md"],
    )
    assert dr.baseline_retrieved_doc_ids == ["a.md", "b.md"]
    assert dr.intervention_retrieved_doc_ids == ["c.md"]


def test_dataclass_diagnosis_result_json_serializable():
    dr = _make_diagnosis_result()
    text = json.dumps(dataclasses.asdict(dr))
    parsed = json.loads(text)
    assert parsed["case_id"] == "fc001"
    assert parsed["baseline_retrieved_doc_ids"] == ["other.md"]
    assert "hit" in parsed["baseline_metrics"]


def test_dataclass_diagnosis_result_fixed_moved_defaults_false():
    dr = DiagnosisResult(
        case_id="x",
        question="q",
        expected_label=LABEL_NO_FAILURE,
        baseline_label=LABEL_NO_FAILURE,
        intervention_label=LABEL_NO_FAILURE,
        baseline_retrieved_doc_ids=[],
        intervention_retrieved_doc_ids=[],
        baseline_metrics={},
        intervention_metrics={},
    )
    assert dr.fixed is False
    assert dr.moved is False


def test_dataclass_diagnosis_report_defaults():
    report = DiagnosisReport(n_cases=0)
    assert report.n_fixed == 0
    assert report.n_moved == 0
    assert report.n_confirmed == 0
    assert report.per_case == []


def test_dataclass_diagnosis_report_json_serializable():
    dr = _make_diagnosis_result()
    report = DiagnosisReport(
        n_cases=1,
        n_fixed=1,
        n_confirmed=1,
        per_case=[dr],
    )
    text = json.dumps(dataclasses.asdict(report))
    parsed = json.loads(text)
    assert parsed["n_cases"] == 1
    assert len(parsed["per_case"]) == 1


def test_dataclass_label_constants_are_strings():
    for label in (
        LABEL_MISSING_EVIDENCE,
        LABEL_LOW_RANK_EVIDENCE,
        LABEL_DISTRACTOR_EVIDENCE,
        LABEL_UNANSWERABLE,
        LABEL_NO_FAILURE,
    ):
        assert isinstance(label, str)
        assert label  # non-empty


def test_dataclass_label_constants_are_distinct():
    labels = [
        LABEL_MISSING_EVIDENCE,
        LABEL_LOW_RANK_EVIDENCE,
        LABEL_DISTRACTOR_EVIDENCE,
        LABEL_UNANSWERABLE,
        LABEL_NO_FAILURE,
    ]
    assert len(set(labels)) == 5


FIXTURE_CASES = Path(__file__).parent / "fixtures" / "failure" / "cases.jsonl"


# ---------------------------------------------------------------------------
# T02 — load_failure_cases
# ---------------------------------------------------------------------------

def test_load_returns_six_cases():
    cases = load_failure_cases(FIXTURE_CASES)
    assert len(cases) == 6


def test_load_all_are_failure_case_instances():
    from tiny_rag_lab.failure import FailureCase
    cases = load_failure_cases(FIXTURE_CASES)
    assert all(isinstance(c, FailureCase) for c in cases)


def test_load_unanswerable_case_not_skipped():
    cases = load_failure_cases(FIXTURE_CASES)
    fc005 = next((c for c in cases if c.case_id == "fc005"), None)
    assert fc005 is not None
    assert fc005.gold_doc_ids == []
    assert fc005.expected_label == LABEL_UNANSWERABLE


def test_load_baseline_deserializes_to_retriever_config():
    cases = load_failure_cases(FIXTURE_CASES)
    for case in cases:
        assert isinstance(case.baseline, RetrieverConfig)
        assert isinstance(case.intervention, RetrieverConfig)
        assert case.baseline.retriever in ("dense", "bm25", "hybrid")
        assert isinstance(case.baseline.top_k, int)


def test_load_case_ids_are_non_empty():
    cases = load_failure_cases(FIXTURE_CASES)
    assert all(c.case_id for c in cases)


def test_load_questions_are_non_empty():
    cases = load_failure_cases(FIXTURE_CASES)
    assert all(c.question for c in cases)


def test_load_skips_empty_case_id(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"case_id": "", "question": "What?"}\n'
        '{"case_id": "fc001", "question": "Real question?"}\n',
        encoding="utf-8",
    )
    cases = load_failure_cases(bad)
    assert len(cases) == 1
    assert cases[0].case_id == "fc001"


def test_load_skips_empty_question(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"case_id": "fc001", "question": ""}\n'
        '{"case_id": "fc002", "question": "Real question?"}\n',
        encoding="utf-8",
    )
    cases = load_failure_cases(bad)
    assert len(cases) == 1
    assert cases[0].case_id == "fc002"


def test_load_skips_malformed_json(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        'not valid json\n'
        '{"case_id": "fc001", "question": "Valid question?"}\n',
        encoding="utf-8",
    )
    cases = load_failure_cases(bad)
    assert len(cases) == 1


def test_load_defaults_when_baseline_absent(tmp_path):
    f = tmp_path / "cases.jsonl"
    f.write_text(
        '{"case_id": "fc001", "question": "Q?"}\n',
        encoding="utf-8",
    )
    cases = load_failure_cases(f)
    assert cases[0].baseline == RetrieverConfig()
    assert cases[0].intervention == RetrieverConfig()
    assert cases[0].gold_doc_ids == []
    assert cases[0].expected_label == LABEL_NO_FAILURE
    assert cases[0].notes == ""


def test_load_empty_gold_doc_ids_not_skipped(tmp_path):
    f = tmp_path / "cases.jsonl"
    f.write_text(
        '{"case_id": "fc001", "question": "Q?", "gold_doc_ids": []}\n',
        encoding="utf-8",
    )
    cases = load_failure_cases(f)
    assert len(cases) == 1
    assert cases[0].gold_doc_ids == []


# ---------------------------------------------------------------------------
# T03 — detect_failure_label
# ---------------------------------------------------------------------------

def test_detect_missing_evidence_no_hit():
    label = detect_failure_label(
        retrieved_doc_ids=["other.md"],
        gold_doc_ids=["gold.md"],
        expected_label=LABEL_MISSING_EVIDENCE,
    )
    assert label == LABEL_MISSING_EVIDENCE


def test_detect_no_failure_gold_at_rank_1():
    label = detect_failure_label(
        retrieved_doc_ids=["gold.md", "other.md"],
        gold_doc_ids=["gold.md"],
        expected_label=LABEL_NO_FAILURE,
    )
    assert label == LABEL_NO_FAILURE


def test_detect_low_rank_evidence_gold_at_rank_4():
    # gold at rank 4 > default threshold 3 → low_rank_evidence
    label = detect_failure_label(
        retrieved_doc_ids=["a.md", "b.md", "c.md", "gold.md"],
        gold_doc_ids=["gold.md"],
        expected_label=LABEL_LOW_RANK_EVIDENCE,
    )
    assert label == LABEL_LOW_RANK_EVIDENCE


def test_detect_low_rank_fires_before_distractor():
    # gold at rank 4 AND precision 1/4=0.25 < 0.5 — must return low_rank, not distractor
    label = detect_failure_label(
        retrieved_doc_ids=["a.md", "b.md", "c.md", "gold.md"],
        gold_doc_ids=["gold.md"],
        expected_label=LABEL_LOW_RANK_EVIDENCE,
    )
    assert label == LABEL_LOW_RANK_EVIDENCE
    assert label != LABEL_DISTRACTOR_EVIDENCE


def test_detect_distractor_evidence_gold_at_rank_1_low_precision():
    # gold at rank 1 (within threshold), but precision 1/4=0.25 < 0.5 → distractor
    label = detect_failure_label(
        retrieved_doc_ids=["gold.md", "a.md", "b.md", "c.md"],
        gold_doc_ids=["gold.md"],
        expected_label=LABEL_DISTRACTOR_EVIDENCE,
    )
    assert label == LABEL_DISTRACTOR_EVIDENCE


def test_detect_distractor_requires_good_rank():
    # gold at rank 3 (at threshold boundary, not exceeding it), low precision → distractor
    label = detect_failure_label(
        retrieved_doc_ids=["a.md", "b.md", "gold.md", "c.md"],
        gold_doc_ids=["gold.md"],
        expected_label=LABEL_DISTRACTOR_EVIDENCE,
    )
    assert label == LABEL_DISTRACTOR_EVIDENCE


def test_detect_unanswerable_empty_gold_with_expected_label():
    label = detect_failure_label(
        retrieved_doc_ids=["a.md", "b.md"],
        gold_doc_ids=[],
        expected_label=LABEL_UNANSWERABLE,
    )
    assert label == LABEL_UNANSWERABLE


def test_detect_no_failure_empty_gold_wrong_expected_label():
    # empty gold + expected != unanswerable → cannot evaluate → no_failure
    label = detect_failure_label(
        retrieved_doc_ids=["a.md"],
        gold_doc_ids=[],
        expected_label=LABEL_MISSING_EVIDENCE,
    )
    assert label == LABEL_NO_FAILURE


def test_detect_custom_low_rank_threshold():
    thresholds = DetectionThresholds(low_rank_threshold=1)
    # gold at rank 2 > threshold 1 → low_rank
    label = detect_failure_label(
        retrieved_doc_ids=["a.md", "gold.md"],
        gold_doc_ids=["gold.md"],
        expected_label=LABEL_LOW_RANK_EVIDENCE,
        thresholds=thresholds,
    )
    assert label == LABEL_LOW_RANK_EVIDENCE


def test_detect_custom_precision_threshold():
    thresholds = DetectionThresholds(distractor_precision_threshold=0.8)
    # gold at rank 1 (≤3), precision 1/2=0.5 < 0.8 → distractor
    label = detect_failure_label(
        retrieved_doc_ids=["gold.md", "noise.md"],
        gold_doc_ids=["gold.md"],
        expected_label=LABEL_DISTRACTOR_EVIDENCE,
        thresholds=thresholds,
    )
    assert label == LABEL_DISTRACTOR_EVIDENCE


def test_detect_uses_reciprocal_rank_for_low_rank_check():
    # reciprocal_rank(["a","b","c","gold"], ["gold"]) = 1/4 = 0.25
    # default threshold=3, 1/threshold=0.333; 0.25 < 0.333 → low_rank
    from tiny_rag_lab.eval import reciprocal_rank
    retrieved = ["a.md", "b.md", "c.md", "gold.md"]
    gold = ["gold.md"]
    assert reciprocal_rank(retrieved, gold) == pytest.approx(0.25)
    label = detect_failure_label(retrieved, gold, LABEL_LOW_RANK_EVIDENCE)
    assert label == LABEL_LOW_RANK_EVIDENCE


def test_detect_no_reimplementation_of_hit_at_k():
    from tiny_rag_lab.eval import hit_at_k
    retrieved = ["a.md", "gold.md"]
    gold = ["gold.md"]
    assert hit_at_k(retrieved, gold) is True
    label = detect_failure_label(retrieved, gold, LABEL_NO_FAILURE)
    assert label != LABEL_MISSING_EVIDENCE  # hit was found


def test_detect_no_reimplementation_of_context_precision():
    from tiny_rag_lab.eval import context_precision_at_k
    retrieved = ["gold.md", "noise1.md", "noise2.md"]
    gold = ["gold.md"]
    assert context_precision_at_k(retrieved, gold) == pytest.approx(1 / 3)
    label = detect_failure_label(retrieved, gold, LABEL_DISTRACTOR_EVIDENCE)
    assert label == LABEL_DISTRACTOR_EVIDENCE
