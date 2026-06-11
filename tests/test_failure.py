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

from tiny_rag_lab.failure import (
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
