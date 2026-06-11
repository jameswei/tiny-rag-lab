"""Failure taxonomy and diagnosis for the RAG pipeline.

Phase 1.8 scope: failure labels, data contracts, case fixture loader,
detection logic, diagnosis runner, and report formatter.

All dataclass fields are JSON-native types so dataclasses.asdict() +
json.dumps() serializes any type without a custom encoder.

Two failure modes are documented here but not heuristically detectable:
  "unsupported_answer" — answer not grounded in retrieved context; requires LLM-as-judge
  "citation_mismatch"  — cited chunk does not support the claim; requires LLM-as-judge
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tiny_rag_lab.embeddings import Embedder
    from tiny_rag_lab.index_loader import LoadedIndex


# ---------------------------------------------------------------------------
# Failure label constants
# ---------------------------------------------------------------------------

LABEL_MISSING_EVIDENCE    = "missing_evidence"     # gold docs absent from retrieved set
LABEL_LOW_RANK_EVIDENCE   = "low_rank_evidence"    # gold present but first hit at rank > threshold
LABEL_DISTRACTOR_EVIDENCE = "distractor_evidence"  # context_precision below threshold
LABEL_UNANSWERABLE        = "unanswerable_query"   # query has no gold in corpus
LABEL_NO_FAILURE          = "no_failure"           # all thresholds met


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class RetrieverConfig:
    """Retrieval configuration for one run of a failure case.

    All fields are JSON-native so RetrieverConfig round-trips through
    dataclasses.asdict() without a custom encoder.
    """

    retriever: str = "dense"    # "dense" | "bm25" | "hybrid"
    top_k: int = 5


@dataclass
class FailureCase:
    """One curated failure scenario with reproducible inputs.

    gold_doc_ids may be [] for unanswerable_query cases. This diverges from
    EvalSample, which skips rows with empty gold_doc_ids. load_failure_cases
    preserves unanswerable cases intentionally — the empty list is meaningful.

    baseline is the retrieval config expected to exhibit the failure.
    intervention is the config that tests whether the failure can be reduced.
    notes records the human explanation of what the failure demonstrates and
    which trace fields are most diagnostic.
    """

    case_id: str
    question: str
    gold_doc_ids: list[str] = field(default_factory=list)
    expected_label: str = LABEL_NO_FAILURE
    baseline: RetrieverConfig = field(default_factory=RetrieverConfig)
    intervention: RetrieverConfig = field(default_factory=RetrieverConfig)
    notes: str = ""


@dataclass
class DetectionThresholds:
    """Thresholds for heuristic failure label assignment.

    low_rank_threshold: first gold hit at rank > this triggers low_rank_evidence.
      Default 3 — gold found at rank 4 or deeper is a buried-evidence failure.
    distractor_precision_threshold: context_precision below this triggers
      distractor_evidence, but only when gold rank is within low_rank_threshold.
      Default 0.5 — more than half the context is noise.
    """

    low_rank_threshold: int = 3
    distractor_precision_threshold: float = 0.5


@dataclass
class DiagnosisResult:
    """Per-case diagnosis from one run of rag diagnose.

    baseline_retrieved_doc_ids and intervention_retrieved_doc_ids carry the
    actual doc_ids returned by retrieval — the trace-backed evidence for the
    assigned labels. They let callers inspect what was retrieved and why.

    fixed is True when the baseline has a failure and the intervention does not.
    moved is True when both runs have failures but with different labels.
    """

    case_id: str
    question: str
    expected_label: str
    baseline_label: str
    intervention_label: str
    baseline_retrieved_doc_ids: list[str]       # doc_ids actually retrieved at baseline
    intervention_retrieved_doc_ids: list[str]   # doc_ids actually retrieved at intervention
    baseline_metrics: dict[str, float]          # keys: hit, reciprocal_rank, context_precision, context_recall
    intervention_metrics: dict[str, float]      # same keys
    fixed: bool = False
    moved: bool = False


@dataclass
class DiagnosisReport:
    """Aggregate diagnosis over all failure cases in one rag diagnose run."""

    n_cases: int
    n_fixed: int = 0
    n_moved: int = 0
    n_confirmed: int = 0        # baseline_label == expected_label
    per_case: list[DiagnosisResult] = field(default_factory=list)
