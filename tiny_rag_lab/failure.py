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
    from tiny_rag_lab.reranker import Reranker


# ---------------------------------------------------------------------------
# Failure label constants
# ---------------------------------------------------------------------------

LABEL_MISSING_EVIDENCE    = "missing_evidence"     # gold docs absent from retrieved set
LABEL_LOW_RANK_EVIDENCE   = "low_rank_evidence"    # gold present but first hit at rank > threshold
LABEL_DISTRACTOR_EVIDENCE = "distractor_evidence"  # context_precision below threshold
LABEL_UNANSWERABLE        = "unanswerable_query"   # query has no gold in corpus
LABEL_NO_FAILURE          = "no_failure"           # all thresholds met
LABEL_UNSUPPORTED_ANSWER  = "unsupported_answer"   # answer not grounded; requires LLM judge (Phase 2.0)
LABEL_CITATION_MISMATCH   = "citation_mismatch"    # cited chunk does not support claim; requires LLM judge (Phase 2.0)


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
    reranker: str = "none"                # Phase 1.9
    rerank_top_n: int | None = None       # Phase 1.9


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
    answer_label_expected: str = ""   # Phase 2.0; "" = not an answer-side case
    baseline_answer: str = ""         # Phase 2.0; non-empty → run_answer_diagnosis skips generator
    intervention_answer: str = ""     # Phase 2.0; non-empty → run_answer_diagnosis skips generator


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


# ---------------------------------------------------------------------------
# Failure case loader
# ---------------------------------------------------------------------------

def load_failure_cases(path: Path) -> list[FailureCase]:
    """Load FailureCase objects from a cases.jsonl file.

    Does NOT skip rows with empty gold_doc_ids — unanswerable_query cases
    legitimately have gold_doc_ids: []. This diverges from load_eval_samples,
    which skips empty gold lists.
    Skips rows with empty case_id or empty question.
    Silently skips malformed JSON rows.
    baseline and intervention default to RetrieverConfig() when absent.
    """
    cases: list[FailureCase] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            case_id = str(row.get("case_id", "")).strip()
            if not case_id:
                continue
            question = str(row.get("question", "")).strip()
            if not question:
                continue

            gold_doc_ids = row.get("gold_doc_ids")
            if not isinstance(gold_doc_ids, list):
                gold_doc_ids = []

            baseline_raw = row.get("baseline") or {}
            baseline = RetrieverConfig(
                retriever=str(baseline_raw.get("retriever", "dense")),
                top_k=int(baseline_raw.get("top_k", 5)),
                reranker=str(baseline_raw.get("reranker", "none")),
                rerank_top_n=baseline_raw.get("rerank_top_n"),
            ) if isinstance(baseline_raw, dict) else RetrieverConfig()

            intervention_raw = row.get("intervention") or {}
            intervention = RetrieverConfig(
                retriever=str(intervention_raw.get("retriever", "dense")),
                top_k=int(intervention_raw.get("top_k", 5)),
                reranker=str(intervention_raw.get("reranker", "none")),
                rerank_top_n=intervention_raw.get("rerank_top_n"),
            ) if isinstance(intervention_raw, dict) else RetrieverConfig()

            cases.append(FailureCase(
                case_id=case_id,
                question=question,
                gold_doc_ids=list(gold_doc_ids),
                expected_label=str(row.get("expected_label", LABEL_NO_FAILURE)),
                baseline=baseline,
                intervention=intervention,
                notes=str(row.get("notes", "")),
                answer_label_expected=str(row.get("answer_label_expected", "")),
                baseline_answer=str(row.get("baseline_answer", "")),
                intervention_answer=str(row.get("intervention_answer", "")),
            ))
    return cases


# ---------------------------------------------------------------------------
# Failure label detection
# ---------------------------------------------------------------------------

def detect_failure_label(
    retrieved_doc_ids: list[str],
    gold_doc_ids: list[str],
    expected_label: str,
    thresholds: DetectionThresholds | None = None,
) -> str:
    """Assign a failure label from retrieved and gold doc IDs.

    Detection order (first match wins):
    1. gold_doc_ids empty + expected_label == LABEL_UNANSWERABLE → LABEL_UNANSWERABLE
    2. gold_doc_ids empty otherwise → LABEL_NO_FAILURE (cannot evaluate)
    3. no hit → LABEL_MISSING_EVIDENCE
    4. first gold hit at rank > low_rank_threshold → LABEL_LOW_RANK_EVIDENCE
    5. context_precision < distractor_precision_threshold → LABEL_DISTRACTOR_EVIDENCE
    6. LABEL_NO_FAILURE

    Steps 4 and 5 are mutually exclusive: low_rank fires when gold is buried
    regardless of precision; distractor fires only when gold is well-ranked but
    surrounding context is noisy. Checking low_rank first prevents a large top_k
    from shadowing a rank-ordering failure with a precision signal.

    Calls hit_at_k, reciprocal_rank, and context_precision_at_k from eval.py.
    rank > threshold is expressed as rr < 1/threshold (since rr = 1/rank and
    a hit is guaranteed at this point, rr > 0).
    """
    from tiny_rag_lab.eval import context_precision_at_k, hit_at_k, reciprocal_rank

    if thresholds is None:
        thresholds = DetectionThresholds()

    if not gold_doc_ids:
        return LABEL_UNANSWERABLE if expected_label == LABEL_UNANSWERABLE else LABEL_NO_FAILURE

    if not hit_at_k(retrieved_doc_ids, gold_doc_ids):
        return LABEL_MISSING_EVIDENCE

    # reciprocal_rank > 0 is guaranteed (hit confirmed above).
    # rank > threshold ⟺ 1/rank < 1/threshold ⟺ rr < 1/threshold.
    rr = reciprocal_rank(retrieved_doc_ids, gold_doc_ids)
    if rr < 1.0 / thresholds.low_rank_threshold:
        return LABEL_LOW_RANK_EVIDENCE

    if context_precision_at_k(retrieved_doc_ids, gold_doc_ids) < thresholds.distractor_precision_threshold:
        return LABEL_DISTRACTOR_EVIDENCE

    return LABEL_NO_FAILURE


# ---------------------------------------------------------------------------
# Diagnosis runner
# ---------------------------------------------------------------------------

def run_diagnosis(
    cases: list[FailureCase],
    index: LoadedIndex,
    embedder: Embedder | None,
    reranker: "Reranker | None" = None,
    thresholds: DetectionThresholds | None = None,
) -> DiagnosisReport:
    """Run baseline and intervention retrieval for every case and return an
    aggregate DiagnosisReport.

    When reranker is provided, cases whose config specifies a non-none reranker
    use it; cases with reranker="none" skip rerank. Raises ValueError if any
    case needs a reranker and reranker is None.

    Builds BM25Retriever once before the case loop when any case uses bm25 or
    hybrid — same build-once pattern as run_retrieval_eval in eval.py.

    Raises ValueError if embedder is None and any case requires dense or hybrid
    retrieval.
    """
    from tiny_rag_lab.bm25 import BM25Retriever
    from tiny_rag_lab.eval import (
        context_precision_at_k,
        context_recall_at_k,
        hit_at_k,
        reciprocal_rank,
    )
    from tiny_rag_lab.hybrid import retrieve_hybrid
    from tiny_rag_lab.retrieval import retrieve_by_vector

    if not cases:
        return DiagnosisReport(n_cases=0)

    needs_embedder = any(
        c.baseline.retriever in ("dense", "hybrid") or
        c.intervention.retriever in ("dense", "hybrid")
        for c in cases
    )
    if needs_embedder and embedder is None:
        raise ValueError(
            "embedder must not be None when any case uses dense or hybrid retrieval"
        )

    needs_bm25 = any(
        c.baseline.retriever in ("bm25", "hybrid") or
        c.intervention.retriever in ("bm25", "hybrid")
        for c in cases
    )
    bm25_retriever = BM25Retriever(index.chunks) if needs_bm25 else None

    # Validate: if any case needs a reranker and none is provided, raise.
    needs_reranker = any(
        config.reranker != "none"
        for c in cases
        for config in (c.baseline, c.intervention)
    )
    if needs_reranker and reranker is None:
        raise ValueError(
            "reranker must not be None when any case uses a non-none reranker"
        )

    # Validate: rerank_top_n must be >= top_k when reranker is active.
    for c in cases:
        for label, config in (("baseline", c.baseline), ("intervention", c.intervention)):
            if config.reranker != "none" and config.rerank_top_n is not None:
                if config.rerank_top_n < config.top_k:
                    raise ValueError(
                        f"case {c.case_id} {label}: rerank_top_n "
                        f"({config.rerank_top_n}) must be >= top_k ({config.top_k})"
                    )

    def _retrieve(question: str, config: RetrieverConfig) -> list[str]:
        # When reranker is active, retrieve rerank_top_n candidates first.
        retrieval_k = (
            config.rerank_top_n
            if config.reranker != "none" and reranker is not None and config.rerank_top_n is not None
            else config.top_k
        )

        if config.retriever == "bm25":
            results = bm25_retriever.retrieve(question, top_k=retrieval_k)
        elif config.retriever == "hybrid":
            results = retrieve_hybrid(
                question, index, embedder, top_k=retrieval_k,
                bm25_retriever=bm25_retriever,
            )
        else:
            query_vec = embedder.embed([question])[0]
            results = retrieve_by_vector(query_vec, index, top_k=retrieval_k)

        if config.reranker != "none" and reranker is not None:
            from tiny_rag_lab.reranker import apply_reranker
            results, _audit = apply_reranker(
                question, results, reranker, config.top_k,
            )

        return [r.chunk.doc_id for r in results]

    def _metrics(retrieved: list[str], gold: list[str]) -> dict[str, float]:
        return {
            "hit": float(hit_at_k(retrieved, gold)),
            "reciprocal_rank": reciprocal_rank(retrieved, gold),
            "context_precision": context_precision_at_k(retrieved, gold),
            "context_recall": context_recall_at_k(retrieved, gold),
        }

    per_case: list[DiagnosisResult] = []
    for case in cases:
        baseline_doc_ids = _retrieve(case.question, case.baseline)
        intervention_doc_ids = _retrieve(case.question, case.intervention)

        baseline_label = detect_failure_label(
            baseline_doc_ids, case.gold_doc_ids, case.expected_label, thresholds,
        )
        intervention_label = detect_failure_label(
            intervention_doc_ids, case.gold_doc_ids, case.expected_label, thresholds,
        )

        fixed = baseline_label != LABEL_NO_FAILURE and intervention_label == LABEL_NO_FAILURE
        moved = (
            baseline_label != LABEL_NO_FAILURE
            and intervention_label != LABEL_NO_FAILURE
            and baseline_label != intervention_label
        )

        per_case.append(DiagnosisResult(
            case_id=case.case_id,
            question=case.question,
            expected_label=case.expected_label,
            baseline_label=baseline_label,
            intervention_label=intervention_label,
            baseline_retrieved_doc_ids=baseline_doc_ids,
            intervention_retrieved_doc_ids=intervention_doc_ids,
            baseline_metrics=_metrics(baseline_doc_ids, case.gold_doc_ids),
            intervention_metrics=_metrics(intervention_doc_ids, case.gold_doc_ids),
            fixed=fixed,
            moved=moved,
        ))

    return DiagnosisReport(
        n_cases=len(per_case),
        n_fixed=sum(r.fixed for r in per_case),
        n_moved=sum(r.moved for r in per_case),
        n_confirmed=sum(r.baseline_label == r.expected_label for r in per_case),
        per_case=per_case,
    )


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

_SEPARATOR = "─" * 44


def format_diagnosis_report(report: DiagnosisReport) -> str:
    """Return a plain-text summary of the diagnosis report.

    No ANSI escape codes. Float values at 3 decimal places.
    Outcome word per case: FIXED, MOVED, CONFIRMED, or UNCHANGED.

    Example:
      Diagnosis report  (n=6)
      ────────────────────────────────────────────
        Confirmed  : 4
        Fixed      : 1
        Moved      : 1
      ────────────────────────────────────────────
      Case fc001  expected=missing_evidence
        baseline   : missing_evidence       hit=0.000  prec=0.000  recall=0.000  mrr=0.000
        interv.    : no_failure             hit=1.000  prec=0.500  recall=1.000  mrr=0.500
        FIXED
    """
    lines = [
        f"Diagnosis report  (n={report.n_cases})",
        _SEPARATOR,
        f"  Confirmed  : {report.n_confirmed}",
        f"  Fixed      : {report.n_fixed}",
        f"  Moved      : {report.n_moved}",
        _SEPARATOR,
    ]
    for dr in report.per_case:
        lines.append(f"Case {dr.case_id}  expected={dr.expected_label}")
        bm = dr.baseline_metrics
        lines.append(
            f"  baseline   : {dr.baseline_label:<24}"
            f"  hit={bm.get('hit', 0.0):.3f}"
            f"  prec={bm.get('context_precision', 0.0):.3f}"
            f"  recall={bm.get('context_recall', 0.0):.3f}"
            f"  mrr={bm.get('reciprocal_rank', 0.0):.3f}"
        )
        im = dr.intervention_metrics
        lines.append(
            f"  interv.    : {dr.intervention_label:<24}"
            f"  hit={im.get('hit', 0.0):.3f}"
            f"  prec={im.get('context_precision', 0.0):.3f}"
            f"  recall={im.get('context_recall', 0.0):.3f}"
            f"  mrr={im.get('reciprocal_rank', 0.0):.3f}"
        )
        if dr.fixed:
            outcome = "FIXED"
        elif dr.moved:
            outcome = "MOVED"
        elif dr.baseline_label == dr.expected_label:
            outcome = "CONFIRMED"
        else:
            outcome = "UNCHANGED"
        lines.append(f"  {outcome}")
        lines.append("")
    return "\n".join(lines).rstrip()