"""Evaluation harness for the retrieval plane.

Phase 1.6 scope: retrieval-quality metrics only (hit rate, MRR, context
precision, context recall). Answer-quality metrics are deferred to a later
phase.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    """One question-answer pair from the evaluation dataset.

    gold_doc_ids are corpus-relative doc_ids identical to Document.doc_id,
    so they can be matched directly against retrieved chunk.doc_id values.
    answer is kept for future answer-quality metrics but unused in Phase 1.6.
    """

    question_id: str
    question: str
    answer: str
    gold_doc_ids: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Per-question retrieval evaluation result."""

    question_id: str
    question: str
    gold_doc_ids: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)  # rank-ordered
    hit: bool = False
    reciprocal_rank: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0


@dataclass
class EvalReport:
    """Aggregate retrieval evaluation over all questions."""

    n_questions: int
    top_k: int
    hit_rate: float = 0.0
    mrr: float = 0.0
    mean_context_precision: float = 0.0
    mean_context_recall: float = 0.0
    per_question: list[EvalResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Eval dataset loader
# ---------------------------------------------------------------------------

def load_eval_samples(path: Path) -> list[EvalSample]:
    """Load EvalSample objects from a qa.jsonl file.

    Skips rows where question is empty or gold_doc_ids is empty.
    Does not raise on malformed rows — they are silently skipped.
    """
    samples: list[EvalSample] = []
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
            question = str(row.get("question", "")).strip()
            if not question:
                continue
            gold_doc_ids = row.get("gold_doc_ids")
            if not isinstance(gold_doc_ids, list) or not gold_doc_ids:
                continue
            samples.append(EvalSample(
                question_id=str(row.get("question_id", "")).strip(),
                question=question,
                answer=str(row.get("answer", "")).strip(),
                gold_doc_ids=list(gold_doc_ids),
            ))
    return samples


# ---------------------------------------------------------------------------
# Retrieval metric functions (T03)
#
# All four functions receive already-sliced lists — the caller passes only the
# top-k slice, not the full ranked list. k is therefore implicit in the length
# of retrieved_doc_ids.
# ---------------------------------------------------------------------------

def hit_at_k(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> bool:
    """True if at least one retrieved doc is in the gold set."""
    return any(d in gold_doc_ids for d in retrieved_doc_ids)


def reciprocal_rank(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    """1 / rank_of_first_hit; 0.0 if no retrieved doc is in the gold set.

    Rank is 1-indexed: the first position yields RR=1.0, second yields 0.5.
    """
    for i, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            return 1.0 / i
    return 0.0


def context_precision_at_k(
    retrieved_doc_ids: list[str], gold_doc_ids: list[str]
) -> float:
    """Fraction of retrieved docs that are in the gold set.

    Returns 0.0 when retrieved_doc_ids is empty.
    Note: each retrieved position is counted independently, so a doc that
    appears twice in the top-k contributes two hits to the numerator.
    """
    if not retrieved_doc_ids:
        return 0.0
    hits = sum(1 for d in retrieved_doc_ids if d in gold_doc_ids)
    return hits / len(retrieved_doc_ids)


def context_recall_at_k(
    retrieved_doc_ids: list[str], gold_doc_ids: list[str]
) -> float:
    """Fraction of gold docs covered by the retrieved set.

    Returns 0.0 when gold_doc_ids is empty.
    """
    if not gold_doc_ids:
        return 0.0
    covered = len(set(retrieved_doc_ids) & set(gold_doc_ids))
    return covered / len(gold_doc_ids)


# ---------------------------------------------------------------------------
# Report formatter (T05)
# ---------------------------------------------------------------------------

_SEPARATOR = "─" * 36


def format_eval_report(report: EvalReport) -> str:
    """Return a plain-text summary of retrieval evaluation metrics.

    No ANSI escape codes. Values are rounded to 3 decimal places.
    """
    header = f"Evaluation report  (n={report.n_questions}, top_k={report.top_k})"
    lines = [
        header,
        _SEPARATOR,
        f"Hit Rate @ {report.top_k:<6}:  {report.hit_rate:.3f}",
        f"MRR               :  {report.mrr:.3f}",
        f"Context Precision :  {report.mean_context_precision:.3f}",
        f"Context Recall    :  {report.mean_context_recall:.3f}",
    ]
    return "\n".join(lines)
