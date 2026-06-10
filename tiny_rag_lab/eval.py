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
