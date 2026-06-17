"""LLM-as-judge contracts for answer quality evaluation.

Phase 2.0 scope: JudgeVerdict data contract, Judge protocol, FakeJudge for
deterministic tests, and heuristic answer-failure label detection.

OpenAIJudge (real LLM path) is added in Phase 2.0 T02.

All dataclass fields are JSON-native types so dataclasses.asdict() + json.dumps()
serializes any judge verdict without a custom encoder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class JudgeVerdict:
    """LLM-as-judge assessment of one RAG answer.

    All float scores are 0.0-1.0:
      faithfulness:      is the answer grounded in the retrieved context?
      answer_relevance:  does the answer address the question?
      citation_support:  do cited chunks actually support the claims made?
      answer_correctness: how close is the answer to the reference_answer?
                         None when no reference_answer was provided.

    latency is wall-clock seconds for the judge call (0.0 for FakeJudge).
    notes is free-form explanation text from the judge.
    """

    faithfulness: float
    answer_relevance: float
    citation_support: float
    answer_correctness: float | None
    judge_name: str
    latency: float
    notes: str = ""


def _default_verdict() -> JudgeVerdict:
    """Passing verdict used as FakeJudge default when no verdict_map entry matches."""
    return JudgeVerdict(
        faithfulness=1.0,
        answer_relevance=1.0,
        citation_support=1.0,
        answer_correctness=None,
        judge_name="fake",
        latency=0.0,
    )


class Judge(Protocol):
    """Second-pass judge over a generated answer.

    Implementations must be deterministic for the same inputs so trace and
    eval outputs are reproducible.
    """

    name: str

    def judge(
        self,
        query: str,
        context: list[str],
        answer: str,
        citations: list[str] | None = None,
        reference_answer: str | None = None,
        expected_facts: list[str] | None = None,
    ) -> JudgeVerdict: ...


@dataclass
class FakeJudge:
    """Deterministic judge for tests.

    verdict_map maps answer string -> JudgeVerdict. When a call's answer
    matches a key, that verdict is returned; otherwise default_verdict is
    returned. Keying on the answer string (not the query) lets fc008/fc009
    fixture cases produce different verdicts for their scripted baseline_answer
    vs intervention_answer even when both use the same question.

    default_verdict is a fully-passing verdict (all scores 1.0,
    answer_correctness None) so tests that do not override the map pass by
    default.
    """

    name: str = "fake"
    default_verdict: JudgeVerdict = field(default_factory=_default_verdict)
    verdict_map: dict[str, JudgeVerdict] | None = None

    def judge(
        self,
        query: str,
        context: list[str],
        answer: str,
        citations: list[str] | None = None,
        reference_answer: str | None = None,
        expected_facts: list[str] | None = None,
    ) -> JudgeVerdict:
        if self.verdict_map is not None and answer in self.verdict_map:
            return self.verdict_map[answer]
        return self.default_verdict


# ---------------------------------------------------------------------------
# Answer-side failure detection
# ---------------------------------------------------------------------------

@dataclass
class AnswerDetectionThresholds:
    """Thresholds for answer-side failure label assignment.

    faithfulness_threshold: faithfulness below this triggers unsupported_answer.
      Default 0.5 — less than half the answer is grounded in context.
    citation_support_threshold: citation_support below this triggers
      citation_mismatch, but only when faithfulness already passes.
      Default 0.5 — more than half the cited passages do not support claims.
    """

    faithfulness_threshold: float = 0.5
    citation_support_threshold: float = 0.5


def detect_answer_failure_label(
    verdict: JudgeVerdict,
    thresholds: AnswerDetectionThresholds | None = None,
) -> str:
    """Assign an answer-side failure label from a judge verdict.

    Detection order (first match wins):
    1. faithfulness < faithfulness_threshold  -> LABEL_UNSUPPORTED_ANSWER
    2. citation_support < citation_support_threshold -> LABEL_CITATION_MISMATCH
    3. -> LABEL_NO_FAILURE

    Imports failure label constants at call time to avoid a module-level
    circular dependency (failure.py imports from judge.py in T05+).
    """
    from tiny_rag_lab.failure import (
        LABEL_CITATION_MISMATCH,
        LABEL_NO_FAILURE,
        LABEL_UNSUPPORTED_ANSWER,
    )

    if thresholds is None:
        thresholds = AnswerDetectionThresholds()

    if verdict.faithfulness < thresholds.faithfulness_threshold:
        return LABEL_UNSUPPORTED_ANSWER
    if verdict.citation_support < thresholds.citation_support_threshold:
        return LABEL_CITATION_MISMATCH
    return LABEL_NO_FAILURE
