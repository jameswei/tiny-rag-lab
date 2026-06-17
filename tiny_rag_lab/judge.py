"""LLM-as-judge contracts for answer quality evaluation.

Phase 2.0 scope: JudgeVerdict data contract, Judge protocol, FakeJudge for
deterministic tests, OpenAIJudge for real LLM judging, and heuristic
answer-failure label detection.

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
# OpenAI-compatible judge (real LLM path)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of RAG (Retrieval-Augmented Generation) answers.
Given a question, the retrieved context passages, and a generated answer, score
the answer on the following dimensions. Respond ONLY with a JSON object — no
prose, no markdown fences.

JSON keys (all required):
  faithfulness       float 0.0-1.0  Is the answer grounded in the context?
  answer_relevance   float 0.0-1.0  Does the answer address the question?
  citation_support   float 0.0-1.0  Do cited passages support the claims?
  answer_correctness float 0.0-1.0 or null  How close to the reference answer?
                     Set to null when no reference_answer is provided.
  notes              string         One-sentence explanation.
"""

_JUDGE_USER_TEMPLATE = """\
Question: {question}

Retrieved context:
{context}

Generated answer: {answer}
{citations_block}{reference_block}{facts_block}"""


class OpenAIJudge:
    """Calls any OpenAI-compatible endpoint with a JSON-mode prompt.

    Construction is lazy — no API call or import of the openai package happens
    until judge() is called. This keeps `import tiny_rag_lab.judge` free of
    the openai dependency.

    Raises ValueError when the model response is not valid JSON or is missing
    required keys.
    """

    DEFAULT_MODEL = "gpt-4o-mini"
    name: str = "openai"

    def __init__(self, model: str, api_key: str, base_url: str | None = None) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url

    def judge(
        self,
        query: str,
        context: list[str],
        answer: str,
        citations: list[str] | None = None,
        reference_answer: str | None = None,
        expected_facts: list[str] | None = None,
    ) -> JudgeVerdict:
        import json as _json
        import time as _time

        from openai import OpenAI

        citations_block = ""
        if citations:
            citations_block = "\nCitations: " + ", ".join(citations) + "\n"

        reference_block = ""
        if reference_answer:
            reference_block = f"\nReference answer: {reference_answer}\n"

        facts_block = ""
        if expected_facts:
            facts_block = "\nExpected facts:\n" + "\n".join(f"- {f}" for f in expected_facts) + "\n"

        context_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context))
        user_msg = _JUDGE_USER_TEMPLATE.format(
            question=query,
            context=context_text,
            answer=answer,
            citations_block=citations_block,
            reference_block=reference_block,
            facts_block=facts_block,
        )

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        t0 = _time.monotonic()
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )
        latency = _time.monotonic() - t0

        raw = response.choices[0].message.content or ""
        try:
            data = _json.loads(raw)
        except _json.JSONDecodeError as exc:
            raise ValueError(f"OpenAIJudge: response is not valid JSON: {raw!r}") from exc

        required = {"faithfulness", "answer_relevance", "citation_support", "answer_correctness", "notes"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"OpenAIJudge: response missing keys {missing}: {raw!r}")

        return JudgeVerdict(
            faithfulness=float(data["faithfulness"]),
            answer_relevance=float(data["answer_relevance"]),
            citation_support=float(data["citation_support"]),
            answer_correctness=float(data["answer_correctness"]) if data["answer_correctness"] is not None else None,
            judge_name=self.name,
            latency=latency,
            notes=str(data.get("notes", "")),
        )


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
