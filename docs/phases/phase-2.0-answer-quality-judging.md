# Phase 2.0 Spec: Answer Quality Judging

**Status:** Draft — pending sign-off
**Authors:** Claude Code
**Based on:** `docs/phases/phase-1.9-2.2-final-roadmap.md`
**Taskboard:** `docs/phases/phase-2.0-taskboard.md`
**Date:** 2026-06-17

---

## Goal

Measure answer quality, not only retrieval quality, using an LLM-as-judge path
behind a fakeable interface. Phase 1.8 explicitly deferred `unsupported_answer`
and `citation_mismatch` failure modes that require an LLM judge. Phase 2.0
closes them and gives every subsequent phase a faithful metric to measure
against.

The judge is opt-in (`--judge none` default). Every existing command remains
identical when `--judge` is absent.

---

## Scope

### In Scope

- `tiny_rag_lab/judge.py`: `JudgeVerdict` dataclass, `Judge` protocol,
  `FakeJudge`, `OpenAIJudge`
- `EvalSample` back-compat extension: optional `reference_answer` and
  `expected_facts` fields (missing in existing JSONL rows defaults to
  `None`/`[]`)
- `AnswerEvalResult` and `AnswerEvalReport` data contracts in `eval.py`
- `run_answer_eval(samples, index, embedder, top_k, retriever, generator, judge,
  reranker=None, rerank_top_n=None) -> AnswerEvalReport` in `eval.py`
- `format_answer_eval_report(report: AnswerEvalReport) -> str` in `eval.py`
- `--generator fake|openai` flag on `cmd_eval`, `cmd_ask`, and `cmd_diagnose`
  (default `openai`); `fake` uses `FakeGenerator`, bypassing LLM calls
- `cmd_eval --judge none|fake|openai`: when active, runs retrieval eval
  followed by answer eval and prints both sections; new flags `--generator`,
  `--model`, `--api-key`, `--base-url` added to `rag eval` (they exist today
  only on `rag ask`)
- `AskTrace.verdict: JudgeVerdict | None = None` field (serializes as `null`
  when absent); `format_ask_trace` renders the verdict block when present
- `cmd_ask --judge none|fake|openai`: judges the generated answer inline
- `_make_judge(name, model, api_key, base_url)` factory in `cli.py`; falls
  back to `OPENAI_API_KEY` env var before raising on missing key
- Failure lab: `LABEL_UNSUPPORTED_ANSWER`, `LABEL_CITATION_MISMATCH` constants;
  `AnswerDetectionThresholds`; `AnswerDiagnosisResult`; `AnswerDiagnosisReport`;
  `detect_answer_failure_label(verdict, thresholds) -> str`;
  `run_answer_diagnosis(cases, index, embedder, generator, judge, reranker,
  thresholds) -> AnswerDiagnosisReport`; `format_answer_diagnosis_report`
- Fixture: fc008 (`unsupported_answer`) and fc009 (`citation_mismatch`) appended
  to `tests/fixtures/failure/cases.jsonl`; both carry `baseline_answer` and
  `intervention_answer` fields so `run_answer_diagnosis` can bypass the
  generator and produce deterministic, distinct verdicts
- `cmd_diagnose --judge none|fake|openai`: when active, also runs
  `run_answer_diagnosis` on answer-side cases and appends the answer diagnosis
  section to the report
- Gated `tests/test_judge_openai.py`: `TINY_RAG_LAB_TEST_JUDGE=1` env flag,
  `pytest.importorskip("openai")` — same two-gate ordering as the Phase 1.9
  cross-encoder fix

### Out Of Scope For Phase 2.0

- RAGAS-style composite scores or G-EVAL claim decomposition
- Structured JSON output from `rag ask` (Phase 2.1)
- Context budget / token counting (Phase 2.1)
- Multi-run comparison reports
- Streaming responses
- Cohere, Anthropic, or other provider judges beyond OpenAI-compatible

---

## Design Decision 1: Judge Is Opt-In, Default `none`

`--judge` defaults to `none`. When absent, every command produces identical
output to Phase 1.9. `AskTrace.verdict` is `None`; it serializes as JSON
`null`. No LLM call is triggered by default.

`--judge fake --generator fake` runs the full pipeline offline using
`FakeJudge` and `FakeGenerator` with scripted outputs — useful for integration
tests and CI. `--judge fake` alone only fakes the judge; generation still calls
the real LLM unless `--generator fake` is also set. `--judge openai` requires
`--api-key` or `OPENAI_API_KEY` env var; `--model` is optional and defaults
to `OpenAIJudge.DEFAULT_MODEL` (`gpt-4o-mini`).

## Design Decision 2: Four Metrics; `answer_correctness` Is Conditional

`faithfulness`, `answer_relevance`, and `citation_support` are always
computed. `answer_correctness` is computed only when `reference_answer` is
provided for the sample. When absent, `JudgeVerdict.answer_correctness` is
`None` and `AnswerEvalReport.mean_answer_correctness` is `None`. This keeps
the score semantically honest — a "0" correctness for a sample with no
reference is meaningless noise.

## Design Decision 3: Retrieval Metrics And Answer Metrics Stay In Separate Reports

`run_retrieval_eval` returns `EvalReport` unchanged. `run_answer_eval` returns
`AnswerEvalReport`. `cmd_eval --judge` runs both and prints both sections
separated by a blank line. This keeps retrieval regressions visible even when
answer quality changes, and avoids polluting `EvalReport` with optional
nullable fields.

## Design Decision 4: Generator Mode Is Explicit (`--generator fake|openai`)

A new `--generator fake|openai` flag (default `openai`) controls which
generator is used. `fake` returns `FakeGenerator` (offline, scripted).
`openai` calls `_make_generator(args)` (the existing factory).

`--model`, `--api-key`, and `--base-url` are shared by both generator and
judge — one set of credentials, one model, for both. If a user needs different
models for generation and judging they can run the commands separately; this
phase does not add per-role model flags.

`cmd_eval` does not currently have `--model`/`--api-key`/`--base-url`; T03
explicitly adds them. `cmd_ask` already has them; T04 adds only `--generator`
and `--judge`.

The reranker parameters thread through `run_answer_eval` so
`--reranker cross-encoder --judge openai` can be combined.

## Design Decision 5: `FakeJudge` Keys On The Answer String, Not The Query

```python
@dataclass
class FakeJudge:
    name: str = "fake"
    default_verdict: JudgeVerdict = field(default_factory=_default_verdict)
    verdict_map: dict[str, JudgeVerdict] | None = None
```

When `verdict_map` is set, the fake judge looks up by the `answer` string
passed to `judge(...)`. When absent it returns `default_verdict` for every
call. Keying on `answer` rather than `query` is necessary because fc008/fc009
use the same question for baseline and intervention — a query key cannot
produce different verdicts. With scripted `baseline_answer`/`intervention_answer`
fields in the fixture (see DD8), each run produces a distinct answer string
and `FakeJudge` can return the correct scripted verdict.

## Design Decision 6: `run_answer_diagnosis` Is A Separate Function, Not An Extension Of `run_diagnosis`

The answer-side loop (retrieve → generate → judge) is architecturally different
from the retrieval-side loop (retrieve only). A separate function keeps each
readable and lets the CLI compose them when `--judge` is set. `cmd_diagnose`
already prints a `format_diagnosis_report`; when `--judge` is active it
appends `format_answer_diagnosis_report` after a separator.

## Design Decision 7: `OpenAIJudge` Uses JSON Mode

The prompt instructs the model to respond in JSON with keys `faithfulness`,
`answer_relevance`, `citation_support`, `answer_correctness` (null when no
reference), and `notes`. The judge sets `response_format={"type":
"json_object"}` where the provider supports it. Invalid JSON raises
`ValueError` with the raw response text in the message.

## Design Decision 8: Fixture Carries Scripted Answers For Answer-Side Cases

`FailureCase` gains two new optional string fields:

```python
baseline_answer: str = ""       # NEW; non-empty → run_answer_diagnosis skips generator
intervention_answer: str = ""   # NEW; non-empty → run_answer_diagnosis skips generator
```

When `baseline_answer` is non-empty, `run_answer_diagnosis` uses it directly
as the baseline answer and skips the generator call for that run. Same for
`intervention_answer`. This is required for fc008/fc009, where the same
question with identical retrieval configs would otherwise produce the same
generated answer and the same verdict for both baseline and intervention.

Scripted answers make the fixture self-contained and the failure mode visible
in the JSON: a reader can see both what the broken answer looks like
(`baseline_answer`) and what a fixed answer looks like (`intervention_answer`).

`load_failure_cases` reads both fields with default `""`.

## Design Decision 9: fc008 And fc009 Use `answer_label_expected`, Not `expected_label`

Existing cases fc001–fc007 set `expected_label` for retrieval-side failure
detection. Answer-side cases use a separate `answer_label_expected` field so
the retrieval pass (`run_diagnosis`) and the answer pass
(`run_answer_diagnosis`) can each use the appropriate field. Missing
`answer_label_expected` in the JSONL defaults to `""` (not an answer case).
`run_answer_diagnosis` skips cases without it.

---

## Data Contracts

### `tiny_rag_lab/judge.py` (new)

```python
@dataclass
class JudgeVerdict:
    """LLM-as-judge assessment of one RAG answer.

    All float scores are 0.0-1.0. answer_correctness is None when no
    reference_answer was provided. latency is wall-clock seconds for the
    judge API call. notes is free-form explanation text from the judge.
    """
    faithfulness: float
    answer_relevance: float
    citation_support: float
    answer_correctness: float | None
    judge_name: str
    latency: float
    notes: str = ""


class Judge(Protocol):
    name: str

    def judge(
        self,
        query: str,
        context: list[str],             # chunk texts, rank order
        answer: str,
        citations: list[str] | None = None,
        reference_answer: str | None = None,
        expected_facts: list[str] | None = None,
    ) -> JudgeVerdict: ...


@dataclass
class FakeJudge:
    name: str = "fake"
    default_verdict: JudgeVerdict = field(default_factory=_default_verdict)
    verdict_map: dict[str, JudgeVerdict] | None = None

    def judge(self, query, context, answer, ...) -> JudgeVerdict:
        # returns verdict_map[answer] when present, else default_verdict


class OpenAIJudge:
    """Calls any OpenAI-compatible endpoint with a JSON-mode prompt."""
    DEFAULT_MODEL = "gpt-4o-mini"
    name: str = "openai"

    def __init__(self, model: str, api_key: str, base_url: str | None = None): ...
    def judge(self, query, context, answer, ...) -> JudgeVerdict: ...
```

`_default_verdict()` returns `JudgeVerdict(faithfulness=1.0,
answer_relevance=1.0, citation_support=1.0, answer_correctness=None,
judge_name="fake", latency=0.0)` — a passing verdict so tests that do not
override the map pass by default.

### Updates To `tiny_rag_lab/eval.py`

```python
@dataclass
class EvalSample:
    question_id: str
    question: str
    answer: str
    gold_doc_ids: list[str]
    reference_answer: str | None = None        # NEW Phase 2.0
    expected_facts: list[str] = field(default_factory=list)  # NEW Phase 2.0


@dataclass
class AnswerEvalResult:
    question_id: str
    question: str
    verdict: JudgeVerdict | None = None


@dataclass
class AnswerEvalReport:
    n_questions: int
    judge: str = "none"
    mean_faithfulness: float = 0.0
    mean_answer_relevance: float = 0.0
    mean_citation_support: float = 0.0
    mean_answer_correctness: float | None = None   # None when no reference_answers
    per_question: list[AnswerEvalResult] = field(default_factory=list)
```

`load_eval_samples` gains two additional field reads with defaults for backward
compatibility — rows without `reference_answer` or `expected_facts` load
unchanged.

### Update To `tiny_rag_lab/trace.py`

```python
@dataclass
class AskTrace:
    ...  # all existing fields unchanged
    verdict: JudgeVerdict | None = None   # NEW Phase 2.0
```

`JudgeVerdict` is a dataclass with all JSON-native fields, so
`dataclasses.asdict()` continues to serialize the full trace without a custom
encoder. When `verdict` is `None` it serializes as `null`.

`format_ask_trace` appends a verdict block after the citations section when
`trace.verdict is not None`:

```
Judge verdict  (judge=openai)
  Faithfulness     : 0.920
  Answer Relevance : 0.880
  Citation Support : 0.750
  Notes            : Answer is mostly grounded; one claim lacks citation.
```

When `answer_correctness` is not None, a fifth line appears:

```
  Answer Correct.  : 0.810
```

### Updates To `tiny_rag_lab/failure.py`

```python
LABEL_UNSUPPORTED_ANSWER = "unsupported_answer"    # NEW
LABEL_CITATION_MISMATCH  = "citation_mismatch"     # NEW


@dataclass
class AnswerDetectionThresholds:
    faithfulness_threshold: float = 0.5
    citation_support_threshold: float = 0.5


@dataclass
class AnswerDiagnosisResult:
    case_id: str
    question: str
    expected_label: str              # from answer_label_expected field
    baseline_label: str
    intervention_label: str
    baseline_verdict: JudgeVerdict | None
    intervention_verdict: JudgeVerdict | None
    fixed: bool = False
    moved: bool = False


@dataclass
class AnswerDiagnosisReport:
    n_cases: int
    n_fixed: int = 0
    n_moved: int = 0
    n_confirmed: int = 0
    per_case: list[AnswerDiagnosisResult] = field(default_factory=list)
```

`FailureCase` gains three new fields:

```python
@dataclass
class FailureCase:
    ...  # existing fields unchanged
    answer_label_expected: str = ""    # NEW; "" means not an answer-side case
    baseline_answer: str = ""          # NEW; non-empty → skip generator in run_answer_diagnosis
    intervention_answer: str = ""      # NEW; non-empty → skip generator in run_answer_diagnosis
```

`load_failure_cases` reads all three with default `""`.

---

## Core Functions

```python
def run_answer_eval(
    samples: list[EvalSample],
    index: LoadedIndex,
    embedder: Embedder | None,
    top_k: int,
    retriever: str,
    generator: Generator,
    judge: Judge,
    reranker: Reranker | None = None,
    rerank_top_n: int | None = None,
) -> AnswerEvalReport:
    """Retrieve -> generate -> judge per sample. Returns AnswerEvalReport.

    Raises ValueError if retriever in ("dense","hybrid") and embedder is None.
    Raises ValueError if reranker is not None and rerank_top_n is None.
    mean_answer_correctness is None when no sample has reference_answer set.
    """
```

```python
def detect_answer_failure_label(
    verdict: JudgeVerdict,
    thresholds: AnswerDetectionThresholds | None = None,
) -> str:
    """Assign an answer-side failure label from a judge verdict.

    Detection order (first match wins):
    1. faithfulness < faithfulness_threshold  -> LABEL_UNSUPPORTED_ANSWER
    2. citation_support < citation_support_threshold -> LABEL_CITATION_MISMATCH
    3. -> LABEL_NO_FAILURE
    """
```

```python
def run_answer_diagnosis(
    cases: list[FailureCase],
    index: LoadedIndex,
    embedder: Embedder | None,
    generator: Generator,
    judge: Judge,
    reranker: Reranker | None = None,
    thresholds: AnswerDetectionThresholds | None = None,
) -> AnswerDiagnosisReport:
    """Retrieve -> generate -> judge baseline and intervention per answer-side case.

    Skips cases where answer_label_expected is "".
    When case.baseline_answer / case.intervention_answer is non-empty, uses it
    directly as the answer for that run and skips the generator call. This lets
    answer-side fixture cases be fully deterministic without a live LLM.
    Raises ValueError if embedder is None and any active case uses dense or hybrid.
    """
```

```python
def format_answer_diagnosis_report(report: AnswerDiagnosisReport) -> str:
    """Plain-text summary of answer diagnosis. Same layout as
    format_diagnosis_report but shows verdict scores instead of retrieval
    metrics."""
```

---

## CLI Changes

### `rag eval`

New flags (all new to `rag eval`; they exist today only on `rag ask`):
- `--judge none|fake|openai` (default `none`)
- `--generator fake|openai` (default `openai`)
- `--model NAME`
- `--api-key KEY`
- `--base-url URL`

One model/key/url pair is shared by both generator and judge.

When `--judge` is not `none`: `cmd_eval` runs `run_retrieval_eval` (unchanged)
then `run_answer_eval`. The formatter prints the retrieval section, a blank
separator, then the answer quality section. When `--judge none`, output is
identical to Phase 1.9. Fully offline: `--judge fake --generator fake`.

### `rag ask`

New flags:
- `--judge none|fake|openai` (default `none`)
- `--generator fake|openai` (default `openai`)

When `--judge` is active: after `generator.generate(prompt)` returns an
answer, calls `judge.judge(query, chunk_texts, answer, citations, ...)`.
Populates `AskTrace.verdict`. `format_ask_trace` renders the verdict block.

### `rag diagnose`

New flags:
- `--judge none|fake|openai` (default `none`)
- `--generator fake|openai` (default `openai`)

When active: runs `run_answer_diagnosis` on cases with
`answer_label_expected != ""`. Appends `format_answer_diagnosis_report` to
the CLI output after the retrieval diagnosis section.

### Factories

```python
def _make_judge(name: str, model: str | None, api_key: str | None, base_url: str | None) -> Judge | None:
    if name == "none":
        return None
    if name == "fake":
        return FakeJudge()
    if name == "openai":
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("--api-key or OPENAI_API_KEY env is required for --judge openai")
        return OpenAIJudge(model=model or OpenAIJudge.DEFAULT_MODEL, api_key=api_key, base_url=base_url)
    raise ValueError(f"Unknown judge: {name!r}")


def _make_generator_from_flag(name: str, args) -> Generator:
    """name is "fake" or "openai". "openai" delegates to existing _make_generator(args)."""
    if name == "fake":
        from tiny_rag_lab.generation import FakeGenerator
        return FakeGenerator()
    return _make_generator(args)   # existing factory; reads model/api_key/base_url from args
```

---

## Failure Fixture Cases

Append to `tests/fixtures/failure/cases.jsonl`:

**fc008** (`unsupported_answer`):

```json
{"case_id": "fc008", "question": "What topics does the sample document cover?", "gold_doc_ids": ["with_h1.md"], "expected_label": "no_failure", "answer_label_expected": "unsupported_answer", "baseline": {"retriever": "dense", "top_k": 3}, "intervention": {"retriever": "dense", "top_k": 3}, "baseline_answer": "The document covers retrieval and also quantum computing.", "intervention_answer": "The document covers several topics useful for retrieval testing.", "notes": "baseline_answer contains an invented claim (quantum computing) absent from context; FakeJudge verdict_map[baseline_answer] returns faithfulness=0.1. intervention_answer is grounded; verdict_map[intervention_answer] returns faithfulness=0.9. run_answer_diagnosis uses scripted answers directly, skipping the generator."}
```

**fc009** (`citation_mismatch`):

```json
{"case_id": "fc009", "question": "Where does the nested document live?", "gold_doc_ids": ["subdir/nested.md"], "expected_label": "no_failure", "answer_label_expected": "citation_mismatch", "baseline": {"retriever": "dense", "top_k": 3}, "intervention": {"retriever": "dense", "top_k": 3}, "baseline_answer": "The nested document lives in the root directory [Source: with_h1.md].", "intervention_answer": "The nested document lives in a subdirectory [Source: subdir/nested.md].", "notes": "baseline_answer cites the wrong source; FakeJudge verdict_map[baseline_answer] returns citation_support=0.2. intervention_answer cites correctly; verdict_map[intervention_answer] returns citation_support=0.9."}
```

Existing cases fc001–fc007 have no `answer_label_expected` field; the loader
defaults it to `""` and `run_answer_diagnosis` skips them.

---

## Required Tests

### `tests/test_judge.py` (new)

- `JudgeVerdict` round-trips through `dataclasses.asdict()` and back
- `JudgeVerdict.answer_correctness` serializes as `null` when `None`
- `FakeJudge()` (no map) returns `default_verdict` for any call
- `FakeJudge(verdict_map={a: v})` returns `v` when `answer == a`, `default_verdict` otherwise
- `FakeJudge` is fully deterministic across repeated calls with same args
- `detect_answer_failure_label` with faithfulness=0.3 → `unsupported_answer`
- `detect_answer_failure_label` with faithfulness=0.8, citation_support=0.3 → `citation_mismatch`
- `detect_answer_failure_label` with faithfulness=0.9, citation_support=0.9 → `no_failure`

### `tests/test_judge_openai.py` (new, gated)

- Gate 1: `if os.environ.get("TINY_RAG_LAB_TEST_JUDGE") != "1": pytest.skip(..., allow_module_level=True)`
- Gate 2: `pytest.importorskip("openai")`
- `OpenAIJudge()` construction has no side effects (no API call)
- `OpenAIJudge.judge(...)` returns valid `JudgeVerdict` with all scores in [0, 1]

### Updates To `tests/test_eval_runner.py`

- `load_eval_samples` on rows with and without `reference_answer`/`expected_facts`
- `run_answer_eval` with `FakeJudge` + `FakeGenerator` returns `AnswerEvalReport`
- `mean_faithfulness`, `mean_answer_relevance`, `mean_citation_support` aggregate correctly
- `mean_answer_correctness` is `None` when no samples have `reference_answer`
- `mean_answer_correctness` is a float when at least one sample has it
- `reranker + rerank_top_n` thread correctly into `run_answer_eval`
- `reranker not None` + `rerank_top_n=None` raises `ValueError`

### Updates To `tests/test_eval_metrics.py`

- `format_answer_eval_report` prints all metric lines including `Answer Correctness` when float
- `format_answer_eval_report` omits `Answer Correctness` line when `mean_answer_correctness` is `None`
- `format_eval_report` (retrieval) is unchanged

### Updates To `tests/test_trace.py`

- `AskTrace` with `verdict=None` serializes `"verdict": null`
- `AskTrace` with `verdict=JudgeVerdict(...)` serializes all verdict fields
- `format_ask_trace` with `verdict=None` omits the verdict block
- `format_ask_trace` with `verdict` populated appends judge verdict block after citations

### Updates To `tests/test_cmd_eval.py`

- `--judge fake --generator fake` exits 0, stdout contains answer quality section
- `--judge none` exits 0, stdout has no answer quality section (identical to Phase 1.9)
- `--judge openai` with missing `--api-key` and `OPENAI_API_KEY` unset exits non-zero with clear message

### Updates To `tests/test_cmd_ask.py`

- `--judge fake --generator fake` exits 0, `format_ask_trace` output contains `Judge verdict` block
- `--judge none` exits 0, no `Judge verdict` block

### Updates To `tests/test_failure.py`

- `load_failure_cases` with fc008/fc009 rows populates `answer_label_expected` correctly
- `load_failure_cases` on fc001–fc007 (no field) sets `answer_label_expected=""`
- `run_answer_diagnosis` with fc008/fc009 + `FakeJudge(verdict_map={answer: verdict})` (no FakeGenerator needed — scripted answers bypass it):
  - fc008: `verdict_map[baseline_answer]` has faithfulness=0.1; baseline → `unsupported_answer`; intervention → `no_failure`; `fixed=True`
  - fc009: `verdict_map[baseline_answer]` has citation_support=0.2; baseline → `citation_mismatch`; intervention → `no_failure`; `fixed=True`
  - `n_confirmed==2`, `n_fixed==2`
- `run_answer_diagnosis` skips cases with `answer_label_expected=""`
- All existing fc001–fc007 retrieval tests still pass

### Updates To `tests/test_cmd_diagnose.py`

- `--judge fake` exits 0 and stdout contains answer diagnosis section
- `--judge none` output identical to Phase 1.9 (no answer section)

### CLI Smoke (Documentation, Not CI)

```
uv run rag eval --qa-file tests/fixtures/eval/qa.jsonl \
  --judge openai --model gpt-4o-mini --api-key $OPENAI_API_KEY \
  --index-dir PATH --retriever dense --top-k 3
```

---

## Acceptance Criteria

Phase 2.0 is complete when:

1. `tiny_rag_lab/judge.py` exists with `JudgeVerdict`, `Judge` protocol,
   `FakeJudge`, `OpenAIJudge`.
2. `EvalSample` gains `reference_answer` and `expected_facts` with back-compat
   defaults; existing `qa.jsonl` rows load unchanged.
3. `run_answer_eval` and `format_answer_eval_report` exist in `eval.py`.
4. `rag eval --judge fake --generator fake` runs fully offline (no LLM call)
   and prints retrieval + answer quality sections.
5. `rag eval --judge none` output identical to Phase 1.9.
6. `rag ask --judge fake --generator fake` runs fully offline; `AskTrace.verdict`
   is populated; formatter shows verdict block.
7. `rag ask --judge none` output identical to Phase 1.9.
8. `LABEL_UNSUPPORTED_ANSWER`, `LABEL_CITATION_MISMATCH`,
   `detect_answer_failure_label`, `run_answer_diagnosis`, and
   `format_answer_diagnosis_report` exist in `failure.py`.
9. fc008 and fc009 in fixture; fc001–fc007 unchanged.
10. `uv run pytest --tb=short -q` green with no regressions.
11. Default `uv run pytest` does not call any real LLM.
12. `tests/test_judge_openai.py` skips without `TINY_RAG_LAB_TEST_JUDGE=1`
    (env flag checked before `openai` import — same two-gate pattern as
    Phase 1.9 cross-encoder fix).

---

## Learning Notes

- The judge introduces LLM-as-evaluator as a distinct concept separate from
  LLM-as-generator. A learner can compare `rag ask` output with and without
  `--judge fake` and see the verdict fields appear.
- Keeping `faithfulness`, `answer_relevance`, and `citation_support` as
  separate scores rather than one composite teaches why each matters: an
  answer can be relevant but hallucinated, or faithful but incomplete.
- The `answer_correctness` conditional design teaches that correctness
  requires a ground truth — it cannot be inferred from the context alone.
- fc008 and fc009 make the two answer-side failure modes concrete: a learner
  can see the scripted verdict change from low faithfulness / low citation
  support to a passing score between baseline and intervention.
- The opt-in design (`--judge none`) keeps Phase 1.9 behavior reachable for
  comparison, supporting the project's "make the mechanics visible" philosophy.
