# Phase 2.0 Taskboard

This file tracks Phase 2.0 implementation tasks, dependencies, ownership, and
status in one lightweight table.

The implementation contract is `docs/phases/phase-2.0-answer-quality-judging.md`.
Phase 2.0 is a candidate phase — it must be reviewed, signed off, and activated
in `docs/phases/README.md` before implementation begins.

## Status Values

- `todo`: not started
- `in_progress`: actively being implemented
- `review`: implementation is ready for review and verification
- `blocked`: cannot proceed; blocker must be written in `Notes`
- `done`: reviewed, tested, and accepted

## Update Rules

- Set `Status` to `in_progress` before starting work.
- Set `Status` to `review` after implementation and local tests pass.
- Set `Status` to `done` only after review and required tests pass.
- The task owner must not mark their own task `done`; a different reviewing
  agent must sign off and make the `done` update.
- When marking `done`, record the reviewing agent and test result in `Notes`.
- Use `blocked` only with a concrete blocker in `Notes`.
- Keep `Owner` as an agent/person name or `unassigned`.
- Do not change task IDs after creation.
- Update `Notes` with skipped tests, setup limits, or follow-up work.

## Taskboard

| ID | Milestone | Task | Depends On | Status | Owner | Acceptance | Notes |
|---|---|---|---|---|---|---|---|
| P2.0-T01 | M2.0.0 | `judge.py`: `JudgeVerdict` dataclass, `Judge` protocol, `FakeJudge` (default_verdict + optional verdict_map keyed by **answer string**), `detect_answer_failure_label(verdict, thresholds) -> str`, `AnswerDetectionThresholds`; extend `EvalSample` with `reference_answer: str \| None = None` and `expected_facts: list[str] = []`; extend `load_eval_samples` with back-compat defaults; add `AnswerEvalResult` and `AnswerEvalReport` to `eval.py`; extend `FailureCase` with `answer_label_expected: str = ""`, `baseline_answer: str = ""`, `intervention_answer: str = ""`; extend `load_failure_cases` to read all three with defaults | — | todo | unassigned | `JudgeVerdict` round-trips through `dataclasses.asdict()`; `FakeJudge()` returns `default_verdict`; `FakeJudge(verdict_map={a: v})` returns `v` when `answer == a`, `default_verdict` otherwise; `detect_answer_failure_label` correctly maps low faithfulness → `unsupported_answer`, low citation_support → `citation_mismatch`, else `no_failure`; `EvalSample` with and without `reference_answer` loads from existing `qa.jsonl`; `FailureCase` with and without new fields loads from existing `cases.jsonl`; `uv run pytest tests/test_judge.py tests/test_eval_runner.py tests/test_failure.py -k "judge or eval_sample or answer_label" --tb=short -q`: N passed | |
| P2.0-T02 | M2.0.1 | `judge.py`: `OpenAIJudge(model, api_key, base_url=None)` — lazy construction (no API call in `__init__`), JSON-mode prompt, parses JSON response into `JudgeVerdict`, raises `ValueError` on unparseable JSON; `_make_judge(name, model, api_key, base_url)` factory: falls back to `OPENAI_API_KEY` env var before raising; `_make_generator_from_flag(name, args)` factory: `fake` → `FakeGenerator()`, `openai` → existing `_make_generator(args)`; gated `tests/test_judge_openai.py` with env-flag-first two-gate pattern (`TINY_RAG_LAB_TEST_JUDGE=1` checked before `pytest.importorskip("openai")`) | P2.0-T01 | todo | unassigned | `OpenAIJudge()` construction has no side effects; `_make_judge("none", ...)` returns `None`; `_make_judge("fake", ...)` returns `FakeJudge()`; `_make_judge("openai", ...)` raises when neither `--api-key` nor `OPENAI_API_KEY` is set; `_make_judge("openai", ...)` succeeds when `OPENAI_API_KEY` is set even without `--api-key`; `_make_generator_from_flag("fake", ...)` returns `FakeGenerator`; gated test skips without `TINY_RAG_LAB_TEST_JUDGE=1`; `import tiny_rag_lab.judge` does not import `openai` | |
| P2.0-T03 | M2.0.2 | `eval.py`: `run_answer_eval(samples, index, embedder, top_k, retriever, generator, judge, reranker=None, rerank_top_n=None) -> AnswerEvalReport`; `format_answer_eval_report(report) -> str`; `cmd_eval`: **add** `--judge none\|fake\|openai`, `--generator fake\|openai`, `--model`, `--api-key`, `--base-url` (none of these exist on eval today); one model/key/url shared by generator and judge; when `--judge` active: run retrieval eval then answer eval and print both sections | P2.0-T01, P2.0-T02 | todo | unassigned | `run_answer_eval` with `FakeJudge` + `FakeGenerator` returns `AnswerEvalReport`; `mean_answer_correctness` is `None` when no `reference_answer` supplied; float when at least one sample has it; `reranker + rerank_top_n` thread correctly; `run_answer_eval(reranker=X, rerank_top_n=None)` raises `ValueError`; `cmd_eval --judge fake --generator fake` exits 0 and stdout contains answer quality section; `cmd_eval --judge none` output identical to Phase 1.9; `uv run pytest tests/test_eval_runner.py tests/test_eval_metrics.py tests/test_cmd_eval.py --tb=short -q`: N passed | |
| P2.0-T04 | M2.0.3 | `trace.py`: `AskTrace.verdict: JudgeVerdict \| None = None`; `format_ask_trace` appends judge verdict block when `verdict is not None`; `cmd_ask`: add `--judge none\|fake\|openai` and `--generator fake\|openai` (`--model`/`--api-key`/`--base-url` already exist); after `generator.generate(prompt)`, when judge is active, calls `judge.judge(...)` and populates `AskTrace.verdict` | P2.0-T01, P2.0-T02 | todo | unassigned | `AskTrace(verdict=None)` serializes `"verdict": null`; `AskTrace(verdict=JudgeVerdict(...))` serializes all verdict fields; `format_ask_trace` with `verdict=None` has no verdict block; with `verdict` populated contains `Judge verdict` header with all four score lines; `cmd_ask --judge fake --generator fake` exits 0; `cmd_ask --judge none` output identical to Phase 1.9; `uv run pytest tests/test_trace.py tests/test_cmd_ask.py --tb=short -q`: N passed | |
| P2.0-T05 | M2.0.4 | `failure.py`: add `LABEL_UNSUPPORTED_ANSWER`, `LABEL_CITATION_MISMATCH` constants; add `AnswerDiagnosisResult`, `AnswerDiagnosisReport`; add `run_answer_diagnosis(cases, index, embedder, generator, judge, reranker=None, thresholds=None) -> AnswerDiagnosisReport` — skips cases where `answer_label_expected==""`; when `case.baseline_answer`/`case.intervention_answer` is non-empty uses them directly, skipping the generator call; add `format_answer_diagnosis_report(report) -> str`; append fc008 and fc009 (with `baseline_answer`/`intervention_answer`) to `tests/fixtures/failure/cases.jsonl`; `cmd_diagnose`: add `--judge none\|fake\|openai` and `--generator fake\|openai`; when active, runs `run_answer_diagnosis` and appends answer diagnosis section | P2.0-T01, P2.0-T02 | todo | unassigned | fc001–fc007 retrieval tests still pass; fc008 + fc009 load with all new fields; `run_answer_diagnosis` with `FakeJudge(verdict_map={fc008_baseline_answer: low_faith_verdict, ...})` — no FakeGenerator needed because scripted answers bypass generator: fc008 baseline=`unsupported_answer` intervention=`no_failure` fixed=True; fc009 baseline=`citation_mismatch` intervention=`no_failure` fixed=True; `n_confirmed==2`, `n_fixed==2`; skips fc001–fc007 silently; `cmd_diagnose --judge fake --generator fake` exits 0 and stdout contains answer diagnosis section; `cmd_diagnose --judge none` output identical to Phase 1.9; `uv run pytest tests/test_failure.py tests/test_cmd_diagnose.py --tb=short -q`: N passed | |
| P2.0-T06 | M2.0.5 | Phase close: update `docs/phases/README.md` to mark Phase 2.0 complete and set Current Phase to next or No active phase; update `docs/roadmap.md` Phase 2.0 entry to "Complete"; run full suite and CLI smokes; verify `import tiny_rag_lab.judge` does not import `openai` | P2.0-T01–T05 | todo | unassigned | All P2.0-T01–T05 `done` with reviewer sign-off; `uv run pytest --tb=short -q`: all passed; `uv run rag eval --help` shows `--judge` flag; `uv run rag eval --judge none --qa-file tests/fixtures/eval/qa.jsonl --index-dir PATH` matches Phase 1.9 output; `python -c "import tiny_rag_lab.judge; assert 'openai' not in __import__('sys').modules"` succeeds; phase index updated | |

## Review-Sensitive Tasks

These tasks require architecture or code review before being marked `done`:

- **P2.0-T01**: `JudgeVerdict` must be fully JSON-native so `dataclasses.asdict()`
  serializes a full `AskTrace` without a custom encoder. `answer_correctness: float | None`
  must serialize as `null`. `FakeJudge.verdict_map` is keyed by answer string (not query).
  All three new `FailureCase` defaults must not break existing `run_diagnosis` calls.
- **P2.0-T02**: Two-gate order in `test_judge_openai.py` — env flag checked before
  `openai` import. `_make_judge("openai", ...)` must fall back to `OPENAI_API_KEY`
  env var before raising. `OpenAIJudge.__init__` must not call the API.
- **P2.0-T03**: All five flags (`--judge`, `--generator`, `--model`, `--api-key`,
  `--base-url`) are **new additions** to `rag eval`; do not assume they exist.
  Retrieval and answer reports stay as separate objects. `--judge none` output
  byte-identical to Phase 1.9.
- **P2.0-T04**: `generator.generate(prompt)` takes a prompt string, not
  `(query, chunks)`. `AskTrace.verdict` default is `None`, not a mutable factory.
  `--judge none` path on `rag ask` byte-identical to Phase 1.9.
- **P2.0-T05**: fc008/fc009 `expected_label` stays `"no_failure"` (retrieval pass
  succeeds). `run_answer_diagnosis` uses `case.baseline_answer`/`case.intervention_answer`
  directly when non-empty, bypassing the generator — tests do not need a FakeGenerator
  for these cases. Skipping fc001–fc007 must be silent.

## Minimum Phase 2.0 Completion

Minimum Phase 2.0 completion requires P2.0-T01 through P2.0-T06.
