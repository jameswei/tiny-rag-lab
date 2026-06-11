# Phase 1.8 Taskboard

This file tracks Phase 1.8 implementation tasks, dependencies, ownership, and
status in one lightweight table.

The implementation contract is `docs/phases/phase-1.8-failure-lab.md`. This
taskboard must stay aligned with that spec. The phase scope proposal was reviewed
and signed off by Codex on 2026-06-11. It becomes active only after owner
acceptance and activation in `docs/phases/README.md`.

## Status Values

- `todo`: not started
- `in_progress`: actively being implemented
- `review`: implementation is ready for review and verification
- `blocked`: cannot proceed; blocker must be written in `Notes`
- `done`: reviewed, tested, and accepted

## Update Rules

- Set `Status` to `in_progress` before starting work.
- Set `Status` to `review` after implementation and local tests.
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
| P1.8-T01 | M1.8.0 | `failure.py`: `LABEL_*` constants + `RetrieverConfig`, `FailureCase`, `DiagnosisResult`, `DiagnosisReport`, `DetectionThresholds` dataclasses | — | done | claude | All five dataclasses and five constants exist; `dataclasses.asdict()` round-trip for all; all fields JSON-native; `FailureCase` with `gold_doc_ids: []` round-trips correctly; `DiagnosisResult` includes `baseline_retrieved_doc_ids` and `intervention_retrieved_doc_ids` as `list[str]`; default values correct; `uv run pytest tests/test_failure.py -k dataclass --tb=short -q`: N passed | Reviewed by Codex on 2026-06-11; no findings; `uv run pytest tests/test_failure.py -k dataclass --tb=short -q`: 18 passed; prior full suite by Claude: `uv run pytest --tb=short -q`: 438 passed |
| P1.8-T02 | M1.8.1 | Fixture: `tests/fixtures/corpus/section_alpha.md` + `section_beta.md`; `tests/fixtures/failure/cases.jsonl` (6 cases); `load_failure_cases(path)` in `failure.py` | P1.8-T01 | done | claude | `load_failure_cases` returns 6 `FailureCase` objects; fc005 (`gold_doc_ids: []`) not skipped; empty `case_id` row skipped; malformed JSON row skipped; `baseline` field deserializes to `RetrieverConfig`; no regressions in existing tests; `uv run pytest tests/test_failure.py -k load --tb=short -q && uv run pytest tests/ --tb=short -q` | Reviewed by Codex on 2026-06-11; no findings; `uv run pytest tests/test_failure.py -k load --tb=short -q`: 11 passed; `uv run pytest --tb=short -q`: 461 passed |
| P1.8-T03 | M1.8.2 | `detect_failure_label(retrieved_doc_ids, gold_doc_ids, expected_label, thresholds)` in `failure.py` | P1.8-T01 | done | claude | Known-input tests for all 6 label paths; `low_rank_evidence` fires before `distractor_evidence` — a test with gold at rank 4 and precision < 0.5 returns `low_rank_evidence` not `distractor_evidence`; `distractor_evidence` fires only when gold rank ≤ threshold but precision < threshold; custom `DetectionThresholds` respected; calls `hit_at_k`, `context_precision_at_k`, `reciprocal_rank` from `eval.py` (no reimplementation); `uv run pytest tests/test_failure.py -k detect --tb=short -q`: N passed | Reviewed by Codex on 2026-06-11; no findings after reciprocal_rank fix; `uv run pytest tests/test_failure.py -k detect --tb=short -q`: 16 passed; `uv run pytest --tb=short -q`: 462 passed |
| P1.8-T04 | M1.8.3 | `run_diagnosis(cases, index, embedder, thresholds) -> DiagnosisReport` in `failure.py` | P1.8-T02, P1.8-T03 | done | claude | Returns `DiagnosisReport` with `n_cases == 6` for fixture; `n_fixed` correct; fc005 gets `unanswerable_query` label; each `DiagnosisResult` carries `baseline_retrieved_doc_ids` and `intervention_retrieved_doc_ids` populated from actual retrieval; empty cases returns `DiagnosisReport(n_cases=0)`; `None` embedder with dense case raises `ValueError`; BM25-only cases accept `None` embedder; `uv run pytest tests/test_failure.py -k runner --tb=short -q`: N passed | Reviewed by Codex on 2026-06-11; no findings; `uv run pytest tests/test_failure.py -k runner --tb=short -q`: 11 passed; `uv run pytest --tb=short -q`: 485 passed |
| P1.8-T05 | M1.8.4 | `format_diagnosis_report(report) -> str` in `failure.py` | P1.8-T01 | done | claude | Output contains `n_cases` count; contains confirmed/fixed/moved counts; each `DiagnosisResult` block has `case_id`, baseline+intervention metric lines, outcome word (FIXED/MOVED/CONFIRMED/UNCHANGED); no ANSI codes (`\x1b` absent); floats at 3 decimal places; `uv run pytest tests/test_failure.py -k format --tb=short -q`: N passed | Reviewed by Codex on 2026-06-11; no findings; `uv run pytest tests/test_failure.py -k format --tb=short -q`: 12 passed; `uv run pytest --tb=short -q`: 485 passed |
| P1.8-T06 | M1.8.5 | `rag diagnose --cases-file PATH [--index-dir PATH]` CLI command; `cmd_diagnose` in `cli.py` | P1.8-T04, P1.8-T05 | done | claude | `rag diagnose --help` exits 0 showing `--cases-file`; missing `--cases-file` exits non-zero; end-to-end with FakeEmbedder prints `Diagnosis report  (n=6)`; no regressions; `uv run pytest tests/test_cmd_diagnose.py --tb=short -q`: N passed; `uv run pytest --tb=short -q`: 420+ passed | Reviewed by Codex on 2026-06-11; no findings; `uv run pytest tests/test_cmd_diagnose.py --tb=short -q`: 12 passed; `uv run rag diagnose --help`: exits 0 and shows `--cases-file`; `uv run pytest --tb=short -q`: 497 passed |
| P1.8-T07 | M1.8.6 | Phase close | P1.8-T01–T06 | todo | unassigned | All P1.8-T01–T06 rows `done` with reviewer sign-off; `docs/phases/README.md` updated; `uv run pytest --tb=short -q`: all passed; `uv run rag diagnose --help` exits 0 | |

## Review-Sensitive Tasks

These tasks require architecture or code review before being marked `done`:

- `P1.8-T01`: data contracts — `FailureCase.gold_doc_ids` may be empty (diverges
  from `EvalSample`); `RetrieverConfig` nesting in `FailureCase` must remain
  JSON-native.
- `P1.8-T03`: detection logic imports from `eval.py`; verify no metric math is
  redefined.
- `P1.8-T04`: `BM25Retriever` build-once pattern and `None`-embedder guard
  must match `run_retrieval_eval` behavior.
- `P1.8-T06`: `cmd_diagnose` must not add `--retriever` or `--top-k` flags in
  this phase (all config comes from per-case JSONL).

## Minimum Phase 1.8 Completion

Minimum Phase 1.8 completion requires `P1.8-T01` through `P1.8-T07`.
