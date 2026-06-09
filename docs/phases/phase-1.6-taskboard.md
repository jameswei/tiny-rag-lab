# Phase 1.6 Taskboard

This file tracks Phase 1.6 implementation tasks, dependencies, ownership, and
status in one lightweight table.

The implementation contract is
`docs/phases/phase-1.6-evaluation-harness.md`. This taskboard must stay aligned
with that spec. It becomes active only after owner acceptance.

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
| P1.6-T01 | M1.6.0 | EvalSample, EvalResult, EvalReport dataclasses | P1-T04 | todo | unassigned | `dataclasses.asdict()` round-trip for all three types; float fields default to 0.0, bool to False, list to []; `uv run pytest tests/test_eval_metrics.py -k dataclass --tb=short -q`: N passed | All three dataclasses in `tiny_rag_lab/eval.py`; no loading or metric logic |
| P1.6-T02 | M1.6.1 | `load_eval_samples()` and eval fixture | P1.6-T01 | todo | unassigned | Fixture loads 3 EvalSample objects without error; rows with empty `question` or empty `gold_doc_ids` are skipped; field mapping correct; `uv run pytest tests/test_eval_runner.py -k load --tb=short -q`: N passed | Creates `tests/fixtures/eval/qa.jsonl`; no metric or retrieval logic |
| P1.6-T03 | M1.6.2 | Retrieval metric functions | P1.6-T01 | todo | unassigned | Known-input unit tests: hit=True/False, RR=0.5 for rank-2 hit, RR=0.0 for miss, precision=0.5 for half-match, recall=0.5 for half-coverage, zero-division safe; `uv run pytest tests/test_eval_metrics.py -k metric --tb=short -q`: N passed | Four pure functions in `tiny_rag_lab/eval.py`; inputs are list[str] only; no I/O |
| P1.6-T04 | M1.6.3 | `run_retrieval_eval()` runner | P1.6-T02, P1.6-T03, P1-T12 | todo | unassigned | Returns EvalReport with n_questions=3 for fixture data; per_question has one EvalResult per sample; mean metrics are correct arithmetic means; EvalReport.top_k equals top_k arg; FakeEmbedder used in tests; `uv run pytest tests/test_eval_runner.py --tb=short -q`: N passed | No formatting or file I/O; receives pre-loaded list[EvalSample] |
| P1.6-T05 | M1.6.4 | `format_eval_report()` formatter | P1.6-T01 | todo | unassigned | Output contains "Hit Rate", "MRR", "Context Precision", "Context Recall", n_questions, top_k; values rounded to 3 decimal places; no ANSI escape codes; `uv run pytest tests/test_eval_metrics.py -k format --tb=short -q`: N passed | Returns str only; no printing or file I/O |
| P1.6-T06 | M1.6.4 | `rag eval` CLI command | P1.6-T04, P1.6-T05 | todo | unassigned | `rag eval --help` exits 0; `--qa-file` required (missing exits non-zero); end-to-end test with fake embedder prints all four metric labels; no regressions; `uv run pytest tests/test_cmd_eval.py --tb=short -q`: N passed; full suite 241+ passed | Adds `cmd_eval` and subparser to `cli.py`; no changes to existing commands |
| P1.6-T07 | M1.6.5 | Phase close | P1.6-T01–T06 | todo | unassigned | All P1.6-T01–T06 rows show `done` with reviewer sign-off; `docs/phases/README.md` updated; `uv run pytest --tb=short -q`: 248+ passed; `uv run rag eval --help` exits 0 | Docs only; no production-code changes |

## Review-Sensitive Tasks

These tasks require architecture or code review before being marked `done`:

- `P1.6-T01`: data contracts and field types.
- `P1.6-T03`: metric correctness and edge-case handling.
- `P1.6-T04`: runner aggregation correctness.
- `P1.6-T06`: CLI surface and end-to-end behavior.

## Minimum Phase 1.6 Completion

Minimum Phase 1.6 completion requires `P1.6-T01` through `P1.6-T07`.
