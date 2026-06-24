# Current Task

Task:         P2.1-T04
Phase:        Phase 2.1
Spec:         docs/phases/phase-2.1-context-budget-structured-answers.md
Taskboard:    docs/phases/phase-2.1-taskboard.md
Owner:        Claude Code
Status:       review
Review Result: signed_off
Reviewer:     Codex
Last Updated: 2026-06-24
Updated By:   Codex

## Findings From Last Review

- none

Previous blocking finding is fixed: negative `--context-budget` is now
validated before the answer-side judge branch in both `cmd_eval` and
`cmd_diagnose`, with tests for the default `--judge none` path.

## Tests Reviewed

- `uv run pytest tests/test_eval_runner.py tests/test_cmd_eval.py tests/test_cmd_diagnose.py --tb=short -q`: 115 passed
- manual CLI smoke confirmed `cmd_eval` and `cmd_diagnose` both raise
  `ValueError` for `--context-budget -1` with default `--judge none`
- `uv run pytest --tb=short -q`: 710 passed, 7 skipped

## Blocker

- none

## Notes

Fix: in `tiny_rag_lab/cli.py`, `context_budget = getattr(args, "context_budget", 0)`
and `if context_budget < 0: raise ValueError(...)` now appear immediately after
`load_index`/`load_failure_cases`, before any conditional branches.
