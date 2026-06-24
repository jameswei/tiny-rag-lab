# Current Task

Task:         P2.1-T03
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

## Tests Reviewed

- `uv run pytest tests/test_cmd_ask.py --tb=short -q`: 45 passed
- manual JSON smoke for `rag ask --context-budget 8192 --output-format json`:
  valid JSON trace; `context_pack` populated; answer contained 3 source markers
- `uv run pytest --tb=short -q`: 695 passed, 7 skipped

## Blocker

- none

## Notes

Files changed:
- `tiny_rag_lab/cli.py`: added `_make_token_counter()` factory; added
  `--context-budget INT` (default 0 = unlimited) and `--output-format text|json`
  (default text) to `cmd_ask`; pipeline applies `pack_context` when
  `context_budget > 0`, filters results to selected chunk_ids, attaches
  `pack_result` to `AskTrace.context_pack`; `--output-format json` prints
  `json.dumps(trace_to_dict(trace), indent=2)` to stdout; `--trace-out` still
  writes JSON regardless of `--output-format`; `--context-budget -1` raises
  `ValueError`
- `tests/test_cmd_ask.py`: 20 new tests for T03 acceptance criteria

Key behaviours verified:
- Default output (budget=0) is identical to Phase 2.0 (no Context packing block)
- budget=8192 shows Context packing block before Answer: in text output
- --output-format json produces valid JSON with answer + context_pack keys
- context_pack=null in JSON when budget=0
- verdict in JSON when --judge fake active
- --trace-out writes JSON file even with --output-format json
- Tight budget (budget=1) filters all chunks → fewer source markers than budget=8192
- --context-budget -1 raises ValueError
