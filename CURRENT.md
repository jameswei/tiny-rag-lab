# Current Task

Task:         P2.0-T04
Phase:        Phase 2.0 — Answer Quality Judging
Spec:         docs/phases/phase-2.0-answer-quality-judging.md
Taskboard:    docs/phases/phase-2.0-taskboard.md
Owner:        Claude Sonnet 4.6
Status:       done
Review Result: signed_off
Reviewer:     codex
Last Updated: 2026-06-17
Updated By:   codex

## Findings From Last Review

No remaining T04 findings.

## Latest Review Result

Signed off by codex on 2026-06-17.

## Tests Reviewed

- `uv run pytest tests/test_trace.py tests/test_cmd_ask.py --tb=short -q`: 73 passed
- `uv run pytest --tb=short -q`: 618 passed, 2 skipped
- CLI smoke with fixture index and `--trace-out`:
  - `--judge none --generator fake`: no `Judge verdict` text; JSON `verdict` is `null`
  - `--judge fake --generator fake`: prints `Judge verdict`; JSON `verdict.judge_name` is `fake`

## Blocker

- none

## Notes

### Files Changed

- `tiny_rag_lab/trace.py`: `AskTrace.verdict: JudgeVerdict | None = None` (default
  None, serializes as null via dataclasses.asdict()); `format_ask_trace` appends
  Judge verdict block after citations when verdict is not None; verdict block
  shows Faithfulness/Answer Relevance/Citation Support, optional Answer Correctness
  (omitted when None), optional Notes (omitted when empty)
- `tiny_rag_lab/cli.py`: `cmd_ask` reads `--judge`/`--generator` flags, calls
  `_make_judge` and `_make_generator_from_flag`; judge called after generation,
  verdict stored in AskTrace; `build_parser` adds `--judge` and `--generator`
  to `p_ask`
- `tests/test_trace.py`: AskTrace.verdict serialization (None→null, populated→all
  fields), format_ask_trace verdict block (no block when None, header/scores/
  correctness/notes when set)
- `tests/test_cmd_ask.py`: parser defaults, `--judge none` no verdict block,
  `--judge fake` shows verdict block after Answer section

### Design Decisions

- `--judge none` path is byte-identical to Phase 1.9 — verdict is None, no block
- `verdict` field is last in AskTrace so existing positional consumers are unaffected
- `JudgeVerdict` is a dataclass with JSON-native fields; `dataclasses.asdict()`
  nests it cleanly under the `verdict` key with no custom encoder needed
