# Current Task

Task:         P1-T15
Phase:        Phase 1
Spec:         docs/phases/phase-1-naive-classic-rag.md
Taskboard:    docs/phases/phase-1-taskboard.md
Owner:        claude
Status:       review
Review Result: signed_off
Reviewer:     codex
Last Updated: 2026-06-08
Updated By:   codex

## Findings From Last Review

- none

## Tests Reviewed

- `uv run pytest tests/test_prompting.py --tb=short -q`: pass, 24 passed in 0.01s
- `uv run pytest --tb=short -q`: pass, 213 passed in 3.41s

## Blocker

- none

## Notes

## Handoff

### Task Summary

Added `tiny_rag_lab/prompting.py` with `assemble_prompt` and `format_source_table`.
The prompt template and context block template are module-level strings —
visible and auditable per spec.

### Files Changed

- `tiny_rag_lab/prompting.py`: new module — `PROMPT_TEMPLATE`, `CONTEXT_BLOCK_TEMPLATE`, `assemble_prompt`, `format_source_table`, `_format_context_block`
- `tests/test_prompting.py`: 24 tests covering required instructions, context block fields, rank order, empty results, and source table output

### Design Decisions

- **Templates as module-level strings**: spec says "project-owned visible template".
  Both `PROMPT_TEMPLATE` and `CONTEXT_BLOCK_TEMPLATE` are named constants,
  not buried inside functions, so a learner can read and modify the exact
  prompt text without digging through logic.
- **`assemble_prompt(question, results)`**: takes `list[RetrievalResult]` directly
  so it composes naturally after `retrieve()`. No intermediate format conversion.
- **Empty results → valid prompt**: produces a prompt with no `[Source: ...]`
  blocks. The instruction "if context is insufficient, say so" still applies
  and the model can respond accordingly.
- **`format_source_table`**: separate from `assemble_prompt` so the CLI can
  print it after the answer without it being part of the LLM context.
- **Citation format**: `[Source: {chunk_id}]` exactly as in the spec. The same
  marker appears in context blocks so the model can pick it up and repeat it.

### Tests Run

- `uv run pytest tests/test_prompting.py --tb=short -q`: 24 passed
- `uv run pytest --tb=short -q`: 213 passed

### Known Gaps

- none

### Questions For Next Agent

- T16 (generation interface and fake generator) depends on T15 — now unblocked.
- T18 (`rag ask`) is the first task that wires all three planes together.
