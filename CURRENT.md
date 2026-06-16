# Current Task

Task:         P1.9-T05
Phase:        Phase 1.9 — Reranking
Spec:         docs/phases/phase-1.9-reranking.md
Taskboard:    docs/phases/phase-1.9-taskboard.md
Owner:        unassigned
Status:       todo
Review Result: pending
Reviewer:     
Last Updated: 2026-06-17
Updated By:   whale

## Findings From Last Review

- none

## Tests Reviewed

- none yet

## Blocker

- none

## Notes

### Prior Tasks

P1.9-T02, T03, T04 are done (signed off by codex 2026-06-17). Full suite:
536 passed, 8 skipped.

### Handoff

T05 is unblocked (depends on T01 and T03, both done). Scope:

- `trace.py`: add `reranker` / `rerank_top_n` fields to `AskTrace`
- `trace.py`: update `format_ask_trace` to render reranker info and
  `pre_rerank_*` fields
- `cli.py`: add `--reranker` / `--rerank-top-n` / `--reranker-model` flags
  to `rag ask` subparser
- `cli.py`: two-stage retrieve flow in `cmd_ask` before `assemble_prompt`
- `tests/test_cmd_ask.py`: new reranker tests

T06 is also unblocked (depends on T01 and T03, both done).

### Files Carrying State From T02-T04

- `tiny_rag_lab/reranker.py`: `CrossEncoderReranker`, `FakeReranker`,
  `apply_reranker`, `chunk_traces_from_rerank`
- `tiny_rag_lab/trace.py`: `RetrieveTrace` already has `reranker` /
  `rerank_top_n`; `format_retrieve_trace` already renders them
- `tiny_rag_lab/cli.py`: `_make_reranker` factory exists
