# Current Task

Task:         P1-T13, P1-T14
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

- `uv run pytest tests/test_cmd_index_retrieve.py --tb=short -q`: pass, 17 passed in 0.13s
- `uv run pytest tests/test_cli.py --tb=short -q`: pass, 7 passed in 0.01s
- `uv run pytest --tb=short -q`: pass, 189 passed in 4.78s
- `uv run rag index --help`: pass
- `uv run rag retrieve --help`: pass

## Blocker

- none

## Notes

## Handoff

### Task Summary

Implemented `cmd_index` (T13) and `cmd_retrieve` (T14) in `tiny_rag_lab/cli.py`,
replacing the `NotImplementedError` stubs. Both use `_make_embedder()` — a
thin factory helper — so tests can patch it with `FakeEmbedder` without
changing the CLI interface or adding any `--embedder` flag.

### Files Changed

- `tiny_rag_lab/cli.py`: implemented `cmd_index`, `cmd_retrieve`, added `_make_embedder` helper
- `tests/test_cmd_index_retrieve.py`: 17 tests for T13 and T14

### Design Decisions

- **`_make_embedder(model_name)`**: a module-level factory function rather than
  a constructor call inline. Tests patch this with `unittest.mock.patch` to
  inject `FakeEmbedder`; the real CLI path goes through
  `SentenceTransformerEmbedder` unchanged. No `--embedder` CLI flag needed.
- **`getattr(embedder, "model_name", backend)`**: `FakeEmbedder` has no
  `model_name`; fallback to `type(embedder).__name__` so `cmd_index` prints
  sensibly for any embedder. Same fallback is stored in the manifest.
- **Manifest `embedding_model`**: `cmd_retrieve` reads the model name back from
  the manifest and recreates the same embedder, keeping index and query
  embeddings from the same model.
- **Preview truncation at 200 chars**: spec says "chunk preview"; 200 chars
  captures the gist without flooding the terminal.

### Tests Run

- `uv run pytest tests/test_cmd_index_retrieve.py --tb=short -q`: 17 passed
- `uv run pytest --tb=short -q`: 189 passed

### Known Gaps

- Comprehensive CLI tests with tiny fixture corpus and fake backends are in T20.
  These tests cover the core path; T20 will add edge cases and the `ask` command.

### Questions For Next Agent

- T15 (prompt assembly) is next; it depends only on T12 (done).
- T16 (generation interface) follows T15.
