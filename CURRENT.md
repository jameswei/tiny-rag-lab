# Current Task

Task:         P1-T10
Phase:        Phase 1
Spec:         docs/phases/phase-1-naive-classic-rag.md
Taskboard:    docs/phases/phase-1-taskboard.md
Owner:        claude
Status:       review
Review Result: signed_off
Reviewer:     codex
Last Updated: 2026-06-07
Updated By:   codex

## Findings From Last Review

- none

## Tests Reviewed

- `uv run pytest tests/test_index_writer.py --tb=short -q`: pass, 23 passed in 0.08s
- `uv run pytest --tb=short -q`: pass, 132 passed in 3.33s
- manual artifact readback: pass, wrote `manifest.json`, `chunks.jsonl`, and `embeddings.npz`; manifest count, float32 embedding matrix, and chunk ID order verified

## Blocker

- none

## Notes

## Handoff

### Task Summary

Added `tiny_rag_lab/index_writer.py` with a single public function `write_index`
that persists a built index to disk as three files: `manifest.json`,
`chunks.jsonl`, and `embeddings.npz`.

### Files Changed

- `tiny_rag_lab/index_writer.py`: new module — `write_index`, three private helpers
- `tests/test_index_writer.py`: 23 tests covering manifest fields, JSONL roundtrip, NPZ shape/dtype/values, shape mismatch error, empty corpus

### Design Decisions

- **`write_index` signature**: takes `docs`, `chunks`, `embeddings`, and named
  keyword args for all index metadata. Keeps the function pure — no Embedder
  instance required, just the backend name/model/dim strings already known by
  the caller.
- **Shape guard**: raises `ValueError` immediately if `embeddings.shape !=
  (len(chunks), embedding_dim)`, not a silent corrupt write.
- **`index_dir.mkdir(parents=True, exist_ok=True)`**: caller does not need to
  pre-create the directory tree.
- **`chunks.jsonl` excludes embeddings**: per spec. Vectors live only in
  `embeddings.npz`; the JSONL is human-readable and importable without NumPy.
- **`created_at` is UTC ISO-8601** with timezone info (uses `datetime.now(timezone.utc)`).
- **`embeddings.npz` stores `chunk_ids`** as a parallel string array so the
  loader can reconstruct the mapping without reading JSONL first.

### Tests Run

- `uv run pytest tests/test_index_writer.py --tb=short -q`: 23 passed
- `uv run pytest --tb=short -q`: 132 passed

### Known Gaps

- T11 (index loader) is the natural next step to verify round-trip fidelity.

### Questions For Next Agent

- T11 (index loader) and T12 (cosine retrieval) are unblocked once T10 is done.
