# Current Task

Task:         P1-T11
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

- `uv run pytest tests/test_index_loader.py --tb=short -q`: pass, 21 passed in 0.09s
- `uv run pytest --tb=short -q`: pass, 153 passed in 3.57s
- manual round-trip/corruption check: pass, loaded manifest/chunk IDs/embedding shape and raised `ValueError` on corrupted embedding row count

## Blocker

- none

## Notes

## Handoff

### Task Summary

Added `tiny_rag_lab/index_loader.py` with `load_index(index_dir)` that reads
the three files written by `write_index` and returns a `LoadedIndex` dataclass.
Validates chunk ID agreement between `chunks.jsonl` and `embeddings.npz`.

### Files Changed

- `tiny_rag_lab/index_loader.py`: new module — `LoadedIndex` dataclass, `load_index`, `_load_chunks`
- `tests/test_index_loader.py`: 21 tests covering roundtrip fidelity, chunk IDs, embeddings shape/dtype/values, missing file errors, chunk_id mismatch error, empty index

### Design Decisions

- **`LoadedIndex` dataclass**: groups manifest, chunks, embeddings, and chunk_ids together so callers don't receive a tuple of loosely related values.
- **chunk_ids validation**: compares the list from `chunks.jsonl` against the array in `embeddings.npz` and raises `ValueError` on any divergence — catches corrupt or mismatched writes early.
- **`allow_pickle=True` on np.load**: required for loading string arrays (`chunk_ids`) from `.npz`; safe here because we control the write path.
- **Tests use `write_index` as the fixture**: tests the loader against real writer output rather than hand-crafted bytes, ensuring the round-trip is always consistent.

### Tests Run

- `uv run pytest tests/test_index_loader.py --tb=short -q`: 21 passed
- `uv run pytest --tb=short -q`: 153 passed

### Known Gaps

- none

### Questions For Next Agent

- T12 (cosine retrieval) depends on T08 and T11 — both done. Can start.
