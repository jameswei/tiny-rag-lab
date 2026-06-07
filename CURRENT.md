# Current Task

Task:         P1-T07, P1-T08
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

- `uv run pytest tests/test_chunking.py tests/test_embeddings.py --tb=short -q`: pass, 33 passed in 0.06s
- `uv run pytest --tb=short -q`: pass, 98 passed in 0.09s
- manual boundary check: pass, lengths 799/800/1480 no longer emit redundant contained tail chunks

## Blocker

- none

## Notes

## Handoff

### Task Summary

T07: deterministic fixed-size character chunker in chunking.py. The slice
invariant (normalized_text[char_start:char_end] == chunk.text) is the
central correctness property and is asserted in four separate tests.

T08: Embedder ABC + FakeEmbedder in embeddings.py. FakeEmbedder uses a
SHA-256-seeded NumPy RNG to produce deterministic float32 unit vectors of
any dimension — no model downloads required.

### Files Changed

- `tiny_rag_lab/chunking.py`: chunk_document, chunk_documents
- `tiny_rag_lab/embeddings.py`: Embedder (ABC), FakeEmbedder
- `tests/test_chunking.py`: 20 tests — slice invariant, overlap, stable IDs, metadata, validation errors
- `tests/test_embeddings.py`: 12 tests — shape/dtype, determinism, unit vectors, fixture retrieval

### Design Decisions

- **step = chunk_size - chunk_overlap**: window advances by this amount each
  iteration; validated to be > 0 (overlap < chunk_size).
- **Whitespace-only chunks skipped**: spec requirement; preserves meaningful
  retrieval units.
- **metadata dict is shared across chunks from the same doc**: all chunks
  from a doc carry the same title/path/format/raw_hash. Immutable in practice.
- **FakeEmbedder dim=8 default**: small enough for fast tests, large enough
  to show correct shapes. Callers can override.
- **Empty list returns (0, dim) shape**: `np.empty((0, self.dim))` avoids
  shape inconsistency that `np.array([])` would produce.
- **T07 is review-sensitive** per taskboard; slice invariant and chunk ID
  stability are the properties Codex should verify.

### Tests Run

- `uv run pytest --tb=short -q`: 98 passed (0 failures, 0 skipped)
- Two test bugs caught and fixed during development:
  - `test_text_exactly_chunk_size_gives_one_chunk` needed `chunk_overlap=0`
    (default 120 produces a second tail chunk for 800-char text)
  - `test_embed_empty_list` required empty-list guard in FakeEmbedder.embed

### Known Gaps

- None. T09 (sentence-transformers embedder) depends on T08 and can start.
  T10 (index writer) depends on T04, T07, T08 and can start.

### Questions For Next Agent

- None.
