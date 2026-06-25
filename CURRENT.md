# Current Task

Task:         P2.2-T02
Phase:        Phase 2.2
Spec:         docs/phases/phase-2.2-structural-semantic-chunking.md
Taskboard:    docs/phases/phase-2.2-taskboard.md
Owner:        Claude Code
Status:       review
Review Result: signed_off
Reviewer:     Codex
Last Updated: 2026-06-25
Updated By:   Codex

## Findings From Last Review

- none

## Tests Reviewed

- `uv run pytest tests/test_chunking.py --tb=short -q`: 44 passed
- manual batching/packing probe: semantic chunking called `embedder.embed`
  once with all 4 sentences, low-threshold packing produced contiguous
  sentence chunks, and oversized-sentence fallback produced overlapping
  windows
- `uv run python -c "import sys; import tiny_rag_lab.chunking; print('tiny_rag_lab.embeddings' in sys.modules)"`:
  printed `False`
- `uv run pytest --tb=short -q`: 732 passed, 7 skipped

## Blocker

- none

## Handoff

### Task Summary

Added `chunk_document_semantic`: splits `normalized_text` into sentences
(reusing `_split_sentences` from P2.2-T01), embeds all sentences in a single
batch call, then packs them in order — closing a chunk when the next
sentence would exceed `chunk_size` or cosine similarity to the previous
sentence drops below `similarity_threshold`. A single sentence that itself
exceeds `chunk_size` falls back to `_chunk_oversized_span` (same tier-3
fallback as structural chunking; the only place `chunk_overlap` is used).

### Files Changed

- `tiny_rag_lab/chunking.py`: added `TYPE_CHECKING` import of `Embedder`
  (matches the lazy-import convention already used in `eval.py`/`failure.py`
  for cross-module type hints) and `chunk_document_semantic`.
- `tests/test_chunking.py`: 10 new tests — slice invariant, unreachable-low
  `similarity_threshold` (chunking driven purely by `chunk_size`),
  unreachable-high `similarity_threshold` (every sentence its own chunk),
  oversized-single-sentence fallback with overlap, **embedder called
  exactly once with all sentences in one batch** (the review-sensitive
  requirement, verified with a counting wrapper around `FakeEmbedder`),
  determinism across repeated calls with a fresh `FakeEmbedder`,
  metadata/`chunk_id` contract, empty/whitespace input, validation errors.

### Design Decisions

- Similarity is always computed between *consecutive sentences in the
  original sequence* (`vectors[i-1] @ vectors[i]`), not between a sentence
  and some representative of the currently-open chunk. This matches the
  spec's literal wording ("the previous one") and means the topic-shift
  decision is independent of how prior sentences happened to get packed.
- After an oversized-sentence fallback (tier 3), the next sentence starts a
  fresh chunk with no similarity check against the oversized sentence —
  there's no "current chunk" embedding to compare against at that point.
  Not explicitly tested; documented here for the next agent.
- Did not generalize `_pack_units` to support a similarity predicate.
  `chunk_document_semantic` has its own small packing loop because the
  extra topic-shift condition doesn't fit `_pack_units`'s "oversized or not"
  shape without adding a feature only this one caller needs.
- `Embedder` is imported under `TYPE_CHECKING` only (chunking.py already has
  `from __future__ import annotations`, so the runtime annotation is a
  string regardless) — avoids importing `tiny_rag_lab.embeddings` at
  runtime just for a type hint, matching `eval.py`/`failure.py`.

### Tests Run

- `uv run pytest tests/test_chunking.py --tb=short -q`: 44 passed
- `uv run pytest --tb=short -q`: 732 passed, 7 skipped (full suite, no
  regressions)

### Known Gaps

- none

### Learning Notes

- `test_semantic_embedder_called_exactly_once_per_document` is the most
  important test in this batch — it's what actually proves indexing speed
  matches the spec's documented cost tradeoff (one embedding pass over
  sentences, not N passes).
- The unreachable-threshold tests (`-2.0` and `2.0`) are a clean way to
  isolate "chunk_size-only" and "similarity-only" behavior without needing
  a real embedding model that produces semantically meaningful similarity
  scores — `FakeEmbedder`'s vectors are deterministic per-text but not
  semantically related, so only the boundary thresholds are reliably
  testable without a real model.

### Questions For Next Agent

- None.
