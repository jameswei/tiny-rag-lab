# Current Task

Task:         P1-T12
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

- `uv run pytest tests/test_retrieval.py --tb=short -q`: pass, 19 passed in 0.06s
- `uv run pytest --tb=short -q`: pass, 172 passed in 3.38s
- manual edge-case check: `top_k=0` returns `[]`; `top_k=-1` raises `ValueError`

## Blocker

- none

## Notes

## Handoff

### Task Summary

Added `tiny_rag_lab/retrieval.py` with two public functions: `retrieve_by_vector`
(core ranking logic with known vectors) and `retrieve` (embeds query then
delegates). Both return `list[RetrievalResult]` ranked highest-first.

### Files Changed

- `tiny_rag_lab/retrieval.py`: new module — `retrieve`, `retrieve_by_vector`, `DEFAULT_TOP_K`
- `tests/test_retrieval.py`: 19 tests using known vectors for deterministic ranking and `top_k` validation

### Design Decisions

- **Two-function split**: `retrieve_by_vector` accepts a pre-computed vector so
  tests can verify exact rankings with known vectors without needing an embedder.
  `retrieve` embeds a text string and delegates — the CLI will call `retrieve`.
- **Zero query vector → empty list**: no meaningful similarity ranking is possible
  so returning empty is preferable to returning arbitrary results.
- **Zero-norm index rows → score 0.0**: divide by 1.0 (safe_norms trick) keeps
  the division safe; the zero mask then explicitly sets those scores to 0.0 so
  the result is deterministic regardless of the unit vector direction.
- **Re-normalization of inputs**: both query and index vectors are normalized
  inside `retrieve_by_vector` so callers may pass raw or pre-normalized vectors
  with identical results. Confirmed by `test_unnormalized_query_gives_same_ranking`.
- **`np.argsort` over `np.argpartition`**: O(n log n) but simpler and correct
  for Phase 1 corpus sizes. Can switch to argpartition later if needed.
- **Scores in [-1, 1]**: cosine similarity of L2-unit vectors. Verified by
  `test_scores_in_valid_range` over 20 random unit vectors.

### Tests Run

- `uv run pytest tests/test_retrieval.py --tb=short -q`: 19 passed
- `uv run pytest --tb=short -q`: 172 passed

### Known Gaps

- none

### Questions For Next Agent

- T13 (`rag index` CLI) and T14 (`rag retrieve` CLI) are next in M1.6.
- T15 (prompt assembly) depends on T12 — now unblocked.
