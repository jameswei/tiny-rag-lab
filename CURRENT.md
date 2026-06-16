# Current Task

Task:         P1.9-T01
Phase:        Phase 1.9 — Reranking
Spec:         docs/phases/phase-1.9-reranking.md
Taskboard:    docs/phases/phase-1.9-taskboard.md
Owner:        claude
Status:       done
Review Result: signed_off
Reviewer:     codex
Last Updated: 2026-06-16
Updated By:   codex

## Findings From Last Review

- none

## Tests Reviewed

- `uv run pytest tests/test_reranker.py --tb=short -q`: 22 passed
- `uv run pytest tests/test_trace.py tests/test_cmd_retrieve.py tests/test_cmd_ask.py --tb=short -q`: 60 passed
- `uv run pytest --tb=short -q`: 519 passed (Phase 1.8 closed at 497; +22 new
  tests from T01, no regressions)

## Blocker

- none

## Notes

### Handoff

T01 introduces the Phase 1.9 reranking data contracts and composition
helpers. No CLI, eval, ask, or failure-lab wiring yet — those are
T03 / T04 / T05 / T06 and depend on T01.

### Files Changed

- `tiny_rag_lab/reranker.py` (new): `RerankResult` dataclass, `Reranker`
  protocol, `FakeReranker`, `apply_reranker`, `chunk_traces_from_rerank`
- `tiny_rag_lab/trace.py`: added optional `pre_rerank_rank` and
  `pre_rerank_score` fields to `ChunkTrace`, both defaulting to `None`
- `tests/test_reranker.py` (new): 22 tests covering dataclass round-trip,
  FakeReranker noop / score-map / ties / missing chunks, apply_reranker
  slicing and clipping, chunk_traces_from_rerank with audit on / off

### Design Decisions

- `apply_reranker` clips with `min(top_k, len(results))` to match
  `retrieve_by_vector` behavior. Only `top_k < 0` raises. The
  `rerank_top_n >= top_k` invariant is enforced at the CLI / runner layer
  (T03 / T04), not inside `apply_reranker`.
- `FakeReranker` uses Python's stable sort twice (first by tie-break key,
  then by primary key) instead of a compound key tuple, so the tie-break
  policy is visible in the call sequence.
- `Reranker` is a `typing.Protocol`, not an abstract base class.
  Implementations don't need to inherit; they only need a `name: str`
  attribute and a `rerank()` method with the right signature.
- The two new `ChunkTrace` fields are positioned at the end of the
  dataclass with `None` defaults so existing trace consumers and
  positional constructions continue to work unchanged.

### Acceptance Criteria Check

Spec / taskboard acceptance:

- ✓ `RerankResult` round-trips through `dataclasses.asdict()`
- ✓ `FakeReranker(score_map=None)` no-op behavior verified
- ✓ `FakeReranker(score_map=...)` deterministic with `pre_rank` tie-break
- ✓ `apply_reranker` clips to `min(top_k, len(results))`, raises only on
  `top_k < 0`, returns `([], [])` on empty input
- ✓ `chunk_traces_from_rerank(audit=None)` leaves `pre_rerank_*` as `None`
- ✓ Target test command passes: `uv run pytest tests/test_reranker.py
  -k 'dataclass or fake or apply or chunk_traces' --tb=short -q`

### Questions For Next Agent

- none

### Phase Activation

Phase 1.9 was activated under "Current Phase" in `docs/phases/README.md`
as part of this work, per owner direction "please start to work on new
tasks without dependencies of new phase 1.9" on 2026-06-16.
