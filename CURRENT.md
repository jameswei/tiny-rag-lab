# Current Task

Task:         P1.6-T04
Phase:        Phase 1.6
Spec:         docs/phases/phase-1.6-evaluation-harness.md
Taskboard:    docs/phases/phase-1.6-taskboard.md
Owner:        claude
Status:       done
Review Result: signed_off
Reviewer:     codex
Last Updated: 2026-06-10
Updated By:   codex

## Findings From Last Review

- none

## Tests Reviewed

- `uv run pytest tests/test_eval_runner.py --tb=short -q`: pass, 24 passed
- `uv run pytest --tb=short -q`: pass, 308 passed (no regressions)
- `uv run python -c "from pathlib import Path; from tiny_rag_lab.eval import load_eval_samples, run_retrieval_eval; from tiny_rag_lab.embeddings import FakeEmbedder; from tiny_rag_lab.documents import load_documents; from tiny_rag_lab.chunking import chunk_documents; from tiny_rag_lab.index_loader import LoadedIndex; corpus=Path('tests/fixtures/corpus'); docs=load_documents(corpus); chunks=chunk_documents(docs, chunk_size=500, chunk_overlap=50); emb=FakeEmbedder(dim=8); index=LoadedIndex(manifest={}, chunks=chunks, embeddings=emb.embed([c.text for c in chunks]), chunk_ids=[c.chunk_id for c in chunks]); samples=load_eval_samples(Path('tests/fixtures/eval/qa.jsonl')); report=run_retrieval_eval(samples, index, emb, top_k=3); print(report.n_questions, report.top_k, len(report.per_question), round(report.hit_rate, 3), round(report.mrr, 3))"`:
  pass, prints `3 3 3 1.0 0.778`

## Blocker

- none

## Notes

Reviewed by codex; T04 accepted. Runner embeds each question, delegates to
`retrieve_by_vector`, records ranked `doc_id` values, computes the four
Phase 1.6 retrieval metrics, and aggregates arithmetic means correctly.

## Handoff

### Task Summary

Implemented `run_retrieval_eval(samples, index, embedder, top_k) -> EvalReport`
in `tiny_rag_lab/eval.py`. For each sample: embeds the question, retrieves
top_k chunks via `retrieve_by_vector`, extracts `chunk.doc_id` for each result,
computes the four metrics, builds an `EvalResult`. Aggregates arithmetic means
into `EvalReport`. Empty samples returns a zero-filled report without dividing.

### Files Changed

- `tiny_rag_lab/eval.py`: added `run_retrieval_eval()`; added `TYPE_CHECKING` imports for `Embedder` and `LoadedIndex`; deferred `retrieve_by_vector` import inside the function
- `tests/test_eval_runner.py`: added 12 runner tests covering report structure, aggregation correctness, top_k, empty samples

### Design Decisions

- `retrieve_by_vector` is imported inside the function body (deferred) to avoid a circular import: `eval.py` → `retrieval.py` → `index_loader.py` → potential back-reference. `TYPE_CHECKING` block handles the type hints without runtime cost.
- `_build_index()` fixture helper builds an in-memory `LoadedIndex` directly, matching the pattern in `test_persistence_roundtrip.py`, no disk I/O needed.
- Aggregation tests verify the formula (`report.hit_rate == mean(r.hit for r in ...)`) rather than hard-coded expected values, so they are independent of FakeEmbedder's specific ranking.

### Tests Run

- `uv run pytest tests/test_eval_runner.py --tb=short -q`: 24 passed
- `uv run pytest --tb=short -q`: 308 passed

### Known Gaps

- none

### Questions For Next Agent

- T06 (`rag eval` CLI) is now fully unblocked — T04 and T05 are both done
