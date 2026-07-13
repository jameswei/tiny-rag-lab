# Phase 1.5 Record: Retrieval Mechanics

**Status:** Complete
**Completion:** Reviewed and closed in the original phase; detailed execution
evidence was consolidated after Phase 3.0.

## Delivered

- Dense, BM25, and dense-plus-BM25 hybrid retrieval for `rag retrieve` and
  `rag eval`.
- A visible whitespace/lowercase BM25 tokenizer and Reciprocal Rank Fusion
  (RRF) for hybrid ranking.
- Retriever identity recorded in evaluation output and traces.

## Durable Decisions

- Dense retrieval remains the default; BM25 and hybrid retrieval are explicit
  comparisons, not hidden fallbacks.
- BM25 scores are raw backend scores, while hybrid ranking uses rank-based RRF
  rather than pretending dense and keyword scores share a scale.
- BM25 is rebuilt from the loaded local index at query time; the persisted
  learning artifact remains the readable NumPy/file index.

## Follow-on Context

Cross-encoder reranking was intentionally separate from base retrieval and was
added in Phase 1.9. The simple tokenizer remains a teaching baseline, not a
language-aware production analyzer.
