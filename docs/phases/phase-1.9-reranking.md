# Phase 1.9 Record: Reranking

**Status:** Complete
**Completion:** Reviewed and closed in the original phase; detailed execution
evidence was consolidated after Phase 3.0.

## Delivered

- An opt-in reranking seam with deterministic fake and local cross-encoder
  implementations.
- `--reranker` and candidate-count controls across retrieval, asking,
  evaluation, and diagnosis.
- Trace and report fields that preserve pre-rerank and post-rerank evidence.

## Durable Decisions

- Base retrieval produces candidates; reranking is a visible second pass, not
  behavior hidden inside a retriever.
- Reranking defaults to `none`; the cross-encoder loads lazily and is never a
  mandatory download for normal use or tests.
- Per-chunk rerank audit information is retained so learners can see both the
  original and final rank.

## Follow-on Context

Provider rerankers, score calibration, and multi-stage reranking are outside
the learning baseline.
