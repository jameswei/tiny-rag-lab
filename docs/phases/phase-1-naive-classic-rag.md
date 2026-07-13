# Phase 1 Record: Classic RAG Baseline

**Status:** Complete
**Completion:** Original verification was recorded before this historical
consolidation.

## Delivered

- The Python `rag` CLI with `index`, `retrieve`, and `ask` commands.
- Local Markdown/plain-text loading, visible normalization, deterministic
  fixed-character chunks, source metadata, and stable chunk IDs.
- Local sentence-transformer embeddings, NumPy cosine retrieval, persisted
  file indexes, grounded prompting, citations, and OpenAI-compatible
  generation.
- Prepared watsonxDocsQA support plus deterministic fake embedder/generator
  paths for offline tests.

## Durable Decisions

- The project owns the document-to-prompt mechanics; high-level RAG framework
  wrappers are not the core implementation.
- NumPy/file indexes are the inspectable baseline, and generated corpora and
  indexes stay outside version control.
- Real generation is optional and OpenAI-compatible; tests must remain able to
  run without model downloads or provider credentials.

## Follow-on Context

BM25, hybrid retrieval, evaluation, traces, failure diagnosis, reranking,
context budgets, alternative chunking, and the browser lab were deliberately
added in later phases rather than hidden in the initial baseline.
