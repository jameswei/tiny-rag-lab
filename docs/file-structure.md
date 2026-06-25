# File Structure

This document gives agents a quick map of the repository. It should stay
concise; detailed behavior belongs in phase specs and architecture docs.

## Planning And Collaboration

```text
AGENTS.md                         agent entrypoint
CURRENT.md                        live task state, created from template when needed
docs/current-task-template.md      template for CURRENT.md
docs/handoff-template.md           implementation handoff note template
docs/agent-guidelines.md           collaboration, review, and handoff rules
docs/phases/README.md              active phase pointer
docs/phases/*.md                   phase specs and taskboards
```

## Project Direction

```text
docs/proposal.md                   project purpose and non-goals
docs/roadmap.md                    proposed phase sequence
docs/architecture.md               conceptual RAG planes and interfaces
```

## Implementation Layout

Phase 1 through 2.2 implementation files:

```text
tiny_rag_lab/
  __init__.py
  bm25.py             (Phase 1.5: BM25Retriever and visible tokenization)
  cli.py              (Phase 2.1: _make_token_counter, --context-budget, --output-format on ask/eval/diagnose; Phase 2.2: --chunking-strategy, --semantic-similarity-threshold on index)
  context.py          (Phase 2.1: TokenCounter protocol, FakeTokenCounter, TiktokenCounter, pack_context, ContextPackResult)
  documents.py
  chunking.py         (Phase 2.2: chunk_document_structural, chunk_document_semantic, chunk_document_with_strategy, chunk_documents_with_strategy)
  embeddings.py
  eval.py              (Phase 1.6 + 1.5 + 2.1: metrics, retriever-aware eval runner, context budget support)
  failure.py           (Phase 1.8 + 2.0 + 2.1: retrieval and answer-side diagnosis, context budget support)
  hybrid.py           (Phase 1.5: RRF and dense+BM25 retrieval)
  index_loader.py
  index_writer.py     (Phase 2.2: chunking_strategy and chunking_params in manifest)
  judge.py             (Phase 2.0: judge contracts, fake judge, OpenAI-compatible judge)
  models.py
  reranker.py          (Phase 1.9: reranker contracts, fake and cross-encoder rerankers)
  retrieval.py
  prompting.py
  generation.py
  trace.py             (Phase 1.7 + 2.0 + 2.1: retrieve/ask trace records, verdict trace, context_pack trace)
scripts/
  prepare_watsonx_docsqa.py
tests/
  fixtures/
    corpus/
    chunking_corpus/   (Phase 2.2: fixed-character vs structural comparison corpus)
    eval/              (Phase 1.6: qa.jsonl fixture)
    failure/           (Phase 1.8: curated failure cases; Phase 2.2: chunking_strategy_cases.jsonl)
  test_*.py
```

Generated data should stay out of version control:

```text
corpus/
.tiny-rag/
```
