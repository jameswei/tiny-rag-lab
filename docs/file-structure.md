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

Phase 1 through 2.0 implementation files:

```text
tiny_rag_lab/
  __init__.py
  bm25.py             (Phase 1.5: BM25Retriever and visible tokenization)
  cli.py
  documents.py
  chunking.py
  embeddings.py
  eval.py              (Phase 1.6 + 1.5: metrics and retriever-aware eval runner)
  failure.py           (Phase 1.8 + 2.0: retrieval and answer-side diagnosis)
  hybrid.py           (Phase 1.5: RRF and dense+BM25 retrieval)
  index_loader.py
  index_writer.py
  judge.py             (Phase 2.0: judge contracts, fake judge, OpenAI-compatible judge)
  models.py
  reranker.py          (Phase 1.9: reranker contracts, fake and cross-encoder rerankers)
  retrieval.py
  prompting.py
  generation.py
  trace.py             (Phase 1.7 + 2.0: retrieve/ask trace records, verdict trace)
scripts/
  prepare_watsonx_docsqa.py
tests/
  fixtures/
    corpus/
    eval/              (Phase 1.6: qa.jsonl fixture)
    failure/           (Phase 1.8: curated failure cases)
  test_*.py
```

Generated data should stay out of version control:

```text
corpus/
.tiny-rag/
```
