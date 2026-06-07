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

## Expected Implementation Layout

Phase 1 has not started yet, but the expected layout is:

```text
tiny_rag_lab/
  __init__.py
  cli.py
  documents.py
  chunking.py
  embeddings.py
  index.py
  retrieval.py
  prompting.py
  generation.py
  tracing.py
scripts/
  prepare_watsonx_docsqa.py
tests/
  fixtures/
    corpus/
```

Generated data should stay out of version control:

```text
corpus/
.tiny-rag/
```
