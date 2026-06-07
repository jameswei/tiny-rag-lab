# Phase Index

This file routes agents to the current implementation contract. It keeps the
default reading path narrow while preserving proposal and review documents as
historical context.

## Current Phase

Phase 1 is review-ready but not active for implementation until the owner
accepts both documents:

- Spec: `docs/phases/phase-1-naive-classic-rag.md`
- Taskboard: `docs/phases/phase-1-taskboard.md`

Per `AGENTS.md`, agents should not claim or start implementation work until the
owner confirms Phase 1 is active.

## Agent Reading Rule

For normal work, agents should read:

1. `AGENTS.md`
2. `docs/file-structure.md`
3. `docs/agent-guidelines.md`
4. this file
5. the active phase spec and taskboard listed in `Current Phase`

Read `docs/proposal.md`, `docs/roadmap.md`, and `docs/architecture.md` when a
task changes scope, architecture, roadmap, or public interfaces.

## Completed Phases

None yet.

## Candidate And Deferred Phases

| Phase | Focus | Status |
|---|---|---|
| Phase 1 | Naive classic RAG | Review-ready |
| Phase 1.5 | Retrieval mechanics: BM25, hybrid retrieval, inspection | Directional |
| Phase 1.6 | Evaluation harness | Directional |
| Phase 1.7 | Observability and debugging | Directional |
| Phase 1.8 | RAG failure lab | Directional |
| Later | Agentic RAG | Deferred |

Directional phases must not be treated as active implementation contracts until
dedicated specs and taskboards are created.
