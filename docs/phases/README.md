# Phase Index

This file routes agents to the current implementation contract. It keeps the
default reading path narrow while preserving proposal and review documents as
historical context.

## Current Phase

No implementation phase is currently active.

Phase 1.6 is complete:

- Spec: `docs/phases/phase-1.6-evaluation-harness.md`
- Taskboard: `docs/phases/phase-1.6-taskboard.md`

Phase 1 is complete:

- Spec: `docs/phases/phase-1-naive-classic-rag.md`
- Taskboard: `docs/phases/phase-1-taskboard.md`

Per `AGENTS.md`, agents should not claim or start new implementation work until
the next phase scope is confirmed and a phase spec/taskboard exists.

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

| Phase | Focus | Completion |
|---|---|---|
| Phase 1.6 | Evaluation harness | Complete; see `docs/phases/phase-1.6-taskboard.md` |
| Phase 1 | Naive classic RAG | Complete; see `docs/phases/phase-1-taskboard.md` |

## Candidate And Deferred Phases

| Phase | Focus | Status |
|---|---|---|
| Phase 1.5 | Retrieval mechanics: BM25, hybrid retrieval, inspection | Directional |
| Phase 1.7 | Observability and debugging | Directional |
| Phase 1.8 | RAG failure lab | Directional |
| Later | Agentic RAG | Deferred |

Directional phases must not be treated as active implementation contracts until
dedicated specs and taskboards are created.
