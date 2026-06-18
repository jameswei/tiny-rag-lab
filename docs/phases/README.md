# Phase Index

This file routes agents to the current implementation contract. It keeps the
default reading path narrow while preserving proposal and review documents as
historical context.

## Current Phase

<!-- Activation checklist:
     - the phase spec and taskboard are signed off
     - this section names both active files
     - stale Draft/Review-ready/candidate wording in the phase spec/taskboard
       is updated
     - CURRENT.md is created or reset only when a concrete task is claimed or
       ready for review -->

**No active phase.**

Phase 2.0 is complete:

- Spec: `docs/phases/phase-2.0-answer-quality-judging.md`
- Taskboard: `docs/phases/phase-2.0-taskboard.md`

Phase 1.9 is complete:

- Spec: `docs/phases/phase-1.9-reranking.md`
- Taskboard: `docs/phases/phase-1.9-taskboard.md`

Phase 1.8 is complete:

- Spec: `docs/phases/phase-1.8-failure-lab.md`
- Taskboard: `docs/phases/phase-1.8-taskboard.md`

Phase 1.7 is complete:

- Spec: `docs/phases/phase-1.7-observability.md`
- Taskboard: `docs/phases/phase-1.7-taskboard.md`

Phase 1.6 is complete:

- Spec: `docs/phases/phase-1.6-evaluation-harness.md`
- Taskboard: `docs/phases/phase-1.6-taskboard.md`

Phase 1.5 is complete:

- Spec: `docs/phases/phase-1.5-retrieval-mechanics.md`
- Taskboard: `docs/phases/phase-1.5-taskboard.md`

Phase 1 is complete:

- Spec: `docs/phases/phase-1-naive-classic-rag.md`
- Taskboard: `docs/phases/phase-1-taskboard.md`

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
| Phase 2.0 | Answer quality judging | Complete; see `docs/phases/phase-2.0-taskboard.md` |
| Phase 1.9 | Reranking | Complete; see `docs/phases/phase-1.9-taskboard.md` |
| Phase 1.8 | RAG failure lab | Complete; see `docs/phases/phase-1.8-taskboard.md` |
| Phase 1.7 | Observability and debugging | Complete; see `docs/phases/phase-1.7-taskboard.md` |
| Phase 1.6 | Evaluation harness | Complete; see `docs/phases/phase-1.6-taskboard.md` |
| Phase 1.5 | Retrieval mechanics | Complete; see `docs/phases/phase-1.5-taskboard.md` |
| Phase 1 | Naive classic RAG | Complete; see `docs/phases/phase-1-taskboard.md` |

## Candidate And Deferred Phases

| Phase | Focus | Status |
|---|---|---|
| Phase 2.1 | Context budget and structured answers | Directional; draft after Phase 2.0 |
| Phase 2.2 | Structural and semantic chunking | Directional; draft after Phase 2.1 |
| Later | Agentic RAG | Deferred |

Candidate and directional phases must not be treated as active implementation
contracts until their scope proposal is reviewed, signed off, and named under
`Current Phase`.

Near-term roadmap decision: `docs/phases/phase-1.9-2.2-final-roadmap.md`.
