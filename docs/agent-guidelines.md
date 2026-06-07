# Agent Collaboration Guidelines

This document defines how multiple agents should collaborate on
`tiny-rag-lab`.

The project is a learning-focused RAG lab, so collaboration should optimize for
shared understanding, explicit decisions, reproducible tests, and readable
implementation over speed alone.

## Source Of Truth

Agents should read the relevant planning documents before changing code.

Use this source-of-truth order:

1. `docs/phases/README.md`: active phase pointer and phase index.
2. Active phase spec and taskboard named by `docs/phases/README.md`.
3. `docs/architecture.md`: RAG planes and subsystem boundaries.
4. `docs/roadmap.md`: proposed phase sequence.
5. `docs/proposal.md`: project purpose, philosophy, and non-goals.
6. Code and tests: implementation truth once implementation begins.
7. Earlier phase docs: historical context unless the active phase points to
   them.

If documents and code disagree, agents should not silently choose one. The
agent should call out the conflict and either update the stale document as part
of the task or ask for clarification if the correct direction is unclear.

## Role Definitions

Each agent should operate in one clear role for a task. A single agent may fill
multiple roles only when the task is small, but the handoff should still state
which responsibilities were covered.

### Main Developer

The main developer implements scoped changes.

Responsibilities:

- confirm the task belongs to the current phase
- follow the active phase spec and taskboard
- keep changes narrow and intentional
- keep RAG mechanics visible rather than hidden behind frameworks
- update docs when behavior, interfaces, or phase scope changes
- run relevant tests before handoff
- produce a concise handoff note

The main developer should not introduce major architectural changes without
updating the phase spec or adding a durable decision note.

### Architecture Reviewer

The architecture reviewer checks whether a change fits the intended RAG design.

Responsibilities:

- verify indexing, retrieval, generation, and evaluation boundaries remain clear
- check that the core RAG mechanics are project-owned and inspectable
- verify phase scope is respected
- flag dependencies that hide learning-critical behavior
- identify decisions that need documentation before implementation continues

The architecture reviewer should focus on design risks rather than formatting
or small implementation style issues.

### Code Reviewer

The code reviewer checks implementation quality.

Responsibilities:

- verify correctness and maintainability
- check that names and data contracts are clear
- look for hidden assumptions, brittle serialization, and missing edge cases
- verify tests cover the important behavior introduced by the change
- flag code that is technically correct but too opaque for learning

The code reviewer should prioritize bugs, behavior regressions, data-contract
breaks, and missing tests.

### Test Verifier

The test verifier runs tests and records reproducibility information.

Responsibilities:

- run the relevant unit tests
- run CLI smoke tests when dependencies and local artifacts are available
- record Python version and relevant dependency versions
- report skipped tests with a clear reason
- report failing commands with enough output to diagnose the issue

The test verifier should not mark a phase complete if required tests were
skipped without an explicit documented reason.

### Learning Reviewer

The learning reviewer is optional but useful for this project.

Responsibilities:

- check whether the code can be read line by line by a learner
- identify places where comments or docs should explain the RAG concept
- flag overly clever code even if it is technically correct
- suggest documentation improvements that help future study

Comments should explain reasoning, invariants, data shapes, and RAG concepts;
they should not narrate trivial assignments.

## Collaboration Flow

Use this flow for normal implementation tasks:

0. If `CURRENT.md` exists, read it first for live task state and any open review
   findings. Then read the phase spec and taskboard for full context.
   `CURRENT.md` is a fast pointer, not a replacement for the spec and taskboard.
1. Read the relevant proposal, phase spec, taskboard, architecture docs, and
   existing code.
2. Confirm the task belongs to the current phase.
3. Claim the task in the taskboard by setting `Status` to `in_progress` and
   `Owner` to the agent/person name.
4. Create or update `CURRENT.md` from `docs/current-task-template.md`.
5. Make the smallest change that satisfies the task.
6. Keep code educational and explicit.
7. Update docs if the change affects behavior, public interfaces, architecture,
   or phase scope.
8. Run relevant tests.
9. Produce a handoff note. Update `CURRENT.md` status to `review`.
10. Reviewer checks the change against the role-specific checklist. Reviewer
    writes findings directly to `CURRENT.md` under "Findings From Last Review",
    tagged `[Blocking]`, `[Non-blocking]`, `[Nit]`, or `[Question]`. Reviewer
    sets `Review Result` to `changes_requested` or `signed_off` and updates
    `Last Updated` / `Updated By`.
11. Test verifier records test results under "Tests Reviewed".
12. A non-owner agent records sign-off in both `CURRENT.md` and the taskboard
    `Notes` before the task is marked `done`.
13. After a task is committed, the agent who committed resets `CURRENT.md` for
    the next task. If no next task exists and the phase is fully closed, delete
    `CURRENT.md`; the taskboard is the permanent record.

For large design changes, architecture review should happen before full
implementation.

## Review And Sign-Off Requirements

Every implementation task or code change must be reviewed and signed off by an
agent other than the owner who made the change.

The implementing owner may:

- claim a task by setting it to `in_progress`
- move a task to `review` after implementation and local tests
- record test commands, skipped tests, known gaps, and handoff notes

The implementing owner must not:

- mark their own task as `done`
- record their own implementation as reviewed or accepted
- close review-sensitive work without another agent's explicit sign-off

The reviewing agent is responsible for deciding whether the task can move from
`review` to `done`. When marking a task `done`, the reviewer must record their
agent name and the review result in the taskboard `Notes`, for example:

```text
reviewed by codex; uv run pytest: 42 passed; no findings
```

If the reviewer requests fixes, the task stays in `review` until the owner
applies the fixes and the reviewer signs off on the updated change.

## Handoff Format

Every substantial implementation task should end with a handoff note.

Use this format:

```markdown
## Handoff

### Task Summary

Briefly describe what changed and why.

### Files Changed

- `path/to/file.py`: short purpose of change

### Design Decisions

- Decision made and reason

### Tests Run

- `command`: pass/fail/skip

### Known Gaps

- Any limitation, skipped test, missing artifact, or incomplete follow-up

### Learning Notes

- Concepts or implementation areas that deserve careful line-by-line reading

### Questions For Next Agent

- Open questions, if any
```

Small documentation-only changes may use a shorter handoff, but they should
still state what changed and whether tests were run.

## Review Gates

Architecture review is required when a change:

- changes indexing, retrieval, generation, or evaluation boundaries
- changes public data contracts
- changes persisted index artifacts
- changes embedding or generation provider interfaces
- changes the phase roadmap or scope
- adds a new major dependency

Code review is required when a change:

- adds or changes runtime behavior
- changes tests or test strategy
- changes CLI behavior
- changes serialization or persistence
- changes public interfaces

Test verification is required when a change:

- claims a feature works
- changes retrieval behavior
- changes prompt or citation behavior
- changes index persistence
- closes a phase completion criterion

Documentation updates are required when a change:

- changes public CLI/API usage
- changes architecture or phase scope
- adds a decision future agents should follow
- introduces known limitations or setup requirements

## Conflict Handling

Agents should handle conflicts explicitly.

If docs and code disagree:

- identify the conflict
- decide whether the code or docs should change
- update the stale source when the correct direction is clear
- ask for clarification when the correct direction is not clear

If phase scope and a requested change disagree:

- do not silently expand the phase
- propose a phase-spec update or record the change as out of scope

If two agents disagree on architecture:

- reduce the disagreement to a concrete decision
- record the final choice in the relevant spec or a durable decision note

If required local artifacts are unavailable:

- skip only artifact-dependent tests
- run all artifact-independent tests
- report the skip reason clearly

## Learning-First Implementation Rules

Agents should:

- prefer explicit control flow over compact cleverness
- keep document, chunk, embedding, retrieval, prompt, and citation mechanics
  inspectable
- use descriptive names for corpus paths, chunk IDs, scores, and source metadata
- preserve intermediate data where it helps explain the pipeline
- use external libraries for commodity primitives, not to hide the core RAG flow

Agents should avoid:

- wrapping LangChain, LlamaIndex, Haystack, or a vector database as the core
  engine in early phases
- optimizing before baseline behavior is clear
- introducing abstractions that are not needed by the current or next phase
- merging code that works but is too opaque for learning

## Testing Expectations

Agents should run the narrowest useful test set during development and broader
tests before handoff.

Use the active phase spec and taskboard named by `docs/phases/README.md` for
phase-specific test expectations.

In Phase 1, tests should not require:

- network access
- OpenAI-compatible API credentials
- Hugging Face model downloads
- full `watsonxDocsQA` or WixQA corpora

Test reports should include:

- command run
- pass, fail, or skip status
- reason for skip, if skipped
- relevant environment details when setup matters

## Current Project Defaults

Unless a later phase spec changes these defaults, agents should assume:

- Python-only Phase 1 implementation
- `uv` and `pyproject.toml`
- `argparse` CLI
- primary corpus: prepared local `watsonxDocsQA`
- stretch corpus: WixQA
- local embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- no API embeddings in Phase 1
- OpenAI-compatible online generation for real `ask`
- fake embedder and fake generator for tests
- `.tiny-rag/index/` for generated index artifacts
- generated corpora and indexes are not committed to git
- readable, teachable code over production completeness
