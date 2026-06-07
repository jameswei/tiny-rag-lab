# Agent Entry Point

If `CURRENT.md` exists at the project root, read it first. It has the live state
of the active task, open review findings, and the reviewer's last result. Then
continue with the full doc reading order below.

Before changing code, read the project docs in this order:

1. `docs/file-structure.md`
2. `docs/agent-guidelines.md`
3. `docs/phases/README.md`

If `docs/phases/README.md` names an active phase, also read that phase's spec
and taskboard before changing code.

If no phase is active, do not claim or start implementation work until the next
phase scope is confirmed and a phase spec/taskboard exists.

Read `docs/proposal.md`, `docs/roadmap.md`, and `docs/architecture.md` when a
task changes architecture, roadmap, public interfaces, or phase scope.

Use the active phase taskboard to claim tasks, update status, record blockers,
and mark review/done state.

Prefer the project philosophy from `tiny-duo-infer`: readable, teachable code
over clever abstractions or production-grade completeness. Frameworks such as
LangChain, LlamaIndex, Ragas, and vector databases may be useful references, but
the core learning implementation should keep the mechanics visible.

`AGENTS.md` is only an entrypoint. Detailed rules, status, scope, review gates,
and handoff expectations belong in the docs above.
