# Phase 3.0 Taskboard: Local Visual RAG Lab

The implementation contract is `docs/phases/phase-3.0-local-visual-rag-lab.md`.

## Status Values

- `todo`: not started
- `in_progress`: actively being implemented
- `review`: ready for independent review
- `blocked`: stopped by a concrete recorded blocker
- `done`: independently reviewed, tested, and accepted

The task owner must not mark their own task `done`. Detailed handoff and
review evidence belong in `CURRENT.md`; this table keeps durable summaries.

## Taskboard

| ID | Milestone | Task | Depends On | Status | Owner | Acceptance | Notes |
|---|---|---|---|---|---|---|---|
| P3.0-T01 | Engine contracts | Extract backend-neutral index/search service seams; retain NumPy compatibility; add manifest defaults and versioned lab-trace contracts with unit tests. | — | done | Codex | Existing NumPy indexes/loaders and CLI defaults remain compatible; a complete immutable retrieve/ask learning artifact can serialize without secrets. | Independently signed off by Kuhn on 2026-07-13. |
| P3.0-T02 | Optional backend | Add optional Qdrant adapter/profile for the local visual lab; verify the same English fixture corpus through NumPy and Qdrant. The established bare CLI remains NumPy-first in Phase 3.0. | T01 | done | Codex | Qdrant remains optional; no host port; backend identity/score semantics appear in trace/manifest; an opt-in integration smoke is documented. | Independently signed off by Kuhn on 2026-07-13; real loopback-only Qdrant smoke passed (`1 passed`) and profile was torn down. Multilingual support deferred. |
| P3.0-T03 | Local API | Add FastAPI local API, data registry, sequential import/index jobs, corpus upload limits, provider status, and replay/run endpoints. | T01 | done | Codex | API has structured non-secret errors; upload and job boundaries are tested. | Independently signed off by Kuhn on 2026-07-13; restart recovery and secret boundaries covered. |
| P3.0-T04 | Learning content | Add starter replay pack, watsonxDocsQA catalog/import integration, and bilingual failure-lesson data. | T03 | done | Codex | Starter works offline; watsonx import is explicit; fixtures stay reproducible. | Independently signed off by Kuhn on 2026-07-13; raw corpus artifacts remain English, authored explanations EN/ZH. |
| P3.0-T05 | Visual client | Add bilingual React client: Start Lab, Corpus Library, Index Explorer, Run Workspace, and Failure Lab. | T03, T04 | done | Codex | Trace playback exposes all agreed stage artifacts and docs links. | Independently signed off by Kuhn on 2026-07-13; interactive stepper and evidence-backed Failure Lab implemented. |
| P3.0-T06 | Packaging | Add full/slim Compose variants, loopback binding, model-cache behavior, optional Qdrant profile, and setup docs. | T02, T03, T05 | done | Codex | Full indexes offline; slim has an explicit download state; Compose smoke tests pass. | Independently signed off by Kuhn on 2026-07-13; Compose config and real Qdrant smoke validated; temporary profile torn down. |
| P3.0-T07 | Review and close | Run complete regression/integration checks; update README, architecture, roadmap, file map, EN/ZH learning materials, and phase router. | T01–T06 | done | Codex | Independent review/sign-off; all completion criteria and stale-reference sweep complete. | Independently signed off by Kuhn on 2026-07-13. Manual browser visual QA is a non-blocking release-quality follow-up. |
| P3.0-T08 | Visual polish | Address owner preview feedback: make pipeline arrows true inter-stage connectors, improve Run Workspace action spacing, and refine the visual hierarchy of the learning client without changing RAG behavior. | T05 | done | Codex | The two observed layout issues are fixed and the UI has a more coherent, intentional visual system; independent review follows before commit. | Independently signed off by Kuhn on 2026-07-13; broad theme/palette exploration is deferred from Phase 3.0. |
| P3.0-T09 | Release preparation | Bump the package version, align README and project site with the local visual-lab milestone, verify cleanup and regression evidence, then prepare the Phase 3.0 milestone commit. | T01–T08 | done | Codex | One reviewed commit carries the verified version, public project story, and complete Phase 3.0 implementation without generated artifacts or stale local services. | Independently signed off by Kuhn on 2026-07-13; v0.2.0, full regression, clean Qdrant gate, and final no-service sweep verified. |
