# Phase 3.4 Taskboard: Interactive Retrieval Mechanics

**Status:** Complete — owner preview accepted and independent final sign-off
recorded 2026-07-17.

| ID | Task | Status | Owner | Notes |
|---|---|---|---|---|
| P3.4-T01 | Review the phase architecture, exact 16-question content manifest, API/artifact direction, model/image contract, and acceptance gates; activate the signed-off phase. | done | codex | Reviewed by `/root/phase34_scope_review` 2026-07-16; revised scope signed off with no remaining findings. |
| P3.4-T02 | Add compatible engine contracts for BM25/dense/hybrid explanations, candidate pools, rerank audit, and lab-run schema 1.1. | done | codex | Reviewed by `/root/phase34_t02_review` 2026-07-16; 77 focused tests and randomized equivalence checks passed; no findings. |
| P3.4-T03 | Add web API contracts for live retrieval/reranking and the bilingual Retrieval shell plus Lexical and Dense modules. | done | codex | Reviewed by `/root/phase34_t03_review` 2026-07-16; five findings remediated; 63 focused Python and 11 web tests passed; no remaining findings. |
| P3.4-T04 | Build “From local vectors to a vector database”: exact-vector fingerprinting, atomic Qdrant prepare/repair, payload/filter inspection, exact NumPy parity, and absent-service guidance. | done | codex | Signed off by `/root/phase34_t04_review` 2026-07-16 after six findings were remediated; 45 focused Python and 14 web tests passed with no remaining findings. |
| P3.4-T05 | Build live Hybrid and Reranking modules and add consistent reranker controls to Explore retrieve/ask flows. | done | codex | Signed off by `/root/phase34_t05_review` 2026-07-16; 67 focused Python and 17 web tests passed with no findings. |
| P3.4-T06 | Add atomic background-job progress, active discovery, cooperative cancellation, and comparison artifact persistence. | done | codex | Signed off by `/root/phase34_t06_review` 2026-07-16 after publication-race, restart-cleanup, and partial-Qdrant-build findings were fixed; 48 focused tests passed. |
| P3.4-T07 | Build the fingerprint-gated 16-question browser A/B evaluation with presets, editable valid configs, four metrics, and per-question inspection. | done | codex | Signed off by `/root/phase34_t07_review` 2026-07-16 after immutable identity, pinned revision, recovery, integration coverage, and localization findings were fixed; 71 focused Python and 20 web tests passed. |
| P3.4-T08 | Promote versioned seed/evaluation assets and implement pinned full/slim embedding and reranker model lifecycle plus Settings UI. | done | codex | Signed off by `/root/phase34_t08_review` 2026-07-16 after slim cache persistence, partial-snapshot readiness, and interrupted-promotion findings were fixed; 96 focused Python and 22 web tests passed. |
| P3.4-T09 | Add paired EN/ZH guides; update README/landing integration and version metadata; add Python/web/guides GitHub Actions. | done | codex | Signed off by `/root/phase34_t09_review` 2026-07-16 after paired documentation-accuracy findings were fixed; guides/dead links, 22 web tests/build, lock/image contracts, and diff check passed. |
| P3.4-T10 | Run full regression, full/slim Compose and Qdrant smoke tests, complete owner preview/remediation, obtain independent final review, and close the phase. | done | codex | Signed off by `/root/phase34_t09_review` 2026-07-17 with no remaining findings; the owner accepted the complete Studio experience including Browser A/B evaluation; 855 Python tests, 24 web tests, both production builds, full/slim/Qdrant runtime smokes, CPU-only image checks, explicit slim model downloads, and isolated-state cleanup passed. |

## Review Gates

- **Gate A — scope activation:** independent architecture reviewer signs off
  T01 before any runtime task is claimed.
- **Gate B — engine/API:** T02 must be signed off before T03–T05 depend on its
  public contracts; T06 must be signed off before browser evaluation depends
  on its job lifecycle.
- **Gate C — owner experience preview:** the owner previews the Retrieval
  journey, Qdrant comparison, Explore reranking, and evaluation before public
  documentation or screenshots are finalized.
- **Gate D — release closeout:** all code tasks have non-owner sign-off, all
  required tests pass, owner findings are resolved, and runtime cleanup is
  recorded before T10 or the phase can be marked complete.
