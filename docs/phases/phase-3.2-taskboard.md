# Phase 3.2 Taskboard: Real-Corpus Guided Learning

**Status:** Complete — owner acceptance and independent remediation review
completed 2026-07-15.

| ID | Task | Status | Owner | Notes |
|---|---|---|---|---|
| P3.2-T01 | Review and approve the pinned Cloudflare corpus/lesson manifest. | done | owner | Approved 2026-07-13; architecture scope signed off by `/root/phase31_arch_review`. |
| P3.2-T02 | Package Cloudflare/watsonxDocsQA seed assets; build structural/fixed indexes; implement versioned digest verification, staging, atomic promotion, recovery, and conflict handling. | done | codex | Reviewed by `/root/phase31_code_review` 2026-07-13; signed off. Full-image Docker smoke remains T05 closeout work. |
| P3.2-T03 | Add catalog, lesson, and complete replay-artifact API contracts; durable index-to-corpus IDs, validated catalog-question runs, gold checks, and provider gating. | done | codex | Reviewed by `/root/phase31_code_review` 2026-07-13; signed off. |
| P3.2-T04 | Build Learn, Home, Explore, Build & Inspect navigation and artifact presentations. | done | codex | Reviewed by `/root/phase31_code_review` 2026-07-13; signed off. |
| P3.2-T05 | Add asset/API/UI tests for seed recovery, catalog-question association, full/slim behavior, responsive previews, docs, review handoff, and phase closeout. | done | codex | Final review signed off; fresh Studio image `65dd3fef…` built and Compose teardown clean. |
| P3.2-T06 | Resolve owner-preview findings: learning clarity, watsonxDocsQA title preparation, catalog/index/build UX, Live Ask reliability, Failure Lab hierarchy, responsive defects, and Qdrant readiness/build feedback. | done | codex | Owner directed Phase 3.2 remediation 2026-07-14. Architecture signed off by `/root/phase32_remediation_arch_signoff`; owner verified Qdrant and custom-corpus paths 2026-07-14. Reviewed and approved by `/root/phase32_remediation_arch_rereview` 2026-07-15. |
| P3.2-T07 | Add remediation regression coverage, rebuild Studio, complete manual preview, obtain independent review, and close out Phase 3.2. | done | codex | Owner verified all active manual-preview paths, including Qdrant, custom corpus, and full Slim pre-/post-download behavior, 2026-07-15. Independently reviewed and approved by `/root/phase32_remediation_arch_rereview` 2026-07-15; final regression: 57 Python passed, 8 web passed, production web build, and `git diff --check` passed. |
