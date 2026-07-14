# Current Task

**Phase:** No active phase.

**Status:** Phase 3.2 — Real-Corpus Guided Learning is complete. The owner
accepted the manual-preview paths, including Qdrant, custom-corpus, and Slim
image behavior; `/root/phase32_remediation_arch_rereview` independently
approved the remediation on 2026-07-15.

## Phase 3.2 Closeout

- Verified by owner: Qdrant profile and custom-corpus preview, including a
  built 40-document/522-chunk Qdrant index, successful Qdrant retrieval, and
  a separate ten-file NATS custom corpus whose expected queue-group source
  ranked first without replacing bundled assets or existing indexes.
- Verified by owner: an isolated Slim image keeps Guided Learn replay and
  BM25 retrieval usable before model download, while Dense/Hybrid retrieval
  and indexing clearly direct the learner to Settings for the explicit
  embedding-model download. The completed download persists and unlocks
  Dense/Hybrid retrieval plus indexing.

## Final Verification

- Python: 788 passed, 8 skipped.
- Web: 8 passed; production build passed.
- Fresh Studio image: `65dd3fef…`.
- Final independent reviewer: `/root/phase32_remediation_arch_rereview`.

## Next Work

Do not start implementation until a new phase proposal is reviewed, signed off,
and activated in `docs/phases/README.md`.
