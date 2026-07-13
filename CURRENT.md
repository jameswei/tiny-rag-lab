# Current Task

Task: P3.2-T02
Phase: Phase 3.2 — Real-Corpus Guided Learning
Spec: docs/phases/phase-3.2-real-corpus-guided-learning.md
Taskboard: docs/phases/phase-3.2-taskboard.md
Owner: codex
Status: done
Review Result: signed_off
Reviewer: /root/phase31_code_review
Last Updated: 2026-07-13
Updated By: /root/phase31_code_review

## Findings From Last Review

- none. The declared-file digest-mismatch and chunk-metadata provenance tests
  address the final review findings.

## Tests Reviewed

- `UV_CACHE_DIR=/tmp/tiny-rag-uv-cache uv run pytest tests/test_seed_assets.py tests/test_index_writer.py tests/test_web_api.py -q`: 50 passed.
- `UV_CACHE_DIR=/tmp/tiny-rag-uv-cache uv run python -m py_compile ...`: passed.
- Temporary seed smoke: all four assets seeded with verified digests; 40
  Cloudflare files, 1,144 watsonxDocsQA files, and two indexes promoted.
- `docker compose build studio`: reached the new asset-copy layer, but the
  managed Docker builder session ended without a final image record; retry is
  required during closeout.

## Blocker

- none

## Handoff

### Task Summary

Added a versioned image-seed layout, real approved Cloudflare and watsonxDocsQA
source snapshots, two prebuilt Cloudflare NumPy indexes, and atomic local
seeding with digest verification and conflict preservation.

### Files Changed

- `tiny_rag_lab/seed_assets.py`: verified staging/promotion lifecycle.
- `assets/seed/v1/`: corpus snapshots, indexes, and file-level manifest.
- `scripts/`: reproducible corpus preparation, index build, and manifest tools.
- `Dockerfile`: copies immutable seed assets into the studio image.
- `tiny_rag_lab/index_writer.py`: persists optional `source_corpus_id`.

### Design Decisions

- Reserved seed assets are copied from the image only after staging digest
  verification; unmanaged or modified targets are reported as conflicts.
- Cloudflare uses the owner-approved pinned revision and both structural and
  fixed-character NumPy indexes.

### Known Gaps

- The Docker build needs a clean retry because its managed builder session did
  not leave a final local image record, despite reaching the new copy layer.
- Reviewer fixes applied: undeclared target files now conflict before upgrades;
  bundled index paths are relocatable under `/data`; lifecycle coverage now
  includes corrupt seed, stale staging, upgrade, and modified-target cases.

### Questions For Next Agent

- T02 is signed off. Full-image Docker smoke remains T05 closeout work.
