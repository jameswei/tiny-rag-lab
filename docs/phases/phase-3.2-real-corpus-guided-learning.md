# Phase 3.2: Real-Corpus Guided Learning

**Status:** Active. The owner approved the content manifest and
`/root/phase31_arch_review` approved the architecture scope on 2026-07-13.

## Goal

Turn the local visual lab into a coherent first-time learning experience built
on real, inspectable documentation rather than the one-document starter replay.
The new Guided Learn path explains a complete classic-RAG run step by step;
Explore remains the workspace for free retrieval and provider-backed live ask.

## Approved Product Decisions

- Add a separate **Learn** navigation area before Explore. A first visit starts
  with Guided Learn; Explore stays an experiment workspace.
- Use a 40-document pinned Cloudflare State & Coordination slice: Workers,
  Durable Objects, Queues, KV, R2, and Workflows. See the companion content
  manifest for exact paths and questions.
- Bundle two ready NumPy indexes for that corpus: structural chunks are the
  canonical guided index and fixed-character chunks are selectable in Explore.
- Teach four complete saved lessons. They progress stage-by-stage on first
  viewing, then allow free revisiting. A saved answer is clearly labelled as a
  recorded lesson result, never as a live model response.
- Bundle watsonxDocsQA source data in the standard image. Its index is an
  explicit background build, then all 75 original questions are selectable in
  Explore with a small featured subset and post-run gold-source reveal.
- Retrieval-only Explore needs no provider. Live Ask requires an
  OpenAI-compatible provider and uses the same chosen index and retrieval
  artifacts.
- Keep EN/Simplified-Chinese UI. Bundled source content and questions remain
  English; multilingual embedding support is not a Phase 3.2 promise.

## In Scope

- Immutable bundled corpus/index/lesson seed assets with a versioned, atomic,
  integrity-checked lifecycle that never overwrites user-created data.
- Catalog and saved-lesson HTTP contracts, complete stage artifacts, and clear
  provider gating for live generation.
- Learn, Explore, Build & Inspect, Home, and navigation redesign needed to
  expose those artifacts. Failure Lab and custom-corpus workflows remain.
- Sign-aware real-vector presentation, source/chunk provenance, candidate and
  context-selection visibility, raw inspection, and reduced-motion behavior.
- Asset, API, Python, React, Compose, full-image smoke, and responsive preview
  verification.

## Out Of Scope

- Browser aggregate evaluation reports, reranking controls, new retrieval
  algorithms, semantic-index comparison lessons, cloud hosting, shareable
  URLs, public image publishing, or multi-user state.
- A guarantee that arbitrary Chinese or other non-English uploaded corpora use
  a multilingual embedding model.

## Public Contract Direction

- `GET /api/lessons` lists ordered saved lesson summaries.
- `GET /api/lessons/{lesson_id}` returns lesson metadata and its immutable,
  complete replay artifacts.
- `GET /api/corpora/{corpus_id}/questions` exposes catalog question text and
  featured status, never gold metadata. Retrieve/ask accepts an optional
  `catalog_question_id`; the server resolves its canonical question text and
  validates that the selected index belongs to that catalog corpus. Only that
  validated run receives a server-produced
  `catalog_check { question_id, expected_document_ids, retrieved_document_ids,
  hit }` result. A free-text run never receives this field.
- Browser replay artifacts gain corpus/source snapshots, chunk references,
  candidate/context decisions, and saved-result provenance. A saved lesson
  records its schema version, source revision, corpus/index IDs and digests,
  retrieval configuration, and `recorded_lesson_result` answer provenance.
  CLI contracts and the NumPy/Qdrant conceptual boundary remain unchanged.
- Index manifests gain an optional durable `source_corpus_id`. Index creation
  writes the selected corpus ID; seeded Cloudflare indexes use
  `cloudflare-state-v1`, and a built watsonxDocsQA index uses
  `watsonxdocsqa-v1`. Legacy manifests without this field remain usable for
  free retrieval but reject catalog-question runs.

## Seed Asset Lifecycle

Image-owned seed material lives under the immutable
`/opt/tiny-rag-lab/seeds/v1/` directory, never under ignored `corpus/` or the
mounted `/data` volume. It contains `corpora/`, `indexes/`, `lessons/`, and a
versioned `seed-manifest.json`. The manifest records `seed_version`, schema
version, reserved corpus/index/lesson IDs, relative paths, and SHA-256 digests
for every shipped file. Reserved IDs are:

- corpus: `cloudflare-state-v1`, `watsonxdocsqa-v1`
- indexes: `cloudflare-state-structural-v1`, `cloudflare-state-fixed-v1`
- lessons: `cloudflare-state-coordination-v1`

At startup the entrypoint verifies image seed digests, copies each missing or
managed-outdated asset to `/data/.seed-staging/<id>`, verifies the staged
digest, and promotes it atomically into `/data/corpora`, `/data/indexes`, or
`/data/lessons`. It records the managed asset's seed version and digests in
`/data/.seed-state.json` only after promotion.

- A stale staging directory is discarded and rebuilt safely.
- A managed asset with a missing file or digest mismatch is replaced from the
  read-only image seed; partial copies never become visible as ready assets.
- A prior managed seed version is upgraded atomically only when its recorded
  files still match its recorded digests.
- An unexpected target that lacks managed seed identity, or a user-modified
  managed target whose digest no longer matches state, is reported as a
  conflict and is never overwritten automatically. Custom upload/index APIs
  reject the reserved IDs.

## Image Variants And Provider Boundary

- **Full image:** ships the default embedding model, so Learn replay and dense
  or hybrid Explore retrieval over bundled indexes work offline immediately.
- **Slim image:** Learn replay remains offline because it reads saved
  artifacts, but dense/hybrid Explore retrieval and any indexing require the
  existing explicit model-download job. BM25 remains usable against a seeded
  index without that download.
- **Live Ask:** remains independent of image variant and requires a usable
  OpenAI-compatible provider from the environment or the current browser
  session (`base_url` or API key under the existing local-provider rule).
  Browser-supplied secrets are never persisted in runs, jobs, seed state, or
  browser storage.

## Required Verification

- Verify seed idempotency, image-manifest digest failure, missing/partial
  managed asset recovery, stale-staging cleanup, safe prior-seed upgrade, and
  conflict preservation for unexpected or user-modified targets.
- Verify all 75 watsonxDocsQA IDs are listed without gold metadata; a valid
  catalog-question run returns the server-produced gold check; free-text runs
  do not; and corpus/index/question mismatches are rejected, including after
  an app restart. Legacy indexes without `source_corpus_id` must still support
  free retrieval but reject catalog-question runs.
- Verify saved lesson provenance, source revision, index ID/digest, retrieval
  configuration, answer provenance, and every stage artifact.
- Verify full-image offline dense/hybrid retrieval; slim-image replay and BM25
  behavior before download; explicit slim dense/hybrid model gating; and
  environment/browser-session provider success and non-persistence.

## Acceptance Criteria

1. A full default image starts with the structural and fixed Cloudflare indexes
   ready, and Guided Learn completes all six classic-RAG stages without an API
   key or network request.
2. Every guided stage presents a distinct real artifact. The active question,
   corpus/index identity, and configuration remain visible throughout.
3. Explore permits free retrieval over either bundled Cloudflare index, a
   built watsonxDocsQA index, or a custom index; Live Ask is unavailable until
   a provider is configured.
4. watsonxDocsQA indexing is a visible restart-safe background job; after it
   succeeds, all 75 questions and their post-run gold-source checks work.
5. Bundled seed assets are attributable, pinned, integrity-checked, atomically
   recover from partial/corrupt managed copies, and never overwrite unexpected
   or user-modified `/data` state.
6. Existing Failure Lab, custom uploads, NumPy/Qdrant selection, raw artifact
   inspection, EN/ZH UI, and reduced-motion behavior remain available.

## Sign-off Record

The owner approved
[the corpus and lessons manifest](phase-3.2-content-manifest.md), and
`/root/phase31_arch_review` approved this proposal and taskboard, on
2026-07-13. The phase was activated in the phase index on the same date.
