# Phase 3.2: Real-Corpus Guided Learning

**Status:** Complete — owner acceptance and independent remediation review
recorded 2026-07-15 in the [taskboard](phase-3.2-taskboard.md).

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

## Owner-Preview Remediation

The owner reviewed the completed high-priority visual-lab paths and directed
Phase 3.2 to remain active until the findings below are resolved. This is
remediation of the existing learning and provider experience, not a new phase
or a broader roadmap expansion.

The manual preview checklist is the required owner product-acceptance gate.
Automated verification and independent code review establish technical
readiness for a preview iteration; they do not close Phase 3.2. The phase
closes only after the owner accepts the high-priority checklist paths or
explicitly defers any remaining items.

- Make provenance, model identity, fixed lesson questions, chunking choices,
  vector/score semantics, grounded prompts, and saved-versus-live answer modes
  understandable without raw JSON.
- Repair watsonxDocsQA placeholder titles deterministically while preserving
  document IDs, QA gold-document mappings, and source provenance; regenerate
  affected managed seed assets and lesson context-selection artifacts.
- Make catalog/free-question interaction explicit, readable, and safe; improve
  gold-source presentation and human-readable index labels.
- Make Live Ask availability, connection testing, in-flight work, success, and
  failure truthful. A generation failure must never appear as an empty answer
  result.
- Add truthful index-job progress/elapsed feedback and fix observed layout,
  notice, Failure Lab hierarchy, and narrow-responsive defects.

Qdrant-profile and custom-corpus owner previews are complete: the lab now
distinguishes a ready local Qdrant service from an unavailable optional
service, prevents a build against the latter, and provides a safe actionable
build failure if the service disappears. A ten-file NATS custom corpus built
and retrieved independently alongside the bundled assets. Slim-image owner
pre-download preview is complete: Guided Learn replay and BM25 retrieval work
before download, while Dense/Hybrid retrieval and indexing clearly require the
explicit model-download action. The completed download persists and unlocks
Dense/Hybrid retrieval plus indexing. This does not expand the Phase 3.2 RAG
boundary.

### Remediation Contract Decisions

**Provider availability and testing.** Resolve each provider field using a
non-empty browser-session override first, then the environment. Live Ask is
available only when the resolved model is non-empty and at least one of the
resolved base URL or API key is non-empty. Provider-status responses expose
only safe booleans, never a secret or the resolved credential.

`POST /api/provider/test` accepts the same optional provider override as Live
Ask and resolves it with the same precedence. It makes one bounded minimal
chat-completions request using that resolved model, rather than relying on a
provider-specific `/models` probe. It does not persist a run, job, seed state,
or browser secret. Its response is either a safe success result or a sanitised,
actionable failure. The UI warns that testing contacts the provider and may
consume quota/cost.

The test has a server-enforced 10-second deadline. Its response is
`{"ok": true, "message": "Provider connection verified"}` on success. Missing
resolved configuration performs no provider request and returns HTTP 409 with
`{"ok": false, "error": "Configure a model and either a provider base URL or API key before testing."}`.
Timeout and provider failures return a safe `ok: false` error category and
actionable message without exposing credentials, request payloads, or provider
response bodies.

A Live Ask run is either generation-successful, with answer and returned
citations, or `generation_failed`, with no answer/citations and a safe error.
The client must render the latter as a failure state, never as an empty answer
or a citations result.

The client has explicit `testing` and `generating` states. During `testing`,
provider fields, Test connection, and Live Ask are disabled; retrieval remains
available. During `generating`, provider fields, Test connection, and all
Explore run controls are disabled to prevent a duplicate or configuration-drift
request. Each request has a monotonic client request ID and abort signal: only
the current, non-aborted request may update UI state. Controls recover in a
`finally` path after success, safe failure, timeout, or abort.

**watsonxDocsQA title normalisation.** Preserve prepared document IDs and all
QA gold-document mappings. Resolve a display title deterministically in this
order: a non-placeholder dataset title; the first non-placeholder Markdown H1;
a readable source-URL path segment; then the original document ID. Preserve the
original title and resolution method in preparation metadata so the correction
is attributable and reproducible.

A placeholder title is an empty/whitespace value or a value matching
`{{ document.title.text }}` after whitespace normalisation. For the H1 fallback,
scan Markdown source lines in order outside fenced code blocks, accept only an
ATX level-one heading, remove its opening marker and optional closing marker,
then apply the same placeholder rule. This keeps title resolution deterministic
without changing document or QA identities.

For the source-URL fallback, parse the URL, take its final non-empty path
segment, percent-decode it as UTF-8, remove only a terminal `.html`, `.htm`,
`.md`, or `.mdx` extension, replace runs of `-` or `_` with one space, and
collapse whitespace while preserving the segment's original letter case. An
absent, unparseable, or empty result falls through to the original document ID.

**Saved context-selection invariant.** Seed generation must assert that the
first Guided Learn lesson has at least one selected and one omitted candidate,
and that selected evidence directly supports the recorded answer and citations.
Every Guided Learn lesson must meet the same selected-and-omitted contrast;
there is no silent all-candidates-fit exception.

Each lesson source definition declares its required supporting document IDs.
Seed generation records `answer_supporting_chunk_ids` and validates that they
are selected chunks, that they cover every declared required support document,
and that every recorded citation resolves to selected evidence. This makes the
support relationship machine-checkable rather than a visual assertion.

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
- `GET /api/backends` reports whether the always-local NumPy backend and the
  optional Qdrant service are currently available. The browser must not offer
  an unavailable Qdrant build; `POST /api/indexes` repeats the readiness check
  to keep the server authoritative.
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

The owner reopened Phase 3.2 for preview remediation on 2026-07-14.
`/root/phase32_remediation_arch_signoff` approved the bounded remediation
contract before implementation resumed.
