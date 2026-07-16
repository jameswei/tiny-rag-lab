# Phase 3.4: Interactive Retrieval Mechanics

**Status:** Complete — owner preview accepted and independent final review
signed off 2026-07-17 in the [taskboard](phase-3.4-taskboard.md).

## Goal

Turn the retrieval half of RAG into a live, inspectable course inside the
Studio. Learners should be able to move from lexical matching through dense
similarity, vector-database storage, hybrid fusion, reranking, and browser-side
evaluation without treating any library or database as a black box.

This phase adds a dedicated **Retrieval** / **检索** area. It complements the
existing end-to-end lessons and Explore workflow rather than replacing them:

- **Learn** keeps the four guided, saved RAG replays.
- **Retrieval** teaches retrieval mechanics through live experiments over a
  reviewed Cloudflare question set.
- **Explore** remains the free-question end-to-end workspace and gains an
  optional reranking control.

No LLM provider is required for the Retrieval area.

## Product Principles

- Show the calculation before the abstraction: tokens, term contributions,
  vector values, cosine similarity, reciprocal-rank fusion, and rank movement
  remain visible.
- Use real retrieval results from the bundled Cloudflare corpus. Do not add
  simulated or saved retrieval replays.
- Keep NumPy/local-file retrieval as the canonical teaching path. Qdrant is a
  substantial comparison module, not a mandatory dependency or a replacement
  for inspectable local mechanics.
- Use curated questions to teach a concept intentionally. Keep unrestricted
  questions in Explore.
- Prefer small typed contracts and project-owned calculations over framework
  orchestration.

## User Journey

The Studio navigation becomes:

1. Home
2. Learn
3. Retrieval
4. Explore
5. Build & Inspect
6. Failure Lab
7. Settings

Retrieval opens on Lexical Search. All six modules remain directly selectable;
the order is a recommended learning path, not a lock:

1. **Lexical Search** — query tokens, matching terms, document frequencies,
   BM25 components, per-term contributions, and final scores.
2. **Dense Retrieval** — query/chunk vector previews, norms, dot products,
   cosine similarity, and final ordering.
3. **From local vectors to a vector database** — the same stored vectors and
   queries through NumPy and optional Qdrant, including payloads and filters.
4. **Hybrid Retrieval** — dense and BM25 ranks, reciprocal-rank-fusion
   contributions, and fused ordering.
5. **Reranking** — a larger first-stage candidate pool, cross-encoder scores,
   rank movement, dropped candidates, and the final top-k.
6. **Evaluation** — two editable retrieval configurations compared across the
   16-question reviewed browser set with aggregate and per-question evidence.

Module completion may be remembered for the current browser session only. It
must not become a new persisted learning-progress system.

## Curated Content Contract

The exact Phase 3.4 question set is defined in
[phase-3.4-content-manifest.md](phase-3.4-content-manifest.md).

- Exactly 16 English questions ship in the versioned seed asset.
- Four questions are assigned to each of Lexical, Dense, Hybrid, and Reranking.
- Each entry has stable ID, category, question, gold document path or paths,
  teaching purpose, and expected observation.
- The content is an educational/evaluation contract, not a claim that one
  retriever always wins.
- The UI shell and explanations remain bilingual. Corpus questions, source
  text, and retrieval artifacts remain in their original English.

## Retrieval Explanation Contract

### Shared result data

Every live retrieval response must identify the query, index, retriever,
requested candidate depth, final top-k, chunk ID, document metadata, source
path, scores, and stable ordering. Existing stored traces remain loadable.

The web API may add typed optional explanation and candidate-pool fields to a
run artifact. Existing `LabRun.evidence` keeps its current meaning: the final
retrieved list after optional reranking but before context packing, with
`selected_for_context` identifying the packed subset for Ask runs. The larger
pre-rerank pool is stored in a separate optional field so older clients and
replay data do not change meaning.

Phase 3.4 increments the lab-run schema from `1.0` to `1.1`. New fields are
additive and optional. Readers must continue to accept `1.0` artifacts and
treat missing candidate/explanation fields as unavailable; they must not
recompute historical explanations from a later model or index.

### Lexical

The project owns the explanation math around the existing BM25 implementation.
The response exposes normalized query tokens and, for each candidate and query
term, match frequency, document frequency, inverse-document-frequency value,
length normalization inputs, contribution, and total score. Explanations must
use the same tokenization and parameters as ranking.

### Dense

The response exposes bounded query and chunk vector previews, dimensions,
norms, dot product, and cosine similarity. Bar charts must preserve sign rather
than silently converting values to absolute magnitude.

### Qdrant comparison

The module always teaches the local NumPy path first. When Qdrant is absent, it
shows that the module is optional and the exact local launch command:

```bash
docker compose --profile qdrant up -d
```

When Qdrant is ready, one idempotent action prepares a deterministic derived
index named `cloudflare-state-structural-qdrant-local`. Preparation copies the
existing structural index's exact chunks and vectors; it must not re-chunk or
re-embed the corpus. A source fingerprint covers ordered chunk IDs, canonical
float32 source-vector bytes, vector dimension, and cosine distance. That SHA is
a local provenance identity stored with the derived index and Qdrant payloads;
it is not recomputed from remote bytes because Qdrant cosine collections
normalize vectors during upload. A prepared collection is reused only when its
provenance value, complete ordered chunk-ID set, point count, dimension, and
every returned vector match the normalized source vector within `1e-6`. A
missing or mismatched collection is rebuilt under a fingerprint-derived
physical name, verified, and then published through the stable collection
alias; an incomplete staging collection is never published.

The parity experiment uses Qdrant's exact-search option, never approximate HNSW
search. It compares the unfiltered NumPy and Qdrant lists by chunk ID with a
score tolerance of `1e-5`. Chunks whose scores differ within that tolerance are
treated as one tie group, so their internal order is reported as equivalent
rather than falsely presented as a mismatch. Filter demonstrations are shown
separately and do not make a parity claim.

New Qdrant collections store inspectable payload fields for `chunk_id`,
`doc_id`, `title`, `path`, `source_group`, and `source_fingerprint`. Existing
collections containing only `chunk_id` remain searchable, but their filters are
explicitly reported as unavailable. The teaching module may filter the newly
prepared Cloudflare collection by Durable Objects, Queues, KV, R2, or
Workflows.

Qdrant is excluded from browser evaluation quality comparisons. The module
teaches index storage, filtering, payloads, and operational tradeoffs—not a
different semantic scoring concept.

### Hybrid

Hybrid explanations show the independent dense and BM25 lists plus reciprocal
rank fusion using the existing constant:

```text
contribution = 1 / (60 + rank)
```

Each fused result exposes both source ranks, each available contribution, the
sum, and the final order.

### Reranking

The default lesson retrieves 20 first-stage candidates and reranks them to a
final top 5. It exposes first-stage rank and score, reranker score, final rank,
rank delta, and whether a candidate moved, stayed, or was dropped.

The default cross-encoder is `cross-encoder/ms-marco-MiniLM-L-6-v2`. Studio
runtime loading is local-only. Network access is allowed only through the
explicit model-download action in Settings.

Explore adds reranker and rerank-depth controls. Retrieval and Live Ask use the
same selected reranking configuration so the visible context and generated
answer cannot silently diverge.

## Browser Evaluation Contract

Evaluation always uses the bundled Cloudflare structural NumPy index and the
16 reviewed questions. It does not accept uploaded corpora, arbitrary indexes,
Qdrant, or free-form questions in this phase.

The initial presets are:

- BM25 vs Dense
- Dense vs Hybrid
- Hybrid vs Hybrid + cross-encoder

After choosing a preset, learners may edit each side's retriever, top-k,
reranker, and rerank candidate depth. Identical configurations are rejected as
non-instructive.

Each comparison reports hit rate, mean reciprocal rank, context precision, and
context recall separately. It must not collapse them into a composite winner.
Aggregate differences and complete per-question result inspection are both
required.

Evaluation runs as a background job with persisted progress, active-job
discovery after navigation or refresh, terminal error details, and cooperative
cancellation. Cancellation moves from `cancel_requested` to `cancelled` after
the current embedding, retrieval, or cross-encoder call returns and before the
next question begins; the UI must not imply that an in-flight model call can be
forcefully interrupted. Progress/result publication uses atomic file
replacement, and cancelled or failed jobs never publish a complete comparison.
Only one resource-heavy local job may run at a time, consistent with the
current Studio job policy.

## API And Artifact Direction

The implementation may add these public web contracts, with final naming kept
small and typed:

- extend retrieve/ask requests with optional `reranker` and
  `rerank_top_n` fields;
- retrieval-material endpoints for curated questions and module defaults;
- live explanation responses for lexical, dense, hybrid, and reranked runs;
- Qdrant prepare, status, payload/filter, and NumPy comparison operations;
- background evaluation comparison create/status/detail operations;
- model status/download support for the reranker; and
- background-job progress, active discovery, and cancellation where needed.

The core engine remains the source of retrieval and evaluation calculations.
Web routes validate and serialize; React components present the returned
artifacts.

## Model And Image Contract

- The default embedding snapshot is
  `sentence-transformers/all-MiniLM-L6-v2` revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- The default reranker snapshot is
  `cross-encoder/ms-marco-MiniLM-L-6-v2` revision
  `c5ee24cb16019beea0893ab7796b1df96625c6b8`.
- The **full** Studio image bundles the default embedding model and default
  cross-encoder so dense, hybrid, and reranking work offline after build.
- The **slim** image starts without either model and presents separate explicit
  downloads in Settings. Both downloads request the exact revisions above.
- Both variants stay CPU-only. Installing the models must not pull CUDA/NVIDIA
  wheels or require a GPU.
- Missing models disable only the affected operations and explain the exact
  next action. Lexical learning remains available.
- A failed or interrupted download produces a recoverable state; it must not
  falsely report the model as ready.

## Seed, Documentation, And CI Contract

- Promote immutable bundled assets to a new versioned seed tree. Preserve
  conflict-safe upgrade behavior and existing user-created data.
- The evaluation asset has an immutable bundle manifest containing the
  question JSONL SHA-256, chunks JSONL SHA-256, embeddings NPZ SHA-256,
  canonical source-vector fingerprint, document/chunk counts, distance metric,
  embedding dimension, and both pinned model revisions. The current reviewed
  structural artifacts are anchored by chunks SHA-256
  `11960c4f72360fdb4dd7fea1f43fbec4dd36a9214e3256bdcffc36ad3aee1f41`
  and embeddings SHA-256
  `c944c09db6e42fbdac3ec3e25dc74c6e6ea8b23802d612705ec6be512fa29604`.
  Seed generation records the final question and canonical-vector hashes after
  serializing the accepted manifest. Evaluation refuses to start if any
  fingerprint, count, metric, dimension, or model revision differs.
- Add the reviewed retrieval material manifest to the Cloudflare corpus and
  regenerate any bundled artifacts whose stable content depends on the seed
  version.
- Add paired English and Simplified-Chinese learning-guide coverage for
  reranking, and update retrieval/evaluation guides, navigation, and roadmap.
- Update README and landing-page wording as an integrated project capability,
  not as a phase changelog. Add a representative screenshot only after owner
  preview acceptance.
- Add GitHub Actions coverage for Python tests, web tests/build, and Learning
  Guides build/dead-link validation.
- Align package versions at `0.5.0` only after implementation and owner preview
  are accepted.

## Out Of Scope

- Generation lessons, prompt engineering, answer judging, agentic RAG, or a
  new LLM-provider contract.
- A second vector database, hosted Qdrant, public multi-user deployment, or
  making Qdrant a Compose hard dependency.
- Saved retrieval replays, persisted course progress, user-authored evaluation
  sets, uploaded-corpus evaluation, or arbitrary evaluation indexes.
- Learned sparse retrieval, metadata-filter authoring beyond the curated
  Qdrant source groups, approximate-index tuning benchmarks, or performance
  leaderboards.
- A replacement for the existing four Learn lessons or a redesign of the core
  RAG planes.

## Required Verification

- Unit tests proving explanation values reproduce actual BM25, cosine, RRF,
  and reranking order.
- API compatibility tests for old run artifacts and pre-Phase-3.4 indexes.
- Qdrant parity tests using exact search and copied vectors, cosine-upload
  normalization, provenance plus complete remote-vector verification,
  tolerance/tie behavior, atomic repair, payload/filter tests, absent service
  behavior, and legacy minimal payloads with filters disabled.
- Evaluation metric, progress, cancellation, refresh-recovery, and per-question
  detail tests.
- Full/slim model readiness and explicit download tests, including checks that
  CPU-only dependencies do not pull NVIDIA packages.
- Bilingual web tests, responsive production build, Learning Guides build, and
  complete Python regression suite.
- Fresh full and slim Compose smoke tests with clean teardown and no stale
  containers, networks, volumes, jobs, or model states.
- Owner preview of every module, Qdrant absent/present paths, reranked Explore,
  evaluation A/B flow, both languages, and responsive layouts.
- Independent architecture/code/test review with all blocking findings fixed.

## Acceptance Criteria

1. A learner can inspect lexical, dense, hybrid, and reranking calculations
   from live results over a reviewed real corpus without an LLM provider.
2. Qdrant has a substantial hands-on module that compares the same vectors with
   NumPy while remaining optional and conceptually secondary to visible math.
3. The 16-question browser evaluation compares two meaningful configurations,
   reports four metrics, and exposes every question's evidence.
4. Explore can retrieve and ask with a selected reranker using one consistent
   candidate-to-context path.
5. Full and slim images honor the offline/explicit-download contract on CPU.
6. Existing indexes, traces, lessons, CLI workflows, custom corpora, and the
   no-Qdrant deployment continue to work.
7. English/Chinese UI and guides, automated CI, package documentation, owner
   preview, and independent sign-off are complete for version `0.5.0`.

## Sign-off Record

The owner accepted the staged roadmap and explicitly approved a dedicated,
substantial Qdrant module while retaining NumPy as the canonical learning path
on 2026-07-16. `/root/phase34_scope_review` independently reviewed the
candidate, requested reproducibility, compatibility, cancellation, task-split,
and Qdrant-normalization clarifications, then signed off the revised scope with
no remaining findings on 2026-07-16. The phase was activated the same day.
