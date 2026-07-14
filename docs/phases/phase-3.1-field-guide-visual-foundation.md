# Phase 3.1: Field-Guide Visual Foundation

**Status:** Complete. Scope was approved by the owner and independently
reviewed by `/root/phase31_arch_review` on 2026-07-13 before activation.
Implementation was accepted by the owner and signed off by
`/root/phase31_code_review` on 2026-07-13.

## Goal

Make the local visual lab clearer, calmer, and more dynamic without adding RAG
mechanics or a Guided Learn mode. The lab should present the project-owned
artifacts as an inspectable field guide rather than a generic dashboard or a
copy of another RAG course.

## In Scope

- A light, accessible field-guide visual system: paper-like neutral surfaces,
  readable ink-like text, and restrained semantic accents.
- Five capability-preserving areas with an explicit mapping: Home is Start Lab;
  Build & Inspect contains Corpus Library and Index Explorer as internal views;
  Explore is Run Workspace; Failure Lab and Settings remain their own areas.
  Existing corpus, index, run, failure, and provider workflows remain available.
- A component-oriented React client with typed, readable presentation models.
- Evidence-led index, retrieval, context, answer, and failure views. Rank,
  score, source, excerpt, context-selection state, and citation strings are
  visible before raw payloads.
- Progressive disclosure for raw manifests, vectors, prompts, and traces;
  nothing currently inspectable is discarded.
- Purposeful, non-looping motion for pipeline progression, evidence ordering,
  and context selection, with a full `prefers-reduced-motion` path.
- EN/Simplified-Chinese UI copy and learning-material links. Corpus questions,
  evidence, and answers retain their original language.
- A minimal executable front-end test stack: Vitest, React Testing Library, and
  jsdom. It covers grouped navigation, EN/ZH switching, starter replay handoff,
  evidence/context state, raw-artifact disclosure, and reduced-motion behavior.
  Existing API-contract coverage remains in place.
- Local-browser preview at desktop, 390px mobile, and 320px narrow-mobile widths
  before sign-off.

## Out Of Scope

- Guided Learn mode, lesson sequencing, or new curated learning corpus.
- Changes to FastAPI endpoints, `LabRun` schema, trace persistence, index
  artifacts, embedding/generation providers, or vector backends.
- Web controls for reranking, a new lexical/BM25 inspector, and browser-based
  evaluation/reporting.
- Dark theme, shareable run URLs, cloud hosting, or a public multi-user model.
- Version/release reconciliation; that remains a closeout decision.

## Durable Decisions

- Phase 3.1 is a front-end and presentation layer only. The web client renders
  existing project artifacts and does not recompute or reinterpret RAG
  mechanics.
- NumPy remains the default inspectable backend and Qdrant remains an optional
  backend. Their conceptual presentation stays aligned.
- Raw JSON is an advanced inspection view, not the default learning surface.
- Motion explains a real state change; it is never autonomous decoration and
  must disappear under reduced-motion preferences.
- Citation presentation is descriptive only: the UI may place returned citation
  strings beside selected evidence and source metadata, but must not infer or
  label citation correctness/support. Curated Failure Lab outcomes remain the
  teaching surface for citation-mismatch cases.
- Typed UI presentation models may normalize each existing view's data for
  rendering, but Failure Lab artifacts must remain distinct from `LabRun` and
  their API shapes must not be changed.
- Phase 3.2 may add Guided Learn + Explore only after the owner previews this
  foundation and accepts its interaction model.

## Acceptance Criteria

1. The local lab is light-theme, keyboard-usable, responsive to 320px, and
   readable without relying on color alone.
2. Navigation maps Start Lab to Home, Corpus Library and Index Explorer to
   separate internal Build & Inspect views, Run Workspace to Explore, and keeps
   Failure Lab and Settings separate, without removing a workflow or changing
   backend/API behavior. Starter replay begins on Home and opens its resulting
   artifact in Explore.
3. Home replay, custom corpus upload/indexing, watsonxDocsQA import, NumPy and
   optional Qdrant selection, retrieve, live ask, failure comparison, and
   settings all still work.
4. Explore and Failure Lab lead with labelled evidence cards and context/citation
   relationships; raw run data remains available on demand.
5. UI copy is localized in EN/ZH while English starter-corpus data stays
   unchanged.
6. Motion is tied to pipeline/evidence/context state and reduced-motion users
   receive an equivalent static experience.
7. Vitest/React Testing Library interaction tests, web build, Python tests,
   Compose configuration, and local preview at desktop, 390px, and 320px pass
   before review.

## Work Breakdown

See [the Phase 3.1 taskboard](phase-3.1-taskboard.md).
