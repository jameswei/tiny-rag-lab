# Phase 3.0 Spec: Local Visual RAG Lab

**Status:** Complete — post-merge correctness repair (P3.0-T10) independently signed off by Claude Sonnet 5 on 2026-07-13
**Authors:** Codex + owner decisions
**Based on:** `docs/proposal.md`, `docs/architecture.md`, `docs/roadmap.md`
**Taskboard:** `docs/phases/phase-3.0-taskboard.md`
**Date:** 2026-07-12

---

## Goal

Make the existing classic-RAG engine approachable as a local, visual learning
laboratory. A learner must be able to inspect what happens to documents,
chunks, embeddings, retrieved evidence, packed context, and citations for a
single RAG run without giving up the project's explicit, teachable mechanics.

The existing `rag` CLI remains a supported, container-free interface. The web
lab is a second local distribution, not a replacement for the engine or a
public multi-user service.

## Scope

### In Scope

- A React + TypeScript browser client and a local FastAPI API, packaged by
  Docker Compose and bound only to loopback.
- English and Simplified Chinese UI and lesson content.
- A bundled, fully offline quick-start corpus with recorded, replayable traces.
- A corpus library with a user-triggered watsonxDocsQA preparation/import path
  and small custom Markdown/plain-text upload (at most 100 files / 100 MiB).
- Indexing, retrieve, and ask flows with saved, visual replay artifacts.
- Evidence-first inspection of chunking, embeddings, NumPy index artifacts,
  retrieval/reranking, context packing, prompt, answer, and citations.
- Existing curated failure cases presented as baseline-versus-intervention
  lessons.
- A `VectorIndexBackend` abstraction: NumPy remains the default; Qdrant is an
  optional local Docker Compose profile and first industry-style comparison.
- Full and slim local image variants. Full contains the default embedding
  model; slim requires an explicit local model download before custom indexing.
- OpenAI-compatible live generation. A Compose environment configuration may
  supply a default provider; the UI may make a non-persistent session override.

### Out Of Scope

- Public hosting, accounts, sharing, telemetry, tenancy, or background cloud
  ingestion.
- PDF, Office, archive, URL, or large-corpus ingestion.
- Bundled local LLM generation. A user must configure an OpenAI-compatible
  local or hosted provider for live `ask` answers.
- A browser replacement for arbitrary `rag eval` batch runs. Evaluation stays
  in the CLI; Phase 3.0 presents only curated failure lessons.
- Additional vector database adapters, ANN benchmarking, and automatic index
  migration between backends.
- Chinese or other multilingual corpus support, including multilingual model
  packaging and retrieval-quality evaluation. This is deferred to a dedicated
  follow-up phase so Phase 3.0 can finish the English-corpus visual lab well.

## Design Decisions

### One engine, two entrypoints

HTTP handlers call extracted project-owned services; they do not shell out to
the CLI or duplicate RAG mechanics in the frontend. CLI defaults and existing
NumPy indexes remain backward compatible. The default backend is inferred as
`numpy` for manifests produced before Phase 3.0.

### Backend changes storage/search, not the lesson

`Document`, `Chunk`, prompt construction, BM25, hybrid RRF, reranking, context
packing, and generation remain backend-neutral. Both vector backends expose
the same ranked-hit contract. A Qdrant index retains canonical local chunk and
inspection-vector artifacts; Qdrant is authoritative only for dense vector
search. Its returned score is labelled with its metric/backend semantics and
is never silently compared with another backend's raw score.

### A run is an immutable learning artifact

The current CLI trace is useful but intentionally compact. A Phase 3.0 lab
run adds a versioned identifier, index snapshot, exact configuration, query
vector, bounded candidate evidence, score components, full evidence text,
context-packing decision, prompt, answer, citations, stage timings, and any
provider-neutral error. Stored runs never include an API key.

### Quick-start is replay; custom work is real

The quick-start pack uses checked-in traces rather than a fake answer that
could be mistaken for a real model response. Custom corpus retrieval works
locally after indexing. Live answers require an OpenAI-compatible provider;
without one, the lab explains the unavailable generation stage while still
supporting retrieval exploration.

### UI language is not corpus language

EN/ZH applies to the lab's controls, labels, authored teaching explanations,
and learning-doc links. Corpus-derived questions, chunks, retrieved evidence,
answers, citations, and observed run outcomes always stay in the corpus's
original language. Phase 3.0's supported corpus path is English; its EN/ZH UI
does not imply a Chinese retrieval-quality promise.

### Compose is complete, but optional infrastructure stays optional

The `studio` service contains the Python API and built frontend. Its persistent
data lives in a local volume. Qdrant runs only under an explicit profile and
is not exposed on a host port. The default `full` image contains the existing
embedding model; the `slim` variant stores a user-downloaded model in the same
local data volume.

## Public Contracts

### Index backend

Add a small project-owned protocol with `build`, `open`, and `search`
operations. `search` returns backend-neutral ranked vector hits containing a
chunk, rank, raw score, score semantics, and optional backend audit fields.
The NumPy implementation preserves exact cosine similarity. Qdrant uses one
local collection per index and payloads containing only canonical chunk
identity/metadata needed to restore the local chunk record.

The index manifest gains `index_backend`, `distance_metric`, and backend
identity fields. Readers must treat a missing `index_backend` as `numpy`.

### Lab API

Expose only local JSON/file endpoints for health, catalog/corpus listing,
small corpus upload, watsonxDocsQA import job creation/status, index creation,
retrieve, ask, saved run retrieval, and provider/model status. Long downloads
and indexing execute as one local job at a time with pollable progress and a
clear restart/failure state. API responses never expose provider secrets.

### Provider precedence

For live generation, a per-run browser-session override takes precedence over
the server's Compose environment defaults. Endpoint/model preferences may live
in browser storage; API keys stay in memory and are sent only to the loopback
API for that run. The server never persists or returns them.

## Learner Surface

- **Start Lab:** bilingual quick-start replay and an explicit prompt to add a
  real learning corpus.
- **Corpus Library:** starter pack, watsonxDocsQA download/import, and custom
  Markdown/text upload.
- **Index Explorer:** source text, chunk boundaries, manifest, embedding
  components, and local index/backend details.
- **Run Workspace:** controls plus a visual pipeline stepper for retrieval or
  ask. It shows exact evidence and a clearly-labelled 2D projection only as
  an intuition aid; raw vector components and cosine calculations remain
  available.
- **Failure Lab:** the existing retrieval and answer-side scenarios, with the
  observed baseline/intervention traces and bilingual explanation.

Every stage links to the stable corresponding EN/ZH learning material, never
to a volatile source-code line.

## Completion Criteria

1. `docker compose up` starts a loopback-only full visual lab; a slim variant
   clearly completes its embedding-model prerequisite before custom indexing.
2. A learner can replay a starter trace; import watsonxDocsQA or upload a
   small Markdown/text corpus; build a NumPy index; inspect it; retrieve; and
   replay a saved run.
3. Live ask works with an OpenAI-compatible provider and remains unavailable
   with a clear non-secret error when none is configured.
4. NumPy stays the default and existing CLI/index tests remain compatible.
5. The optional Qdrant profile builds/searches an equivalent fixture index
   through the same visible pipeline.
6. EN/ZH UI, failure lessons, docs, automated tests, Compose smoke checks,
   architecture review, code review, and independent test verification pass.
