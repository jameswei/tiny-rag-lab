# Phase 3.0 Record: Local Visual RAG Lab

**Status:** Complete, including the post-merge correctness repair signed off by
Claude Sonnet 5 on 2026-07-13.

## Delivered

- A local React learning client and FastAPI API packaged as one Compose studio
  service beside the existing CLI.
- Offline starter replay, custom small Markdown/text corpora, guided
  watsonxDocsQA import, visual retrieve/ask/context-packing playback, and
  curated failure lessons.
- Inspectable NumPy/file-index vectors by default, with an optional local
  Qdrant backend behind the same learning flow.
- English and Simplified Chinese UI plus full/slim image variants and optional
  OpenAI-compatible live generation.

## Durable Decisions

- The CLI and browser are two entrypoints to one project-owned RAG engine; the
  web client does not reimplement or hide the pipeline.
- A saved lab run is an immutable, replayable learning artifact containing its
  configuration, evidence, context packing, and answer outcome without secrets.
- NumPy remains the default inspectable backend. Qdrant changes vector
  persistence/search only, not chunks, embeddings, prompts, citations, or
  trace vocabulary.
- UI language and corpus language are independent. The supported default corpus
  path is English; bilingual UI does not imply multilingual retrieval quality.
- Local Compose is complete without making optional infrastructure mandatory:
  Qdrant is profile-gated and live generation requires user configuration.

## Repair and Deferred Context

The signed-off repair added replayable non-secret live-generation failures,
bounded streamed uploads, visible one-job admission, Qdrant error boundaries,
diagnostic logging, shared citation extraction, backend-factory use, and legacy
manifest coverage. Index/model caching and defensive backend-identity tightening
remain deliberate follow-up decisions rather than silently bundled changes.
