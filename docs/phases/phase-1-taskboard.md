# Phase 1 Taskboard

This file tracks Phase 1 implementation tasks, dependencies, ownership, and
status in one lightweight table.

The candidate implementation contract is
`docs/phases/phase-1-naive-classic-rag.md`. This taskboard should stay aligned
with that spec. It becomes active only after owner acceptance.

## Status Values

- `todo`: not started
- `in_progress`: actively being implemented
- `review`: implementation is ready for review and verification
- `blocked`: cannot proceed; blocker must be written in `Notes`
- `done`: reviewed, tested, and accepted

## Update Rules

- Set `Status` to `in_progress` before starting work.
- Set `Status` to `review` after implementation and local tests.
- Set `Status` to `done` only after review and required tests pass.
- The task owner must not mark their own task `done`; a different reviewing
  agent must sign off and make the `done` update.
- When marking `done`, record the reviewing agent and test result in `Notes`.
- Use `blocked` only with a concrete blocker in `Notes`.
- Keep `Owner` as an agent/person name or `unassigned`.
- Do not change task IDs after creation.
- Update `Notes` with skipped tests, setup limits, or follow-up work.
- Keep acceptance criteria short; detailed requirements belong in the phase spec.

## Taskboard

| ID | Milestone | Task | Depends On | Status | Owner | Acceptance | Notes |
|---|---|---|---|---|---|---|---|
| P1-T00 | M1.0 | Project scaffolding | none | todo | unassigned | `uv sync`, import works, pytest runs | Python package, `pyproject.toml`, dev deps, initial test harness |
| P1-T01 | M1.0 | CLI entry point | P1-T00 | todo | unassigned | `rag --help` works | Use `argparse`; no RAG behavior required yet |
| P1-T02 | M1.1 | watsonxDocsQA preparation script | P1-T00 | todo | unassigned | local dataset converts to Markdown, manifest, QA JSONL | Script lives under `scripts/`; download path optional |
| P1-T03 | M1.1 | Corpus git-ignore rules | P1-T00 | todo | unassigned | generated corpora and index dirs ignored | Commit scripts and fixtures only |
| P1-T04 | M1.2 | Core data contracts | P1-T00 | todo | unassigned | dataclasses serialize cleanly | `Document`, `Chunk`, `RetrievalResult`, `RagTrace` |
| P1-T05 | M1.2 | Document loader | P1-T04 | todo | unassigned | `.md` and `.txt` load with title/hash | Markdown H1 title fallback to filename |
| P1-T06 | M1.2 | Text normalization | P1-T04 | todo | unassigned | normalization tests pass | Line endings, trailing spaces, blank-line collapse |
| P1-T07 | M1.3 | Character chunker | P1-T04, P1-T06 | todo | unassigned | offset and stable-ID tests pass | Preserve slice invariant into normalized text |
| P1-T08 | M1.4 | Embedding interface and fake embedder | P1-T04 | todo | unassigned | deterministic fixture retrieval possible | Tests must not require model downloads |
| P1-T09 | M1.4 | Local sentence-transformers embedder | P1-T08 | todo | unassigned | real embedder can embed sample text | Use `sentence-transformers/all-MiniLM-L6-v2` |
| P1-T10 | M1.5 | Index writer | P1-T04, P1-T07, P1-T08 | todo | unassigned | manifest, chunks JSONL, embeddings NPZ written | Include corpus file hashes and embedding metadata |
| P1-T11 | M1.5 | Index loader | P1-T10 | todo | unassigned | saved fixture index loads | Validate chunk IDs and embedding row order |
| P1-T12 | M1.6 | Cosine retrieval | P1-T08, P1-T11 | todo | unassigned | known-vector ranking tests pass | Handle zero vectors deliberately |
| P1-T13 | M1.6 | `rag index` | P1-T02, P1-T05, P1-T07, P1-T09, P1-T10 | todo | unassigned | CLI builds index artifacts | Chunk flags belong here |
| P1-T14 | M1.6 | `rag retrieve` | P1-T09, P1-T11, P1-T12 | todo | unassigned | CLI prints ranked chunks and scores | Include chunk ID, title/path, preview |
| P1-T15 | M1.7 | Prompt assembly | P1-T12 | todo | unassigned | prompt includes question, context, source markers | Project-owned visible template |
| P1-T16 | M1.7 | Generation interface and fake generator | P1-T15 | todo | unassigned | fake answer includes source marker | No network/API key in tests |
| P1-T17 | M1.7 | OpenAI-compatible generator | P1-T16 | todo | unassigned | provider smoke path is configurable | Tests may skip real provider without credentials |
| P1-T18 | M1.7 | `rag ask` | P1-T14, P1-T15, P1-T16 | todo | unassigned | CLI prints answer, citations, source table | Use fake backend in CLI tests |
| P1-T19 | M1.8 | Persistence round-trip test | P1-T10, P1-T11, P1-T12 | todo | unassigned | loaded index preserves expected retrieval | Required acceptance criterion |
| P1-T20 | M1.8 | CLI test coverage | P1-T13, P1-T14, P1-T18 | todo | unassigned | index/retrieve/ask CLI tests pass | Use tiny fixture corpus and fake backends |
| P1-T21 | Phase close | Handoff and verification | P1-T00-P1-T20 | todo | unassigned | handoff complete, required tests recorded | Reviewer must verify before marking done |

## Review-Sensitive Tasks

These tasks require architecture/code review before being marked `done`:

- `P1-T04`: data contracts and serialization shape.
- `P1-T07`: chunk offset semantics and stable chunk IDs.
- `P1-T10`: persisted index format and manifest contents.
- `P1-T12`: cosine ranking and zero-vector behavior.
- `P1-T15`: prompt contract and citation/source marker format.
- `P1-T18`: end-to-end `ask` behavior and trace output.

## Minimum Phase 1 Completion

Minimum Phase 1 completion requires `P1-T00` through `P1-T21`.

## Stretch

These tasks are useful after the primary corpus path works, but they do not
block Phase 1 completion.

| ID | Milestone | Task | Depends On | Status | Owner | Acceptance | Notes |
|---|---|---|---|---|---|---|---|
| P1-S01 | Stretch | WixQA preparation script | P1-T02 | todo | unassigned | local WixQA converts to Markdown, manifest, QA JSONL | Customer-support corpus for later experiments |
| P1-S02 | Stretch | WixQA indexing smoke test | P1-S01, P1-T13 | todo | unassigned | `rag index` works on prepared WixQA | Does not block Phase 1 |

## Next Phase

Phase 1.5 should focus on retrieval mechanics: configurable experiments,
keyword/BM25 baseline, hybrid retrieval, optional reranking, and richer
retrieval inspection. It is not active until a dedicated spec and taskboard are
created.
