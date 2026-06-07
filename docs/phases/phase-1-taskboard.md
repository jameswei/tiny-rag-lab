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
| P1-T00 | M1.0 | Project scaffolding | none | done | claude | `uv sync`, import works, pytest runs | reviewed by codex; `uv run pytest --tb=short -q`: 1 passed; import and dependency smoke checks pass; no findings |
| P1-T01 | M1.0 | CLI entry point | P1-T00 | done | claude | `rag --help` works | reviewed by codex; `uv run pytest --tb=short -q`: 8 passed; `rag --help` and subcommand help pass; no findings |
| P1-T02 | M1.1 | watsonxDocsQA preparation script | P1-T00 | done | claude | local dataset converts to Markdown, manifest, QA JSONL | reviewed by codex; `uv run pytest --tb=short -q`: 41 passed; default dataset/schema and doc/QA ID linkage verified; no findings |
| P1-T03 | M1.1 | Corpus git-ignore rules | P1-T00 | done | claude | generated corpora and index dirs ignored | reviewed by codex; `.gitignore` excludes `corpus/` and `.tiny-rag/`; no findings |
| P1-T04 | M1.2 | Core data contracts | P1-T00 | done | claude | dataclasses serialize cleanly | reviewed by codex; `uv run pytest --tb=short -q`: 33 passed; dataclass contracts and serialization tests pass; no findings |
| P1-T05 | M1.2 | Document loader | P1-T04 | done | claude | `.md` and `.txt` load with title/hash | reviewed by codex; `uv run pytest tests/test_documents.py --tb=short -q`: 24 passed; doc IDs, titles, formats, hashes verified; no findings |
| P1-T06 | M1.2 | Text normalization | P1-T04 | done | claude | normalization tests pass | reviewed by codex; `uv run pytest --tb=short -q`: 65 passed; line endings, trailing whitespace, and blank-line collapse verified; no findings |
| P1-T07 | M1.3 | Character chunker | P1-T04, P1-T06 | done | claude | offset and stable-ID tests pass | reviewed by codex; `uv run pytest tests/test_chunking.py tests/test_embeddings.py --tb=short -q`: 33 passed; slice invariant, stable IDs, and boundary chunks verified; no findings |
| P1-T08 | M1.4 | Embedding interface and fake embedder | P1-T04 | done | claude | deterministic fixture retrieval possible | reviewed by codex; `uv run pytest tests/test_chunking.py tests/test_embeddings.py --tb=short -q`: 30 passed; fake embedder deterministic shape/dtype/unit-vector behavior verified; no findings |
| P1-T09 | M1.4 | Local sentence-transformers embedder | P1-T08 | done | claude | real embedder can embed sample text | reviewed by codex; `uv run pytest tests/test_sentence_transformer_embedder.py --tb=short -q`: 11 passed; local-files-only test path verified; no findings |
| P1-T10 | M1.5 | Index writer | P1-T04, P1-T07, P1-T08 | done | claude | manifest, chunks JSONL, embeddings NPZ written | reviewed by codex; `uv run pytest tests/test_index_writer.py --tb=short -q`: 23 passed; manifest, chunks JSONL, embeddings NPZ, and chunk ID order verified; no findings |
| P1-T11 | M1.5 | Index loader | P1-T10 | done | claude | saved fixture index loads | reviewed by codex; `uv run pytest tests/test_index_loader.py --tb=short -q`: 21 passed; round-trip, chunk ID order, missing files, and corrupt row count verified; no findings |
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
