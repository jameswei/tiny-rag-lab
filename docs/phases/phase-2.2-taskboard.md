# Phase 2.2 Taskboard

This file tracks Phase 2.2 implementation tasks, dependencies, ownership, and
status in one lightweight table.

The implementation contract is
`docs/phases/phase-2.2-structural-semantic-chunking.md`.

## Status Values

- `todo`: not started
- `in_progress`: actively being implemented
- `review`: implementation is ready for review and verification
- `blocked`: cannot proceed; blocker must be written in `Notes`
- `done`: reviewed, tested, and accepted

## Update Rules

- Set `Status` to `in_progress` before starting work.
- Set `Status` to `review` after implementation and local tests pass.
- Set `Status` to `done` only after review and required tests pass.
- The task owner must not mark their own task `done`; a different reviewing
  agent must sign off and make the `done` update.
- When marking `done`, record the reviewing agent and test result in `Notes`.
- Use `blocked` only with a concrete blocker in `Notes`.
- Keep `Owner` as an agent/person name or `unassigned`.
- Do not change task IDs after creation.
- Keep `Notes` concise. Detailed handoff notes, review findings, and full
  test evidence belong in `CURRENT.md`.

## Taskboard

| ID | Milestone | Task | Depends On | Status | Owner | Acceptance | Notes |
|---|---|---|---|---|---|---|---|
| P2.2-T01 | M2.2.0 | `chunking.py`: extract `_validate_chunk_params(chunk_size, chunk_overlap)` from `chunk_document` (identical `ValueError` text); add `_chunk_oversized_span(doc, start, end, chunk_size, chunk_overlap) -> list[Chunk]` shared sliding-window fallback restricted to a span (tier 3, consumes `chunk_overlap`); add `_split_sentences(text) -> list[tuple[int,int]]` (regex sentence splitter; introduced here so it is shared, used by both this task's fallback and P2.2-T02); add `_split_blocks(text) -> list[tuple[int,int]]` (blank-line splitter; a block that is a single ATX heading line merges with the next block); add `chunk_document_structural(doc, chunk_size=800, chunk_overlap=120) -> list[Chunk]` — three-tier packing: (1) pack whole blocks up to `chunk_size`, non-overlapping; (2) a block that alone exceeds `chunk_size` is split via `_split_sentences` and packed at sentence granularity instead of going straight to character windowing; (3) only a single sentence that itself exceeds `chunk_size` falls back to `_chunk_oversized_span`. Tests in `tests/test_chunking.py`. | — | done | Claude Code | Slice invariant holds for all structural chunks; heading-only block never forms a standalone chunk when a following block exists; an oversized block is packed at sentence granularity (tier 2), not character-windowed directly; only an oversized single sentence triggers `_chunk_oversized_span` (tier 3) and that fallback still satisfies the slice invariant + respects `chunk_overlap`; tier 1 and tier 2 chunks are contiguous (`chunk[i].char_end == chunk[i+1].char_start`); metadata/`chunk_id` identical contract to `chunk_document`; empty/whitespace document yields zero chunks; all existing `chunk_document`/`chunk_documents` tests still pass unmodified; `uv run pytest tests/test_chunking.py --tb=short -q`: 34 passed | Reviewed by Codex 2026-06-25; signed_off; focused chunking tests 34 passed; full suite 722 passed, 7 skipped. |
| P2.2-T02 | M2.2.1 | `chunking.py`: add `chunk_document_semantic(doc, embedder, chunk_size=800, chunk_overlap=120, similarity_threshold=0.5) -> list[Chunk]` — reuses `_split_sentences` from P2.2-T01 as its primary unit; embeds all sentences in one batch call, packs sentences greedily, closes a chunk when the next sentence would exceed `chunk_size` or cosine similarity (dot product of L2-normalized vectors) to the previous sentence drops below `similarity_threshold`; falls back to `_chunk_oversized_span` for a single oversized sentence. Tests in `tests/test_chunking.py` using the existing `FakeEmbedder`. | P2.2-T01 | done | Claude Code | Slice invariant holds for all semantic chunks; `similarity_threshold=-2.0` (unreachable) yields chunking driven purely by `chunk_size`; a single sentence exceeding `chunk_size` triggers the shared fallback; deterministic across repeated calls with the same `FakeEmbedder`; metadata/`chunk_id` identical contract to `chunk_document`; empty document yields zero chunks; `uv run pytest tests/test_chunking.py --tb=short -q`: 44 passed | Reviewed by Codex 2026-06-25; signed_off; focused chunking tests 44 passed; full suite 732 passed, 7 skipped. |
| P2.2-T03 | M2.2.2 | `chunking.py`: add `chunk_document_with_strategy(doc, strategy="fixed_character", chunk_size=800, chunk_overlap=120, embedder=None, similarity_threshold=0.5) -> list[Chunk]` and `chunk_documents_with_strategy(docs, ...)` dispatching to the three chunkers by name; raise `ValueError` for an unknown strategy or `strategy="semantic"` with `embedder=None`. `index_writer.py`: `write_index`/`_write_manifest` gain `chunking_strategy: str = "fixed_character"`, `chunking_params: dict \| None = None`, persisted into `manifest.json` (`chunking_params` always serialized, `{}` when not semantic). `cli.py`: `cmd_index` gains `--chunking-strategy {fixed_character,structural,semantic}` (default `fixed_character`) and `--semantic-similarity-threshold FLOAT` (default `0.5`); embedder construction moves before chunking **only** when `strategy == "semantic"` — `fixed_character` and `structural` keep today's chunk-then-embed order unchanged; dispatches via `chunk_documents_with_strategy`; passes `chunking_strategy`/`chunking_params` to `write_index`; post-index summary print includes the chosen strategy for every strategy (an intentional stdout change, including for the default). Tests in `tests/test_chunking.py`, `tests/test_index_writer.py`, `tests/test_cmd_index_retrieve.py`. | P2.2-T01, P2.2-T02 | todo | unassigned | Default `rag index` (no new flags) produces chunks and embeddings byte-identical to Phase 2.1, embedder constructed after chunking exactly as in Phase 2.1, and a manifest identical to Phase 2.1 plus the two new keys at their defaults (only the printed summary line differs, intentionally); `rag index --chunking-strategy structural` and `--chunking-strategy semantic --semantic-similarity-threshold 0.7` both exit 0 with correct `manifest.json` fields; `rag retrieve` against a semantic-chunked index returns results; unknown strategy and `semantic` + no embedder both raise `ValueError`; a test asserts the embedder is constructed before chunking for `semantic` and after chunking for `fixed_character`/`structural`; `uv run pytest tests/test_chunking.py tests/test_index_writer.py tests/test_cmd_index_retrieve.py --tb=short -q`: N passed | |
| P2.2-T04 | M2.2.3 | Add new fixture corpus content (or extend an existing fixture) containing a clear case where fixed-character chunking at a deliberately tight `chunk_size` splits a necessary sentence/instruction across two chunks while structural chunking at the same `chunk_size` keeps it whole. Build both index variants with `rag index --chunking-strategy fixed_character` / `--chunking-strategy structural` against the same corpus and document the before/after `rag retrieve`/`rag diagnose` comparison. No changes to `failure.py`, `eval.py`, `FailureCase`, or `RetrieverConfig`. | P2.2-T03 | todo | unassigned | Fixture corpus committed; building both index variants and running the same query/case against each shows a measurable difference in retrieved/omitted context attributable to chunking strategy alone; comparison is reproducible from documented CLI commands; no production code changed outside fixtures/docs; `uv run pytest --tb=short -q`: N passed, no regressions | |
| P2.2-T05 | M2.2.4 | Phase close: update `docs/phases/README.md` (mark Phase 2.2 complete, "No active phase"), `docs/roadmap.md` (Phase 2.2 → Complete), `README.md` (Phase 2.2 result + CLI examples), `docs/file-structure.md` (chunking.py/index_writer.py/cli.py phase tags), `corpus/gaps.md` §1.1 (structural/semantic chunking no longer purely a gap — note what is and isn't implemented), and EN/ZH learning materials (new doc or extension covering structural/semantic chunking, plus learning-roadmap entries). Run full suite and CLI smokes. | P2.2-T01–T04 | todo | unassigned | All P2.2-T01–T04 `done` with reviewer sign-off; `uv run pytest --tb=short -q`: all passed; `uv run rag index --help` shows `--chunking-strategy` and `--semantic-similarity-threshold`; stale-reference check for Phase 2.1-only wording in README, file-structure, gaps.md, and learning roadmaps finds no matches; phase index and roadmap updated | |

## Review-Sensitive Tasks

- **P2.2-T01**: The validation extracted into `_validate_chunk_params` must
  raise byte-identical `ValueError` message text to what `chunk_document`
  raises today, or every existing test asserting on that text breaks.
  The fallback order matters: an oversized block must be packed at sentence
  granularity (tier 2) before `_chunk_oversized_span` (tier 3) is ever
  considered — going straight from block to character window would silently
  drop the roadmap's "sentence boundaries where practical" requirement and
  reintroduce the mid-sentence split this phase exists to fix.
  `_chunk_oversized_span` must be the *only* place either new strategy
  consumes `chunk_overlap` — tier 1 and tier 2 chunks must be exactly
  contiguous, not overlapping.
- **P2.2-T02**: `chunk_document_semantic` must call `embedder.embed(...)`
  exactly once per document (one batch call for all sentences), not once per
  sentence — a per-sentence call would silently make indexing far slower
  than documented. Cosine similarity must be computed as a dot product
  (vectors are already L2-normalized); do not re-normalize or call a
  separate similarity library.
- **P2.2-T03**: Default `rag index` behavior (no new flags, i.e.
  `fixed_character`) must produce chunks and embeddings byte-identical to
  Phase 2.1, and a manifest identical to Phase 2.1 except for the two added
  keys at their documented defaults — only the printed summary line may
  differ. Embedder construction must move before chunking **only** when
  `strategy == "semantic"`; `fixed_character` and `structural` must keep
  today's chunk-then-embed order exactly, so the default path never
  constructs a real embedding model before chunking validates/completes,
  same as Phase 2.1.
- **P2.2-T04**: Must not modify `failure.py`, `eval.py`, `FailureCase`, or
  `RetrieverConfig` — the comparison is a fixture + documented CLI
  walkthrough, not new comparison code.

## Minimum Phase 2.2 Completion

Minimum Phase 2.2 completion requires P2.2-T01 through P2.2-T05.
