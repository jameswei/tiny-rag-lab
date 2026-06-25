# Phase 2.2 Spec: Structural And Semantic Chunking

**Status:** Active
**Authors:** Claude Code
**Based on:** `docs/phases/phase-1.9-2.2-final-roadmap.md`, `docs/roadmap.md`
**Taskboard:** `docs/phases/phase-2.2-taskboard.md`
**Date:** 2026-06-25

---

## Goal

After Phase 2.1, the pipeline can measure whether retrieval is accurate,
whether answers are faithful, and how much retrieved context reaches the
prompt. None of that says anything about how the chunks themselves were
formed. Phase 1 through 2.1 used a fixed-size character sliding window for
every document — correct as a learning baseline, but indifferent to document
structure. It can split a heading from its body, or a sentence across a
chunk boundary, with no awareness that it has done so.

Phase 2.2 adds two new chunking strategies alongside the existing
fixed-character chunker, which remains the default and the fallback:

- **Structural**: pack Markdown-aware blocks (headings, paragraphs, lists)
  so chunk boundaries fall between units of meaning instead of mid-sentence.
- **Semantic** (experimental): embed sentences and split where consecutive
  sentence similarity drops below a threshold — a topic-shift boundary.

This continues the project's "keep mechanics visible" philosophy
(`docs/architecture.md`): chunking strategy becomes an explicit, recorded,
comparable choice instead of a hidden constant. It does not change how
chunks are consumed once produced — `Chunk`, `RetrievalResult`,
`format_ask_trace`, citations, and `assemble_prompt` are all untouched.
Chunking strategy is purely an indexing-time decision, recorded once in the
index manifest.

---

## Scope

### In Scope

- `tiny_rag_lab/chunking.py`: `chunk_document_structural`,
  `chunk_document_semantic`, `chunk_document_with_strategy`,
  `chunk_documents_with_strategy` (new). Existing `chunk_document` /
  `chunk_documents` keep their exact current signature and behavior.
- `tiny_rag_lab/index_writer.py`: `write_index` / `_write_manifest` gain
  `chunking_strategy: str = "fixed_character"` and
  `chunking_params: dict | None = None`, persisted into `manifest.json`.
- `tiny_rag_lab/cli.py`: `cmd_index` gains `--chunking-strategy
  {fixed_character,structural,semantic}` (default `fixed_character`) and
  `--semantic-similarity-threshold FLOAT` (default `0.5`, consumed only when
  strategy is `semantic`). For `--chunking-strategy semantic` only,
  embedder construction in `cmd_index` moves before chunking so the
  semantic chunker can use it; the `fixed_character` and `structural` paths
  keep today's chunk-then-embed order unchanged.
- New fixture corpus content plus a documented before/after example showing a
  chunking-boundary failure under fixed-character chunking that structural
  chunking avoids, built by running existing `rag index` / `rag retrieve` /
  `rag diagnose` against two indices that differ only in chunking strategy.
- Tests: updates to `tests/test_chunking.py`, `tests/test_index_writer.py`,
  `tests/test_cmd_index_retrieve.py`.
- Phase close: `docs/phases/README.md`, `docs/roadmap.md`, `README.md`,
  `docs/file-structure.md`, `corpus/gaps.md` §1.1, EN/ZH learning materials.

### Out Of Scope For Phase 2.2

- Any change to `Chunk`, `RetrievalResult`, `AskTrace`/`RetrieveTrace`,
  citations, or `assemble_prompt`.
- Any change to `eval.py`, `failure.py`, or the `FailureCase` /
  `RetrieverConfig` data contracts. Comparing chunking strategies is a usage
  pattern — build two indices with different `--chunking-strategy`, run the
  existing `rag eval` / `rag diagnose` against each, compare the reports —
  not new code. `RetrieverConfig` has no `index_dir` field and gains none.
- Per-chunk strategy metadata. Strategy is recorded once, in the manifest,
  not duplicated onto every `Chunk.metadata`.
- Incremental re-chunking or partial re-indexing.
- Recursive-character chunking, agentic/LLM-driven chunking boundaries
  (catalogued in `corpus/gaps.md` but not implemented here).
- Query rewriting, agentic RAG, or anything deferred by
  `docs/phases/phase-1.9-2.2-final-roadmap.md`.

---

## Design Decision 1: Chunking Strategy Is An Indexing-Time CLI Choice

No `--chunking-strategy` flag is added to `rag retrieve`, `rag ask`,
`rag eval`, or `rag diagnose`. Chunking happens once, at `rag index` time;
every downstream command reads whatever chunks are already on disk. Adding
the flag only to `cmd_index` keeps the change local and matches how
`--chunk-size` / `--chunk-overlap` already work. Comparing strategies means
building two index directories and pointing existing commands at each one —
no command other than `rag index` needs to know strategies exist.

## Design Decision 2: Three-Tier Packing — Blocks, Then Sentences, Then Raw Character Window

The roadmap's structural scope is explicitly four items: headings,
paragraphs, lists, **and sentence boundaries where practical**. A naive
"pack blocks, fall back to character windowing for an oversized block" design
would silently drop the fourth item — an oversized block would be split
mid-sentence by the character fallback, the exact failure mode (a sentence
severed across a chunk boundary) this phase exists to avoid. So both new
strategies use a three-tier fallback, not two:

1. **Structural**: pack whole blocks (from `_split_blocks`) into chunks up to
   `chunk_size`. **Semantic**: pack whole sentences (from `_split_sentences`)
   into chunks up to `chunk_size`, subject to the similarity-threshold split
   in Design Decision 4. Packed chunks at this tier are contiguous and
   non-overlapping (`chunk[i].char_end == chunk[i+1].char_start`).
2. If a single **block** (structural mode only) exceeds `chunk_size` on its
   own, split that block into sentences via `_split_sentences` and pack
   *those* sentences up to `chunk_size` — still non-overlapping. This is the
   "sentence boundaries where practical" tier: it only activates when block
   packing alone cannot satisfy the budget.
3. If a single **sentence** still exceeds `chunk_size` on its own (a
   run-on sentence, or a code fence with no sentence punctuation, in either
   structural or semantic mode), it is handed to a shared private helper,
   `_chunk_oversized_span`, which reuses `chunk_document`'s existing sliding
   window — restricted to that sentence's character range — and is the only
   place either new strategy applies `chunk_overlap`.

This guarantees the slice invariant holds even for pathological input
(tier 3 always terminates), keeps sentence-aware splitting genuinely
sentence-aware (tier 2), and reuses code at every tier instead of
re-implementing windowing logic three times. `_split_sentences` is therefore
introduced once, in P2.2-T01, and reused by both the structural fallback and
the semantic chunker (P2.2-T02).

## Design Decision 3: A Markdown Heading Never Forms A Chunk With No Body

The structural splitter groups `normalized_text` into blocks at blank-line
boundaries (the boundary `normalize_text` already produces between
paragraphs). If a block's entire content is a single Markdown ATX heading
line (matches `^#{1,6}\s`), it is merged with the next block before packing,
so a heading is never left to dangle alone at the end of a chunk with its
body pushed into the next one.

## Design Decision 4: Semantic Chunking Needs An `Embedder` At Chunk Time And Is Opt-In

`chunk_document_semantic(doc, embedder, chunk_size, chunk_overlap,
similarity_threshold)` splits `normalized_text` into sentences, embeds all of
them in one batch call (`embedder.embed([...])`), then walks sentences in
order: a chunk closes (and a new one starts) when adding the next sentence
would exceed `chunk_size`, **or** the cosine similarity between the current
sentence's embedding and the previous one drops below
`similarity_threshold` (cosine similarity is a plain dot product since
`Embedder` vectors are L2-normalized — no new math dependency).

This requires embedding every sentence — a finer-grained, additional
embedding pass beyond the existing chunk-level embedding — making semantic
chunking markedly slower to index than fixed-character or structural. It is
documented as an experimental, opt-in mode, consistent with
`docs/phases/phase-1.9-2.2-final-roadmap.md`'s "semantic chunking as
experimental mode when useful." `chunk_document_with_strategy` raises
`ValueError` if `strategy="semantic"` and `embedder is None`.

## Design Decision 5: New Chunkers Are Invisible Past The Indexing Boundary

`chunk_document_structural` and `chunk_document_semantic` both reuse
`make_chunk_id()` and populate `Chunk.metadata` identically to
`chunk_document()` (`title`, `path`, `format`, `raw_hash`). No new field is
added to `Chunk`. This means retrieval, trace formatting, citations, and
prompting require zero changes — a chunk produced by any strategy looks
identical to consumers downstream of chunking.

## Design Decision 6: Manifest Gains Two Defaulted Fields; Every Existing Caller Keeps Working

`write_index()` and `_write_manifest()` gain `chunking_strategy: str =
"fixed_character"` and `chunking_params: dict | None = None`. Existing
callers and tests that don't pass these keep producing the exact same
manifest shape as Phase 2.1 plus two new keys at their defaults — this is the
same "optional, defaulted parameter" pattern used for `counter` /
`context_budget` in Phase 2.1's `run_answer_eval` and `run_answer_diagnosis`.
`chunking_params` is always serialized (`{}` for `fixed_character` /
`structural`; `{"similarity_threshold": <float>}` for `semantic`) so the key
is always present and consumers never need to branch on its absence.

---

## Data Contracts

### Update to `tiny_rag_lab/chunking.py`

```python
def chunk_document_structural(
    doc: Document,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Pack Markdown-aware blocks into chunks up to chunk_size.

    Blocks are split at blank-line boundaries; a block that is a single
    ATX heading line is merged with the following block. Packed chunks
    are contiguous and non-overlapping. A block that alone exceeds
    chunk_size is split into sentences (_split_sentences) and packed at
    sentence granularity instead of falling straight to character
    windowing. Only a single sentence that itself exceeds chunk_size
    falls back to _chunk_oversized_span (uses chunk_overlap).
    """


def chunk_document_semantic(
    doc: Document,
    embedder: Embedder,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    similarity_threshold: float = 0.5,
) -> list[Chunk]:
    """Pack sentences into chunks, splitting at topic shifts.

    Splits normalized_text into sentences, embeds them in one batch call,
    and closes the current chunk when the next sentence would exceed
    chunk_size, or cosine similarity to the previous sentence's embedding
    drops below similarity_threshold. A sentence that alone exceeds
    chunk_size falls back to _chunk_oversized_span (uses chunk_overlap).
    """


def chunk_document_with_strategy(
    doc: Document,
    strategy: str = "fixed_character",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    embedder: "Embedder | None" = None,
    similarity_threshold: float = 0.5,
) -> list[Chunk]:
    """Dispatch to chunk_document / chunk_document_structural /
    chunk_document_semantic by strategy name.

    Raises ValueError for an unrecognized strategy, and if strategy is
    "semantic" and embedder is None.
    """


def chunk_documents_with_strategy(
    docs: list[Document],
    strategy: str = "fixed_character",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    embedder: "Embedder | None" = None,
    similarity_threshold: float = 0.5,
) -> list[Chunk]:
    """chunk_document_with_strategy applied to each document, in order."""
```

Internal helpers (not part of the public contract, but load-bearing):
`_split_blocks(text) -> list[tuple[int, int]]` (blank-line + heading-merge
splitter, used by structural's tier 1); `_split_sentences(text) ->
list[tuple[int, int]]` (regex sentence splitter, introduced once in
P2.2-T01, used by structural's tier 2 fallback *and* by the semantic
chunker's primary unit in P2.2-T02); `_chunk_oversized_span(doc, start, end,
chunk_size, chunk_overlap) -> list[Chunk]` (shared sliding-window fallback —
tier 3 for both new strategies, the only place either consumes
`chunk_overlap`); and `_validate_chunk_params(chunk_size, chunk_overlap)` —
the validation currently inline in `chunk_document`, extracted so all three
public chunkers share it verbatim. **The extracted helper must raise the
exact same `ValueError` message text `chunk_document` raises today** — existing
tests assert on that text.

### Update to `tiny_rag_lab/index_writer.py`

```python
def write_index(
    index_dir: Path,
    docs: list[Document],
    chunks: list[Chunk],
    embeddings: np.ndarray,
    *,
    corpus_root: Path,
    embedding_backend: str,
    embedding_model: str,
    embedding_dim: int,
    chunk_size: int,
    chunk_overlap: int,
    chunking_strategy: str = "fixed_character",   # NEW Phase 2.2
    chunking_params: dict | None = None,            # NEW Phase 2.2
) -> None:
    ...
```

`manifest.json` gains two keys:

```json
{
  "...": "... existing fields unchanged ...",
  "chunk_size": 800,
  "chunk_overlap": 120,
  "chunking_strategy": "structural",
  "chunking_params": {}
}
```

For `chunking_strategy="semantic"`, `chunking_params` is
`{"similarity_threshold": 0.5}` (or whatever value was passed).

---

## CLI Changes

### `rag index`

New flags:
- `--chunking-strategy {fixed_character,structural,semantic}` (default
  `fixed_character`)
- `--semantic-similarity-threshold FLOAT` (default `0.5`; used only when
  `--chunking-strategy semantic`)

Pipeline change in `cmd_index`, scoped by strategy so the default path is
untouched:
1. If `args.chunking_strategy == "semantic"`: construct the embedder
   **before** chunking (only this strategy needs it during chunking), then
   call `chunk_documents_with_strategy(docs, strategy="semantic",
   chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
   embedder=embedder, similarity_threshold=args.semantic_similarity_threshold)`.
   Otherwise (`fixed_character` or `structural`): keep today's existing
   order exactly — call `chunk_documents_with_strategy(docs,
   strategy=args.chunking_strategy, chunk_size=args.chunk_size,
   chunk_overlap=args.chunk_overlap)` (no embedder needed) **first**, then
   construct the embedder afterward, same as Phase 2.1. The embedder is
   never constructed before chunking for `fixed_character` or `structural` —
   only `semantic` reorders.
2. Pass `chunking_strategy=args.chunking_strategy` and the appropriate
   `chunking_params` dict to `write_index(...)`.
3. Print the chosen strategy in the existing post-index summary (e.g.
   `Chunked into N chunk(s) (strategy=structural, size=800, overlap=120)`).
   This is an intentional, documented stdout text change for every
   invocation, including the default — it is not covered by the
   "byte-identical" claim below, which is scoped to chunks, embeddings, and
   manifest content.

Default invocation (no new flags) must produce: chunks and embeddings
byte-identical to Phase 2.1; a manifest identical to Phase 2.1 except for
the two new `chunking_strategy`/`chunking_params` keys at their defaults;
and the same chunk-then-embed construction order as Phase 2.1 (embedder
construction is not moved for the default `fixed_character` path). The
printed summary line gains a strategy annotation (step 3 above) — that line,
and only that line, differs from Phase 2.1's stdout.

`--chunk-overlap`'s help text must note that for `structural` and `semantic`
strategies it only takes effect on the rare oversized-unit fallback (Design
Decision 2, tier 3) — most chunks produced by those strategies have no
overlap at all, unlike `fixed_character` chunks.

`rag retrieve`, `rag ask`, `rag eval`, `rag diagnose` are unchanged.

---

## Required Tests

### `tests/test_chunking.py`

- Structural: slice invariant across a fixture with headings, paragraphs,
  and lists; a heading-only block is merged with its following block and
  never forms a standalone chunk; a block exceeding `chunk_size` is packed at
  sentence granularity (tier 2) rather than going straight to character
  windowing — verify the resulting chunks align with sentence boundaries, not
  arbitrary character offsets; only a single sentence that itself exceeds
  `chunk_size` falls back to `_chunk_oversized_span` (tier 3) and that
  fallback's chunks satisfy the slice invariant and respect `chunk_overlap`;
  packed (tier 1 and tier 2) chunks are contiguous
  (`chunk[i].char_end == chunk[i+1].char_start`); metadata and `chunk_id`
  contract identical to `chunk_document`; empty/whitespace-only document
  yields zero chunks.
- Semantic: same slice-invariant, metadata, and empty-document coverage,
  using the existing `FakeEmbedder` (`tiny_rag_lab/embeddings.py`) for
  determinism; `similarity_threshold` set below any achievable cosine value
  (e.g. `-2.0`) yields chunking driven purely by `chunk_size`; a single very
  long sentence exceeding `chunk_size` falls back to
  `_chunk_oversized_span`; `chunk_document_semantic` is deterministic across
  repeated calls with the same `FakeEmbedder` seed.
- Dispatch: `chunk_document_with_strategy` / `chunk_documents_with_strategy`
  route to the right chunker for each of the three strategy names; an
  unknown strategy raises `ValueError`; `strategy="semantic"` with
  `embedder=None` raises `ValueError`.
- Regression: every existing `chunk_document` / `chunk_documents` test
  continues to pass unmodified — the validation refactor must not change
  observable behavior or error text.

### `tests/test_index_writer.py`

- `write_index(...)` without `chunking_strategy`/`chunking_params` writes
  `"chunking_strategy": "fixed_character"`, `"chunking_params": {}` into
  `manifest.json` (backward-compatible default).
- `write_index(..., chunking_strategy="semantic", chunking_params={
  "similarity_threshold": 0.7})` round-trips both fields through
  `manifest.json` exactly.

### `tests/test_cmd_index_retrieve.py`

- `rag index --chunking-strategy structural` exits 0; `manifest.json`
  contains `"chunking_strategy": "structural"`.
- `rag index --chunking-strategy semantic --semantic-similarity-threshold
  0.7` exits 0; `manifest.json` contains `"chunking_strategy": "semantic"`
  and `"chunking_params": {"similarity_threshold": 0.7}`; a subsequent
  `rag retrieve` against that index returns results (end-to-end smoke,
  proves semantic-chunked chunks are retrievable like any other).
- Default `rag index` invocation (no new flags) produces a manifest
  identical to Phase 2.1 except for the two new keys at their defaults.

---

## Acceptance Criteria

Phase 2.2 is complete when:

1. `chunk_document_structural`, `chunk_document_semantic`,
   `chunk_document_with_strategy`, `chunk_documents_with_strategy` exist in
   `tiny_rag_lab/chunking.py`; `chunk_document`/`chunk_documents` are
   unchanged in signature and behavior.
2. Every chunk from every strategy satisfies
   `doc.normalized_text[chunk.char_start:chunk.char_end] == chunk.text`.
   Structural chunking covers all four roadmap sub-items — headings,
   paragraphs, lists, and sentence boundaries where block packing alone
   cannot fit the budget (Design Decision 2, tier 2) — not just the first
   three.
3. `rag index` (no chunking flags, i.e. default `fixed_character`) produces
   chunks and embeddings byte-identical to Phase 2.1, and a manifest
   identical to Phase 2.1 except for the two new `chunking_strategy`/
   `chunking_params` keys at their defaults. Embedder construction stays
   after chunking for this default path, exactly as in Phase 2.1 — only the
   `semantic` strategy reorders construction before chunking. The printed
   summary line is the only stdout difference from Phase 2.1 (it gains a
   strategy annotation).
4. `rag index --chunking-strategy structural` and `--chunking-strategy
   semantic` both exit 0 and produce a retrievable index (`rag retrieve`
   against it returns results).
5. `manifest.json` records `chunking_strategy` and `chunking_params` for
   every strategy, including the default.
6. A documented example (fixture + walkthrough) shows fixed-character
   chunking splitting necessary context that structural chunking keeps
   intact, reproduced by building two indices and comparing
   `rag retrieve`/`rag diagnose` output — with no changes to `failure.py`.
7. `uv run pytest --tb=short -q`: all passed, no regressions.
8. `docs/phases/README.md`, `docs/roadmap.md`, `README.md`,
   `docs/file-structure.md`, `corpus/gaps.md` §1.1, and EN/ZH learning
   materials all reflect Phase 2.2 completion with no stale Phase 2.1-only
   wording.

---

## Learning Notes

- Fixed-size chunking is mechanically simple but structurally blind: it has
  no idea where a sentence, list, or section ends. Structural chunking shows
  that respecting document structure is often a bigger lever than tuning
  `chunk_size`/`chunk_overlap`.
- A heading with no body is a classic edge case that naive blank-line
  splitting gets wrong — Design Decision 3 exists specifically because of it.
  It's a good worked example of why "split on blank lines" alone is not the
  whole algorithm.
- Semantic chunking trades indexing speed (one embedding call per sentence,
  not per chunk) for boundary quality. A learner can run the same corpus
  through `fixed_character`, `structural`, and `semantic` and compare
  indexing latency directly.
- The chunking-boundary failure example demonstrates the same lesson as the
  Phase 1.8 failure lab: a retrieval failure is sometimes not a retrieval
  problem at all — it's an indexing-time decision made before retrieval ever
  runs.
- Comparing chunking strategies needed no new comparison machinery —
  `rag eval` and `rag diagnose` already work on any index directory. The
  lesson is architectural: keeping retrieval-time tools decoupled from
  indexing-time decisions means a new indexing strategy is "free" to evaluate
  with tools that already exist.
