# Structural And Semantic Chunking

Phase 2.2 adds two chunking strategies alongside the existing fixed-character
baseline. Chunking strategy is an **indexing-time decision** — you choose it
when running `rag index`, and every downstream command (`retrieve`, `ask`,
`eval`, `diagnose`) works with whatever chunks are already on disk.

---

## Why Chunking Strategy Matters

The slice invariant guarantees that every chunk's text is a correct slice of
the document's `normalized_text`. But being a correct slice doesn't mean the
slice is a *useful* one.

A fixed-size character window can cut in the middle of a sentence:

```
Chunk A: "Before bulk import, set timeout_ms to 150"    ← sentence incomplete
Chunk B: "00 and set retry_mode to manual."             ← depends on A
```

One chunk may still match a query about `timeout_ms`, but neither chunk can
support the full answer on its own because neither carries the complete
instruction. Structural and semantic chunking solve this by placing boundaries
at meaningful document positions instead of at arbitrary character offsets.

---

## Strategy 1: Fixed-Character (the baseline)

```
chunk_document(doc, chunk_size=800, chunk_overlap=120)
```

Slides a window of `chunk_size` characters forward by `chunk_size -
chunk_overlap` characters at each step. Consecutive chunks share
`chunk_overlap` characters.

**When to use**: quick experiments, verifying the pipeline end to end, or
when documents have no structure (plain prose, unformatted logs).

---

## Strategy 2: Structural

```
chunk_document_structural(doc, chunk_size=800, chunk_overlap=120)
```

Three-tier packing (Design Decision 2 in the Phase 2.2 spec):

1. **Tier 1 — whole blocks.** Split `normalized_text` at blank-line
   boundaries (`_split_blocks`). A block that is a single ATX heading line
   (`# …`) is merged with the following block so a heading never stands alone.
   Pack whole blocks greedily until adding the next block would exceed
   `chunk_size`.

2. **Tier 2 — sentences.** If a single block is wider than `chunk_size` on
   its own, split it into sentences (`_split_sentences`) and pack those
   sentences the same way. This is the "sentence boundaries where practical"
   tier — it only activates when block packing alone can't fit the budget.

3. **Tier 3 — character window.** If a single sentence is wider than
   `chunk_size` on its own (run-on sentence or a code fence with no
   punctuation), fall back to `_chunk_oversized_span` — the same sliding
   window as `chunk_document`, but restricted to that sentence's range.
   This is the **only** tier that consumes `chunk_overlap`.

Tier 1 and tier 2 chunks are **contiguous** (`chunk[i].char_end ==
chunk[i+1].char_start`) — no overlap, no gaps. Tier 3 chunks overlap as
with fixed-character chunking.

**When to use**: Markdown-formatted corpora (documentation, wikis, runbooks)
where heading–body and paragraph boundaries are meaningful.

---

## Strategy 3: Semantic (experimental)

```
chunk_document_semantic(doc, embedder, chunk_size=800, chunk_overlap=120,
                        similarity_threshold=0.5)
```

Steps:

1. Split `normalized_text` into sentences (`_split_sentences`).
2. Embed **all sentences in one batch call** — `embedder.embed([s1, s2, …])`.
   One embedder call, not one call per sentence. This is still an extra
   sentence-level embedding step before the normal chunk-level embedding that
   writes the index.
3. Walk sentences in order. Close the current chunk and start a new one when:
   - adding the next sentence would exceed `chunk_size`, **or**
   - cosine similarity between the current sentence's embedding and the
     previous sentence's embedding drops below `similarity_threshold`.

Cosine similarity is a plain dot product because `Embedder` vectors are
L2-normalized — no extra normalization needed.

A single sentence wider than `chunk_size` falls back to
`_chunk_oversized_span` (tier 3).

**When to use**: narrative text without explicit structure where topic shifts
are detectable through embedding similarity but not through blank lines.

**Key cost**: semantic chunking must embed every sentence — a finer-grained,
extra embedding pass beyond the existing chunk-level embedding step. It is
documented as experimental and opt-in.

---

## The Shared Private Helpers

The chunkers reuse a small set of private helpers for correctness and code
reuse. The structural and semantic strategies add the boundary-aware helpers:

| Helper | Purpose |
|---|---|
| `_validate_chunk_params` | Raises `ValueError` for invalid `chunk_size` / `chunk_overlap` — same text as `chunk_document` always raised |
| `_chunk_metadata(doc)` | Returns `{title, path, format, raw_hash}` metadata dict |
| `_split_sentences(text)` | Regex sentence splitter; returns `[(start, end), …]` spans that cover the whole string with no gaps |
| `_split_blocks(text)` | Blank-line splitter with ATX heading merge; returns `[(start, end), …]` spans |
| `_chunk_oversized_span(doc, start, end, chunk_size, chunk_overlap)` | Tier-3 fallback: slides a character window over `text[start:end]` |

---

## Strategy Dispatch

```python
chunk_document_with_strategy(
    doc,
    strategy="fixed_character",  # or "structural" or "semantic"
    chunk_size=800,
    chunk_overlap=120,
    embedder=None,                # required when strategy="semantic"
    similarity_threshold=0.5,
)
```

Raises `ValueError` for an unknown strategy name, or if `strategy="semantic"`
and `embedder is None`.

`chunk_documents_with_strategy(docs, ...)` applies `chunk_document_with_strategy`
to each document in order.

---

## Manifest Recording

The chunking strategy is recorded **once** in the index manifest — not on
every chunk. This keeps the `Chunk` dataclass unchanged and means retrieval,
trace formatting, citations, and prompting require zero changes:

```json
{
  "chunking_strategy": "structural",
  "chunking_params": {}
}
```

For `semantic`:

```json
{
  "chunking_strategy": "semantic",
  "chunking_params": {"similarity_threshold": 0.5}
}
```

`chunking_params` is always serialized (never absent), so consumers never
need to branch on its presence.

---

## CLI

```bash
# default: fixed-character (unchanged from all prior phases)
rag index --corpus PATH --index-dir .tiny-rag/index

# structural chunking
rag index --corpus PATH --index-dir .tiny-rag/structural \
    --chunking-strategy structural

# semantic chunking with custom threshold
rag index --corpus PATH --index-dir .tiny-rag/semantic \
    --chunking-strategy semantic \
    --semantic-similarity-threshold 0.7
```

`--chunk-overlap` still applies to the tier-3 character-window fallback for
`structural` and `semantic`; most chunks produced by those strategies have no
overlap at all.

---

## Before/After Comparison

Phase 2.2 ships a concrete demonstration in
`tests/fixtures/chunking_corpus/` and a walkthrough in
`docs/phases/phase-2.2-chunking-comparison.md`.

The key result: at `chunk_size=75`, fixed-character chunking splits the gold
instruction across two chunks, so BM25 retrieves the distractor document
first. Structural chunking keeps the sentence whole, and `rag diagnose` shows
the failure stops reproducing:

| Index strategy | `rag diagnose` result |
|---|---|
| `fixed_character` | `Confirmed: 1` — failure reproduces |
| `structural` | `Confirmed: 0` — failure stops |

---

## What This Teaches

- Chunk boundaries are a design choice, not just an implementation detail.
- The same `Chunk`, `RetrievalResult`, `AskTrace`, and `rag diagnose`
  infrastructure is reused without change — chunking is purely an
  indexing-time decision.
- Comparing strategies is a usage pattern: build two indices with different
  `--chunking-strategy`, run the same queries against each, and compare the
  reports.
- Semantic chunking is not always better — it adds embedding cost at index
  time and depends on the embedding model's ability to detect topic shifts.
  Always measure before adopting.
