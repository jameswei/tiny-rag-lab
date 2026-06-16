# Retrieval Mechanics — Dense, BM25, and Hybrid Search

Phase 1 delivered one retriever: dense cosine-similarity search. Phase 1.5 adds
two more — keyword-based BM25 and a hybrid that fuses dense + BM25 via
Reciprocal Rank Fusion. Two new modules do this work: `bm25.py` (keyword
retrieval) and `hybrid.py` (fusion logic).

---

## Why Three Retrievers?

Dense retrieval finds semantically similar chunks: "automobile maintenance" and
"car repair" score high even though they share no words. But it sometimes misses
exact matches: a search for "API key rotation" might rank a document about "key
management" above the literal API-key-rotation page.

BM25 is the opposite. It cares about exact word overlap — if the query word
"rotation" appears many times in a document, BM25 boosts that document. But it
has no understanding of synonyms or paraphrases.

Hybrid retrieval combines both: dense catches the semantics, BM25 catches the
exact terms, and Reciprocal Rank Fusion merges their ranked lists into one.

---

## BM25 Keyword Retrieval (`bm25.py`)

### What BM25 does

BM25 (Best Matching 25) is a ranking function from the TF-IDF family. For each
chunk, it computes a score based on:

- **Term frequency (TF)**: how often each query word appears in the chunk.
  More occurrences → higher score, but with diminishing returns.
- **Inverse document frequency (IDF)**: how rare each query word is across the
  entire corpus. Words that appear in many chunks (like "the" or "and") get
  down-weighted.

The implementation uses `rank_bm25`, a pure-Python BM25 library with no native
extensions, no GPU, and no model downloads.

### Tokenization: `_tokenize`

```python
def _tokenize(text: str) -> list[str]:
    return text.lower().split()
```

This is the most important single line in the module. It splits on whitespace
after lowercasing — no stemming, no stopword removal, no punctuation stripping.
This means:

- `"Watson?"` and `"watson"` are **different tokens** — the question mark stays
  attached. A query for "watson" will not match a chunk that says "Watson?".
- `"API"` and `"api"` are the **same token** — case is normalized away.
- Chinese/Japanese text without spaces between words is **not split** — the
  entire sentence becomes one token.

This tokenizer is deliberately simple. The goal is to expose the BM25 mechanic
clearly, not to maximize retrieval scores. A production system would strip
punctuation or use a proper tokenizer; that is a future experiment.

### `BM25Retriever` class

```python
class BM25Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        # Tokenizes every chunk and builds a rank_bm25.BM25Okapi index
        # If all chunks tokenize to empty, self._bm25 stays None

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        # Returns rank-ordered results with raw BM25 score in the .score field
```

Key behaviors:

- **Returns `RetrievalResult` objects** — the same dataclass the rest of the
  pipeline uses. `score` is the raw BM25 score (not normalized to [-1,1] like
  cosine similarity).
- **Empty corpus → `[]`**: if `_bm25` is `None`, `retrieve()` returns an empty
  list.
- **Empty query → `[]`**: a blank or whitespace-only query returns no results.
- **`top_k < 0` → `ValueError`**: consistent with the dense retriever contract.

### Building the index once, reusing it

`BM25Retriever.__init__` builds an inverted index over all chunks — it
tokenizes every chunk and pre-computes document statistics. This is O(N) in
corpus size and should be done once, not per query. Callers that run many
queries (like `rag eval`) build one `BM25Retriever` and reuse it. The CLI
builds a fresh one per command because single-query overhead is low.

### When BM25 wins over dense

BM25 excels when the query contains precise, distinctive terms:

| Query | Dense might... | BM25 tends to... |
|---|---|---|
| "`POST /v2/accounts`" | Retrieve general API docs | Find the exact endpoint docs |
| "`ValueError: chunk_size must be positive`" | Retrieve generic error handling | Find the specific error message |
| "`ibm cloud api key rotation`" | Spread across topics | Focus on "rotation" + "key" |

The Phase 1.6 evaluation harness can confirm these intuitions with numbers —
run `rag eval` with `--retriever dense` and `--retriever bm25` on the same QA
set and compare hit rates.

---

## Reciprocal Rank Fusion (`hybrid.py`)

### The RRF formula

```python
def reciprocal_rank_fusion(
    results_lists: list[list[RetrievalResult]],
    top_k: int,
    k: int = 60,
) -> list[RetrievalResult]:
```

For each unique chunk that appears in any result list, the fused score is:

```
rrf_score(chunk) = sum( 1 / (k + rank_i)  for each list i where chunk appears )
```

`rank_i` is 1-indexed, matching `RetrievalResult.rank`. The constant `k = 60`
is a smoothing parameter from the original RRF paper — it prevents rank-1 in one
list from completely dominating rank-2 in another.

**Example.** Suppose chunk A is rank 1 in dense and rank 3 in BM25; chunk B is
rank 2 in dense only:

```
rrf(A) = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
rrf(B) = 1/(60+2)               = 0.01613
```

Chunk A wins because it appears in both lists, even though chunk B ranks higher
in the dense list than A does in the BM25 list.

### How fusion works step by step

1. **Accumulate scores.** Iterate over every result list. For each
   `RetrievalResult`, add `1/(k + result.rank)` to the chunk's running total.
2. **Track first occurrence.** The `chunk` object is taken from the first list
   where the chunk appears — all lists hold identical `Chunk` references from
   the same index, so this is cosmetic.
3. **Sort descending.** Chunks are sorted by fused score, highest first.
4. **Re-rank.** `rank` is re-assigned 1-indexed in the fused order.

### Tie-breaking

When two chunks have equal fused score, Python's stable `sorted` preserves
their relative order from the first results list. Since dense results are always
passed as the first list, ties are broken in favor of the chunk with the higher
dense rank.

### Why RRF instead of score normalization

Dense cosine similarity and BM25 scores live in different ranges. Dense ranges
from roughly -1 to 1 (usually 0.0 to 0.7 in practice); BM25 is unbounded and
can be 0 to 20+. Normalizing scores to a common scale would require assumptions
about their distribution. RRF avoids normalization entirely — it only uses
ranks, which are always comparable.

---

## Hybrid Retrieval: `retrieve_hybrid`

```python
def retrieve_hybrid(
    query: str,
    index: LoadedIndex,
    embedder: Embedder,
    top_k: int = 5,
    bm25_retriever: BM25Retriever | None = None,
) -> list[RetrievalResult]:
```

This function:

1. **Builds BM25** (if `bm25_retriever` is `None`): `BM25Retriever(index.chunks)`.
   Callers running many queries should build the retriever once and inject it
   via the `bm25_retriever` parameter to avoid rebuilding per query.
2. **Runs dense retrieval**: calls `retrieve(query, index, embedder, top_k=top_k)`
   from `retrieval.py`.
3. **Runs BM25 retrieval**: calls `bm25_retriever.retrieve(query, top_k=top_k)`.
4. **Fuses results**: calls `reciprocal_rank_fusion([dense_results, bm25_results], top_k=top_k)`.
5. **Returns** exactly `top_k` results with fused RRF scores and re-assigned
   1-indexed ranks.

The returned `RetrievalResult.score` is the fused RRF score — a small positive
number, not a cosine similarity.

---

## CLI Usage

### `rag retrieve` with retriever selection

```bash
# Dense (default)
rag retrieve "what is watson assistant?" --retriever dense

# BM25 keyword retrieval — no embedder needed
rag retrieve "what is watson assistant?" --retriever bm25

# Hybrid: dense + BM25 via RRF
rag retrieve "what is watson assistant?" --retriever hybrid
```

When `--retriever bm25` is used, the CLI skips loading the embedding model
entirely — BM25 does not use embeddings.

### `rag eval` with retriever selection

```bash
rag eval --qa-file qa.jsonl --retriever dense   # Phase 1 baseline
rag eval --qa-file qa.jsonl --retriever bm25    # keyword-only
rag eval --qa-file qa.jsonl --retriever hybrid  # combined
```

The evaluation report header includes the retriever name:

```
Evaluation report  (n=847, top_k=5, retriever=hybrid)
──────────────────────────────────────────────────────
Hit Rate @ 5      :  0.751
MRR               :  0.603
Context Precision :  0.328
Context Recall    :  0.667
```

### `rag diagnose` with per-case retrievers

The failure lab (`rag diagnose`) does not take a `--retriever` flag. Each
failure case in `cases.jsonl` defines its own `baseline` and `intervention`
retriever configurations, so dense, BM25, and hybrid can be compared
case-by-case within a single diagnosis run.

---

## Source Module Map

| Module | What it provides |
|---|---|
| `tiny_rag_lab/bm25.py` | `_tokenize()`, `BM25Retriever` class |
| `tiny_rag_lab/hybrid.py` | `reciprocal_rank_fusion()`, `retrieve_hybrid()` |
| `tiny_rag_lab/retrieval.py` | `retrieve()`, `retrieve_by_vector()` — dense path used by hybrid |

No changes were needed to `models.py`, `index_loader.py`, or `index_writer.py`
— the `RetrievalResult` and `Chunk` dataclasses already support all three
retrievers without modification.

---

## What to Inspect Next

After reading this document, the most useful experiments are:

1. **Compare retrievers on a known query.** Run `rag retrieve "exact term"` with
   `--retriever dense`, then `--retriever bm25`. Look at which chunks appear in
   both lists and which are unique to one retriever.
2. **Check the eval numbers.** Run `rag eval` with all three retrievers on the
   same QA set. Look at which questions each retriever gets right — it tells you
   which types of queries benefit from keyword vs. semantic search.
3. **Trace a hybrid run.** Add `--retriever hybrid` to `rag retrieve` and
   inspect the trace output. The dense and BM25 scores are visible separately
   in the trace before RRF fuses them.

---

## Related Docs

- [Retrieval and Generation](retrieval-and-generation.md) — the dense cosine
  retrieval path explained in detail.
- [Evaluating Retrieval](evaluating-retrieval.md) — how to measure which
  retriever works best.