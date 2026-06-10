# Phase 1.5 Spec: Retrieval Mechanics

**Status:** Complete — closed by Codex on 2026-06-11
**Authors:** Claude Code + owner decisions
**Based on:** `docs/proposal.md`, `docs/roadmap.md`, `docs/architecture.md`
**Taskboard:** `docs/phases/phase-1.5-taskboard.md`
**Date:** 2026-06-10

---

## Goal

Add BM25 keyword retrieval and hybrid (dense + BM25) retrieval to the existing RAG
pipeline so that all three retrieval strategies can be compared side-by-side using the
Phase 1.6 evaluation harness. This makes the mechanics of keyword vs. semantic vs.
combined retrieval directly visible and measurable.

---

## Scope

### In scope

- `BM25Retriever` class backed by `rank_bm25` (pure-Python, no heavy deps)
- Visible `_tokenize(text)` helper (whitespace + lowercase, no external tokenizer)
- `reciprocal_rank_fusion(results_lists, top_k, k=60)` standalone function
- `retrieve_hybrid(query, index, embedder, top_k)` combining dense + BM25 via RRF
- `--retriever {dense,bm25,hybrid}` flag for `rag retrieve` (default: `dense`)
- `--retriever {dense,bm25,hybrid}` flag for `rag eval` (default: `dense`)
- `retriever: str` field added to `EvalReport` (records which strategy was used)
- `retriever: str = "dense"` parameter added to `run_retrieval_eval()`
- Unit tests for BM25 retriever with known inputs and outputs
- Unit tests for RRF function with known rank lists
- CLI tests for `--retriever` flag on both `rag retrieve` and `rag eval`

### Out of scope for Phase 1.5

- Reranking (cross-encoder or otherwise) — deferred
- Configurable chunking strategies (heading-aware, sentence-aware) — deferred
- Comparison report across configurations (pretty table of all three metrics) — Phase 1.7
- BM25 index persistence to disk — deferred; BM25 index is built in memory at query time
- Tunable RRF k parameter via CLI — deferred
- Metadata filtering — deferred

---

## New Dependency

Add `rank_bm25` to `pyproject.toml` runtime dependencies.

`rank_bm25` is a pure-Python BM25 implementation. No native extensions, no GPU, no
model downloads.

---

## Module Layout

Two new modules under `tiny_rag_lab/`:

```
tiny_rag_lab/bm25.py     — BM25Retriever class
tiny_rag_lab/hybrid.py   — RRF function + retrieve_hybrid()
```

No changes to `tiny_rag_lab/models.py`, `tiny_rag_lab/retrieval.py`,
`tiny_rag_lab/index_loader.py`, or `tiny_rag_lab/index_writer.py`.

---

## BM25 Retriever

File: `tiny_rag_lab/bm25.py`

```python
from tiny_rag_lab.models import Chunk, RetrievalResult

def _tokenize(text: str) -> list[str]:
    return text.lower().split()

class BM25Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        ...  # builds rank_bm25.BM25Okapi over tokenized chunk texts

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        ...  # returns rank-ordered RetrievalResult list
```

- `_tokenize` is a module-level function so it stays visible and testable.
- `BM25Okapi` from `rank_bm25` does the scoring; `BM25Retriever` wraps it to return
  the same `RetrievalResult` dataclass the rest of the pipeline uses.
- `score` field in each `RetrievalResult` is the raw BM25 score (not normalized).
- Empty corpus → `retrieve()` returns `[]`.
- Empty query → `retrieve()` returns `[]`.

**Known tokenizer limitation:** `_tokenize` splits on whitespace after lowercasing, so
punctuation stays attached (e.g. `"watson?"` ≠ `"watson"`). This directly affects BM25
hit rate on the watsonxDocsQA eval set. It is acceptable for Phase 1.5 because the goal
is to expose the BM25 mechanic, not maximize score. Stripping punctuation or using a
proper tokenizer is a future experiment.

---

## Hybrid Retrieval

File: `tiny_rag_lab/hybrid.py`

```python
from tiny_rag_lab.bm25 import BM25Retriever
from tiny_rag_lab.embeddings import Embedder
from tiny_rag_lab.index_loader import LoadedIndex
from tiny_rag_lab.models import RetrievalResult

def reciprocal_rank_fusion(
    results_lists: list[list[RetrievalResult]],
    top_k: int,
    k: int = 60,
) -> list[RetrievalResult]:
    ...

def retrieve_hybrid(
    query: str,
    index: LoadedIndex,
    embedder: Embedder,
    top_k: int = 5,
    bm25_retriever: BM25Retriever | None = None,
) -> list[RetrievalResult]:
    ...
```

**Reciprocal Rank Fusion formula:**

For each unique chunk appearing in any result list, its fused score is:

```
rrf_score(chunk) = sum(1 / (k + rank_i) for each list i where chunk appears)
```

Chunks are then sorted by fused score descending, and the top `top_k` are returned.

`rank_i` is 1-indexed (as in `RetrievalResult.rank`).

The `score` field in returned `RetrievalResult` objects is the fused RRF score.
`rank` is re-assigned 1-indexed in the fused result order.
`chunk` is taken from the first occurrence across lists (the objects are identical).

**Tie-breaking:** when two chunks have equal fused score, Python's stable `sorted` means
the chunk that appeared earlier in the first results list wins. Dense results are always
passed first, so ties are broken by dense rank order.

`retrieve_hybrid`:
1. If `bm25_retriever` is `None`, builds `BM25Retriever(index.chunks)` internally
2. Runs `retrieve(query, index, embedder, top_k=top_k)` (dense, from `retrieval.py`)
3. Calls `bm25_retriever.retrieve(query, top_k=top_k)`
4. Calls `reciprocal_rank_fusion([dense_results, bm25_results], top_k=top_k)`

The `bm25_retriever` parameter exists so callers that run many queries (e.g.
`run_retrieval_eval`) can build the BM25 index once and inject it, avoiding one rebuild
per query. Single-query CLI callers can omit it and accept the build cost.

---

## Changes to Existing Modules

### `tiny_rag_lab/eval.py`

Add `retriever: str = "dense"` field to `EvalReport`:

```python
@dataclass
class EvalReport:
    n_questions: int
    top_k: int
    retriever: str = "dense"  # NEW: "dense" | "bm25" | "hybrid"
    hit_rate: float = 0.0
    mrr: float = 0.0
    mean_context_precision: float = 0.0
    mean_context_recall: float = 0.0
    per_question: list[EvalResult] = field(default_factory=list)
```

`retriever` is a run-parameter (like `top_k`), so it sits alongside `top_k` before the
result aggregates. It must have a default of `"dense"` so that existing construction
calls such as `EvalReport(n_questions=0, top_k=top_k)` continue to work without change.

Add `retriever: str = "dense"` parameter to `run_retrieval_eval()` and make `embedder`
nullable:

```python
def run_retrieval_eval(
    samples: list[EvalSample],
    index: LoadedIndex,
    embedder: Embedder | None,
    top_k: int,
    retriever: str = "dense",
) -> EvalReport:
    ...
```

**Validation contract:** if `retriever` is `"dense"` or `"hybrid"` and `embedder` is
`None`, the function must raise `ValueError` immediately before entering the per-sample
loop. The `"bm25"` path never calls `embedder.embed()` and must work when `embedder` is
`None`.

The function constructs any strategy-specific objects **once before the per-sample loop**,
then reuses them across all samples. This is the correct place to learn that BM25 builds
an inverted index once over the corpus, not once per query:

- `"dense"`: embed each query with `embedder`, call `retrieve_by_vector(query_vec, index, top_k)`
- `"bm25"`: build `BM25Retriever(index.chunks)` once before the loop; call `.retrieve(query, top_k)` per sample
- `"hybrid"`: build `BM25Retriever(index.chunks)` once before the loop; call `retrieve_hybrid(query, index, embedder, top_k, bm25_retriever=bm25_retriever)` per sample — the pre-built instance is injected so `retrieve_hybrid` does not rebuild it

`format_eval_report()` adds one line showing the retriever name in the header.

### `tiny_rag_lab/cli.py`

Add `--retriever` flag to `rag retrieve`:

```
--retriever {dense,bm25,hybrid}   retrieval strategy (default: dense)
```

Add `--retriever` flag to `rag eval`:

```
--retriever {dense,bm25,hybrid}   retrieval strategy (default: dense)
```

**Embedder loading:** `_make_embedder()` loads `sentence-transformers/all-MiniLM-L6-v2`
(~100 MB, slow on first run). For the pure BM25 path no embedder is needed. Both
`cmd_retrieve` and `cmd_eval` must skip `_make_embedder()` when `args.retriever == "bm25"`.
Pass `embedder=None` or simply omit it; the BM25 path never calls `embedder.embed()`.

No changes to `rag index` or `rag ask`.

---

## CLI Examples

```bash
# Dense (existing behavior, now explicit)
rag retrieve "what is watson assistant?" --retriever dense

# BM25 keyword retrieval
rag retrieve "what is watson assistant?" --retriever bm25

# Hybrid (dense + BM25 via RRF)
rag retrieve "what is watson assistant?" --retriever hybrid

# Evaluate with each strategy
rag eval --qa-file corpus/watsonx-docsqa/qa.jsonl --retriever dense
rag eval --qa-file corpus/watsonx-docsqa/qa.jsonl --retriever bm25
rag eval --qa-file corpus/watsonx-docsqa/qa.jsonl --retriever hybrid
```

Example output for `rag eval --retriever hybrid`:

```
Evaluation report  (n=847, top_k=5, retriever=hybrid)
──────────────────────────────────────────────────────
Hit Rate @ 5      :  0.751
MRR               :  0.603
Context Precision :  0.328
Context Recall    :  0.667
```

---

## Required Tests

**BM25 unit tests** (`tests/test_bm25.py`):

- `_tokenize("Hello World")` → `["hello", "world"]`
- A chunk containing a unique term ranks first when that term is queried
- `top_k` clipping: requesting more than corpus size returns all chunks
- Empty corpus: `retrieve()` returns `[]`
- Empty query string: `retrieve()` returns `[]`

**Hybrid / RRF unit tests** (`tests/test_hybrid.py`):

- `reciprocal_rank_fusion` with two lists: chunk in rank-1 of both lists scores higher
  than a chunk in rank-2 of one list only
- `reciprocal_rank_fusion` with a chunk appearing only in one list still appears in output
- `retrieve_hybrid` returns exactly `top_k` results when corpus is large enough
- Returned `RetrievalResult.rank` values are 1-indexed and contiguous

**CLI tests** (extend `tests/test_cmd_index_retrieve.py` or new `tests/test_cmd_retriever.py`):

- `rag retrieve --help` shows `--retriever` flag
- `rag retrieve <query> --retriever bm25` returns results (fake embedder not needed for bm25)
- `rag retrieve <query> --retriever hybrid` returns results
- Invalid `--retriever foo` exits non-zero with error message

**Eval CLI tests** (extend `tests/test_cmd_eval.py`):

- `rag eval --help` shows `--retriever` flag
- `rag eval --qa-file ... --retriever bm25` prints all four metric labels
- `rag eval --qa-file ... --retriever hybrid` prints all four metric labels
- Output contains `retriever=bm25` or `retriever=hybrid` in the report header

---

## Acceptance Criteria

Phase 1.5 is complete when:

1. `rag retrieve "question" --retriever bm25` returns ranked results without an embedder.
2. `rag retrieve "question" --retriever hybrid` returns ranked results combining dense
   and BM25 via RRF.
3. `rag eval --qa-file qa.jsonl --retriever {dense,bm25,hybrid}` each print a valid
   four-metric report with the retriever name in the header.
4. All new and existing tests pass without network access, model downloads, or API
   credentials.
5. `EvalReport` includes a `retriever` field.
