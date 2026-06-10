# Phase 1.6 Spec: Evaluation Harness

**Status:** Review-ready
**Authors:** Claude Code + owner decisions
**Based on:** `docs/roadmap.md`, `docs/phases/phase-1-naive-classic-rag.md`
**Taskboard:** `docs/phases/phase-1.6-taskboard.md`
**Date:** 2026-06-09

---

## Goal

Add a `rag eval` command that runs the existing retriever against the prepared
`qa.jsonl` eval set and reports objective retrieval quality metrics. This
establishes a measurable baseline before any retrieval changes are made in
Phase 1.5.

---

## Scope

### In scope

- `EvalSample`, `EvalResult`, `EvalReport` dataclasses
- `load_eval_samples(path) -> list[EvalSample]` loader
- Four deterministic metric functions: `hit_at_k`, `reciprocal_rank`,
  `context_precision_at_k`, `context_recall_at_k`
- `run_retrieval_eval(samples, index, embedder, top_k) -> EvalReport` runner
- `format_eval_report(report) -> str` plain-text formatter
- `rag eval --qa-file PATH --index-dir PATH --top-k INT` CLI command
- Unit tests for all metric functions with known inputs and outputs
- Integration test with a small fixture eval dataset
- CLI test using fake embedder backend — no model downloads or API credentials

### Out of scope for Phase 1.6

- Answer quality metrics (faithfulness, answer correctness, answer relevance) —
  require LLM-as-judge, non-deterministic, deferred
- Comparison reports across retriever or chunking configurations — deferred to
  Phase 1.5 (requires multiple retriever implementations)
- BM25 or hybrid retrieval evaluation — Phase 1.5
- Persisting eval run artifacts to disk — Phase 1.7

---

## Data Contracts

```python
@dataclass
class EvalSample:
    question_id: str
    question: str
    answer: str             # reference answer; kept for future answer metrics
    gold_doc_ids: list[str] # corpus-relative doc_ids matching Document.doc_id
```

```python
@dataclass
class EvalResult:
    question_id: str
    question: str
    gold_doc_ids: list[str]
    retrieved_doc_ids: list[str]  # doc_id per retrieved chunk, rank-ordered
    hit: bool                     # True if any retrieved doc is in gold set
    reciprocal_rank: float        # 1/rank_of_first_hit; 0.0 if no hit
    context_precision: float      # fraction of retrieved docs that are gold
    context_recall: float         # fraction of gold docs covered by retrieved
```

```python
@dataclass
class EvalReport:
    n_questions: int
    top_k: int
    hit_rate: float               # mean hit across all questions
    mrr: float                    # mean reciprocal rank
    mean_context_precision: float
    mean_context_recall: float
    per_question: list[EvalResult]
```

All dataclasses live in `tiny_rag_lab/eval.py`.

---

## Metric Definitions

Given `retrieved_doc_ids: list[str]` (top-k slice, rank-ordered) and
`gold_doc_ids: list[str]`:

**hit_at_k**
```
any(d in gold_doc_ids for d in retrieved_doc_ids)
```
Returns `False` for empty retrieval.

**reciprocal_rank**
```
1 / i  where i is the 1-based position of the first hit
0.0    if no hit
```

**context_precision_at_k**
```
sum(1 for d in retrieved_doc_ids if d in gold_doc_ids) / len(retrieved_doc_ids)
0.0 if retrieved_doc_ids is empty
```

**context_recall_at_k**
```
len(set(retrieved_doc_ids) & set(gold_doc_ids)) / len(gold_doc_ids)
0.0 if gold_doc_ids is empty
```

The `k` parameter is implicit — callers pass only the already-sliced top-k
list. Functions operate on whatever list they receive.

---

## Eval Dataset Format

`qa.jsonl` was prepared in `P1-T02` by
`scripts/prepare_watsonx_docsqa.py`. Each line is a JSON object:

```json
{
  "question_id": "...",
  "question": "...",
  "answer": "...",
  "gold_doc_ids": ["docs/example.md"]
}
```

`gold_doc_ids` are corpus-relative paths identical to `Document.doc_id` values
in the index — direct string match, no translation needed.

`load_eval_samples` must skip rows where `question` is empty or `gold_doc_ids`
is empty. It must not raise on malformed rows.

---

## Runner

```python
def run_retrieval_eval(
    samples: list[EvalSample],
    index: LoadedIndex,
    embedder: Embedder,
    top_k: int,
) -> EvalReport:
    ...
```

For each sample:
1. Embed the question using `embedder.embed([sample.question])[0]`
2. Retrieve via `retrieve_by_vector(query_vec, index, top_k=top_k)`
3. Extract `chunk.doc_id` for each `RetrievalResult` to form `retrieved_doc_ids`
4. Compute the four metrics against `sample.gold_doc_ids`
5. Build an `EvalResult`

Aggregate means over all per-question results to build `EvalReport`.

---

## CLI

```bash
rag eval --qa-file corpus/watsonx-docsqa/qa.jsonl \
         --index-dir .tiny-rag/index \
         --top-k 5
```

Flags:

- `--qa-file PATH` — required; path to `qa.jsonl`
- `--index-dir PATH` — default `.tiny-rag/index`
- `--top-k INT` — default `5`

Example output:

```
Evaluation report  (n=847, top_k=5)
────────────────────────────────────
Hit Rate @ 5      :  0.723
MRR               :  0.581
Context Precision :  0.312
Context Recall    :  0.651
```

The `_make_embedder` factory from `cli.py` is reused unchanged, so tests can
patch it with `FakeEmbedder` without touching the CLI interface.

---

## Required Tests

**Metric unit tests** (`tests/test_eval_metrics.py`):

- `hit_at_k(["a", "b"], ["b"])` → `True`
- `hit_at_k(["a", "b"], ["c"])` → `False`
- `reciprocal_rank(["a", "b", "c"], ["b"])` → `0.5`
- `reciprocal_rank(["a", "b"], ["c"])` → `0.0`
- `context_precision_at_k(["a", "b"], ["a"])` → `0.5`
- `context_precision_at_k([], ["a"])` → `0.0`
- `context_recall_at_k(["a", "b"], ["a", "c"])` → `0.5`
- `context_recall_at_k(["a"], [])` → `0.0`
- `format_eval_report` output contains all four metric labels and values

**Loader and runner tests** (`tests/test_eval_runner.py`):

- `load_eval_samples` loads a 3-record fixture without error
- Rows with empty `question` or `gold_doc_ids` are skipped
- `run_retrieval_eval` returns `EvalReport` with correct `n_questions`
- Mean metrics are the correct arithmetic means of per-question values
- FakeEmbedder used — no model download required

**CLI test** (`tests/test_cmd_eval.py`):

- `rag eval --help` exits 0
- `--qa-file` is required (missing it exits non-zero)
- End-to-end test with fake embedder: output contains all four metric labels

**Test fixture**: `tests/fixtures/eval/qa.jsonl` — 3 hand-written records
referencing doc_ids from the existing `tests/fixtures/corpus/`.

---

## Acceptance Criteria

Phase 1.6 is complete when:

1. `rag eval --qa-file qa.jsonl --index-dir .tiny-rag/index` prints a
   four-metric report (hit rate, MRR, context precision, context recall).
2. All metric functions have unit tests with known inputs and outputs.
3. A fixture-based integration test covers load → index → eval → report.
4. Tests pass without network access, model downloads, or API credentials.
