# Evaluating Retrieval — Measuring Before You Optimize

Phase 1.6 adds a measurement layer to the RAG pipeline. Before you tune chunk
size, swap embedding models, or add BM25, you need a number to improve. This
document explains what that number is, how it's computed, and what it tells you.

---

## Why Measure First?

After Phase 1, you can ask questions and get answers. But you have no way to
know whether the retriever found the right chunks. Maybe it always retrieves
the correct document — or maybe it misses 40% of the time and the LLM is
covering up for it with hallucinations.

The `qa.jsonl` file was prepared in Phase 1 (`P1-T02`) specifically for this
moment. It contains questions whose correct source documents are known in
advance. This is the evaluation set: a ground truth you can compare retrieved
results against.

---

## The Evaluation Dataset (`qa.jsonl`)

Each line in `qa.jsonl` is a JSON object:

```json
{
  "question_id": "q001",
  "question": "What topics does the sample document cover?",
  "answer": "It covers several topics useful for retrieval testing.",
  "gold_doc_ids": ["with_h1.md"]
}
```

The critical field is `gold_doc_ids`. These are the corpus-relative paths
(identical to `Document.doc_id` values in the index) of the documents that
actually contain the answer. When the retriever is working correctly, at least
one retrieved chunk should come from one of these documents.

This direct matching — `chunk.doc_id` against `gold_doc_ids` — is what makes
the metrics deterministic. No LLM judgment required.

---

## Data Contracts (`eval.py`)

Three dataclasses carry evaluation state through the pipeline:

```python
@dataclass
class EvalSample:
    question_id: str
    question: str
    answer: str               # kept for future answer-quality metrics
    gold_doc_ids: list[str]   # what the retriever should find
```

```python
@dataclass
class EvalResult:
    question_id: str
    question: str
    gold_doc_ids: list[str]
    retrieved_doc_ids: list[str]  # what the retriever actually found, rank-ordered
    hit: bool                     # did any retrieved doc match a gold doc?
    reciprocal_rank: float        # 1/rank_of_first_hit; 0.0 if no hit
    context_precision: float      # fraction of retrieved docs that are relevant
    context_recall: float         # fraction of gold docs that were retrieved
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

Notice the layering: `EvalSample` is input (from the dataset), `EvalResult` is
per-question output (from one retrieval run), `EvalReport` is the aggregate
(across all questions). Each type has exactly the fields it needs and no more.

---

## The Four Metrics

All four metric functions in `eval.py` are pure functions. They take two lists
of strings and return a number. The caller is responsible for slicing the
retrieval results to the desired `k` before passing them in — `k` is implicit
in the list length.

### Hit Rate @ k

```python
def hit_at_k(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> bool:
    return any(d in gold_doc_ids for d in retrieved_doc_ids)
```

The simplest question: did the retriever find *anything* useful in the top k?
A hit is `True` if at least one retrieved chunk came from a gold document.

**Intuition:** Think of this as pass/fail per question. Hit rate is the
fraction of questions that passed.

### MRR — Mean Reciprocal Rank

```python
def reciprocal_rank(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    for i, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            return 1.0 / i
    return 0.0
```

Hit rate doesn't distinguish between the answer being at rank 1 vs rank 5. MRR
does. The reciprocal rank (RR) for one question is `1/rank_of_first_hit`:

| First hit at rank | RR |
|---|---|
| 1 | 1.000 |
| 2 | 0.500 |
| 3 | 0.333 |
| 5 | 0.200 |
| not found | 0.000 |

MRR is the mean of RR across all questions. A system that reliably puts the
right chunk first has MRR close to 1.0. A system that often buries the answer
deep in the ranking or returns no hit in the top-k has MRR close to 0.

**Intuition:** MRR measures "how high up in the results was the first useful
chunk?" It penalizes a retriever that finds the answer but only at rank 4.

### Context Precision @ k

```python
def context_precision_at_k(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    if not retrieved_doc_ids:
        return 0.0
    hits = sum(1 for d in retrieved_doc_ids if d in gold_doc_ids)
    return hits / len(retrieved_doc_ids)
```

Of the k chunks retrieved, what fraction came from a gold document? A
retriever that fetches mostly irrelevant chunks wastes the LLM's context
window and can distract it.

**One subtlety:** precision is computed at the chunk level (each retrieved
position is counted independently), not the document level. If the same
gold document appears at positions 2 and 4, both count as hits. This is a
deliberate choice for a v1 baseline — it reflects the actual prompt content
rather than unique document coverage.

**Intuition:** High precision means the LLM sees a clean signal. Low precision
means the prompt is full of noise.

### Context Recall @ k

```python
def context_recall_at_k(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    if not gold_doc_ids:
        return 0.0
    covered = len(set(retrieved_doc_ids) & set(gold_doc_ids))
    return covered / len(gold_doc_ids)
```

Of the gold documents, how many were covered by the top-k retrieval? This uses
set intersection — duplicate retrievals of the same doc count only once.

**Intuition:** High recall means the retriever found all the evidence the LLM
needs. Low recall means some required evidence was not retrieved in the top-k,
so the LLM cannot give a complete answer regardless of how good its generation is.

### Precision vs Recall

These two metrics pull in opposite directions as you increase `k`:

| Increasing k | Effect |
|---|---|
| Recall ↑ | More gold docs covered (good) |
| Precision ↓ | More irrelevant chunks included (bad) |

Choosing `k` is a trade-off. The evaluation harness makes this trade-off
visible so you can pick a value informed by data rather than intuition.

---

## The Runner (`run_retrieval_eval`)

```python
def run_retrieval_eval(
    samples: list[EvalSample],
    index: LoadedIndex,
    embedder: Embedder,
    top_k: int,
) -> EvalReport:
```

The runner loops over every `EvalSample`, runs retrieval, and aggregates:

```python
for sample in samples:
    query_vec = embedder.embed([sample.question])[0]
    results = retrieve_by_vector(query_vec, index, top_k=top_k)
    retrieved_doc_ids = [r.chunk.doc_id for r in results]

    hit = hit_at_k(retrieved_doc_ids, sample.gold_doc_ids)
    rr  = reciprocal_rank(retrieved_doc_ids, sample.gold_doc_ids)
    cp  = context_precision_at_k(retrieved_doc_ids, sample.gold_doc_ids)
    cr  = context_recall_at_k(retrieved_doc_ids, sample.gold_doc_ids)
    ...
```

Three things to notice:

1. **It reuses the existing retrieval path exactly.** `retrieve_by_vector` is
   the same function `rag ask` uses. The evaluation measures the same behavior
   the user experiences.

2. **It extracts `doc_id`, not `chunk_id`.** Multiple chunks may come from the
   same document; the metrics care about document-level coverage.

3. **`retrieve_by_vector` is imported inside the function body** (deferred
   import). This avoids a circular import: `eval.py` → `retrieval.py` →
   `index_loader.py` → back to types defined elsewhere. The `TYPE_CHECKING`
   block at the top of `eval.py` handles the type hints without any runtime cost.

---

## Reading the Output

The following baseline was generated locally from the cached
`ibm-research/watsonxDocsQA` snapshot:

```text
Documents : 1144
Chunks    : 8648
Questions : 75
Model     : sentence-transformers/all-MiniLM-L6-v2
```

```
Evaluation report  (n=75, top_k=5)
────────────────────────────────────
Hit Rate @ 5     :  0.867
MRR               :  0.756
Context Precision :  0.365
Context Recall    :  0.867
```

How to read this:

- **Hit Rate 0.867** — 87% of questions had at least one relevant chunk in the
  top 5. 13% of questions had no relevant chunk in the top 5.
- **MRR 0.756** — the first relevant chunk usually appears near the top of the
  list. Many first hits are at rank 1, with some later hits pulling the mean
  down.
- **Context Precision 0.365** — about 37% of the top-5 chunks were from a gold
  document. The rest were distractors in the prompt.
- **Context Recall 0.867** — 87% of the gold documents were covered by the
  top 5.

This is a reproducible local baseline, not a benchmark claim. It depends on the
prepared snapshot, chunking settings, embedding model, and `top_k`.

The same local run also shows the precision/recall trade-off:

| top_k | Hit Rate | MRR | Context Precision | Context Recall |
|---:|---:|---:|---:|---:|
| 1 | 0.680 | 0.680 | 0.680 | 0.680 |
| 3 | 0.840 | 0.749 | 0.453 | 0.840 |
| 5 | 0.867 | 0.756 | 0.365 | 0.867 |
| 10 | 0.907 | 0.762 | 0.249 | 0.907 |

As `k` increases, hit rate and recall improve because the retriever has more
chances to include a gold document. Precision falls because more non-gold
chunks enter the prompt. Phase 1.5 (BM25, hybrid retrieval) will try to improve
these numbers, and now you have a way to tell whether it worked.

---

## The CLI

```bash
rag eval --qa-file corpus/watsonx-docsqa/qa.jsonl \
         --index-dir .tiny-rag/index \
         --top-k 5
```

`cmd_eval` in `cli.py` follows the same pattern as `cmd_retrieve`:

```python
def cmd_eval(args):
    index = load_index(Path(args.index_dir))
    embedder = _make_embedder(index.manifest.get("embedding_model"))
    samples = load_eval_samples(Path(args.qa_file))
    report = run_retrieval_eval(samples, index, embedder, top_k=args.top_k)
    print(format_eval_report(report))
```

The `_make_embedder` factory is the same one used by `cmd_retrieve` and
`cmd_ask`. Tests patch it with `FakeEmbedder` — no model download, no network.

---

## What This Teaches

Evaluation answers the question your Phase 1 pipeline left open: **is the
retriever actually working?**

The key insight: retrieval quality and answer quality are separate. A bad
retriever with a good LLM produces fluent-sounding wrong answers. A good
retriever with a mediocre LLM can at least produce grounded correct ones. The
eval harness separates the two by measuring retrieval alone — no LLM needed.

The four metrics each reveal a different failure mode:

| Metric | What a low score tells you |
|---|---|
| Hit Rate | The retriever often returns no relevant document in the top-k |
| MRR | The relevant document is retrieved but buried — rank 4 instead of rank 1 |
| Context Precision | The top-k is full of noise — the LLM is swimming in distractors |
| Context Recall | Some required evidence is not retrieved in the top-k |

These failure modes have different fixes. Improving recall might mean
increasing `k` or using a hybrid retriever. Improving precision might mean
decreasing `k` or using a reranker. Without metrics, you're guessing.
