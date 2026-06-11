# Observability and Debugging — Making One RAG Run Explainable

Phase 1.7 adds an observability layer for single `retrieve` and `ask` runs.
The goal is not a dashboard or an evaluation report. The goal is a trustworthy
record of one run: what query was used, which chunks were returned, what prompt
was sent to the generator, what answer came back, and how long each stage took.

---

## Why Traces Matter

A RAG answer can be wrong for several different reasons:

- The retriever returned the wrong chunks.
- The right chunks were returned, but at low rank.
- The prompt did not include enough context.
- The model ignored the context.
- The model cited chunks that do not support the answer.

Without a trace, these failures collapse into one vague symptom: "the answer is
bad." With a trace, you can inspect the pipeline stage by stage.

Phase 1.6 evaluation answers a batch question: "How good is retrieval across a
dataset?" Phase 1.7 tracing answers an interactive question: "What happened in
this one run?"

---

## The Trace Types

All trace types live in `tiny_rag_lab/trace.py`. They are intentionally plain
dataclasses with JSON-native fields so `dataclasses.asdict()` and `json.dumps()`
are enough to serialize them.

### `ChunkTrace`

`ChunkTrace` is a compact view of one retrieved chunk:

```python
@dataclass
class ChunkTrace:
    rank: int
    chunk_id: str
    doc_id: str
    title: str
    path: str
    score: float
    text_preview: str
```

It keeps the fields a learner needs when debugging retrieval: rank, score,
identity, source document, and a short preview. It does not store the full
chunk text, because traces should stay readable and small.

### `RetrieveTrace`

`RetrieveTrace` records one `rag retrieve` run:

```python
@dataclass
class RetrieveTrace:
    query: str
    retriever: str
    top_k: int
    chunks: list[ChunkTrace]
    latency_by_stage: dict[str, float]
```

The latency keys are:

| Retriever | Latency keys |
|---|---|
| `dense` | `load`, `embed`, `retrieve` |
| `hybrid` | `load`, `embed`, `retrieve` |
| `bm25` | `load`, `retrieve` |

BM25 omits `embed` because it does not use an embedding model.

### `AskTrace`

`AskTrace` records one `rag ask` run:

```python
@dataclass
class AskTrace:
    query: str
    retriever: str
    top_k: int
    chunks: list[ChunkTrace]
    prompt: str
    answer: str
    citations: list[str]
    latency_by_stage: dict[str, float]
```

The latency keys are `load`, `embed`, `retrieve`, `prompt_assembly`, and
`generate`. This splits the full RAG pipeline into inspectable stages.

---

## Terminal Output vs JSON Output

Both `rag retrieve` and `rag ask` now always print formatter-backed trace
output:

```bash
rag retrieve "how do I deploy a model" --index-dir .tiny-rag/index
rag ask "how do I deploy a model" --index-dir .tiny-rag/index
```

Adding `--trace-out` writes the same trace contract to disk as JSON:

```bash
rag retrieve "how do I deploy a model" \
  --index-dir .tiny-rag/index \
  --trace-out /tmp/retrieve-trace.json

rag ask "how do I deploy a model" \
  --index-dir .tiny-rag/index \
  --trace-out /tmp/ask-trace.json
```

`--trace-out` does not change the run. It only adds a JSON artifact.

---

## What To Look For In A Retrieve Trace

Start with the header:

- `retriever`: confirms whether the run used dense, BM25, or hybrid retrieval.
- `top_k`: confirms how many chunks the command asked for.
- `latency`: shows whether time was spent loading, embedding, or ranking.

Then read the chunk list:

- A high `score` at rank 1 usually means the query and chunk are close in the
  embedding space.
- A low score across all chunks suggests the query may not match the corpus.
- A correct `doc_id` at low rank suggests the retriever found the answer but
  did not rank it confidently.
- Repeated `doc_id` values can be useful, but can also crowd out other evidence.

Retrieve traces are the fastest way to debug retrieval without involving an
LLM.

---

## What To Look For In An Ask Trace

An ask trace includes everything from retrieval plus the prompt and answer.

Inspect it in this order:

1. **Chunks**: did retrieval bring in useful evidence?
2. **Prompt**: did prompt assembly include the expected source markers?
3. **Answer**: did the generator follow the context?
4. **Citations**: do answer citations point to retrieved chunk IDs?
5. **Latency**: which stage dominates the run?

This order matters. If the chunks are wrong, the model never had the right
evidence. If the chunks are right but the answer is wrong, the issue is likely
prompting or generation.

---

## How This Connects To Phase 1.8

Phase 1.8 will study RAG failures. The trace schema from Phase 1.7 is the input
contract for that work. A failure lab needs durable artifacts: query, chunks,
scores, prompt, answer, citations, and latency. The trace files provide exactly
that without adding a database or report UI.

The key learning point: observability is not just logging. It is choosing the
small set of fields that lets you explain system behavior later.

