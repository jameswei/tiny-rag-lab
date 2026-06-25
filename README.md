# tiny-rag-lab

**Project site:** https://jameswei.github.io/tiny-rag-lab/

`tiny-rag-lab` is a learning-first RAG engine/laboratory for understanding how
classic retrieval-augmented generation works end to end.

The goal is to keep the RAG lifecycle visible:
document loading, text normalization, chunking, metadata, embeddings, local
vector search, retrieval, prompt assembly, answer generation, citations,
evaluation, and failure inspection.

## Current Status

Phase 1 through Phase 2.2 are complete. No phase is currently active.

- **Phase 1 — Naive Classic RAG**: full pipeline from corpus to grounded answers with citations
- **Phase 1.5 — Retrieval Mechanics**: BM25 keyword retrieval, hybrid retrieval, and retriever comparison flags
- **Phase 1.6 — Evaluation Harness**: retrieval quality metrics (`rag eval`) against a prepared QA set
- **Phase 1.7 — Observability And Debugging**: retrieve/ask traces, stage latency, and optional JSON trace output
- **Phase 1.8 — RAG Failure Lab**: curated failure cases and `rag diagnose` for baseline vs. intervention retrieval
- **Phase 1.9 — Reranking**: fake and cross-encoder reranker interfaces with retrieve/eval/ask/diagnose integration
- **Phase 2.0 — Answer Quality Judging**: fake and OpenAI-compatible judge paths for answer metrics and answer-side failure diagnosis
- **Phase 2.1 — Context Budget And Structured Answers**: token-budget context packing, inspectable omitted chunks, and optional JSON answer output
- **Phase 2.2 — Structural And Semantic Chunking**: structural (Markdown-aware three-tier) and semantic (embedding-based topic-shift) chunking alongside the existing fixed-character baseline

Completed phase contracts:

- [Phase index](docs/phases/README.md)
- [Phase 1 spec](docs/phases/phase-1-naive-classic-rag.md) · [taskboard](docs/phases/phase-1-taskboard.md)
- [Phase 1.5 spec](docs/phases/phase-1.5-retrieval-mechanics.md) · [taskboard](docs/phases/phase-1.5-taskboard.md)
- [Phase 1.6 spec](docs/phases/phase-1.6-evaluation-harness.md) · [taskboard](docs/phases/phase-1.6-taskboard.md)
- [Phase 1.7 spec](docs/phases/phase-1.7-observability.md) · [taskboard](docs/phases/phase-1.7-taskboard.md)
- [Phase 1.8 spec](docs/phases/phase-1.8-failure-lab.md) · [taskboard](docs/phases/phase-1.8-taskboard.md)
- [Phase 1.9 spec](docs/phases/phase-1.9-reranking.md) · [taskboard](docs/phases/phase-1.9-taskboard.md)
- [Phase 2.0 spec](docs/phases/phase-2.0-answer-quality-judging.md) · [taskboard](docs/phases/phase-2.0-taskboard.md)
- [Phase 2.1 spec](docs/phases/phase-2.1-context-budget-structured-answers.md) · [taskboard](docs/phases/phase-2.1-taskboard.md)
- [Phase 2.2 spec](docs/phases/phase-2.2-structural-semantic-chunking.md) · [taskboard](docs/phases/phase-2.2-taskboard.md)

## Phase 1 Result

Phase 1 delivers a minimal but complete CLI-first RAG baseline:

```text
local corpus -> documents -> normalized text -> chunks -> embeddings
-> local vector index -> query embedding -> cosine retrieval
-> grounded prompt -> generated answer with citations
```

Key decisions:

- Python implementation
- `argparse` CLI
- primary corpus: IBM `watsonxDocsQA`
- local embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- OpenAI-compatible online generation for real answers
- fake embedder and fake generator for tests
- local index files under `.tiny-rag/index/`
- no vector database in Phase 1
- no LangChain/LlamaIndex/Haystack wrapper in Phase 1

## Phase 1.5 Result

Phase 1.5 adds inspectable retrieval strategies to compare dense vector search,
BM25 keyword search, and hybrid retrieval with Reciprocal Rank Fusion.

```text
query + index -> dense retrieval | BM25 retrieval -> optional RRF fusion
-> ranked chunks and eval reports tagged with retriever=dense|bm25|hybrid
```

## Phase 1.6 Result

Phase 1.6 adds a `rag eval` command that measures retrieval quality against the
prepared `qa.jsonl` evaluation set. Four deterministic metrics are reported:
hit rate @ k, MRR, context precision, and context recall.

```text
qa.jsonl + index -> embed questions -> retrieve top-k -> compare to gold docs
-> hit rate, MRR, context precision, context recall
```

## Phase 1.7 Result

Phase 1.7 adds trace records and human-readable trace output for retrieve and
ask flows. Traces expose the retriever, top-k, ranked chunks, scores, citations,
prompt/answer context, and stage latency.

```text
query + retrieval/ask flow -> trace fields -> readable trace and optional JSON
```

## Phase 1.8 Result

Phase 1.8 adds a failure lab for curated retrieval failure scenarios. The
`rag diagnose` command compares each case's baseline and intervention retrieval
config, labels heuristic failure modes, and reports whether failures were
confirmed, fixed, moved, or unchanged.

```text
failure cases + index -> baseline retrieval + intervention retrieval
-> failure labels, metrics, and diagnosis report
```

## Phase 1.9 Result

Phase 1.9 adds a reranker abstraction and optional second-pass reranking for
retrieve, eval, ask, and diagnose workflows. The default `none` path remains
unchanged; fake rerankers keep tests offline, and the cross-encoder path is
lazy and gated.

```text
initial candidates -> optional reranker -> final top-k chunks
-> traces and reports with reranker metadata
```

## Phase 2.0 Result

Phase 2.0 adds answer-quality judging behind a fakeable interface. `rag eval`
can print retrieval metrics plus answer metrics, `rag ask` can include a judge
verdict in the trace, and `rag diagnose` can cover answer-side failures such
as unsupported answers and citation mismatches.

```text
retrieved context + generated answer -> judge verdict
-> answer metrics, trace verdicts, and answer-side diagnosis
```

## Phase 2.1 Result

Phase 2.1 adds generation-side context budgeting and optional structured JSON
output. A token budget controls which retrieved chunks actually reach the
prompt; omitted chunks are recorded in the trace so budget pressure is visible
and debuggable. The default path (no `--context-budget` flag) is identical to
Phase 2.0.

```text
retrieved chunks -> pack_context (greedy by token count)
-> selected chunks in prompt, omitted chunks in trace
-> optional --output-format json prints full AskTrace as JSON
```

## Phase 2.2 Result

Phase 2.2 adds two new indexing-time chunking strategies alongside the existing
fixed-character baseline. The chunking strategy is chosen at `rag index` time
and recorded in the manifest; all downstream commands are unchanged.

- **Structural**: pack Markdown-aware blocks (headings, paragraphs, lists)
  so chunk boundaries fall on structural boundaries instead of mid-sentence.
  Falls back to sentence-level packing for oversized blocks.
- **Semantic** (experimental): embed all sentences in one batch call, then
  split where consecutive sentence cosine similarity drops below a threshold —
  a topic-shift boundary.

```text
corpus + --chunking-strategy structural  -> blocks/sentence-aware chunks
corpus + --chunking-strategy semantic    -> topic-shift chunks (batch embed)
corpus + --chunking-strategy fixed_character  -> original sliding window (default)
all three -> manifest records chunking_strategy + chunking_params
```

## CLI

```bash
rag index --corpus PATH --index-dir .tiny-rag/index --chunk-size 800 --chunk-overlap 120
rag index --corpus PATH --index-dir .tiny-rag/index --chunking-strategy structural
rag index --corpus PATH --index-dir .tiny-rag/index --chunking-strategy semantic --semantic-similarity-threshold 0.5
rag retrieve "question text" --index-dir .tiny-rag/index --top-k 5 --retriever dense
rag retrieve "question text" --index-dir .tiny-rag/index --top-k 5 --retriever bm25
rag retrieve "question text" --index-dir .tiny-rag/index --top-k 5 --retriever hybrid
rag ask "question text" --index-dir .tiny-rag/index --top-k 5
rag ask "question text" --index-dir .tiny-rag/index --context-budget 8192
rag ask "question text" --index-dir .tiny-rag/index --context-budget 8192 --output-format json
rag eval --qa-file corpus/watsonx-docsqa/qa.jsonl --index-dir .tiny-rag/index --top-k 5 --retriever dense
rag eval --qa-file corpus/watsonx-docsqa/qa.jsonl --index-dir .tiny-rag/index --top-k 5 --retriever bm25
rag eval --qa-file corpus/watsonx-docsqa/qa.jsonl --index-dir .tiny-rag/index --top-k 5 --retriever hybrid
rag eval --qa-file corpus/watsonx-docsqa/qa.jsonl --index-dir .tiny-rag/index --judge fake --generator fake
rag eval --qa-file corpus/watsonx-docsqa/qa.jsonl --index-dir .tiny-rag/index --judge fake --generator fake --context-budget 8192
rag diagnose --cases-file tests/fixtures/failure/cases.jsonl --index-dir .tiny-rag/index
rag diagnose --cases-file tests/fixtures/failure/cases.jsonl --index-dir .tiny-rag/index --judge fake --generator fake
rag diagnose --cases-file tests/fixtures/failure/cases.jsonl --index-dir .tiny-rag/index --judge fake --generator fake --context-budget 8192
```

Help is available for each command:

```bash
uv run rag --help
uv run rag index --help
uv run rag retrieve --help
uv run rag ask --help
uv run rag eval --help
uv run rag diagnose --help
```

## Development

Install/sync dependencies:

```bash
uv sync --group dev
```

Run tests:

```bash
uv run pytest --tb=short -q
```

Prepare the primary corpus after dependencies are installed:

```bash
uv run python scripts/prepare_watsonx_docsqa.py --inspect
uv run python scripts/prepare_watsonx_docsqa.py --output-dir corpus/watsonx-docsqa
```

Generated corpora and indexes are intentionally ignored by git:

```text
corpus/
.tiny-rag/
```

## Docs

- [Proposal](docs/proposal.md): project purpose, philosophy, and non-goals
- [Roadmap](docs/roadmap.md): directional phase sequence
- [Architecture](docs/architecture.md): conceptual RAG planes and boundaries
- [Agent guidelines](docs/agent-guidelines.md): collaboration, review, and handoff workflow
- [File structure](docs/file-structure.md): quick repository map
- [Phase docs](docs/phases/README.md): active phase pointer and phase contracts

For implementation work, the phase spec and taskboard under `docs/phases/` are
the source of truth.
