# tiny-rag-lab

**Project site:** https://jameswei.github.io/tiny-rag-lab/

`tiny-rag-lab` is a learning-first RAG engine/laboratory for understanding how
classic retrieval-augmented generation works end to end.

> **v0.2.0 / Phase 3.0 milestone:** the project now includes a local visual
> RAG lab alongside its inspectable CLI. Learners can replay a starter run,
> inspect chunks, vectors, retrieval and context packing, compare curated
> failures, and run a small local corpus through the same visible mechanics.

The goal is to keep the RAG lifecycle visible:
document loading, text normalization, chunking, metadata, embeddings, local
vector search, retrieval, prompt assembly, answer generation, citations,
evaluation, and failure inspection.

## The pipeline

```text
local corpus -> documents -> normalized text -> chunks -> embeddings
-> local vector index -> query embedding -> cosine retrieval
-> grounded prompt -> generated answer with citations
```

## What it covers

**Retrieval**
- Dense vector search, BM25 keyword retrieval, and hybrid fusion (Reciprocal Rank Fusion)
- Optional second-pass reranking — fake or cross-encoder

**Evaluation**
- `rag eval`: hit rate @ k, MRR, context precision, context recall
- LLM-as-judge answer metrics: faithfulness, relevance, correctness

**Observability**
- Per-query trace output: retriever, scores, ranked chunks, stage latency, prompt context
- `rag diagnose`: curated failure cases with baseline vs. intervention comparison

**Generation**
- Token-budget context packing; omitted chunks recorded in trace
- Optional `--output-format json` for structured answer output

**Chunking**
- `fixed_character`: sliding window (default)
- `structural`: Markdown-aware block boundaries
- `semantic`: embedding-based topic-shift detection (experimental)

## Tech stack

- Python · `argparse` CLI · `uv`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local)
- Vector index: NumPy (no vector database)
- Generation: OpenAI-compatible API
- Test backends: fake embedder + fake generator (fully offline)
- Corpus: IBM `watsonxDocsQA`
- No LangChain / LlamaIndex / Haystack wrapper

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

## Local Visual Lab (Phase 3.0 / v0.2.0)

The CLI remains the lightweight, direct interface. The local visual lab is a
second entrypoint that serves a React learning client and a FastAPI API from
one `studio` service.

The lab is designed for local learning, not public hosting: the UI supports
English and Simplified Chinese while preserving the source corpus's language in
questions, evidence, answers, and citations. It includes an offline starter
replay, small Markdown/text uploads, watsonxDocsQA import, visual trace
playback, curated failure lessons, and inspectable NumPy vectors. Optional
Qdrant changes only the dense-index backend; it does not hide the conceptual
RAG flow.

```bash
cp .env.example .env
docker compose up --build
```

Open http://127.0.0.1:8000. The default `full` image includes the default
embedding model; use `LAB_IMAGE_VARIANT=slim` in `.env` to defer that model
download until a custom corpus is indexed. The optional local Qdrant backend
is started only when requested:

```bash
docker compose --profile qdrant up --build
```

Verify the optional adapter against the same English fixture corpus through a
temporary loopback-only test profile (the normal profile remains internal):

```bash
docker compose --profile qdrant-test up -d qdrant-test
QDRANT_URL=http://127.0.0.1:6333 uv run --extra qdrant pytest tests/test_qdrant_integration.py -q
docker compose --profile qdrant-test down
```

The established bare CLI stays NumPy-first in Phase 3.0; Qdrant is a local
visual-lab comparison backend, not a CLI deployment dependency.

The lab accepts up to 100 Markdown/plain-text files totalling 100 MiB. Its
starter trace can replay offline; live Ask requires an OpenAI-compatible
provider through `.env` or the browser's session-only Settings panel.

## Docs

- [Proposal](docs/proposal.md): project purpose, philosophy, and non-goals
- [Roadmap](docs/roadmap.md): directional phase sequence
- [Architecture](docs/architecture.md): conceptual RAG planes and boundaries
- [File structure](docs/file-structure.md): repository map
- [Phase docs](docs/phases/README.md): phase specs and taskboards
