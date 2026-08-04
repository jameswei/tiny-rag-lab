# tiny-rag-lab

[简体中文](README.zh-CN.md) · [Project site](https://jameswei.github.io/tiny-rag-lab/)

> A learning-first, inspectable classic RAG lab — readable Python, a browser
> Studio with a live retrieval course, a direct CLI, real-corpus traces, and
> bilingual Learning Guides.

`tiny-rag-lab` makes the full path from question to cited answer visible and
inspectable — document corpus, retrieved evidence, packed context, and a
grounded answer. Readable Python keeps the mechanics explicit; a browser
Studio turns them into guided replays and a live retrieval course covering
BM25 term scores, dense cosine math, hybrid RRF fusion, cross-encoder
reranking, and the same vectors in NumPy versus optional Qdrant; a direct CLI
supports repeatable inspection. Searchable bilingual Learning Guides open
beside the lab for deeper reading, and none of this requires an LLM provider
— a tested OpenAI-compatible provider is only needed for Live Ask generation.

It's a learning tool, not a production RAG platform: visible mechanics over
framework magic, evaluation before optimization, failure analysis before
advanced features.

![Guided Learn replay showing real retrieved evidence](website/assets/screenshots/guided-retrieval.jpg)

## Two ways to learn, one core

The web Studio and the CLI are complementary views of the same project-owned
RAG core — not separate products with separate mechanics. Both work from the
same documents, chunks, embeddings, retrieval results, context, prompts,
citations, and traces.

- **Studio (recommended start):** guided real-corpus replays with every
  intermediate artifact inspectable, plus hands-on retrieval, indexing,
  failure, and provider experiments.
- **CLI:** the direct, scriptable entrypoint — repeat a configuration,
  compare results, inspect raw output, follow the mechanics command by
  command.

Learners start from four saved, provider-free lessons over a pinned
40-document Cloudflare corpus rather than a synthetic one-document demo —
every lesson keeps the real source documents, chunks, vectors, ranked
candidates, selected and omitted context, prompts, answers, and citations
that connect one RAG stage to the next. From there, run your own retrievals,
build indexes, bring a small corpus, compare the NumPy index with optional
Qdrant, or connect a tested provider for Live Ask. When a stage needs deeper
explanation, bilingual **Learning Guides** open beside the experiment without
leaving for GitHub.

Classic RAG here means one visible path — retrieve evidence, pack context,
generate a cited answer. A tested OpenAI-compatible provider completes
generation for Live Ask; it does not turn the project into agentic or
multi-step RAG.

## What makes it different

Most RAG examples stop at a framework call or a single happy-path answer.
`tiny-rag-lab` keeps the concepts connected:

- **Implementation and experience stay connected.** The CLI and Studio
  render artifacts from the same RAG core, not an abstract diagram
  disconnected from runnable code.
- **Real artifacts are the lesson.** Guided replay shows the actual
  documents, chunks, vectors, ranked candidates, selected and omitted
  context, prompt, answer, citations, and timing behind a result.
- **Guidance leads to experimentation.** Start from a stable, provider-free
  replay, then change retrieval, build an index, upload a corpus, or connect
  a provider for Live Ask.
- **Failure is part of learning.** Evaluation, trace inspection, and curated
  failure scenarios explain *why* a result happened, not just whether an
  answer came back.

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

Docker Compose serves the Studio at `http://127.0.0.1:8000` by default
(override with `TINY_RAG_LAB_PORT`). The Learning Guides are linked from
every lab stage and are also published at the
[project site](https://jameswei.github.io/tiny-rag-lab/) for reading without
running the lab. The Studio itself stays local-only — no account, no public
deployment — and Live Ask contacts only the provider you configure.

First visit:

1. **Home → Start guided lesson** — replay one of four saved lessons over the
   pinned Cloudflare State & Coordination corpus.
2. **Learn** — step through corpus, chunks, embedding vector, retrieved
   candidates, selected context, answer, and citations.
3. **Retrieval** — six live modules: lexical and dense mechanics, NumPy vs.
   Qdrant, hybrid fusion, reranking, and a 16-question browser A/B
   evaluation.
4. **Explore** — ask from the 75-question watsonxDocsQA catalog or a
   free-form question, compare Dense/BM25/Hybrid retrieval, optionally
   rerank, then inspect the trace. Add a tested OpenAI-compatible provider
   only for Live Ask.
5. **Build & Inspect** — build an index from a bundled corpus or a small
   Markdown/plain-text upload (up to 100 files, 100 MiB), then inspect
   chunks, vectors, and provenance.
6. **Failure Lab** — compare curated failure scenarios against their fixes.

Every stage links **Read the learning guide**, opening the matching guide in
a new tab without losing your place. The interface is bilingual
(English/简体中文); bundled corpus content and recorded answers keep their
original language.

Runs on CPU by default — the `full` image ships pinned embedding and
cross-encoder snapshots, no GPU required. For a smaller image:

```bash
LAB_IMAGE_VARIANT=slim docker compose up --build
```

Guided Learn and BM25 retrieval work out of the box in `slim`; the Settings
page adds the embedding model and reranker as separate downloads when you
need Dense/Hybrid retrieval or cross-encoder reranking.

For the optional Qdrant comparison backend:

```bash
docker compose --profile qdrant up --build
```

## The classic RAG pipeline

```text
local corpus -> documents -> normalized text -> chunks -> embeddings
-> local vector index -> query embedding -> retrieval
-> selected context -> grounded prompt -> answer with citations
```

The project makes each stage inspectable:

- **Indexing:** document loading, normalization, fixed-character, structural,
  and experimental semantic chunking, metadata, embeddings, and a local index.
- **Retrieval:** dense cosine search, BM25 keyword search, hybrid Reciprocal
  Rank Fusion, and optional second-pass reranking.
- **Generation:** explicit context budgets, prompt assembly, an
  OpenAI-compatible generation boundary, citations, and abstention when the
  evidence is insufficient.
- **Evaluation and observability:** retrieval metrics, LLM-as-judge answer
  metrics, replayable traces, and curated failure diagnosis.

## CLI

The direct, repeatable companion to the Studio — same mechanics, compact
commands:

```bash
rag index --corpus PATH --index-dir .tiny-rag/index --chunk-size 800 --chunk-overlap 120
rag index --corpus PATH --index-dir .tiny-rag/index --chunking-strategy structural
rag index --corpus PATH --index-dir .tiny-rag/index --chunking-strategy semantic --semantic-similarity-threshold 0.5

rag retrieve "question text" --index-dir .tiny-rag/index --top-k 5 --retriever dense
rag retrieve "question text" --index-dir .tiny-rag/index --top-k 5 --retriever bm25
rag retrieve "question text" --index-dir .tiny-rag/index --top-k 5 --retriever hybrid
rag retrieve "question text" --index-dir .tiny-rag/index --top-k 5 --retriever hybrid --reranker cross-encoder --rerank-top-n 20

rag ask "question text" --index-dir .tiny-rag/index --top-k 5
rag ask "question text" --index-dir .tiny-rag/index --context-budget 8192 --output-format json

rag eval --qa-file corpus/watsonx-docsqa/qa.jsonl --index-dir .tiny-rag/index --top-k 5 --retriever hybrid
rag eval --qa-file corpus/watsonx-docsqa/qa.jsonl --index-dir .tiny-rag/index --judge fake --generator fake

rag diagnose --cases-file tests/fixtures/failure/cases.jsonl --index-dir .tiny-rag/index
```

Each command has focused help:

```bash
uv run rag --help
uv run rag index --help
uv run rag retrieve --help
uv run rag ask --help
uv run rag eval --help
uv run rag diagnose --help
```

## Development

```bash
uv sync --group dev
uv run pytest --tb=short -q
```

Run the two browser surfaces from source in separate terminals:

```bash
npm --prefix learning_materials install
npm --prefix learning_materials run dev

npm --prefix web install
npm --prefix web run dev
```

The React development server proxies `/docs` requests to the VitePress dev
server started by `npm --prefix learning_materials run dev`, matching the
packaged same-origin route.

Prepare the watsonxDocsQA corpus for the standalone CLI when needed:

```bash
uv run python scripts/prepare_watsonx_docsqa.py --inspect
uv run python scripts/prepare_watsonx_docsqa.py --output-dir corpus/watsonx-docsqa
```

Generated local corpora and indexes are intentionally ignored by Git:

```text
corpus/
.tiny-rag/
```

## Tech stack

- Python · `argparse` CLI · FastAPI · React + TypeScript · VitePress · Docker Compose
- Local embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Default index: inspectable NumPy files; optional local Qdrant adapter
- Generation: OpenAI-compatible Chat Completions API
- Offline testing: fake embedder + fake generator
- No LangChain, LlamaIndex, or Haystack wrapper around the learning-critical
  RAG mechanics

## Docs

- [Learning Guides](learning_materials/en/learning-roadmap.md): conceptual
  companion to the CLI and visual lab — served locally by Studio and also
  published at the project site
- [Proposal](docs/proposal.md): project purpose, philosophy, and non-goals
- [Architecture](docs/architecture.md): conceptual RAG planes and boundaries
- [File structure](docs/file-structure.md): repository map
