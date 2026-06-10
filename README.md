# tiny-rag-lab

`tiny-rag-lab` is a learning-first RAG engine/laboratory for understanding how
classic retrieval-augmented generation works end to end.

The goal is to keep the RAG lifecycle visible:
document loading, text normalization, chunking, metadata, embeddings, local
vector search, retrieval, prompt assembly, answer generation, citations,
evaluation, and failure inspection.

## Current Status

Phase 1 and Phase 1.6 are complete.

- **Phase 1 — Naive Classic RAG**: full pipeline from corpus to grounded answers with citations
- **Phase 1.6 — Evaluation Harness**: retrieval quality metrics (`rag eval`) against a prepared QA set

Completed phase contracts:

- [Phase index](docs/phases/README.md)
- [Phase 1 spec](docs/phases/phase-1-naive-classic-rag.md) · [taskboard](docs/phases/phase-1-taskboard.md)
- [Phase 1.6 spec](docs/phases/phase-1.6-evaluation-harness.md) · [taskboard](docs/phases/phase-1.6-taskboard.md)

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

## Phase 1.6 Result

Phase 1.6 adds a `rag eval` command that measures retrieval quality against the
prepared `qa.jsonl` evaluation set. Four deterministic metrics are reported:
hit rate @ k, MRR, context precision, and context recall.

```text
qa.jsonl + index -> embed questions -> retrieve top-k -> compare to gold docs
-> hit rate, MRR, context precision, context recall
```

## CLI

```bash
rag index --corpus PATH --index-dir .tiny-rag/index --chunk-size 800 --chunk-overlap 120
rag retrieve "question text" --index-dir .tiny-rag/index --top-k 5
rag ask "question text" --index-dir .tiny-rag/index --top-k 5
rag eval --qa-file corpus/watsonx-docsqa/qa.jsonl --index-dir .tiny-rag/index --top-k 5
```

Help is available for each command:

```bash
uv run rag --help
uv run rag index --help
uv run rag retrieve --help
uv run rag ask --help
uv run rag eval --help
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
