# Phase 1 Spec: Naive Classic RAG

**Status:** Review-ready  
**Authors:** Codex + Claude Code + owner decisions  
**Based on:** `docs/proposal.md`, `docs/roadmap.md`, `docs/architecture.md`  
**Taskboard:** `docs/phases/phase-1-taskboard.md`  
**Date:** 2026-06-07

---

## Goal

Implement the smallest complete, inspectable classic RAG path:

```
local corpus -> documents -> normalized text -> chunks -> embeddings
-> local vector index -> query embedding -> cosine retrieval
-> grounded prompt -> generated answer with citations
```

By the end of Phase 1, a user can prepare the primary corpus, build a local
index, inspect retrieved chunks for a query, and ask a question that returns an
answer with source references.

---

## Scope

### In scope

- Python project using `uv` and `pyproject.toml`
- `rag` CLI using Python's standard `argparse`
- Markdown and plain-text document loading
- Prepared local export of IBM `watsonxDocsQA` as the required corpus
- Dataset preparation script for `watsonxDocsQA`
- Simple visible text normalization
- Deterministic fixed-size character chunking
- Source metadata and stable chunk IDs
- Local embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Deterministic fake embedder for tests
- NumPy cosine retrieval
- Local index persistence under `.tiny-rag/index/`
- Grounded prompt assembly
- OpenAI-compatible online generation through the `openai` Python client
- Fake generator for tests
- CLI commands: `rag index`, `rag retrieve`, `rag ask`
- Minimal terminal trace output
- Unit and CLI tests using fake backends

### Out of scope for Phase 1

- API embeddings
- BM25 and hybrid retrieval
- Reranking
- Metadata filtering beyond source display
- Full evaluation metrics and reports
- Saved trace stores
- Failure taxonomy
- UI
- Agentic RAG
- Vector databases
- LangChain, LlamaIndex, Haystack, or other high-level RAG framework wrappers
- WixQA as a completion requirement

WixQA remains a stretch corpus after the primary `watsonxDocsQA` path works.

---

## Runtime And Tooling

Use Python with `uv`.

Initial runtime dependencies:

- `numpy`
- `sentence-transformers`
- `datasets`
- `openai`

Initial dev dependency:

- `pytest`

Go is not used in Phase 1. It can be reconsidered later for a comparison
implementation or a small serving binary after the RAG mechanics are clear.

---

## Corpus Preparation

Phase 1 must not depend on live web crawling. Open datasets are prepared into
local files before indexing.

Required prepared layout:

```text
corpus/
  watsonx-docsqa/
    docs/
      <doc_id>.md
      ...
    dataset-manifest.json
    qa.jsonl
```

The preparation script lives under `scripts/` and should support converting
already-downloaded local dataset files when possible. It may also support
downloading through Hugging Face for convenience, but conversion into the local
layout is the core responsibility.

Each prepared document is normal Markdown:

```markdown
# Original Document Title

Original document content...
```

`dataset-manifest.json` records:

- dataset name
- source URL
- license
- preparation timestamp
- document count
- records mapping `doc_id`, local path, title, and original URL when available

`qa.jsonl` preserves dataset-provided questions, answers, and gold document
labels when available. Phase 1 does not evaluate against this file; it is kept
so Phase 1.6 can add evaluation cleanly.

Generated corpus files are not committed to git. Commit preparation code and
small test fixtures only.

---

## Data Contracts

Use dataclasses or equivalent simple typed structures. Keep fields serializable
where practical.

```python
@dataclass
class Document:
    doc_id: str
    path: str
    title: str
    format: str
    raw_text: str
    normalized_text: str
    raw_hash: str
```

- `doc_id` is the POSIX-style path relative to the `--corpus PATH` root passed
  to `rag index`, such as `docs/example.md`.
- `doc_id` must not include the project root, absolute path prefixes, or the
  prepared corpus directory name. For example, indexing
  `--corpus corpus/watsonx-docsqa` should produce `doc_id` values like
  `docs/example.md`, not `corpus/watsonx-docsqa/docs/example.md`.
- `path` is the filesystem path used for loading.
- `title` comes from the first Markdown H1 when present, otherwise the filename.
- `format` is `markdown` or `text`.
- `raw_hash` is the SHA-256 hash of `raw_text`.

```python
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int
    metadata: dict[str, Any]
```

- `char_start` and `char_end` are Python string character offsets into
  `Document.normalized_text`.
- Required invariant:

```python
document.normalized_text[chunk.char_start:chunk.char_end] == chunk.text
```

- `chunk_id` is deterministic:

```text
sha256(doc_id + ":" + str(char_start) + ":" + chunk_text)[:16]
```

- `metadata` includes at least `title`, `path`, `format`, and `raw_hash`.

```python
@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int
```

- `score` is cosine similarity.
- `rank` is 1-indexed.

```python
@dataclass
class RagTrace:
    query: str
    retrieved_chunks: list[RetrievalResult]
    prompt: str
    answer: str
    citations: list[str]
    latency_by_stage: dict[str, float]
```

Phase 1 traces are printed to the terminal, not persisted as a trace store.

---

## Text Normalization And Chunking

Normalization rules:

- Normalize line endings to `\n`.
- Strip trailing whitespace from lines.
- Collapse runs of more than two blank lines to two blank lines.
- Preserve Markdown headings and most punctuation.

Chunking rules:

- Use deterministic character-based chunking over `normalized_text`.
- Default `chunk_size`: 800 characters.
- Default `chunk_overlap`: 120 characters.
- Skip empty or whitespace-only chunks.
- Offsets must refer to the exact slice in `normalized_text`.
- Chunking parameters are index-time settings and must be written to the index
  manifest.

Heading-aware, sentence-aware, and token-aware chunking are deferred.

---

## Embeddings

Interface:

```python
class Embedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        ...
```

Rules:

- Return shape is `(len(texts), dim)`.
- Retrieval code must not know which backend produced the vectors.
- All tests use the deterministic fake embedder.

Real Phase 1 embedder:

- `sentence-transformers/all-MiniLM-L6-v2`

Notes:

- The model runs locally after its weights are downloaded.
- A fresh machine may need network access on first real indexing use.
- No embedding model needs to be downloaded before coding starts.
- Model name and embedding dimension are written to the manifest.
- API embeddings are deferred.

---

## Index Artifacts

Persist the index under:

```text
.tiny-rag/index/
  manifest.json
  chunks.jsonl
  embeddings.npz
```

`manifest.json` includes:

- `schema_version`
- `corpus_root`
- `created_at`
- `document_count`
- `chunk_count`
- `chunk_size`
- `chunk_overlap`
- `embedding_backend`
- `embedding_model`
- `embedding_dim`
- `corpus_files`

Each `corpus_files` record includes:

```json
{
  "doc_id": "docs/example.md",
  "path": "/absolute/or/project/path/docs/example.md",
  "raw_hash": "..."
}
```

`chunks.jsonl` stores one serialized chunk per line and does not include
embedding vectors.

`embeddings.npz` stores:

- embedding matrix
- parallel `chunk_ids`

The embedding row order must match `chunk_ids`.

---

## Retrieval

Use cosine similarity over NumPy arrays.

Defaults:

- `top_k`: 5

Rules:

- Normalize vectors safely.
- Handle zero vectors deliberately.
- Return ranked `RetrievalResult` values.
- `rag retrieve` displays rank, score, chunk ID, title/path, and chunk preview.

---

## Generation And Prompting

Generation interface:

```python
class Generator:
    def generate(self, prompt: str) -> str:
        ...
```

Real provider:

- Use OpenAI or an OpenAI-compatible API through the `openai` Python client.
- Base URL, API key, and model are configurable by CLI flag or environment
  variable.
- Tests must not depend on provider credentials or network access.

Fake generator:

- Returns a structured answer with at least one citation-like source marker.
- Good enough for tests to verify prompt assembly and citation extraction.
- Example marker: `[Source: <chunk_id>]`.

Prompt must include:

- user question
- retrieved context blocks
- source marker for each context block
- instruction to answer only from provided context
- instruction to say when context is insufficient
- instruction to cite sources using the provided markers

Starting prompt template:

```text
You are a retrieval-augmented assistant. Answer the question using only the
provided context.

If the context is insufficient, say that the provided context does not contain
enough information to answer. Do not use outside knowledge.

Cite every factual claim with the source marker for the context block that
supports it.

Question:
{question}

Context:
{context_blocks}

Answer:
```

Each context block should use this format:

```text
[Source: {chunk_id}]
Title: {title}
Path: {path}

{chunk_text}
```

Phase 1 citation format:

```text
[Source: <chunk_id>]
```

The CLI also prints a source table mapping chunk IDs to document paths and
titles.

---

## CLI

Index:

```bash
rag index --corpus PATH --index-dir .tiny-rag/index --chunk-size 800 --chunk-overlap 120
```

Responsibilities:

- load documents
- normalize text
- chunk documents
- embed chunks
- write index artifacts
- print document count, chunk count, embedding model, and index path

Retrieve:

```bash
rag retrieve "question text" --index-dir .tiny-rag/index --top-k 5
```

Responsibilities:

- load index
- embed query
- retrieve top-k chunks
- print rank, score, chunk ID, title/path, and chunk preview

Ask:

```bash
rag ask "question text" --index-dir .tiny-rag/index --top-k 5
```

Responsibilities:

- load index
- retrieve top-k chunks
- assemble prompt
- generate answer
- print answer
- print citations/source table
- print minimal stage timings

Chunking flags belong only to `rag index`.

---

## Required Tests

Chunking:

- known normalized text produces expected chunk text
- every chunk satisfies
  `normalized_text[char_start:char_end] == chunk.text`
- chunk IDs are stable across repeated runs

Cosine retrieval:

- known vectors produce expected ranking and scores
- zero vectors are handled deliberately

Persistence:

- build an index from a fixture corpus
- save artifacts
- load artifacts through the normal index loading path
- retrieve and assert the expected document or chunk is top-ranked

CLI:

- `rag index` creates expected files
- `rag retrieve` prints rank, score, and source metadata
- `rag ask` works with fake backends and includes a citation-like source marker

---

## Acceptance Criteria

Phase 1 is complete when:

1. `watsonxDocsQA` can be prepared into local Markdown files,
   `dataset-manifest.json`, and `qa.jsonl`.
2. `rag index` can build `.tiny-rag/index/manifest.json`,
   `.tiny-rag/index/chunks.jsonl`, and `.tiny-rag/index/embeddings.npz`.
3. `rag retrieve` shows ranked chunks with scores and source metadata.
4. `rag ask` returns an answer plus citation/source references.
5. Tests pass without network access or external API credentials.
6. A persistence round-trip test proves that loading a saved index preserves
   expected retrieval behavior.
