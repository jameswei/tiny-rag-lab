# The RAG Data Flow — How Data Moves From File to Answer

The implementation is built around a small set of dataclasses. They are not just
type definitions — they are the **contract** between every stage of the
pipeline. If you understand these types and how they flow through the CLI,
you understand the architecture.

```
┌──────────┐     ┌──────────┐     ┌──────────────────┐
│ Document │ ──► │  Chunk   │ ──► │ RetrievalResult  │
└──────────┘     └──────────┘     └──────────────────┘
  indexing           indexing           retrieval

                       ┌──────────────────────────────┐
                       │ RetrieveTrace / AskTrace     │
                       └──────────────────────────────┘
                         observability
```

Each arrow is a transformation with a clear rule. Let's walk through every one.

---

## Document — A File Loaded Into Memory

`Document` represents one source file from your corpus. You can find it in
`tiny_rag_lab/models.py`.

```python
@dataclass
class Document:
    doc_id: str           # e.g. "docs/getting-started.md"
    path: str             # e.g. "/home/user/corpus/docs/getting-started.md"
    title: str            # e.g. "Getting Started with watsonx"
    format: str           # "markdown" or "text"
    raw_text: str         # exact file contents, unchanged
    normalized_text: str  # cleaned-up version for chunking
    raw_hash: str         # SHA-256 of raw_text, 64 hex characters
```

### Why `doc_id` is a POSIX path, not an absolute path

When you run `rag index --corpus corpus/watsonx-docsqa`, the code computes
`doc_id` relative to that corpus root:

```python
doc_id = path.relative_to(corpus_root).as_posix()
# Input:  /home/user/tiny-rag-lab/corpus/watsonx-docsqa/docs/getting-started.md
# Output: "docs/getting-started.md"
```

This matters for two reasons. First, the index becomes portable — you can move
the corpus and rebuild the index without changing any identifiers. Second, when
the CLI prints retrieval results, you see a clean path like `docs/faq.md`
instead of a long absolute path nobody wants to read.

### Why keep both `raw_text` and `normalized_text`

`raw_text` is the exact file on disk. `normalized_text` is what the chunker
actually works with. They differ because the pipeline cleans up line endings,
trailing spaces, and blank-line runs (details in `normalize_text`, covered in
the indexing deep-dive). But you still want `raw_text` around — the `raw_hash`
is computed from it, so if the file changes on disk you can detect the drift
without re-reading everything.

### How the title is extracted

For Markdown files, the loader scans for the **first `#` heading**:

```python
if stripped.startswith("# "):
    return stripped[2:].strip()
```

If no H1 exists (or the file is plain text), the title falls back to the
**filename stem** — `faq.md` becomes `"faq"`. This is simple but intentional:
Phase 1 doesn't need fancy title extraction. The point is that every document
has *some* human-readable label for trace output.

---

## Chunk — The Atomic Retrieval Unit

A `Document` can be pages long. An LLM prompt can only hold so much context. So
we break each document into overlapping **chunks** — the smallest unit the
retrieval system can return.

```python
@dataclass
class Chunk:
    chunk_id: str      # e.g. "a1b2c3d4e5f67890"
    doc_id: str        # which document this chunk came from
    text: str          # the actual chunk content
    char_start: int    # where this chunk starts in normalized_text
    char_end: int      # where it ends (exclusive, Python-slice style)
    metadata: dict     # always includes title, path, format, raw_hash
```

### The critical invariant

Every chunk must satisfy:

```python
document.normalized_text[chunk.char_start:chunk.char_end] == chunk.text
```

This is tested in `tests/test_chunking.py` and enforced by the chunker's
implementation. If this invariant ever breaks, retrieval results point to wrong
text and citations become lies. The invariant is the single truth-anchor
between the chunk and its source document.

### Deterministic `chunk_id`

The id is produced by a SHA-256 hash with a predictable input:

```python
def make_chunk_id(doc_id: str, char_start: int, chunk_text: str) -> str:
    raw = f"{doc_id}:{char_start}:{chunk_text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

A few things to notice:

- The id only depends on the **doc_id, offset, and text** — not on random
  seeds, timestamps, or the embedding model.
- Re-indexing the same corpus with the same chunk size produces the **same
  chunk IDs**. This is essential for citations to remain stable across rebuilds.
- It uses the first 16 hex characters (64 bits). The probability of collision
  across a few thousand chunks is negligible, and 16 characters is easier to
  read in a terminal than a full 64-character hash.

### What metadata must carry

The `metadata` dict always includes:

| Key | Value | Why |
|---|---|---|
| `title` | Document title | Shows in trace output |
| `path` | Filesystem path | Lets the user find the source |
| `format` | `"markdown"` or `"text"` | Preserves original file type |
| `raw_hash` | SHA-256 of raw_text | Detect source changes later |

---

## RetrievalResult — A Ranked Chunk With a Score

When the retrieval engine searches the index, it ranks chunks by similarity to
the user's query. Each result bundles the chunk, its score, and its rank:

```python
@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float    # cosine similarity, range [-1, 1]
    rank: int       # 1-indexed (rank 1 = best match)
```

### Why rank is 1-indexed

Ranks start at 1 because that's what humans expect — "the top result" means
rank 1. The retrieval code generates them with `enumerate(results, start=1)`.
This is a small thing, but it matters in the CLI output: `Rank 1  score=0.9234`
is immediately readable.

### Where the score comes from

`score` is cosine similarity between the query vector and the chunk's embedding
vector. Values near 1.0 mean the chunk is semantically similar to the question;
values near 0 mean unrelated; negative values mean opposite direction
(theoretically possible but rare with real embeddings). The exact math is
covered in the retrieval deep-dive.

---

## RetrieveTrace and AskTrace — Records Of One Run

Phase 1.7 moved observability types into `tiny_rag_lab/trace.py`.
`rag retrieve` builds a `RetrieveTrace`, and `rag ask` builds an `AskTrace`.
Both commands print formatter-backed trace output, and both can write JSON with
`--trace-out`.

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

```python
@dataclass
class RetrieveTrace:
    query: str
    retriever: str
    top_k: int
    chunks: list[ChunkTrace]
    latency_by_stage: dict[str, float]
```

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

`RetrieveTrace` is for inspecting search without generation. `AskTrace` is for
inspecting the full RAG path: retrieval, prompt assembly, generation, and
citations.

`latency_by_stage` makes stage boundaries explicit:

| Key | What it measures |
|---|---|
| `"load"` | Time to load the index from disk |
| `"embed"` | Time to compute the query embedding |
| `"retrieve"` | Time to rank chunks |
| `"prompt_assembly"` | Time to build the prompt for `rag ask` |
| `"generate"` | Time for the LLM to produce an answer |

BM25 retrieve traces omit `"embed"` because BM25 does not use an embedding
model. Dense and hybrid retrieve traces include it.

The numbers are visible in retrieve and ask traces, with different keys:

```
Retrieve trace:
latency   : load=0.006s  embed=0.012s  retrieve=0.001s

Ask trace:
latency   : load=0.006s  embed=0.012s  retrieve=0.001s  prompt_assembly=0.000s  generate=1.234s
```

This tells you immediately where the pipeline spends time. In Phase 1 with a
local embedder and an API generator, `generate` will dominate by orders of
magnitude.

---

## How the CLI Wires Everything Together

The CLI is in `tiny_rag_lab/cli.py`. Three commands, each building on the last:

### `rag index` — Build the index

```
load_documents(corpus_root)          → list[Document]
  └─ chunk_documents(docs, ...)      → list[Chunk]
       └─ embedder.embed(texts)      → np.ndarray
            └─ write_index(...)      → .tiny-rag/index/
```

The key insight: indexing is a **pipeline**, not a monolith. Each function does
one thing. The CLI just calls them in order with the user's arguments.

### `rag retrieve` — Search without generating

```
load_index(index_dir)               → LoadedIndex
  └─ embed query if needed          → query vector
       └─ retrieve top-k chunks     → list[RetrievalResult]
            └─ build RetrieveTrace  → terminal output / optional JSON
```

Retrieval is the heart of RAG. `rag retrieve` lets you inspect the search
results *before* any LLM gets involved. This is the debugging entry point: if
the wrong chunks are returned, no prompt can fix it.

### `rag ask` — Full end-to-end

```
load_index → embed query → retrieve → assemble prompt → generate → build AskTrace
```

`cmd_ask` calls `retrieve_by_vector` (the pure-ranking function) rather than
`retrieve` (which also embeds), because it measures embedding and retrieval
timings separately. The prompt assembly and generation are fully decoupled from
retrieval — they only depend on `list[RetrievalResult]`.

---

## What This Teaches

The dataclasses are the **real architecture** of this learning lab. The CLI is just
a thin shell that calls them in order. If you can draw the Document → Chunk →
RetrievalResult → RetrieveTrace / AskTrace chain from memory, you can reason
about any change to the pipeline — new chunkers, different retrievers, better
prompts, richer traces — because you know exactly what data each stage receives
and produces.

The next deep-dives zoom into the stages one at a time:

- **The Indexing Plane** — how documents become chunks and vectors
- **Retrieval and Generation** — how similarity search and prompt assembly work
- **Persistence and Testing** — how the index is saved, loaded, and verified
- **Observability and Debugging** — how trace records explain one run
