# Persistence and Testing — Saving the Index and Proving It Works

The index is expensive to build (embedding thousands of chunks with a real
model takes time) but cheap to reuse. Two modules handle persistence:
`index_writer.py` saves the index, `index_loader.py` reads it back.

The whole pipeline is testable without network or API keys. This document
covers the index format, the round-trip contract, and how fake backends make
testing possible.

---

## The Index Format: Three Files, One Contract

After `rag index` finishes, `.tiny-rag/index/` contains three files:

```
.tiny-rag/index/
  manifest.json     — metadata about the index run
  chunks.jsonl      — one chunk per line, no vectors
  embeddings.npz    — float32 matrix + parallel chunk_ids
```

### Why three files instead of one

Separating chunks (JSONL) from embeddings (NPZ) lets you inspect the chunks
with `cat` or `head` — they're plain text. The embeddings are binary floats
in a compressed NumPy format — not human-readable, but compact and fast to
load.

If they were combined, you'd need a Python script just to see what's in the
index. For a learning lab, inspectability matters more than minimal file count.

### `manifest.json` — the index's birth certificate

```json
{
  "schema_version": "1.0",
  "corpus_root": "/home/user/tiny-rag-lab/corpus/watsonx-docsqa",
  "created_at": "2026-06-09T10:30:00+00:00",
  "document_count": 42,
  "chunk_count": 312,
  "chunk_size": 800,
  "chunk_overlap": 120,
  "embedding_backend": "SentenceTransformerEmbedder",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dim": 384,
  "corpus_files": [
    {"doc_id": "docs/auth.md", "path": "/home/...", "raw_hash": "abc123..."},
    ...
  ]
}
```

The manifest records everything that produced the index — corpus path, chunk
parameters, embedding model, and a hash of every source file. This means you
can look at an index you built three weeks ago and know exactly how it was
made without digging through shell history.

The `schema_version` field ("1.0") exists so future phases can evolve the
format. If Phase 1.5 adds BM25 scores to the index, it can bump the version and
the loader can handle both old and new formats.

### `chunks.jsonl` — one JSON object per line

Each line is a serialized `Chunk`:

```json
{"chunk_id":"a1b2c3...","doc_id":"docs/auth.md","text":"To authenticate...","char_start":0,"char_end":800,"metadata":{"title":"API Auth","path":"/home/...","format":"markdown","raw_hash":"abc123..."}}
```

JSONL (one JSON object per line, newline-delimited) is chosen over a JSON array
for two reasons. First, it's streamable — you can read line-by-line without
loading the entire file into memory. Second, it's trivially inspectable with
`head chunks.jsonl` or `wc -l chunks.jsonl` (the line count is the chunk
count).

Embedding vectors are **not** stored in chunks.jsonl — they live in the NPZ
file. This separation keeps the human-readable side light and the binary side
efficient.

### `embeddings.npz` — the vector matrix

```python
np.savez(
    "embeddings.npz",
    embeddings=embeddings,    # float32 array, shape (N, dim)
    chunk_ids=chunk_ids,      # string array, shape (N,)
)
```

Two arrays stored in one compressed file. The critical contract: **row i in
`embeddings` must correspond to `chunk_ids[i]`**. The writer stores `chunk_ids`
in the same order as the chunks list, and the loader verifies this order by
comparing `chunk_ids` from the NPZ with `chunk_id` fields from chunks.jsonl.

### The row-order contract

This is the most important constraint in the entire persistence layer:

```python
# In the writer:
chunk_ids = np.array([c.chunk_id for c in chunks])
np.savez(..., embeddings=embeddings, chunk_ids=chunk_ids)

# In the loader — verified:
chunk_ids_jsonl = [c.chunk_id for c in chunks]
chunk_ids_npz = [str(cid) for cid in data["chunk_ids"]]
if chunk_ids_jsonl != chunk_ids_npz:
    raise ValueError("chunk_ids mismatch between chunks.jsonl and embeddings.npz")
```

If this order ever drifts, retrieval would return wrong chunks — chunk 3's
embedding row would be matched against chunk 7's text, and citations would be
lies. The loader catches this at read time rather than silently producing wrong
results.

---

## Round-Trip Integrity: The Most Important Test

`tests/test_persistence_roundtrip.py` (P1-T19) proves that save-then-load
doesn't corrupt retrieval:

```
1. Build an index from a fixture corpus using FakeEmbedder
2. Write it to disk with write_index()
3. Load it back with load_index()
4. Retrieve a query
5. Assert the same chunk is top-ranked
```

This is one test that covers four modules (writer, loader, retrieval, models)
and catches whole-class bugs: wrong serialization, order drift, missing fields,
or encoding issues. If this test passes, you can trust that the index format
works.

The test only uses the fixture corpus under `tests/fixtures/corpus/` — a few
small Markdown and plain text files checked into git. No `watsonxDocsQA`
download, no real embeddings, no network. This is intentional: the round-trip
test should always run, even on a fresh clone with zero setup.

---

## The Fake Backend Pattern

Phase 1 uses two fake backends: `FakeEmbedder` and `FakeGenerator`. Together
they let all 241 tests run without network, without API keys, and without
downloading an 80MB transformer model.

### Why this pattern works

Both fakes share the same design:

1. **Narrow interface.** `Embedder.embed()` and `Generator.generate()` are the
   only methods the pipeline calls. The fakes only need to satisfy these.
2. **Deterministic.** Same input always produces same output. This means tests
   can assert exact values, not just "close enough" ranges.
3. **Self-contained.** No external dependencies beyond Python's stdlib and
   NumPy. No downloads, no config files, no environment variables.

### What you can test with fakes

| Test concern | How the fake enables it |
|---|---|
| Chunk ID stability | FakeEmbedder doesn't affect IDs; IDs are hash-based |
| Chunk → embedding pipeline | FakeEmbedder.embed() called with correct texts |
| Retrieval ranking | FakeEmbedder produces known vectors → known ranking |
| Prompt assembly | FakeGenerator echoes source markers from prompt |
| Citation extraction | FakeGenerator includes markers → regex finds them |
| CLI command wiring | FakeEmbedder and FakeGenerator patched via monkeypatch |
| Index persistence | FakeEmbedder creates small deterministic NPZ files |
| Empty result handling | FakeGenerator's no-marker path tests abstention |

### What you can't test with fakes

| Limitation | Why |
|---|---|
| Embedding quality | Fake vectors are random hashes, not semantic representations |
| Real retrieval relevance | Cosine similarity on fake vectors is meaningless |
| Answer faithfulness | FakeGenerator doesn't read the context — it echoes markers |
| LLM prompt compliance | The fake doesn't follow groundedness instructions |
| Embedding model behavior | Only `SentenceTransformerEmbedder` tests cover this |

These gaps are intentional. Phase 1.6 (Evaluation Harness) will add real-corpus
tests with real embeddings to measure actual quality. The fake backends are for
**correctness** — does the code work? Real backends are for **quality** — does
it work well?

### How the CLI tests swap backends

CLI tests use pytest's `monkeypatch` to replace the factory functions:

```python
# In conftest or test file:
monkeypatch.setattr(
    "tiny_rag_lab.cli._make_embedder",
    lambda model_name=None: FakeEmbedder(dim=8)
)
monkeypatch.setattr(
    "tiny_rag_lab.cli._make_generator",
    lambda args: FakeGenerator()
)
```

This is why `cli.py` has `_make_embedder()` and `_make_generator()` as separate
functions rather than inlining the construction. It's a deliberate seam — a
point where the real implementation can be swapped for a test double without
changing the CLI logic.

---

## Validation in the Loader: Defense in Depth

`load_index()` validates more than just the row-order contract:

```python
# All three files must exist
for p in (manifest_path, chunks_path, embeddings_path):
    if not p.exists():
        raise FileNotFoundError(f"Index file not found: {p}")

# Embedding row count must match chunk count
if embeddings.shape[0] != len(chunks):
    raise ValueError(
        f"embeddings row count {embeddings.shape[0]} != chunk count {len(chunks)}"
    )
```

These checks catch common failure modes:

- **Missing file.** Someone ran `rag retrieve` before `rag index`, or deleted a
  file manually.
- **Row count mismatch.** The NPZ file was truncated during a failed write, or
  an old chunks.jsonl was paired with a new embeddings.npz.
- **ID order mismatch.** A bug in the writer stored chunk IDs out of order.

Each check produces a specific error message with the actual values, so you
know exactly what's wrong without guessing.

---

## What This Teaches

The persistence layer is about 150 lines (writer + loader combined), but it
teaches two important system-design concepts:

**Separate human-readable data from binary data.** chunks.jsonl is for humans
and simple tools; embeddings.npz is for machines. Merging them would make both
harder to use.

**Verify on load, not just on save.** The writer assumes correctness; the
loader proves it. Every invariant checked at load time is a bug caught before
it silently produces wrong answers. The three-file format with cross-validation
is a miniature case study in defensive I/O.

The fake backend pattern — narrow interface, deterministic implementation,
deliberate seams for testing — is the reason 241 tests can run in 3.6 seconds
on a fresh clone. It's a pattern that scales to any pipeline where real
backends are slow, expensive, or non-deterministic.
