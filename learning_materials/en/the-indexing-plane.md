# The Indexing Plane — Documents, Normalization, Chunking, Embeddings

The indexing plane turns raw files on disk into searchable vectors. Three
modules do this work: `documents.py` (load + clean), `chunking.py` (split into
pieces), and `embeddings.py` (turn text into numbers).

---

## Step 1: Loading Documents (`documents.py`)

### File discovery

The pipeline only accepts `.md` and `.txt` files:

```python
_SUPPORTED_SUFFIXES = {".md", ".txt"}
```

When you run `rag index --corpus corpus/watsonx-docsqa`, the loader walks the
entire directory tree with `corpus_root.rglob("*")`, picks every file whose
suffix is in the allowed set, sorts them by path (for deterministic order), and
loads each one. Sorting matters — without it, the chunk order in the index
could change between runs even if no files changed.

### What `load_document` does for each file

```python
raw_text = path.read_text(encoding="utf-8")
fmt = "markdown" if path.suffix.lower() == ".md" else "text"
```

Then it builds a `Document` with seven fields:

| Field | How it's set |
|---|---|
| `doc_id` | `path.relative_to(corpus_root).as_posix()` — clean relative path |
| `path` | Absolute path as a string |
| `title` | First `#` heading (Markdown) or filename stem |
| `format` | `"markdown"` or `"text"` |
| `raw_text` | Exact file bytes decoded as UTF-8 |
| `normalized_text` | `normalize_text(raw_text)` — cleaned version |
| `raw_hash` | `sha256(raw_text)` — 64 hex chars |

A subtle point: `raw_hash` is computed from `raw_text`, not `normalized_text`.
This means the hash captures the file exactly as it exists on disk. If you
change a file's line endings or trailing whitespace, the hash changes — you'll
know the source drifted even if the normalization step produces the same
`normalized_text`.

---

## Step 2: Normalizing Text

Normalization lives in the same file so the whole text-preparation story is
visible in one place. Four rules, applied in order:

```python
# Rule 1: Normalize line endings
text = text.replace("\r\n", "\n").replace("\r", "\n")

# Rule 2: Strip trailing whitespace from each line
lines = [line.rstrip() for line in text.split("\n")]

# Rule 3: Collapse runs of >2 blank lines to exactly 2
for line in lines:
    if line == "":
        blank_run += 1
        if blank_run <= 2:
            result.append(line)
    else:
        blank_run = 0
        result.append(line)

# Rule 4: Preserve Markdown headings and punctuation (implicit — we don't
# modify any non-whitespace characters)
```

### Why each rule exists

**Rule 1 — line endings.** A corpus prepared on Windows (`\r\n`) and indexed
on macOS (`\n`) should produce identical chunks. Without this, character
offsets drift.

**Rule 2 — trailing whitespace.** Trailing spaces are invisible noise. They
widen chunks without adding information and can cause chunk boundaries to land
differently across editors.

**Rule 3 — blank line collapse.** A document might have sections separated by 5
blank lines (readable for humans) or 0 blank lines (compact). Collapsing to ≤2
blank lines gives the chunker consistent paragraph boundaries without removing
the visual separation.

**Rule 4 — preserve content.** We never touch actual text characters.
Headings, code blocks, citations, and special punctuation all survive
untouched. This is a character-level pipeline — no tokenization, no parsing.

### Before and after


Before (exaggerated for clarity):
```
# Introduction\r\n\r\n\r\n\r\nFirst paragraph.   \r\n\r\nSecond.\r\n\r\n\r\n\r\nThird.
```

After:
```
# Introduction

First paragraph.

Second.

Third.
```

The normalized version is shorter, more predictable, and ready for chunking.

---

## Step 3: Character Chunking (`chunking.py`)

### The sliding window algorithm

`chunk_document` takes a document's `normalized_text` and walks through it with
a sliding window:

```python
step = chunk_size - chunk_overlap  # e.g. 800 - 120 = 680

start = 0
while start < len(text):
    end = min(start + chunk_size, len(text))
    chunk_text = text[start:end]

    if chunk_text.strip():          # skip empty/whitespace-only windows
        chunks.append(Chunk(...))

    if end == len(text):
        break                       # reached the end — stop

    start += step
```

With defaults (`chunk_size=800`, `chunk_overlap=120`), the step size is 680
characters. This means each chunk shares 120 characters with the next one.

### Why overlap matters

Imagine a document where a key fact spans characters 790–810 — right across the
boundary of two non-overlapping chunks. With overlap, that fact appears in the
tail of chunk 1 (chars 680–800 don't reach it, but wait) and the head of
chunk 2. More concretely:

```
Chunk 1: chars [0,   800)
Chunk 2: chars [680, 1480)   ← overlaps with chunk 1 by 120 chars
Chunk 3: chars [1360, 2160)
```

The 120-character overlap means any sentence that starts near the end of one
chunk will also start the next one — the retrieval system gets two chances to
find it.

The trade-off: more overlap means more chunks (which means a larger index and
slower retrieval), but fewer "split in the middle" misses. Phase 1 uses 120 as
a reasonable default; Phase 1.5 will make this configurable for experiments.

### The slice invariant in action

Every chunk stores `char_start` and `char_end` — Python-style half-open
offsets into `normalized_text`. The test in `test_chunking.py` verifies:

```python
assert document.normalized_text[chunk.char_start:chunk.char_end] == chunk.text
```

This is true by construction — the chunker literally slices `text[start:end]` —
but verifying it in tests catches regressions if someone later adds
normalization steps between chunking and storage.

### Edge cases handled

**Small documents.** If a document is shorter than `chunk_size`, you get one
chunk containing the whole text — `end = min(start + chunk_size, len(text))`
handles this naturally. The loop then sees `end == len(text)`, produces the
chunk, and breaks.

**Whitespace-only chunks.** `if chunk_text.strip()` skips windows that contain
only spaces, tabs, or newlines. This prevents the index from filling up with
noise that can never be retrieved usefully.

**The tail window.** When the final window ends exactly at `len(text)`, the
loop breaks immediately — it does not slide forward to create a redundant tail
chunk wholly contained in the current window.

### chunk_size vs chunk_overlap invariant

The implementation enforces:

```python
if chunk_overlap >= chunk_size:
    raise ValueError(...)
```

If overlap equals or exceeds chunk size, the step becomes zero or negative, and
the sliding window never advances. This is caught at the start so the error
message is clear.

---

## Step 4: Embeddings (`embeddings.py`)

### The interface contract

Every embedder promises one thing:

```python
class Embedder(ABC):
    def embed(self, texts: list[str]) -> np.ndarray:
        # Returns shape (len(texts), dim), dtype float32
```

The retrieval code never checks which class produced the vectors. It only
cares about the shape. This is the **interface boundary** between the indexing
plane and the retrieval plane.

### FakeEmbedder — deterministic and free

The fake embedder turns any text into a unit vector using a hash:

```python
seed = int.from_bytes(sha256(text).digest()[:4], byteorder="little")
rng = np.random.default_rng(seed)
vec = rng.standard_normal(dim).astype(np.float32)
vec = vec / np.linalg.norm(vec)  # L2-normalize to unit length
```

Key properties:
- **Same text → same vector.** Always. The hash is deterministic, and NumPy's
  `default_rng(seed)` is reproducible across runs.
- **Different texts → uncorrelated vectors.** SHA-256's avalanche effect means
  even "hello" and "hello!" produce unrelated seeds.
- **Vectors are unit length.** L2-normalization makes cosine similarity work as
  a simple dot product — no division by norms needed at query time.
- **No downloads, no network, no GPU.** Pure Python + NumPy. All 241 Phase 1
  tests use this embedder.

### SentenceTransformerEmbedder — the real thing

```python
class SentenceTransformerEmbedder(Embedder):
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name=..., local_files_only=False):
        self._model = SentenceTransformer(model_name, local_files_only=...)
```

A few design notes:

**Why `all-MiniLM-L6-v2`?** It's one of the smallest viable embedding models
(384 dimensions, ~80MB), runs on CPU, and produces reasonable semantic vectors.
For a learning lab, small and local beats large and API-dependent.

**`normalize_embeddings=True`** is set in the `encode` call:

```python
vecs = self._model.encode(
    texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
)
```

This makes the real embedder's output consistent with the fake embedder's —
both produce unit vectors, so the same retrieval code works with either.

**`dim` is dynamically discovered** from the loaded model via
`get_embedding_dimension()` rather than hardcoded. If you later switch to a
different model, the dimension updates automatically.

**Network is needed once.** On first use, the model weights download from
Hugging Face and cache locally. After that, all subsequent runs are offline.

---

## What This Teaches

The indexing plane has three clean stages: load → chunk → embed. Each stage has
a single function that does one thing well. The contracts between them are
simple types (`Document`, `Chunk`, `np.ndarray`) — no callbacks, no registries,
no framework wiring. This is what "readable code" means in practice: you can
open `documents.py`, read `load_document` top to bottom, and understand exactly
what happens to one file.

The design choice to keep the embedder behind an ABC means the same retrieval
code works with a 10-line hash function (tests) and an 80MB transformer model
(production). This pattern — a narrow interface plus a deterministic fake —
recurs throughout Phase 1 and makes the entire pipeline testable without
network access.
