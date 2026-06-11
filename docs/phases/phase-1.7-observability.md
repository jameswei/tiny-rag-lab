# Phase 1.7 Spec: Observability and Debugging

**Status:** Scope signed off by Codex on 2026-06-11; not active until owner activation in `docs/phases/README.md`
**Authors:** Claude Code + owner decisions
**Based on:** `docs/roadmap.md`, `docs/phases/phase-1.6-evaluation-harness.md`
**Taskboard:** `docs/phases/phase-1.7-taskboard.md`
**Date:** 2026-06-11

---

## Goal

Make each single RAG run explainable by adding trustworthy trace records to the
`retrieve` and `ask` flows. The trace schema established here serves as the
stable data contract that Phase 1.8 (RAG failure lab) will consume.

---

## Scope

### In scope

- `ChunkTrace`, `RetrieveTrace`, `AskTrace` dataclasses in a new
  `tiny_rag_lab/trace.py`
- Trace serialization: `trace_to_dict()` + `write_trace_json(trace, path)`
- Human-readable trace formatters: `format_retrieve_trace()` and
  `format_ask_trace()`
- Latency tracking by stage in both `retrieve` and `ask` flows
- `--trace-out PATH` optional flag on `rag retrieve` and `rag ask`
- Updated `cmd_retrieve` and `cmd_ask` output via the new formatters
- Unit tests for dataclasses, serialization, and formatters
- CLI tests for `--trace-out` on both commands

### Out of scope for Phase 1.7

- `rag eval --trace-out` — eval-run artifact storage is deferred
- `rag index` traces — not in roadmap
- Failure classification taxonomy — Phase 1.8
- Token budget estimation — requires model-specific tokenizer, deferred
- Comparison UI or multi-run reporting — deferred
- Answer-quality metrics (faithfulness, correctness) — deferred

---

## Design Decision: `RagTrace` in `models.py`

`RagTrace` currently lives in `models.py` and will be removed. All trace types
(`ChunkTrace`, `RetrieveTrace`, `AskTrace`) live in the new `tiny_rag_lab/trace.py`.
`RagTrace` is replaced by `AskTrace`. The single internal import in `cli.py`
(`from tiny_rag_lab.models import RagTrace`) and the existing `RagTrace` tests in
`tests/test_models.py` are updated or removed as part of T05.

This mirrors the `eval.py` precedent: observability types belong in an
observability module, not in the core data-contract module (`models.py` stays
focused on `Document`, `Chunk`, and `RetrievalResult`).

---

## Data Contracts

All types live in `tiny_rag_lab/trace.py`. All fields are JSON-native types
(`str`, `int`, `float`, `list`, `dict`, `bool`) so `dataclasses.asdict()` +
`json.dumps()` serializes the full trace without a custom encoder.

```python
@dataclass
class ChunkTrace:
    """Compact, JSON-safe summary of one retrieved chunk.

    text_preview is the first 120 characters of chunk.text. It is not in the
    roadmap's enumerated fields but is needed for the human-readable formatter
    without embedding full chunk text in the JSON trace.
    """
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
    """Full trace record for one rag retrieve call.

    latency_by_stage keys: "load", "embed" (dense and hybrid only), "retrieve".
    BM25-only runs omit "embed".
    """
    query: str
    retriever: str            # "dense" | "bm25" | "hybrid"
    top_k: int
    chunks: list[ChunkTrace]
    latency_by_stage: dict[str, float]
```

```python
@dataclass
class AskTrace:
    """Full trace record for one rag ask call. Replaces models.RagTrace.

    latency_by_stage keys: "load", "embed", "retrieve", "prompt_assembly",
    "generate".
    """
    query: str
    retriever: str            # currently always "dense"
    top_k: int
    chunks: list[ChunkTrace]
    prompt: str
    answer: str
    citations: list[str]
    latency_by_stage: dict[str, float]
```

All three dataclasses live in `tiny_rag_lab/trace.py`.

---

## Serialization

```python
def trace_to_dict(trace: RetrieveTrace | AskTrace) -> dict:
    """Convert a trace to a plain dict suitable for json.dumps.

    Uses dataclasses.asdict() which recurses into nested dataclasses.
    All field types are JSON-native so no custom encoder is needed.
    """

def write_trace_json(trace: RetrieveTrace | AskTrace, path: Path) -> None:
    """Serialize trace to JSON and write to path.

    Creates parent directories if they do not exist.
    Writes UTF-8 with indent=2 for human readability.
    """
```

---

## Human-Readable Formatters

```python
def format_retrieve_trace(trace: RetrieveTrace) -> str:
    """Return a plain-text debug view of a retrieve trace.

    Example:
      Retrieve trace
      ────────────────────────────────────────
        query     : "how to deploy a model"
        retriever : dense
        top_k     : 5
        latency   : load=0.051s  embed=0.012s  retrieve=0.003s
      ────────────────────────────────────────
      Rank 1  score=0.8432  chunk_id=abc123def456...
        doc_id  : docs/deploy.md
        title   : Deploying Models
        path    : /corpus/docs/deploy.md
        preview : First 120 chars of chunk text...
      ...
    """

def format_ask_trace(trace: AskTrace) -> str:
    """Return a plain-text debug view of an ask trace.

    Sections: header (query/retriever/top_k/latency), retrieved chunks,
    answer, citations.
    No ANSI escape codes.
    """
```

Formatters must not emit ANSI escape codes. Values are printed with 4 decimal
places for scores and 3 decimal places for latencies.

---

## CLI Changes

### `rag retrieve`

New flag:

```
--trace-out PATH   write JSON trace to PATH (optional)
```

`cmd_retrieve` is updated to:
1. Track `load`, `embed` (dense/hybrid only), and `retrieve` latency with
   `time.perf_counter()`.
2. Build a `RetrieveTrace` from the results.
3. Print via `format_retrieve_trace(trace)` — this replaces the current
   ad-hoc loop. The formatter covers the same fields (rank, score, chunk_id,
   title, path, preview) plus the latency header.
4. If `--trace-out` is set, call `write_trace_json(trace, Path(args.trace_out))`.

### `rag ask`

New flag:

```
--trace-out PATH   write JSON trace to PATH (optional)
```

`cmd_ask` is updated to:
1. Track `load`, `embed`, `retrieve`, `prompt_assembly`, and `generate` latency
   with `time.perf_counter()`.
2. Build an `AskTrace` instead of a `RagTrace`.
3. Print via `format_ask_trace(trace)` — this replaces the current ad-hoc
   print statements. The formatter covers answer, source chunks, citations, and
   latency (the same information as before, structured as a trace).
4. If `--trace-out` is set, call `write_trace_json(trace, Path(args.trace_out))`.

The `--trace-out` flag is optional. Omitting it still produces the full
human-readable trace output from `format_ask_trace`.

---

## Required Tests

**Trace unit tests** (`tests/test_trace.py`):

Dataclass tests:
- `ChunkTrace` round-trips through `dataclasses.asdict()` with correct field values
- `RetrieveTrace` serializes to valid JSON containing `retriever`, `top_k`, `chunks`
- `AskTrace` serializes to valid JSON containing `prompt`, `answer`, `citations`

Serialization tests:
- `write_trace_json` writes a file; `json.loads(file.read_text())` succeeds
- Written dict contains `chunks` list where each element has `rank`, `score`,
  `chunk_id`, `doc_id`, `title`, `path`

Formatter tests:
- `format_retrieve_trace` output contains query, retriever name, top_k,
  `load` and `retrieve` latency values, rank, score, doc_id
- `format_ask_trace` output contains query, answer, citation strings
- Neither formatter emits ANSI codes (no `\x1b` in output)

**CLI tests** (`tests/test_cmd_retrieve.py`, `tests/test_cmd_ask.py`):

- `rag retrieve "q" --trace-out /tmp/r.json` writes valid JSON trace
- Written retrieve trace contains `retriever`, `top_k`, `chunks` array
- `rag retrieve "q"` (no `--trace-out`) prints `format_retrieve_trace` output
- `rag ask "q" --trace-out /tmp/a.json` writes valid JSON trace
- Written ask trace contains `prompt`, `answer`, `citations`, `latency_by_stage`
  with keys `load`, `embed`, `retrieve`, `prompt_assembly`, `generate`
- `rag ask "q"` (no `--trace-out`) prints `format_ask_trace` output containing
  answer, chunk list, and latency
- FakeEmbedder and FakeGenerator used — no model downloads or API credentials

---

## Acceptance Criteria

Phase 1.7 is complete when:

1. `rag retrieve "q" --trace-out out.json` writes a valid JSON file with
   `retriever`, `top_k`, `chunks` (each with rank, score, chunk_id, doc_id,
   title, path), and `latency_by_stage`.
2. `rag ask "q" --trace-out out.json` writes a valid JSON file with all retrieve
   fields plus `prompt`, `answer`, `citations`, and `latency_by_stage`.
3. Both commands print human-readable trace output via the formatters whether
   or not `--trace-out` is set. `--trace-out` only controls whether a JSON
   file is also written.
4. All trace types, serialization functions, and formatters have tests with
   known inputs and outputs.
5. Tests pass without network access, model downloads, or API credentials.
6. The full test suite passes without regression.

---

## Schema Stability Note

The `ChunkTrace`, `RetrieveTrace`, and `AskTrace` field sets are intentionally
minimal. Phase 1.8 will read these JSON traces to label failure modes. Any field
added or renamed in Phase 1.7 should be treated as a durable contract.
