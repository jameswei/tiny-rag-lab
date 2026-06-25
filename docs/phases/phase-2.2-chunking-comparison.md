# Phase 2.2 T04: Fixed-Character vs Structural Chunking Walkthrough

This walkthrough uses a tiny dedicated fixture corpus to show the exact failure
Phase 2.2 is meant to make visible: a fixed-size character window can split one
necessary instruction across two chunks, while structural chunking keeps that
instruction together.

## Fixture Inputs

- Corpus: `tests/fixtures/chunking_corpus/`
- Diagnose cases: `tests/fixtures/failure/chunking_strategy_cases.jsonl`
- Query / case question: `Before bulk import, what timeout_ms and retry_mode should I set?`

The corpus contains:

1. `bulk_import_runbook.md` — the gold document with the correct instruction:
   `set timeout_ms to 15000 and set retry_mode to manual`
2. `dry_run_defaults.md` — a distractor document with the wrong settings:
   `set timeout_ms to 5000 and set retry_mode to automatic`

The comparison uses a deliberately tight budget:

- `--chunk-size 75`
- `--chunk-overlap 0`

At that size, fixed-character chunking slices the gold instruction across two
chunks, but structural chunking keeps the sentence whole.

## Build The Two Indices

```bash
uv run rag index \
  --corpus tests/fixtures/chunking_corpus \
  --index-dir .tiny-rag/fixed-chunking-demo \
  --chunk-size 75 \
  --chunk-overlap 0 \
  --chunking-strategy fixed_character

uv run rag index \
  --corpus tests/fixtures/chunking_corpus \
  --index-dir .tiny-rag/structural-chunking-demo \
  --chunk-size 75 \
  --chunk-overlap 0 \
  --chunking-strategy structural
```

## Compare `rag retrieve`

```bash
uv run rag retrieve \
  "Before bulk import, what timeout_ms and retry_mode should I set?" \
  --index-dir .tiny-rag/fixed-chunking-demo \
  --retriever bm25 \
  --top-k 2

uv run rag retrieve \
  "Before bulk import, what timeout_ms and retry_mode should I set?" \
  --index-dir .tiny-rag/structural-chunking-demo \
  --retriever bm25 \
  --top-k 2
```

Observed result on 2026-06-25:

| Index | Rank 1 doc | Rank 1 preview |
|---|---|---|
| fixed-character | `dry_run_defaults.md` | `Before bulk import dry runs, set timeout_ms to 5000 and` |
| structural | `bulk_import_runbook.md` | `Before bulk import, set timeout_ms to 15000 and set retry_mode to manual.` |

## Compare `rag diagnose`

```bash
uv run rag diagnose \
  --cases-file tests/fixtures/failure/chunking_strategy_cases.jsonl \
  --index-dir .tiny-rag/fixed-chunking-demo

uv run rag diagnose \
  --cases-file tests/fixtures/failure/chunking_strategy_cases.jsonl \
  --index-dir .tiny-rag/structural-chunking-demo
```

Observed result on 2026-06-25:

| Index | Diagnose summary |
|---|---|
| fixed-character | `Case fc010 expected=missing_evidence` → baseline/intervention both `missing_evidence`; report summary `Confirmed: 1` |
| structural | `Case fc010 expected=missing_evidence` → baseline/intervention both `no_failure`; report summary `Confirmed: 0` |

## Why This Example Matters

Nothing except the chunking strategy changes between the two indices:

- same corpus
- same `chunk_size`
- same `chunk_overlap`
- same query
- same `bm25` retriever
- same diagnose case

That makes the before/after difference attributable to chunk boundaries alone.
