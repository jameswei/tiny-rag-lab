# Current Task

Task:         P2.2-T05
Phase:        Phase 2.2
Spec:         docs/phases/phase-2.2-structural-semantic-chunking.md
Taskboard:    docs/phases/phase-2.2-taskboard.md
Owner:        unassigned
Status:       todo
Review Result: pending
Reviewer:     
Last Updated: 2026-06-25
Updated By:   GitHub Copilot CLI

## Findings From Last Review

- none

## Tests Reviewed

- `uv run rag index --corpus tests/fixtures/chunking_corpus --index-dir .tiny-rag/fixed-chunking-demo --chunk-size 75 --chunk-overlap 0 --chunking-strategy fixed_character` — passed
- `uv run rag index --corpus tests/fixtures/chunking_corpus --index-dir .tiny-rag/structural-chunking-demo --chunk-size 75 --chunk-overlap 0 --chunking-strategy structural` — passed
- `uv run rag retrieve 'Before bulk import, what timeout_ms and retry_mode should I set?' --index-dir .tiny-rag/fixed-chunking-demo --retriever bm25 --top-k 2` — passed; rank 1 was `dry_run_defaults.md`
- `uv run rag retrieve 'Before bulk import, what timeout_ms and retry_mode should I set?' --index-dir .tiny-rag/structural-chunking-demo --retriever bm25 --top-k 2` — passed; rank 1 was `bulk_import_runbook.md`
- `uv run rag diagnose --cases-file tests/fixtures/failure/chunking_strategy_cases.jsonl --index-dir .tiny-rag/fixed-chunking-demo` — passed; case `fc010` reproduced `missing_evidence`
- `uv run rag diagnose --cases-file tests/fixtures/failure/chunking_strategy_cases.jsonl --index-dir .tiny-rag/structural-chunking-demo` — passed; case `fc010` became `no_failure`
- `git diff -- tiny_rag_lab/failure.py tiny_rag_lab/eval.py` — passed; no diff
- temporary-index reproduction under `/tmp/tiny-rag-t04-review-0Uyhor` — passed; fixed-character rank 1 was `dry_run_defaults.md`, structural rank 1 was `bulk_import_runbook.md`; fixed diagnose confirmed `missing_evidence`, structural diagnose returned `no_failure`
- `uv run pytest --tb=short -q` — passed (746 passed, 7 skipped)

## Blocker

- none

## Notes

- Codex review signed off T04. The new fixture corpus, diagnose case, and
  walkthrough stay within the fixture/docs-only scope and reproduce the
  documented fixed-character versus structural behavior with the existing CLI.

## Handoff

### Task Summary

Completed P2.2-T04 by adding a dedicated fixture corpus, a matching diagnose
case, and a phase-specific walkthrough that demonstrates a real before/after
difference between fixed-character and structural chunking using only existing
CLI commands.

### Files Changed

- `tests/fixtures/chunking_corpus/bulk_import_runbook.md`: gold document whose
  key instruction is split by fixed-character chunking at `chunk_size=75`
- `tests/fixtures/chunking_corpus/dry_run_defaults.md`: distractor document
  with wrong settings that can outrank the split gold instruction under BM25
- `tests/fixtures/failure/chunking_strategy_cases.jsonl`: one reproduceable
  diagnose case (`fc010`) using `bm25` + `top_k=1`
- `docs/phases/phase-2.2-chunking-comparison.md`: reproducible `rag index` /
  `rag retrieve` / `rag diagnose` walkthrough with observed results

### Design Decisions

- Kept the chunking demo in a dedicated fixture corpus so the long-standing
  `tests/fixtures/corpus/` counts and semantics stay unchanged.
- Used the same corpus, same query, same `bm25` retriever, same `top_k`, and
  the same diagnose case across both indices so the observed difference is
  attributable to chunk boundaries alone.
- Used `expected_label="missing_evidence"` in the diagnose case because that is
  the fixed-character failure we want to reproduce; running the same case
  against the structural index then shows the failure no longer reproduces.

### Tests Run

- `uv run rag index --corpus tests/fixtures/chunking_corpus --index-dir .tiny-rag/fixed-chunking-demo --chunk-size 75 --chunk-overlap 0 --chunking-strategy fixed_character`: passed
- `uv run rag index --corpus tests/fixtures/chunking_corpus --index-dir .tiny-rag/structural-chunking-demo --chunk-size 75 --chunk-overlap 0 --chunking-strategy structural`: passed
- `uv run rag retrieve 'Before bulk import, what timeout_ms and retry_mode should I set?' --index-dir .tiny-rag/fixed-chunking-demo --retriever bm25 --top-k 2`: passed
- `uv run rag retrieve 'Before bulk import, what timeout_ms and retry_mode should I set?' --index-dir .tiny-rag/structural-chunking-demo --retriever bm25 --top-k 2`: passed
- `uv run rag diagnose --cases-file tests/fixtures/failure/chunking_strategy_cases.jsonl --index-dir .tiny-rag/fixed-chunking-demo`: passed
- `uv run rag diagnose --cases-file tests/fixtures/failure/chunking_strategy_cases.jsonl --index-dir .tiny-rag/structural-chunking-demo`: passed
- `uv run pytest --tb=short -q`: passed (746 passed, 7 skipped)

### Known Gaps

- none

### Learning Notes

- The important subtlety is that `rag diagnose` still works at the document-hit
  level, so the fixture needs a distractor document strong enough to outrank the
  split gold chunks under `fixed_character`; a same-doc partial-chunk failure
  alone would not surface in diagnose metrics.

### Questions For Next Agent

- Please verify the task stayed within scope: fixtures/docs only, with no
  production-code changes and no edits to `failure.py`, `eval.py`,
  `FailureCase`, or `RetrieverConfig`.
