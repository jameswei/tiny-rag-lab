# Current Task

Task:         P1.5-T04
Phase:        Phase 1.5 — Retrieval Mechanics
Spec:         docs/phases/phase-1.5-retrieval-mechanics.md
Taskboard:    docs/phases/phase-1.5-taskboard.md
Owner:        claude
Status:       done
Review Result: signed_off
Reviewer:     codex
Last Updated: 2026-06-11
Updated By:   codex

## Findings From Last Review

- none

## Tests Reviewed

- `uv run pytest tests/test_eval_runner.py tests/test_cmd_eval.py tests/test_eval_metrics.py --tb=short -q`: 95 passed
- Direct invalid-retriever probe with empty samples: raised `ValueError`

## Blocker

- none

## Notes

- P1.5-T04 signed off by Codex. `EvalReport` records `retriever`, `run_retrieval_eval()` supports dense/BM25/hybrid with direct invalid-value validation, BM25 eval works without an embedder, dense/hybrid reject missing embedders, hybrid injects the pre-built BM25 retriever, and `rag eval --retriever` prints reports with the retriever name.
