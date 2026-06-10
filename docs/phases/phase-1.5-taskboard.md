# Phase 1.5 Taskboard

This file tracks Phase 1.5 implementation tasks, dependencies, ownership, and
status in one lightweight table.

The implementation contract is
`docs/phases/phase-1.5-retrieval-mechanics.md`. This taskboard must stay aligned
with that spec. The phase scope proposal was reviewed and signed off by Codex on
2026-06-10. Phase 1.5 was closed by Codex on 2026-06-11.

## Status Values

- `todo`: not started
- `in_progress`: actively being implemented
- `review`: implementation is ready for review and verification
- `blocked`: cannot proceed; blocker must be written in `Notes`
- `done`: reviewed, tested, and accepted

## Update Rules

- Set `Status` to `in_progress` before starting work.
- Set `Status` to `review` after implementation and local tests.
- Set `Status` to `done` only after review and required tests pass.
- The task owner must not mark their own task `done`; a different reviewing
  agent must sign off and make the `done` update.
- When marking `done`, record the reviewing agent and test result in `Notes`.
- Use `blocked` only with a concrete blocker in `Notes`.
- Keep `Owner` as an agent/person name or `unassigned`.
- Do not change task IDs after creation.
- Update `Notes` with skipped tests, setup limits, or follow-up work.

## Taskboard

| ID | Milestone | Task | Depends On | Status | Owner | Acceptance | Notes |
|---|---|---|---|---|---|---|---|
| P1.5-T01 | M1.5.0 | `rank_bm25` dependency + `BM25Retriever` in `bm25.py` | none | done | claude | `_tokenize` visible and tested; unique-term hit ranks first; empty corpus/query return `[]`; `uv run pytest tests/test_bm25.py --tb=short -q`: N passed | reviewed by codex; `uv run pytest tests/test_bm25.py tests/test_retrieval.py --tb=short -q`: 32 passed; signed off |
| P1.5-T02 | M1.5.1 | `reciprocal_rank_fusion()` + `retrieve_hybrid()` in `hybrid.py` | P1.5-T01 | done | claude | dual-top-1 chunk scores highest after fusion; hybrid returns top_k items; ranks 1-indexed contiguous; `uv run pytest tests/test_hybrid.py --tb=short -q`: N passed | reviewed by codex; `uv run pytest tests/test_hybrid.py tests/test_cmd_index_retrieve.py --tb=short -q`: 36 passed; signed off |
| P1.5-T03 | M1.5.2 | `--retriever` flag for `rag retrieve` | P1.5-T01, P1.5-T02 | done | claude | `--help` shows flag; bm25 and hybrid modes return results with fake-backed index; invalid value exits non-zero; `uv run pytest tests/test_cmd_index_retrieve.py --tb=short -q`: N passed | reviewed by codex; `uv run pytest tests/test_cmd_index_retrieve.py tests/test_hybrid.py --tb=short -q`: 36 passed; signed off |
| P1.5-T04 | M1.5.2 | `EvalReport.retriever` field + `--retriever` flag for `rag eval` | P1.5-T01, P1.5-T02 | done | claude | `EvalReport` has `retriever` field; `run_retrieval_eval()` accepts retriever param; `rag eval --retriever bm25` and `--retriever hybrid` print valid reports with retriever name; `uv run pytest tests/test_cmd_eval.py tests/test_eval_runner.py --tb=short -q`: N passed | reviewed by codex; `uv run pytest tests/test_eval_runner.py tests/test_cmd_eval.py tests/test_eval_metrics.py --tb=short -q`: 95 passed; signed off |
| P1.5-T05 | M1.5.3 | Phase close | P1.5-T01–T04 | done | codex | All P1.5-T01–T04 rows show `done` with reviewer sign-off; `docs/phases/README.md` updated; `uv run pytest --tb=short -q`: 319+ passed; all three `rag eval --retriever` modes exit 0 | closed by codex; `uv run pytest --tb=short -q`: 371 passed; `rag eval --retriever dense`, `bm25`, and `hybrid` exited 0 on fixture index |

## Review-Sensitive Tasks

These tasks require architecture or code review before being marked `done`:

- `P1.5-T01`: BM25 scoring correctness and tokenizer design.
- `P1.5-T02`: RRF fusion formula correctness and tie-breaking behavior.
- `P1.5-T04`: `EvalReport` field addition (data contract change).

## Minimum Phase 1.5 Completion

Minimum Phase 1.5 completion requires `P1.5-T01` through `P1.5-T05`.
