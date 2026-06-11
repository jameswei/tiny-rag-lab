# Phase 1.7 Taskboard

This file tracks Phase 1.7 implementation tasks, dependencies, ownership, and
status in one lightweight table.

The implementation contract is
`docs/phases/phase-1.7-observability.md`. This taskboard must stay aligned with
that spec. The phase scope proposal was reviewed and signed off by Codex on
2026-06-11. It becomes active only after owner acceptance and activation in
`docs/phases/README.md`.

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
| P1.7-T01 | M1.7.0 | `trace.py`: `ChunkTrace`, `RetrieveTrace`, `AskTrace` dataclasses | — | done | claude | `dataclasses.asdict()` round-trip for all three types; `json.dumps(dataclasses.asdict(trace))` produces valid JSON; all field types are JSON-native; `uv run pytest tests/test_trace.py -k dataclass --tb=short -q`: N passed | reviewed by codex; `uv run pytest tests/test_trace.py -k dataclass --tb=short -q`: 13 passed, 20 deselected; signed off |
| P1.7-T02 | M1.7.1 | `trace_to_dict()` and `write_trace_json(trace, path)` in `trace.py` | P1.7-T01 | done | claude | Written file is valid JSON; contains `retriever`, `top_k`, `chunks` list where each element has `rank`/`score`/`chunk_id`/`doc_id`/`title`/`path`; parent dirs created if missing; `uv run pytest tests/test_trace.py -k serial --tb=short -q`: N passed | reviewed by codex; `uv run pytest tests/test_trace.py -k serial --tb=short -q`: 10 passed, 23 deselected; signed off |
| P1.7-T03 | M1.7.2 | `format_retrieve_trace()` and `format_ask_trace()` in `trace.py` | P1.7-T01 | done | claude | `format_retrieve_trace` output contains query, retriever, top_k, latency keys, rank, score, doc_id; `format_ask_trace` output contains answer and citations; no ANSI escape codes (`\x1b` absent); `uv run pytest tests/test_trace.py -k format --tb=short -q`: N passed | reviewed by codex; `uv run pytest tests/test_trace.py -k format --tb=short -q`: 13 passed, 20 deselected; signed off |
| P1.7-T04 | M1.7.3 | `rag retrieve`: build `RetrieveTrace`, latency tracking, `--trace-out PATH` | P1.7-T01, P1.7-T02, P1.7-T03 | done | claude | `--trace-out /tmp/out.json` writes valid JSON trace with `load` and `retrieve` latency keys (plus `embed` for dense/hybrid); output uses `format_retrieve_trace` for all runs; BM25 runs omit `embed` key; `uv run pytest tests/test_cmd_retrieve.py --tb=short -q`: N passed | reviewed by codex; `uv run pytest tests/test_cmd_retrieve.py --tb=short -q`: 10 passed; `uv run pytest tests/test_cmd_index_retrieve.py tests/test_cmd_retrieve.py --tb=short -q`: 32 passed; `uv run pytest tests/test_trace.py tests/test_cmd_retrieve.py --tb=short -q`: 43 passed; signed off |
| P1.7-T05 | M1.7.4 | `rag ask`: build `AskTrace`, replace `RagTrace` usage, `--trace-out PATH` | P1.7-T01, P1.7-T02, P1.7-T03 | done | claude | `--trace-out /tmp/out.json` writes valid JSON with `prompt`, `answer`, `citations`, `latency_by_stage` keys `load`/`embed`/`retrieve`/`prompt_assembly`/`generate`; output uses `format_ask_trace`; `RagTrace` removed from `models.py` and its tests in `tests/test_models.py` updated or removed; `uv run pytest tests/test_cmd_ask.py tests/test_models.py --tb=short -q`: N passed | reviewed by codex; `uv run pytest tests/test_cmd_ask.py tests/test_models.py --tb=short -q`: 28 passed; `uv run pytest --tb=short -q`: 420 passed; signed off |
| P1.7-T06 | M1.7.5 | Phase close | P1.7-T01–T05 | done | codex | All P1.7-T01–T05 rows show `done` with reviewer sign-off; `docs/phases/README.md` updated; `uv run pytest --tb=short -q`: 319+ passed; `rag retrieve --help` and `rag ask --help` show `--trace-out` flag | closed by codex; spec coverage reviewed; `uv run pytest --tb=short -q`: 420 passed; `uv run rag retrieve --help`: shows `--trace-out`; `uv run rag ask --help`: shows `--trace-out`; signed off |

## Review-Sensitive Tasks

These tasks require architecture or code review before being marked `done`:

- `P1.7-T01`: data contracts and field types — schema is durable; Phase 1.8
  consumes these shapes.
- `P1.7-T02`: serialization correctness and parent-dir creation behavior.
- `P1.7-T04`: latency tracking correctness; BM25 omits embed stage.
- `P1.7-T05`: `RagTrace` removal from `models.py` and `tests/test_models.py`; ask output now uses `format_ask_trace`.

## Minimum Phase 1.7 Completion

Minimum Phase 1.7 completion requires `P1.7-T01` through `P1.7-T06`.
