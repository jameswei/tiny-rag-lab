# Phase 2.1 Taskboard

This file tracks Phase 2.1 implementation tasks, dependencies, ownership, and
status in one lightweight table.

The implementation contract is
`docs/phases/phase-2.1-context-budget-structured-answers.md`.

## Status Values

- `todo`: not started
- `in_progress`: actively being implemented
- `review`: implementation is ready for review and verification
- `blocked`: cannot proceed; blocker must be written in `Notes`
- `done`: reviewed, tested, and accepted

## Update Rules

- Set `Status` to `in_progress` before starting work.
- Set `Status` to `review` after implementation and local tests pass.
- Set `Status` to `done` only after review and required tests pass.
- The task owner must not mark their own task `done`; a different reviewing
  agent must sign off and make the `done` update.
- When marking `done`, record the reviewing agent and test result in `Notes`.
- Use `blocked` only with a concrete blocker in `Notes`.
- Keep `Owner` as an agent/person name or `unassigned`.
- Do not change task IDs after creation.
- Keep `Notes` concise. Detailed handoff notes, review findings, and full test
  evidence belong in `CURRENT.md`.

## Taskboard

| ID | Milestone | Task | Depends On | Status | Owner | Acceptance | Notes |
|---|---|---|---|---|---|---|---|
| P2.1-T01 | M2.1.0 | `context.py` (new): `PROMPT_OVERHEAD = 100` constant; `ContextPackResult` dataclass (selected, omitted, estimated_tokens, budget, counter_name — all JSON-native); `TokenCounter` protocol (name: str, count(text) -> int); `FakeTokenCounter(tokens_per_char=0.25)` — returns `int(len(text) * tokens_per_char)`; `TiktokenCounter(model="gpt-4o-mini")` — lazy-imports tiktoken, raises `ImportError` if absent; `pack_context(results, budget, counter, question="") -> ContextPackResult` — reuses `CONTEXT_BLOCK_TEMPLATE` from `prompting.py` for formatting blocks so token counts match actual prompt; deducts `PROMPT_OVERHEAD + counter.count(question)` from budget, greedily selects chunks in rank order; raises `ValueError` if budget < 0. Add `tiktoken` to `pyproject.toml` optional extras. Tests in `tests/test_context.py`. | — | done | Claude Code | `FakeTokenCounter().count(text)` returns `int(len(text) * 0.25)`; `pack_context` block counting matches `CONTEXT_BLOCK_TEMPLATE` format; `pack_context` with large budget selects all chunks; `pack_context` with tight budget omits lowest-ranked chunks; question token deduction verified; `pack_context(budget=-1, ...)` raises `ValueError`; `ContextPackResult` round-trips via `dataclasses.asdict()`; TiktokenCounter tests gated with `pytest.importorskip("tiktoken")`; reviewed by Codex 2026-06-24; `uv run pytest tests/test_context.py --tb=short -q`: 18 passed, 5 skipped; `uv run pytest --tb=short -q`: 667 passed, 7 skipped |
| P2.1-T02 | M2.1.1 | `trace.py`: add `context_pack: ContextPackResult | None = None` to `AskTrace` (default `None`); `format_ask_trace` appends context packing block after sources section when `trace.context_pack is not None` — shows budget, counter_name, selected count + token estimate, omitted count + chunk_ids; verify `trace_to_dict()` round-trip via `dataclasses.asdict()`. | P2.1-T01 | done | Claude Code | `AskTrace(context_pack=None)` serialises `"context_pack": null`; `AskTrace(context_pack=ContextPackResult(...))` serialises all pack fields; `format_ask_trace` without pack: no `"Context packing"` line; with pack: contains `"Context packing"`, `"Selected"`, `"Omitted"` lines; reviewed by Codex 2026-06-24; `uv run pytest tests/test_trace.py --tb=short -q`: 56 passed; `uv run pytest --tb=short -q`: 679 passed, 7 skipped |
| P2.1-T03 | M2.1.2 | `cli.py`: add `_make_token_counter() -> TokenCounter` — tries `TiktokenCounter()`, falls back to `FakeTokenCounter()` on `ImportError`; add `--context-budget INT` (default `0` = unlimited, must be >= 0) and `--output-format text\|json` (default `text`) to `cmd_ask`; validate budget >= 0, raise if negative; pipeline: when `context_budget > 0`, call `pack_context(results, context_budget, counter, question=query)`, filter results to selected chunk_ids, then call `assemble_prompt` on filtered list, attach `pack_result` to `AskTrace.context_pack`; when `context_budget == 0`, skip packing (`context_pack=None`); when `--output-format json`, print `json.dumps(trace_to_dict(trace), indent=2)` instead of `format_ask_trace(trace)`. | P2.1-T01, P2.1-T02 | done | Claude Code | Default `cmd_ask` output identical to Phase 2.0 (no packing); `cmd_ask --context-budget 8192 --output-format text` exits 0 and trace shows `"Context packing"` block; `cmd_ask --context-budget 500 --generator fake` answer has fewer source markers than `--context-budget 8192 --generator fake` on same query, proving budget filtered chunks; `cmd_ask --context-budget 8192 --output-format json` exits 0 and stdout is valid JSON with `"answer"` and `"context_pack"` keys; `cmd_ask --context-budget -1` exits non-zero with clear error; `--output-format json --judge fake --generator fake --context-budget 8192` includes verdict in JSON; reviewed by Codex 2026-06-24; `uv run pytest tests/test_cmd_ask.py --tb=short -q`: 45 passed; `uv run pytest --tb=short -q`: 695 passed, 7 skipped |
| P2.1-T04 | M2.1.3 | `eval.py`: `run_answer_eval` gains `counter: TokenCounter | None = None` and `context_budget: int | None = None`; when both non-None and `context_budget > 0`, apply `pack_context` per sample before `assemble_prompt`; when counter is None or budget is 0/None, skip packing (Phase 2.0 behaviour); validate budget >= 0, raise if negative. `failure.py`: same optional parameters added to `run_answer_diagnosis`. `cli.py`: add `--context-budget INT` (default `0` = unlimited, must be >= 0) to `cmd_eval` and `cmd_diagnose`; validate budget >= 0; pass `counter=_make_token_counter()` and `context_budget` when budget > 0, else `counter=None`. | P2.1-T01, P2.1-T02 | todo | unassigned | Default `cmd_eval` and `cmd_diagnose` output identical to Phase 2.0; `run_answer_eval(counter=None)` or `run_answer_eval(context_budget=0)` identical to Phase 2.0; `run_answer_eval(counter=FakeTokenCounter(), context_budget=200)` with typical multi-chunk retrieval: some chunks omitted; answer text contains fewer source markers than `context_budget=8192` on same eval set; `run_answer_eval(context_budget=-1, ...)` raises `ValueError`; `cmd_eval --context-budget 8192 --judge fake --generator fake` exits 0; `cmd_eval --context-budget 500 --judge fake --generator fake` produces shorter answers; `cmd_eval --context-budget -1` exits non-zero; `cmd_diagnose --context-budget 8192 --judge fake` exits 0; `cmd_diagnose --context-budget -1` exits non-zero; all existing fc001–fc009 retrieval and answer tests pass; `uv run pytest tests/test_eval_runner.py tests/test_cmd_eval.py tests/test_cmd_diagnose.py --tb=short -q`: N passed | |
| P2.1-T05 | M2.1.4 | Phase close: update `docs/phases/README.md` to mark Phase 2.1 complete and set Current Phase to "No active phase"; update `docs/roadmap.md` Phase 2.1 entry to "Complete"; run full suite and CLI smokes; verify `import tiny_rag_lab.context` does not import tiktoken unless tiktoken is installed. | P2.1-T01–T04 | todo | unassigned | All P2.1-T01–T04 `done` with reviewer sign-off; `uv run pytest --tb=short -q`: all passed; `uv run rag ask --help` shows `--context-budget` and `--output-format` flags; `uv run rag eval --help` shows `--context-budget`; `uv run rag diagnose --help` shows `--context-budget`; `python -c "import tiny_rag_lab.context; import sys; assert 'tiktoken' not in sys.modules"` succeeds; phase index and roadmap updated | |

## Review-Sensitive Tasks

- **P2.1-T01**: `ContextPackResult` must be entirely JSON-native (lists of
  strings, int, str) so `dataclasses.asdict()` serialises a full `AskTrace`
  without a custom encoder. **CRITICAL:** `pack_context` must use the exact
  same block formatting as `CONTEXT_BLOCK_TEMPLATE` from `prompting.py` so
  token counts match the actual prompt; otherwise budget estimates will drift.
  `pack_context` must deduct question tokens before chunk selection, not after.
  `pack_context(budget=-1, ...)` must raise `ValueError`. `TiktokenCounter.__init__`
  must not call the tiktoken API beyond loading the encoding.
- **P2.1-T02**: `format_ask_trace` must not break when `context_pack=None` —
  the Phase 2.0 output must be byte-for-byte identical for the `--context-budget
  0` path.
- **P2.1-T03**: Default behavior (no flags) must be identical to Phase 2.0.
  `--context-budget` must default to `0` (unlimited). With tight budget (e.g.
  500 tokens), `FakeGenerator` answers must contain fewer source markers than
  high budget, proving chunks were omitted. `--context-budget -1` must exit
  with clear error. `--output-format json` must write to stdout, not stderr.
  `--trace-out` must still write JSON even when `--output-format text`.
  `_make_token_counter()` must not raise even when tiktoken is absent — it
  silently falls back.
- **P2.1-T04**: `--context-budget` defaults to `0` (unlimited). Existing
  `run_answer_eval` callers (tests) must continue to work without passing
  `counter` or `context_budget`. The new parameters must have defaults that
  reproduce Phase 2.0 behaviour exactly. With `FakeGenerator` and tight budget,
  generated answers must have fewer source markers than high budget, proving
  packing was effective. `run_answer_eval(context_budget=-1, ...)` must raise
  `ValueError`.

## Minimum Phase 2.1 Completion

Minimum Phase 2.1 completion requires P2.1-T01 through P2.1-T05.
