# Current Task

Task:         P1-T17, P1-T18
Phase:        Phase 1
Spec:         docs/phases/phase-1-naive-classic-rag.md
Taskboard:    docs/phases/phase-1-taskboard.md
Owner:        claude
Status:       review
Review Result: signed_off
Reviewer:     codex
Last Updated: 2026-06-08
Updated By:   codex

## Findings From Last Review

- none

## Tests Reviewed

- `uv run pytest tests/test_openai_generator.py tests/test_cmd_ask.py --tb=short -q`: pass, 18 passed in 0.08s
- `uv run pytest --tb=short -q`: pass, 240 passed in 3.51s
- `uv run python -c "import openai; print(openai.__version__)"`: pass, 2.41.0
- `uv run rag ask --help`: pass
- manual fake-backed `cmd_index` + `cmd_ask`: pass, printed answer, source table, and timings

## Blocker

- none

## Notes

## Handoff

### Task Summary

**T17**: Added `OpenAIGenerator` to `tiny_rag_lab/generation.py`. Calls any
OpenAI-compatible chat completions endpoint. Credentials and endpoint are
injected at construction time via constructor args (priority) or SDK env vars.
`from openai import OpenAI` is deferred so the module imports cleanly.

**T18**: Implemented `cmd_ask` in `tiny_rag_lab/cli.py`. Runs the full pipeline
(embed → retrieve → assemble_prompt → generate), builds a `RagTrace`, and
prints answer + source table + stage timings. Added `--model`, `--api-key`,
`--base-url` flags to the `ask` subparser.

### Files Changed

- `tiny_rag_lab/generation.py`: added `OpenAIGenerator` class
- `tiny_rag_lab/cli.py`: implemented `cmd_ask`, added `_make_generator`, added `--model`/`--api-key`/`--base-url` flags to `ask` subparser
- `tests/test_openai_generator.py`: 8 tests — interface contract, API call correctness, constructor credential forwarding (all mocked)
- `tests/test_cmd_ask.py`: 10 tests — answer/source-table/timings output, source markers, top_k, new parser flags

### Design Decisions

- **`_make_generator(args)`**: same pattern as `_make_embedder` — factory tests
  can patch without touching the CLI interface or adding a `--fake` flag.
- **`_CITATION_RE` at module level**: shared regex for extracting `[Source: id]`
  from the answer to populate `RagTrace.citations`. Defined once in `cli.py`.
- **Separate embed/retrieve timing**: `embed_query` and `retrieve_by_vector`
  are called separately so each stage's latency is measured independently and
  stored in `RagTrace.latency_by_stage`.
- **`OpenAIGenerator` passes only non-None kwargs**: `api_key` and `base_url`
  are only forwarded if provided, letting the SDK read its own env vars when
  not set. This keeps the `openai.OpenAI()` call minimal.

### Tests Run

- `uv run pytest tests/test_openai_generator.py tests/test_cmd_ask.py --tb=short -q`: 18 passed
- `uv run pytest --tb=short -q`: 240 passed

### Known Gaps

- T17 has no integration test hitting a real provider (tests are fully mocked).
  That is intentional per spec: "Tests must not depend on provider credentials."

### Questions For Next Agent

- T19 (persistence round-trip test) and T20 (CLI test coverage) are both
  unblocked. T19 depends on T10/T11/T12 (all done). T20 depends on T13/T14/T18
  (all done). Either can go next.
