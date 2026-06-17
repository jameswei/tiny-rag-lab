# Current Task

Task:         P2.0-T02
Phase:        Phase 2.0 — Answer Quality Judging
Spec:         docs/phases/phase-2.0-answer-quality-judging.md
Taskboard:    docs/phases/phase-2.0-taskboard.md
Owner:        Claude Sonnet 4.6
Status:       done
Review Result: signed_off
Reviewer:     codex
Last Updated: 2026-06-17
Updated By:   codex

## Findings From Last Review

No remaining T02 findings.

## Latest Review Result

Signed off by codex on 2026-06-17.

## Tests Reviewed

- `uv run pytest tests/test_judge_openai.py --tb=short -q`: 1 skipped (env flag not set — correct)
- `uv run pytest --tb=short -q`: 576 passed, 2 skipped
- `uv run python -c "import tiny_rag_lab.judge; import sys; assert 'openai' not in sys.modules"`: OK
- `TINY_RAG_LAB_TEST_JUDGE=1 uv run pytest tests/test_judge_openai.py --tb=short -q`: 14 passed, 1 skipped

## Notes

### Files Changed

- `tiny_rag_lab/judge.py`: added `OpenAIJudge` (lazy openai import inside `judge()`),
  `_JUDGE_SYSTEM_PROMPT`, `_JUDGE_USER_TEMPLATE`
- `tiny_rag_lab/cli.py`: added `_make_judge(name, model, api_key, base_url)` and
  `_make_generator_from_flag(name, args)` factories
- `tests/test_judge_openai.py`: gated file — Gate 1 env flag before Gate 2 importorskip

### Design Decisions

- `OpenAIJudge.__init__` stores credentials only; no openai import until `judge()` is called
- `_make_judge("openai", ...)` falls back to `OPENAI_API_KEY` env var before raising
- `--model` is optional; defaults to `OpenAIJudge.DEFAULT_MODEL` ("gpt-4o-mini")
- `response_format={"type": "json_object"}` used for JSON mode; missing keys raise ValueError
