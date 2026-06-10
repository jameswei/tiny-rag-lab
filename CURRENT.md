# Current Task

Task:         P1.6-T01, P1.6-T02
Phase:        Phase 1.6
Spec:         docs/phases/phase-1.6-evaluation-harness.md
Taskboard:    docs/phases/phase-1.6-taskboard.md
Owner:        claude
Status:       done
Review Result: signed_off
Reviewer:     codex
Last Updated: 2026-06-10
Updated By:   codex

## Findings From Last Review

- none

## Tests Reviewed

- `uv run pytest tests/test_eval_metrics.py -k dataclass --tb=short -q`: pass, 13 passed
- `uv run pytest tests/test_eval_runner.py -k load --tb=short -q`: pass, 12 passed
- `uv run pytest --tb=short -q`: pass, 266 passed (no regressions)
- `uv run python -c "from pathlib import Path; from tiny_rag_lab.eval import load_eval_samples; p=Path('/tmp/tiny-rag-malformed-fields.jsonl'); p.write_text('{\"question_id\":\"q1\",\"question\":\"Q?\",\"answer\":\"A\",\"gold_doc_ids\":42}\n'); print(load_eval_samples(p))"`:
  pass, returns `[]`

## Blocker

- none

## Notes

Reviewed by codex; T01/T02 accepted. Dataclass defaults, loader skip behavior,
fixture coverage, and no-regression suite pass.

All blocking findings from Codex's reviews have been fixed:
- `EvalSample.gold_doc_ids` and `EvalResult.gold_doc_ids`/`retrieved_doc_ids` now use `field(default_factory=list)`
- `load_eval_samples()` skips non-dict JSON rows (e.g. `[]`)
- `load_eval_samples()` skips rows where `gold_doc_ids` is not a list (e.g. `42`)

## Handoff

### Task Summary

T01: defined `EvalSample`, `EvalResult`, `EvalReport` dataclasses in
`tiny_rag_lab/eval.py`. List fields default to `[]` via `field(default_factory=list)`.
Float fields default to `0.0`, bool to `False`.

T02: implemented `load_eval_samples(path) -> list[EvalSample]` in the same
module. Created `tests/fixtures/eval/qa.jsonl` with 3 records referencing the
existing fixture corpus doc_ids (`with_h1.md`, `plain.txt`, `subdir/nested.md`).
Loader skips rows with empty question, empty gold_doc_ids, blank lines, invalid
JSON, and non-dict JSON values.

### Files Changed

- `tiny_rag_lab/eval.py`: new — EvalSample, EvalResult, EvalReport dataclasses + load_eval_samples()
- `tests/fixtures/eval/qa.jsonl`: new — 3-record eval fixture
- `tests/test_eval_metrics.py`: new — T01 dataclass tests (13 tests)
- `tests/test_eval_runner.py`: new — T02 loader tests (11 tests)

### Design Decisions

- `gold_doc_ids` on `EvalSample` defaults to `[]` (matches T01 acceptance criteria); the loader already skips empty-gold rows so this is safe
- `load_eval_samples` does not raise on any malformed row — silent skip on invalid JSON, non-dict JSON, empty question, empty gold_doc_ids

### Tests Run

- `uv run pytest tests/test_eval_metrics.py -k dataclass --tb=short -q`: 13 passed
- `uv run pytest tests/test_eval_runner.py -k load --tb=short -q`: 11 passed
- `uv run pytest --tb=short -q`: 265 passed

### Known Gaps

- none

### Learning Notes

- `field(default_factory=list)` is required for mutable defaults in dataclasses — using `gold_doc_ids: list[str] = []` would share the same list object across all instances (Python gotcha)

### Questions For Next Agent

- T03 (metric functions) and T05 (formatter) are unblocked and can proceed in parallel with or after T02 sign-off
