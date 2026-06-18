# Current Task

Task:         P2.0-T05
Phase:        Phase 2.0 — Answer Quality Judging
Spec:         docs/phases/phase-2.0-answer-quality-judging.md
Taskboard:    docs/phases/phase-2.0-taskboard.md
Owner:        Claude Sonnet 4.6
Status:       review
Review Result: signed_off
Reviewer:     codex
Last Updated: 2026-06-18
Updated By:   codex

## Findings From Last Review

- [Blocking] ✅ fc009 now includes source markers: baseline `[Source: with_h1.md]`, intervention `[Source: subdir/nested.md]`
- [Blocking] ✅ `run_answer_diagnosis` extracts citations from `[Source: ...]` markers and passes them to judge via `citations=...` parameter; added focused spy-judge test `test_run_answer_diagnosis_extracts_and_passes_citations` to prevent regression

No remaining T05 findings.

## Latest Review Result

Signed off by codex on 2026-06-18.

## Tests Reviewed

- `uv run pytest tests/test_failure.py tests/test_cmd_diagnose.py --tb=short -q`: 118 passed
- `uv run pytest --tb=short -q`: 649 passed, 2 skipped
- New regression test: `test_run_answer_diagnosis_extracts_and_passes_citations` verifies citations are extracted and passed to judge
- CLI smoke with fixture index:
  - `--judge none`: no answer diagnosis section
  - `--judge fake --generator fake`: answer diagnosis section present; `n=2`; fc008/fc009 labels present

## Notes

### Files Changed

- `tiny_rag_lab/failure.py`:
  - Added `import re` for citation extraction
  - `run_answer_diagnosis`: extracts citations from `[Source: ...]` markers using regex; passes `citations=...` to judge.judge()
- `tests/fixtures/failure/cases.jsonl`:
  - fc009: updated baseline_answer to include `[Source: with_h1.md]` (wrong citation)
  - fc009: updated intervention_answer to include `[Source: subdir/nested.md]` (correct citation)
- `tests/test_failure.py`:
  - Updated FC009_BASELINE_ANSWER and FC009_INTERVENTION_ANSWER constants to include source markers
  - Added new test `test_run_answer_diagnosis_extracts_and_passes_citations` using spy judge to verify citations are extracted and passed
- `tests/test_cmd_diagnose.py`:
  - Updated fc009_baseline and fc009_intervention in `test_diagnose_judge_fake_prints_answer_diagnosis_section` to include source markers

### Design Decisions

- run_answer_diagnosis skips cases where answer_label_expected == "" (silent, no warning)
- FakeJudge.verdict_map keyed by answer string; fc008/fc009 pre-script both sides so no generator call needed
- fc008/fc009 expected_label stays "no_failure" (retrieval pass succeeds); only answer_label_expected triggers run_answer_diagnosis
- format_answer_diagnosis_report shows verdict scores (faith/cit) instead of retrieval metrics
- cmd_diagnose runs retrieval diagnosis then answer diagnosis when --judge active; n= counts are independent
