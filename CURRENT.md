# Current Task

Task:         P1-T05, P1-T06
Phase:        Phase 1
Spec:         docs/phases/phase-1-naive-classic-rag.md
Taskboard:    docs/phases/phase-1-taskboard.md
Owner:        claude
Status:       review
Review Result: signed_off
Reviewer:     codex
Last Updated: 2026-06-07
Updated By:   codex

## Findings From Last Review

- none

## Tests Reviewed

- `uv run pytest tests/test_documents.py --tb=short -q`: pass, 24 passed in 0.01s
- `uv run pytest --tb=short -q`: pass, 65 passed in 0.04s
- manual document/normalization smoke check: pass, nested `doc_id` is `subdir/nested.md`, title extraction and normalization behave as expected

## Blocker

- none

## Notes

## Handoff

### Task Summary

Implemented document loading (T05) and text normalization (T06) together
in tiny_rag_lab/documents.py. Normalization is called inside load_document
so Document.normalized_text is always populated. Added four test fixture
files (three at root, one in a subdirectory) to test recursive loading and
doc_id path computation.

### Files Changed

- `tiny_rag_lab/documents.py`: normalize_text, _extract_title, load_document, load_documents
- `tests/test_documents.py`: 24 tests covering both tasks
- `tests/fixtures/corpus/with_h1.md`: fixture with H1 title
- `tests/fixtures/corpus/no_h1.md`: fixture without H1 (filename fallback)
- `tests/fixtures/corpus/plain.txt`: plain text fixture
- `tests/fixtures/corpus/subdir/nested.md`: nested fixture for recursive load test
- (also removes `tests/fixtures/corpus/.gitkeep` — directory now has real content)

### Design Decisions

- **Both tasks in documents.py**: normalization has no independent module in the file layout; it belongs in documents.py as the natural home for text preparation. T07 (chunker) will import normalize_text from here.
- **normalize_text is pure**: takes a string, returns a string — no I/O, easy to test and reuse in chunking.py if needed.
- **raw_hash computed from raw_text before normalization**: the hash identifies the source file content, not the processed form. This is consistent with the spec and with change detection (if raw_text changes, the hash changes).
- **load_documents sorted by path**: deterministic order makes tests stable and index builds reproducible.
- **.gitkeep not excluded by suffix filter**: `_SUPPORTED_SUFFIXES = {".md", ".txt"}` naturally skips `.gitkeep` with no special case.

### Tests Run

- `uv run pytest tests/test_documents.py --tb=short -q`: 24 passed
- `uv run pytest --tb=short -q`: 65 passed (full suite)

### Known Gaps

- None. T07 (chunker) depends on T04 and T06 and can now start.
- T08 (embedding interface) depends only on T04 and can also start.

### Questions For Next Agent

- None.
