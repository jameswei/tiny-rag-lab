# Current Task

Task:         P1-T02, P1-T03, P1-T04
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

- `uv run pytest --tb=short -q`: pass, 41 passed in 0.03s
- `uv run python scripts/prepare_watsonx_docsqa.py --help`: pass, default dataset is `ibm-research/watsonxDocsQA`
- synthetic conversion smoke check: pass, manifest `doc_id` and QA `gold_doc_ids` both resolve to `docs/abcdef123.md`, and the Markdown file exists

## Blocker

- none

## Notes

## Handoff

### Task Summary

Implemented T03 (gitignore), T04 (data contracts), and T02 (corpus prep
script) together. All three are independent of each other.

### Files Changed

- `.gitignore`: added `corpus/` and `.tiny-rag/` to keep generated artifacts out of git (T03)
- `tiny_rag_lab/models.py`: Document, Chunk, RetrievalResult, RagTrace dataclasses + make_chunk_id (T04)
- `scripts/prepare_watsonx_docsqa.py`: corpus preparation script with Hub download + local conversion + --inspect mode (T02)
- `tests/test_models.py`: 14 tests — chunk_id determinism, slice invariant, serialization for all 4 types (T04)
- `tests/test_prepare_watsonx_docsqa.py`: 15 tests — conversion functions, I/O helpers, all using synthetic rows with no network access (T02)

### Design Decisions

- **models.py as a single contracts file**: all 4 dataclasses live together so the full data shape of the pipeline is visible in one place. Downstream modules (documents.py, chunking.py, etc.) will import from models.py, avoiding circular imports.
- **make_chunk_id in models.py**: the chunk ID formula is part of the Chunk contract, so it belongs alongside Chunk rather than in chunking.py.
- **Dataset schema mapping**: the prep script targets the real `ibm-research/watsonxDocsQA` fields. Corpus rows use `doc_id`, `title`, `md_document`/`document`, and `url`; QA rows use `question_id`, `question`, `correct_answer`, and `ground_truths_contexts_ids`.
- **Prepared IDs**: manifest `doc_id` and `qa.jsonl` `gold_doc_ids` both use the same corpus-relative prepared path, preserving later evaluation linkage.
- **scripts/ import in tests**: `sys.path.insert` is used to import from scripts/ since scripts are not part of the installable package. This is intentional — scripts are one-off tools, not library code.

### Tests Run

- `uv run pytest --tb=short -q`: 41 passed
- No network access required

### Known Gaps

- T02 acceptance "local dataset converts to Markdown, manifest, QA JSONL" requires running the script against the actual dataset (or a local copy). That can't be tested in CI without credentials/network. The conversion logic is tested with synthetic fixtures.

### Questions For Next Agent

- T05 (document loader) and T06 (text normalization) can now start — both depend on T04.
