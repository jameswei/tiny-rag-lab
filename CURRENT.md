# Current Task

Task:         P2.2-T01
Phase:        Phase 2.2
Spec:         docs/phases/phase-2.2-structural-semantic-chunking.md
Taskboard:    docs/phases/phase-2.2-taskboard.md
Owner:        Claude Code
Status:       review
Review Result: signed_off
Reviewer:     Codex
Last Updated: 2026-06-25
Updated By:   Codex

## Findings From Last Review

- none

Previous blocking finding is fixed: consecutive heading-only blocks now merge
through to the following body block for the reviewed repro case, so
`# H1\n\n## H2\n\nBody paragraph with enough words.` at `chunk_size=20`
no longer emits `'# H1\n\n## H2\n\n'` as a standalone chunk.

## Tests Reviewed

- `uv run pytest tests/test_chunking.py --tb=short -q`: 34 passed
- manual reproduction:
  `chunk_document_structural("# H1\n\n## H2\n\nBody paragraph with enough words.", chunk_size=20, chunk_overlap=3)`
  emits the first chunk as `'# H1\n\n## H2\n\nBody pa'`, not a standalone
  heading-only chunk
- `uv run pytest --tb=short -q`: 722 passed, 7 skipped

## Blocker

- none

## Handoff

### Task Summary

Added the structural chunker with three-tier packing per Design Decision 2:
(1) pack whole Markdown blocks up to `chunk_size`, (2) a block that alone
exceeds `chunk_size` is split into sentences and packed at sentence
granularity, (3) only a single sentence that itself exceeds `chunk_size`
falls back to the character sliding window (the only tier that consumes
`chunk_overlap`). `chunk_document`/`chunk_documents` are unchanged in
signature and behavior.

Round 2: fixed the blocking finding from Codex's first review — consecutive
heading-only blocks (`# H1\n\n## H2\n\n...`) now merge as a whole run into
the next body block, instead of only merging one step forward and leaving a
still-heading-only merged block standalone.

### Files Changed

- `tiny_rag_lab/chunking.py`: extracted `_validate_chunk_params` from
  `chunk_document` (byte-identical `ValueError` text); added
  `_chunk_metadata`, `_split_sentences`, `_split_blocks`,
  `_is_heading_only_block`, `_chunk_oversized_span`, `_pack_units`,
  `chunk_document_structural`. `_split_blocks`'s merge loop now uses a
  two-pointer scan: it advances `j` past every consecutive heading-only
  block before merging `[idx, j]` together, instead of merging only
  `[idx, idx+1]` and never re-checking whether the result is still
  heading-only. If a heading run reaches the end of the document with
  nothing to merge into, the whole run becomes one block (same fallback
  as the original single-trailing-heading case).
- `tests/test_chunking.py`: 12 new tests for the structural chunker —
  10 from round 1 (slice invariant across sizes, heading-merge behavior
  including the no-following-block edge case, tier-2 sentence packing vs.
  tier-3 character-window fallback, contiguity, metadata/chunk_id contract,
  empty/whitespace input, validation errors, large-chunk-size single-chunk
  case), plus 2 new in round 2:
  `test_structural_consecutive_headings_never_form_a_heading_only_chunk`
  (Codex's exact repro case) and
  `test_structural_trailing_consecutive_headings_with_no_body` (the
  no-body-anywhere edge case for a heading run). Existing 22
  `chunk_document`/`chunk_documents` tests pass unmodified.

### Design Decisions

- `_pack_units` is a single generic greedy packer used recursively: tier 1
  packs blocks with `_pack_oversized_block` (defined inline in
  `chunk_document_structural`) as its oversized-unit handler; that handler
  packs sentences via `_pack_units` again, with `_chunk_oversized_span` as
  *its* oversized-unit handler. This reuses one packing algorithm for both
  tier 1 and tier 2 instead of duplicating greedy-packing logic.
- `_split_blocks` and `_split_sentences` return contiguous spans covering
  the whole input string (separators/whitespace are attached to the end of
  the preceding span), so concatenating spans always reproduces the
  original text exactly — this is what makes packed (non-fallback) chunks
  exactly contiguous.
- A run of consecutive heading-only blocks can only merge into the next
  block that follows the *whole run*, not just the immediate next block.
  If the run reaches the end of the document with no body to merge into,
  it has nothing to merge into and the whole run becomes one standalone
  block — covered by `test_structural_heading_with_no_body_stays_standalone`
  (single heading) and `test_structural_trailing_consecutive_headings_with_no_body`
  (multi-heading run).

### Tests Run

- `uv run pytest tests/test_chunking.py --tb=short -q`: 34 passed
- `uv run pytest --tb=short -q`: 722 passed, 7 skipped (full suite, no
  regressions)

### Known Gaps

- none remaining from round 1 — the multi-heading edge case Codex flagged
  is now fixed and covered by regression tests.

### Learning Notes

- `_pack_units`'s recursive reuse (tier 1's oversized handler calls into
  tier 2, tier 2's oversized handler is tier 3 directly) is the
  line-by-line place to see how the three-tier fallback in the spec maps to
  code — there's no separate tier-specific packing loop.
- `test_structural_oversized_block_packs_by_sentence_not_character_window`
  is the test that proves tier 2 is real and not just a redundant path to
  tier 3: it constructs a block with several *short* sentences whose sum
  exceeds `chunk_size`, so packing must stop at sentence boundaries, not
  slide a character window.
- The round-2 fix is a good example of why "merge one step forward" and
  "merge the whole run" are different algorithms even though they look
  similar for the single-heading case — the bug only shows up with two or
  more consecutive headings, which the round-1 test suite didn't exercise.

### Questions For Next Agent

- None.
