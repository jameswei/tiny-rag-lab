# Phase 1.8 Record: RAG Failure Lab

**Status:** Complete
**Completion:** Reviewed and closed in the original phase; detailed execution
evidence was consolidated after Phase 3.0.

## Delivered

- Curated, reproducible `rag diagnose` cases and reports for retrieval-side
  failure modes, including missing evidence, distractors, ambiguity, and
  unanswerable questions.
- A small failure taxonomy, JSONL case loader, and baseline-versus-intervention
  diagnosis built on the evaluation and trace foundations.

## Durable Decisions

- Retrieval failures are diagnosed from retrieved evidence and expected source
  IDs; they are distinct from answer-quality failures.
- Unanswerable questions are first-class learning cases, not silent misses.
- Curated fixtures make a failure explanation reproducible instead of relying
  on anecdotal output from a changing corpus.

## Follow-on Context

Unsupported answers and citation mismatches require answer-level judging and
were added in Phase 2.0. The visual lab later presents these lessons in the
browser.
