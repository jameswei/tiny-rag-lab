# Phase 2.2 Record: Structural and Semantic Chunking

**Status:** Complete
**Completion:** Reviewed and closed in the original phase; detailed execution
evidence was consolidated after Phase 3.0.

## Delivered

- Structural chunking that packs Markdown-aware blocks, then sentences, and
  falls back to character windows only for oversized spans.
- Experimental semantic chunking that uses adjacent sentence embedding
  similarity to identify topic shifts.
- Index-time `--chunking-strategy` controls and manifest recording of strategy
  plus parameters.

## Durable Decisions

- Chunking is selected when an index is built; retrieval, prompting, traces,
  and citations consume the resulting chunks without strategy-specific paths.
- Fixed-character chunking stays the baseline and fallback for compatibility.
- Structural chunks keep meaningful boundaries where practical; overlap applies
  only to the oversized character-window fallback.
- Semantic chunking is opt-in and adds a sentence-level embedding pass, so it
  must be measured rather than assumed superior.

## Learning Context

The reproducible fixed-character versus structural comparison is maintained in
the EN and ZH structural/semantic chunking lessons, alongside the commands and
fixture explanation.
