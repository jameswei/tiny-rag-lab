# Final Roadmap: Phase 1.9-2.2 Classic RAG Quality

**Status:** Final roadmap decision - directional, not active
**Authors:** Codex + Claude Code review + owner decision
**Date:** 2026-06-16
**Based on:** `docs/proposal.md`, `docs/roadmap.md`, `docs/gaps.md`,
`docs/phases/new-phases-proposal_claude.md`,
`docs/phases/phase-1.9-2.1-quality-and-production-roadmap-codex.md`, and
`docs/phases/phase-1.9-2.1-quality-and-production-roadmap-codex-review-by-claude.md`

This document records the agreed near-term roadmap after cross-review. It is
not an implementation contract. Each phase still needs its own spec and
taskboard, and only one phase becomes active when `docs/phases/README.md`
names it under "Current Phase".

---

## Decision

The next roadmap stays focused on classic RAG quality and practical production
mechanics. Agentic RAG, query rewriting, multi-step retrieval, memory, and tool
use remain deferred.

Final sequence:

1. **Phase 1.9 - Reranking**
2. **Phase 2.0 - Answer Quality Judging**
3. **Phase 2.1 - Context Budget And Structured Answers**
4. **Phase 2.2 - Structural And Semantic Chunking**

The first implementation proposal to draft next is **Phase 1.9 - Reranking**.
Phases 2.0 through 2.2 stay directional until earlier phases land and their
interfaces are stable.

---

## Phase 1.9 - Reranking

Goal: add a second-pass reranking layer so retrieved candidates can be reordered
by query-document relevance before evaluation and answer generation.

Why now:

- It is the clearest retrieval-quality upgrade from the current baseline.
- Existing retrieval metrics already measure its effect.
- It directly targets `low_rank_evidence` failures from the failure lab.
- It teaches the bi-encoder versus cross-encoder distinction without starting
  agentic RAG.

Expected scope for the future spec:

- `Reranker` interface, `RerankResult` data shape, and `FakeReranker`.
- Local cross-encoder reranker for real runs, lazy-loaded and gated so tests do
  not download or load the model.
- CLI flags such as `--reranker none|cross-encoder` and `--rerank-top-n INT`.
- `rag retrieve` and `rag eval` integration first.
- `rag ask` reranked-context integration later in the same phase.
- Trace fields showing pre-rerank and post-rerank order.
- Failure fixture coverage for reranking fixing buried evidence.

Suggested task split:

1. Reranker interface, result contract, and fake reranker.
2. Cross-encoder reranker backend and model-gating behavior.
3. `rag retrieve` integration and retrieve trace updates.
4. `rag eval` integration and report fields.
5. `rag ask` integration and ask trace updates.
6. Failure fixture update, CLI smoke checks, docs sync, and phase close.

---

## Phase 2.0 - Answer Quality Judging

Goal: measure answer quality, not only retrieval quality, using an
LLM-as-judge path behind a fakeable interface.

Why after reranking:

- Reranking changes which context reaches generation.
- The next question is whether the final answer becomes more faithful,
  relevant, correct, and citation-supported.
- Phase 1.8 explicitly deferred answer-side failure modes that require a judge.

Expected scope:

- `Judge` interface with `FakeJudge` for tests and an OpenAI-compatible judge
  for real runs.
- Explicit judge mode, for example `--judge fake|openai`, default off.
- Answer-level metrics:
  - faithfulness to retrieved context
  - answer relevance to the question
  - answer correctness against a reference answer when available
  - citation support
- Optional eval JSONL fields such as `reference_answer` and
  `expected_facts`, with backward-compatible behavior when missing.
- Failure-lab support for `unsupported_answer` and `citation_mismatch`.
- Reports that keep retrieval metrics and answer metrics separate.

---

## Phase 2.1 - Context Budget And Structured Answers

Goal: add generation-side production mechanics that control prompt context size
and optionally return machine-readable answers.

Why after judging:

- Reranking improves the order of candidate chunks.
- Judging measures whether answer quality improved.
- Context budgeting then controls which ranked chunks actually enter the prompt
  and how much cost, latency, and noise the generation step accepts.

Expected scope:

- `rag ask --context-budget INT`, measured in tokens when a tokenizer is
  available.
- Context packing records:
  - selected chunk IDs
  - omitted chunk IDs
  - estimated budget used
- Ask trace updates for context packing decisions.
- Optional structured output such as `--output-format json`.
- Plain text remains the default output.

Recommended initial budget defaults for the future spec:

- default retrieved-context budget: 8K tokens
- configurable higher budget: 16K or 32K tokens
- use token budgets rather than character budgets so English and Chinese are
  handled consistently by the model tokenizer

---

## Phase 2.2 - Structural And Semantic Chunking

Goal: replace fixed-character chunking as the only indexing strategy with
chunking modes that better preserve document structure and meaning.

Why after the first three phases:

- Fixed-character chunking was correct for the learning baseline, but it is
  usually not good enough as the final production strategy.
- Reranking, judging, and context budgeting create a stronger measurement
  surface before chunking experiments begin.
- Better chunking can then be evaluated with retrieval metrics, answer-quality
  metrics, and failure cases.

Expected scope:

- Keep the existing fixed-character chunker as the baseline and fallback.
- Add structural chunking first:
  - Markdown headings
  - paragraphs
  - lists
  - sentence boundaries where practical
- Add semantic chunking as an experimental mode when useful:
  - embedding-based topic-shift detection
  - clear reports comparing it against structural and fixed modes
- Record chunking strategy and parameters in the index manifest.
- Add failure-lab coverage for bad chunking splitting necessary context.
- Compare chunking strategies through `rag eval` and selected diagnosis cases.

---

## Deferred

The following remain out of scope for this roadmap window:

- query rewriting, HyDE, and query decomposition
- multi-step or multi-hop retrieval
- tool-assisted retrieval
- memory and conversation-aware retrieval
- GraphRAG or structured knowledge retrieval
- vector databases and ANN indexes
- production security, access control, and prompt-injection hardening

These are valid future directions, but they should wait until classic RAG
quality and production prompt mechanics are better understood and measured.

---

## Roadmap Document Policy

`docs/roadmap.md` should not be overwritten. It is the durable project roadmap
and historical phase sequence, so it should be updated in place with a concise
near-term roadmap section that points here. This final roadmap document carries
the detailed decision record for agent handoff and cross-review alignment.
