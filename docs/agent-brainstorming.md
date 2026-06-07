# Agent Brainstorming Guide

This document is for Codex, Claude Code, and other agents helping shape `tiny-rag-lab` before implementation begins.

## Current Task For Agents

Do not implement yet. First, critique and refine the project plan until the first phase is decision-complete.

A good output would be:

- clarified scope
- concrete first corpus
- dependency recommendations
- data schemas for documents, chunks, index records, questions, and traces
- CLI proposal
- evaluation plan
- first phase spec
- first phase taskboard

## Questions To Challenge

### Scope

- Is classic RAG the right first target?
- Which advanced topics must be explicitly deferred?
- What is the smallest useful end-to-end version?
- Should the first version be CLI-only?

### Corpus

- Should the initial corpus be `tiny-duo-infer` docs?
- Should personal notes be included immediately or later?
- What document formats should be supported first?
- How should source metadata be represented?

### Chunking

- What is the first chunking strategy?
- Should chunking be character-based, token-aware, Markdown-heading-aware, or sentence-aware?
- What chunk size and overlap defaults are good for learning?
- How should chunk IDs remain stable across re-indexing?

### Embeddings And Index

- Should embeddings be local, API-backed, or both behind an interface?
- Should the first vector store be NumPy-only?
- When should FAISS, LanceDB, Chroma, or sqlite-vss be considered?
- How should index artifacts be saved and versioned?

### Retrieval

- What retrieval metrics should be implemented first?
- Should BM25 be included in Phase 1 or Phase 1.5?
- How should hybrid retrieval be explained and evaluated?
- What trace output would make retrieval understandable?

### Generation

- What should the default grounded prompt look like?
- How should citations be represented in the answer?
- Should the model be instructed to abstain when evidence is missing?
- How should unanswerable questions be tested?

### Evaluation

- What should the evaluation dataset schema be?
- Should expected answers be exact strings, fact lists, source IDs, or a combination?
- Which metrics can be deterministic without an LLM judge?
- Where is an LLM judge useful, and how should its limitations be documented?

### Observability

- What should a saved RAG trace contain?
- How should the system classify failures?
- What reports would help compare experiments?
- How should token counts, latency, and cost be recorded?

## Suggested First Phase Output

Agents should converge on a Phase 1 spec that includes:

- accepted document formats
- corpus directory layout
- chunk schema
- index schema
- embedding interface
- retrieval algorithm
- prompt template
- citation format
- CLI commands and flags
- tests and acceptance criteria
- explicit non-goals

## Collaboration Style

Follow the `tiny-duo-infer` pattern:

- keep docs and taskboards current
- make assumptions explicit
- prefer readable implementations
- require tests for behavior that teaches a concept
- record handoffs and review results when a phase closes
