# tiny-rag-lab Proposal

## Purpose

`tiny-rag-lab` is a learning-first project for understanding retrieval-augmented generation by building the classic RAG lifecycle directly in Python.

The project should answer a simple question:

> What actually happens between a user question, a document corpus, retrieved context, and a grounded LLM answer?

The goal is not to build a production RAG platform. The goal is to learn the mechanics deeply enough to reason about RAG quality, failures, evaluation, and later agentic extensions.

## Motivation

RAG is a foundational pattern in the current AI and agentic software wave. It appears in knowledge assistants, enterprise AI tools, coding systems, document search, memory systems, customer support, and agent tool use.

Many tutorials hide the important parts behind framework calls. This project should make those parts visible, similar to how `tiny-duo-infer` makes LLM inference concepts visible.

The owner wants to keep learning practical AI system concepts to maintain and supplement long-term engineering capability and competitiveness.

## Learning-First Philosophy

This project should prefer:

- explicit control flow over framework magic
- inspectable intermediate data over opaque pipelines
- small corpora the owner understands over impressive demos
- evaluation before optimization
- failure analysis before advanced features
- readable code and docs over production completeness

Frameworks such as LangChain, LlamaIndex, Ragas, and vector databases can be used as references or optional comparison points, but the core implementation should show the pipeline directly.

## First Learning Target: Classic RAG

The early project should make the classic single-query RAG lifecycle visible:

1. Load documents from a local corpus.
2. Extract and normalize text.
3. Split documents into chunks.
4. Attach source metadata.
5. Embed chunks.
6. Store embeddings in a simple searchable index.
7. Embed a user query.
8. Retrieve top-k relevant chunks.
9. Optionally rerank retrieved chunks.
10. Pack context into a prompt.
11. Call an LLM to generate an answer.
12. Return citations or source references.
13. Evaluate retrieval and answer quality.
14. Inspect failures with enough detail to explain what went wrong.

Phase 1 implements the first naive end-to-end slice. Evaluation, richer
observability, and failure analysis are deferred to later dedicated phases.

## Initial Corpus Direction

Start with a real, open documentation QA corpus rather than only personal
notes. The Phase 1 spec selects:

- primary corpus: IBM `watsonxDocsQA`, prepared into local Markdown files
- stretch corpus: WixQA, after the primary corpus path works

Personal or project documents can still be useful as optional ad hoc corpora,
but they are not the default Phase 1 corpus. Public documentation QA datasets
make the lab more reproducible and closer to common customer-support RAG
scenarios.

## Target User

The primary user is the owner as a learner-builder. Secondary users are collaborating agents such as Codex CLI and Claude Code.

The project should be useful as:

- a study artifact
- an experiment bench
- a source of reusable mental models
- a future base for more advanced RAG and agentic retrieval work

## Non-Goals For The First Version

The first version should not try to be:

- a production RAG service
- a chatbot product
- a LangChain or LlamaIndex wrapper
- a complex UI application
- a GraphRAG system
- a multi-agent system
- a large-scale ingestion platform
- a benchmark chasing state-of-the-art results

These topics can be revisited after classic RAG is understood and measured.
