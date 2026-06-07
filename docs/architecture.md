# tiny-rag-lab Architecture

This document describes the proposed conceptual structure for `tiny-rag-lab`. It is not yet an implementation contract.

## Purpose

The architecture should make the RAG lifecycle visible. Each stage should have clear inputs, outputs, invariants, and debug artifacts.

The central design idea is to split the system into four planes:

1. Indexing plane
2. Retrieval plane
3. Generation plane
4. Evaluation and observability plane

## Indexing Plane

The indexing plane turns raw files into searchable retrieval units.

Responsibilities:

- discover local documents
- load Markdown and plain text
- normalize extracted text
- split text into chunks
- attach metadata
- compute embeddings
- persist or rebuild the local index

Important concepts:

- document: a source file or logical source
- chunk: the atomic retrieval unit
- metadata: path, title, section, offsets, hash, and other source identifiers
- embedding: vector representation of a chunk
- index: searchable collection of chunk embeddings and metadata

Learning boundary:

The first implementation should avoid hiding this behind a production document loader or vector database. A simple local index is preferable until the mechanics are understood.

## Retrieval Plane

The retrieval plane selects context for a user query.

Responsibilities:

- normalize the user query
- compute query embedding
- search the vector index
- optionally run keyword/BM25 retrieval
- optionally combine semantic and keyword results
- optionally rerank candidates
- return ranked chunks with scores and metadata

Important concepts:

- top-k retrieval
- cosine similarity
- score normalization
- metadata filtering
- reranking
- retrieval trace

Learning boundary:

Retrieval should be inspectable. A user should be able to run a command that shows exactly which chunks were retrieved and why they were selected.

## Generation Plane

The generation plane turns retrieved context into an answer.

Responsibilities:

- pack retrieved chunks into a context window
- format a grounded prompt
- call an LLM through a narrow interface
- generate an answer
- include citations or source references
- handle unanswerable questions when context is insufficient

Important concepts:

- context packing
- prompt template
- citation format
- groundedness instruction
- answer abstention
- model interface

Learning boundary:

The prompt assembly should be project-owned and visible. The LLM call can be delegated to a provider or local model, but the system should keep the request and response boundary simple.

## Evaluation And Observability Plane

The evaluation plane measures whether RAG worked and helps explain failures.

Responsibilities:

- load evaluation questions
- run retrieval-only evaluation
- run full answer evaluation
- calculate metrics
- store traces and reports
- classify failures

Suggested retrieval metrics:

- hit rate
- mean reciprocal rank
- context precision
- context recall

Suggested answer metrics:

- faithfulness to retrieved context
- answer correctness against expected facts
- answer relevance to the question
- citation support

Important concepts:

- retrieval failure versus generation failure
- answerable versus unanswerable questions
- distractor context
- unsupported claims
- regression comparison across configurations

## Proposed Interfaces

The first implementation should likely expose CLI commands before any UI:

- `index`: build or rebuild an index for a corpus
- `retrieve`: show ranked chunks for a query
- `ask`: run the full RAG pipeline for one query
- `eval`: run an evaluation set and report metrics
- `inspect`: open or print saved traces from previous runs

Exact flags and schemas should be decided in a phase spec before implementation.

## Dependency Philosophy

Use dependencies for commodity primitives when they do not hide the core lesson:

- text and Markdown parsing can use small libraries if needed
- embeddings can be local or provider-backed behind a small interface
- LLM generation can be provider-backed behind a small interface
- NumPy is acceptable for vector math

Avoid starting with high-level RAG frameworks as the core engine. They can be references or later comparison targets.
