# tiny-rag-lab Roadmap

This roadmap is a proposal, not an active implementation contract. Future agents should refine it into phase specs and taskboards before coding.

## Phase 0: Project Definition

Goal: turn this proposal into a decision-complete first phase.

Expected outputs:

- project scope and non-goals
- initial corpus choice
- dependency policy
- CLI surface proposal
- evaluation dataset format
- first phase spec and taskboard

Status: proposal only.

## Phase 1: Naive Classic RAG

Goal: implement the simplest complete RAG path end to end.

Capabilities:

- load Markdown and plain text files from a local corpus
- split text into deterministic chunks
- preserve source metadata such as path, title, and character offsets
- embed chunks through a small embedding abstraction
- store vectors in a simple local index, likely NumPy-backed for learning
- retrieve top-k chunks by cosine similarity
- assemble a grounded prompt from retrieved chunks
- call an LLM provider or local model through a narrow generation interface
- return answer text plus citations
- expose simple CLI commands: `index`, `retrieve`, `ask`

Learning questions:

- What is a document, a chunk, and a retrieval unit?
- How does chunk size affect answer quality?
- What information must be preserved for citation and debugging?
- What does cosine similarity actually compare?

## Phase 1.5: Retrieval Mechanics

Goal: make retrieval behavior inspectable and comparable.

Capabilities:

- configurable chunk size and overlap
- metadata filtering
- keyword/BM25 retrieval as a non-vector baseline
- hybrid retrieval that combines semantic and keyword scores
- optional simple reranking
- retrieval inspection command showing chunks, scores, and metadata

Learning questions:

- When does semantic retrieval miss exact terms?
- When does keyword retrieval outperform embeddings?
- How do chunk boundaries create false negatives?
- What does reranking fix, and what does it not fix?

## Phase 1.6: Evaluation Harness

Goal: evaluate RAG behavior before optimizing it.

Capabilities:

- versioned evaluation dataset, likely JSONL or YAML
- questions with expected source documents or chunks
- optional reference answer facts
- retrieval metrics: hit rate, MRR, context precision, context recall
- answer metrics: faithfulness, answer correctness, answer relevance
- deterministic eval runs where possible
- comparison reports across chunking, retriever, and prompt configurations

Learning questions:

- Did the retriever fetch the needed evidence?
- Did generation use the evidence faithfully?
- Did a change improve retrieval or merely change the answer style?
- Which failures are retrieval failures versus generation failures?

## Phase 1.7: Observability And Debugging

Goal: make each RAG run explainable.

Capabilities:

- per-query trace: query, rewritten query if any, retrieved chunks, scores, prompt, answer, citations
- token estimates for prompt/context budget
- latency by stage: loading, embedding, retrieval, reranking, generation, evaluation
- failure classifications such as missing evidence, distractor evidence, unsupported answer, citation mismatch, and unanswerable query
- saved run artifacts for later comparison

Learning questions:

- Where did the pipeline spend time?
- Which retrieved chunks influenced the answer?
- Why did a bad answer happen?
- How can failures be grouped into useful categories?

## Phase 1.8: RAG Failure Lab

Goal: intentionally create and study common RAG failure modes.

Scenarios:

- bad chunking splits necessary context
- top-k too small misses evidence
- top-k too large adds distractors
- stale documents conflict with newer documents
- ambiguous query retrieves the wrong topic
- answer uses model prior instead of context
- citation points to a related but unsupported source
- query is unanswerable from the corpus

Learning questions:

- How should the system behave when evidence is missing?
- How can prompts reduce unsupported answers?
- Which failures can be caught by retrieval metrics?
- Which failures require LLM-as-judge or human review?

## Later: Agentic RAG

Only after classic RAG is clear, consider advanced phases:

- query rewriting
- multi-step retrieval
- multi-hop question answering
- tool-assisted retrieval
- self-checking and answer revision
- memory and conversation-aware retrieval
- GraphRAG or structured knowledge retrieval
- production ingestion and background indexing

These should not be part of the first implementation phase.
