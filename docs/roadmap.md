# tiny-rag-lab Roadmap

This roadmap is directional. Active implementation contracts live under
`docs/phases/`.

## Phase 0: Project Definition

Goal: turn this proposal into a decision-complete first phase.

Expected outputs:

- project scope and non-goals
- initial corpus choice
- dependency policy
- CLI surface proposal
- evaluation dataset format
- first phase spec and taskboard

Status: Phase 1, Phase 1.5, Phase 1.6, Phase 1.7, Phase 1.8, and Phase 1.9 are complete
under `docs/phases/`. No implementation phase is currently active. The
near-term roadmap decision is recorded in
`docs/phases/phase-1.9-2.2-final-roadmap.md`.

## Phase 1: Naive Classic RAG

Goal: implement the simplest complete RAG path end to end.

Capabilities:

- prepare IBM `watsonxDocsQA` into local Markdown files
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
- retriever-aware reports for a single eval run; broader comparison reports
  are deferred until trace and artifact shapes are stable

Learning questions:

- Did the retriever fetch the needed evidence?
- Did generation use the evidence faithfully?
- Did a change improve retrieval or merely change the answer style?
- Which failures are retrieval failures versus generation failures?

## Phase 1.7: Observability And Debugging

Goal: make each single RAG run explainable by adding trustworthy trace
records before building broader failure analysis or reporting features.

Included capabilities:

- per-query trace data model for retrieve and ask flows
- retriever-aware trace fields: retriever name, top-k, ranks, scores, chunk ids,
  doc ids, titles, and paths
- prompt, answer, citations, and source table inputs for answer-producing runs
- latency by major stage: loading, embedding, retrieval, prompt assembly, and
  generation where applicable
- optional JSON trace output such as `--trace-out PATH`
- concise human-readable trace output for interactive debugging

Learning questions:

- Where did the pipeline spend time?
- Which retrieved chunks influenced the answer?
- Which retriever produced which evidence?
- What exact prompt and sources were given to generation?
- What trace fields are stable enough for later failure analysis?

Deferred from Phase 1.7:

- failure classification taxonomy
- full eval-run artifact storage
- detailed token budget estimation if it requires model-specific tokenizer
  decisions
- comparison UI or multi-run reporting

## Phase 1.8: RAG Failure Lab

Goal: intentionally create and study common RAG failure modes using the trace
foundation from Phase 1.7.

Scenarios:

- bad chunking splits necessary context
- top-k too small misses evidence
- top-k too large adds distractors
- stale documents conflict with newer documents
- ambiguous query retrieves the wrong topic
- answer uses model prior instead of context
- citation points to a related but unsupported source
- query is unanswerable from the corpus

Expected capabilities:

- curated failure examples with reproducible inputs
- trace-backed notes explaining what failed and why
- lightweight failure labels grounded in observed traces, such as missing
  evidence, distractor evidence, unsupported answer, citation mismatch, and
  unanswerable query
- prompt or retrieval experiments that demonstrate whether a failure is fixed
  or only moved

Learning questions:

- How should the system behave when evidence is missing?
- How can prompts reduce unsupported answers?
- Which failures can be caught by retrieval metrics?
- Which failures require LLM-as-judge or human review?
- Which trace fields are most useful for diagnosing each failure?

Status: Complete; see `docs/phases/phase-1.8-failure-lab.md` and
`docs/phases/phase-1.8-taskboard.md`.

## Near-Term Roadmap: Classic RAG Quality

The next roadmap window is decided but not active. Each phase still needs its
own spec and taskboard before implementation starts.

### Phase 1.9: Reranking

Goal: add a second-pass reranking layer so retrieved candidates can be
reordered by query-document relevance before evaluation and answer generation.

Expected capabilities:

- `Reranker` interface with fake and local cross-encoder implementations
- explicit reranker CLI controls such as `--reranker` and `--rerank-top-n`
- `rag retrieve` and `rag eval` integration first
- `rag ask` reranked-context support later in the same phase
- trace fields showing pre-rerank and post-rerank order
- failure-lab coverage for buried evidence fixed by reranking

Status: Complete; see `docs/phases/phase-1.9-reranking.md` and
`docs/phases/phase-1.9-taskboard.md`.

### Phase 2.0: Answer Quality Judging

Goal: measure generated answer quality with an LLM-as-judge path behind a
fakeable interface.

Expected capabilities:

- `Judge` interface with fake and OpenAI-compatible implementations
- answer-level metrics for faithfulness, relevance, correctness, and citation
  support
- optional eval fields such as `reference_answer` and `expected_facts`
- failure-lab coverage for `unsupported_answer` and `citation_mismatch`
- reports that keep retrieval metrics and answer metrics separate

### Phase 2.1: Context Budget And Structured Answers

Goal: control prompt context size and optionally return machine-readable
answers.

Expected capabilities:

- `rag ask` context budget controls, measured in tokens when possible
- trace fields for selected chunks, omitted chunks, and estimated budget used
- optional structured JSON answer output
- plain-text answer output remains the default

### Phase 2.2: Structural And Semantic Chunking

Goal: compare fixed-character chunking with chunking strategies that better
preserve document structure and meaning.

Expected capabilities:

- keep fixed-character chunking as the baseline and fallback
- add structural chunking based on Markdown headings, paragraphs, lists, and
  sentence boundaries where practical
- add semantic chunking as an experimental mode when useful
- record chunking strategy and parameters in the index manifest
- evaluate chunking modes with `rag eval` and failure-lab cases

## Later: Reporting, Artifacts, And Agentic RAG

After the near-term quality roadmap is clearer, consider broader observability
and workflow features:

- full eval-run artifact storage with run ids, schemas, retention rules, and
  reproducibility metadata
- comparison reports across chunking, retriever, prompt, and generation
  configurations
- richer token budget estimation with explicit tokenizer/model choices
- UI or notebook-style reports for multi-run inspection

Only after classic RAG quality and production prompt mechanics are measured,
consider advanced agentic phases:

- query rewriting
- multi-step retrieval
- multi-hop question answering
- tool-assisted retrieval
- self-checking and answer revision
- memory and conversation-aware retrieval
- GraphRAG or structured knowledge retrieval
- production ingestion and background indexing

These should not be part of the first implementation phase.
