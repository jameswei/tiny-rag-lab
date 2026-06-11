# Learning Roadmap

This is the recommended reading order for the learning materials in this
directory. Start at the top and work down.

---

## Reading Order

| # | Document | Why here |
|---|----------|----------|
| 1 | [The RAG Data Flow](the-rag-data-flow.md) | Vocabulary and data contracts for everything below |
| 2 | [The Indexing Plane](the-indexing-plane.md) | How documents become searchable vectors |
| 3 | [Retrieval and Generation](retrieval-and-generation.md) | How queries find chunks and become answers |
| 4 | [Persistence and Testing](persistence-and-testing.md) | The on-disk format, round-trip integrity, and fake backends |
| 5 | [Evaluating Retrieval](evaluating-retrieval.md) | Metrics that tell you whether the retriever works |
| 6 | [Observability and Debugging](observability-and-debugging.md) | Per-run traces that explain one retrieve or ask command |

---

## High-Level Architecture

The pipeline has two main planes, plus infrastructure and measurement:

```
                   ┌──────────────────────┐
                   │   1. Indexing Plane  │
                   │                      │
     corpus ──────►│ load → normalize →   │
     (.md,.txt)    │ chunk → embed        │────► .tiny-rag/index/
                   │                      │       (manifest.json,
                   │ documents.py         │        chunks.jsonl,
                   │ chunking.py          │        embeddings.npz)
                   │ embeddings.py        │
                   └──────────────────────┘

                   ┌──────────────────────-┐
                   │   2. Retrieval &      │
                   │      Generation Plane │
                   │                       │
     user ────────►│ embed query →         │
     question      │ cosine search →       │────► printed answer
                   │ pack prompt →         │     + citations
                   │ call LLM              │     + trace output
                   │                       │     + optional JSON trace
                   │ retrieval.py          │
                   │ prompting.py          │
                   │ generation.py         │
                   └──────────────────────-┘

                   ┌──────────────────────-┐
                   │   3. Evaluation Layer │
                   │                       │
     qa.jsonl ────►│ embed questions →     │────► EvalReport
                   │ retrieve → compare    │     (hit rate, MRR,
                   │ to gold_doc_ids       │      precision, recall)
                   │                       │
                   │ eval.py               │
                   └──────────────────────-┘

                   ┌──────────────────────-┐
                   │   4. Observability    │
                   │                       │
 retrieve / ask ──►│ collect chunks,       │────► RetrieveTrace /
                   │ prompt, answer,       │     AskTrace
                   │ citations, latency    │     (terminal + JSON)
                   │                       │
                   │ trace.py              │
                   └──────────────────────-┘
```

The two planes meet at the index on disk — the indexing plane writes it, the
retrieval plane reads it. The evaluation layer reuses the retrieval plane
exactly as the user experiences it. The observability layer records what
happened in one `retrieve` or `ask` run so you can debug it later.

---

## Data Flow: Document → Answer

The core pipeline uses three data dataclasses, then the observability layer
turns command output into trace dataclasses. Each arrow is a transformation.

```
┌──────────┐     ┌──────────┐     ┌──────────────────┐
│ Document │ ──► │  Chunk   │ ──► │ RetrievalResult  │
└──────────┘     └──────────┘     └──────────────────┘
  indexing          indexing            retrieval

                       ┌──────────────────────────────┐
                       │ RetrieveTrace / AskTrace     │
                       └──────────────────────────────┘
                         observability
```

| Type | Fields (key ones) | Created by | Consumed by |
|---|---|---|---|
| **Document** | `doc_id`, `normalized_text`, `raw_hash`, `title`, `format` | `documents.load_document()` | Chunker |
| **Chunk** | `chunk_id`, `doc_id`, `text`, `char_start`, `char_end`, `metadata` | `chunking.chunk_document()` | Embedder, Retriever |
| **RetrievalResult** | `chunk`, `score`, `rank` (1-indexed) | `retrieval.retrieve_by_vector()` | Prompt assembler |
| **RetrieveTrace** | `query`, `retriever`, `top_k`, `chunks`, `latency_by_stage` | `cli.cmd_retrieve()` | Terminal output, optional JSON trace |
| **AskTrace** | `query`, `retriever`, `top_k`, `chunks`, `prompt`, `answer`, `citations`, `latency_by_stage` | `cli.cmd_ask()` | Terminal output, optional JSON trace |

The critical invariant: `document.normalized_text[chunk.char_start:chunk.char_end] == chunk.text`.
If this breaks, citations point to wrong text.

---

## CLI Surface

```
rag index --corpus PATH --index-dir .tiny-rag/index --chunk-size 800 --chunk-overlap 120
rag retrieve "question" --index-dir .tiny-rag/index --top-k 5 --trace-out /tmp/retrieve.json
rag ask "question" --index-dir .tiny-rag/index --top-k 5 --trace-out /tmp/ask.json
rag eval --qa-file qa.jsonl --index-dir .tiny-rag/index --top-k 5
```

Each command reuses the output of the previous one. `index` builds the index.
`retrieve` searches it. `ask` runs the full pipeline. `eval` measures retrieval
quality.

---

## How the Learning Docs Map to the Pipeline

| Learning doc | Pipeline stage | Source modules it covers |
|---|---|---|
| The RAG Data Flow | Architecture overview | `models.py`, `cli.py` |
| The Indexing Plane | Loading, normalizing, chunking, embedding | `documents.py`, `chunking.py`, `embeddings.py` |
| Retrieval and Generation | Cosine search, prompt assembly, LLM call | `retrieval.py`, `prompting.py`, `generation.py` |
| Persistence and Testing | Save/load index, round-trip integrity, fake backends | `index_writer.py`, `index_loader.py`, test suite |
| Evaluating Retrieval | Retrieval quality metrics | `eval.py` |
| Observability and Debugging | Per-run trace records and JSON artifacts | `trace.py`, `cli.py` |
