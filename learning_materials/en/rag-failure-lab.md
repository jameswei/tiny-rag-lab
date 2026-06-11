# RAG Failure Lab — Turning Retrieval Mistakes Into Test Cases

Phase 1.8 adds a small failure lab. The goal is to stop treating bad answers as
one vague problem and instead classify common retrieval-side failures with
repeatable cases.

---

## What The Failure Lab Adds

The core pieces live in `tiny_rag_lab/failure.py`:

| Piece | Purpose |
|---|---|
| `FailureCase` | One curated question, gold document list, expected label, baseline config, and intervention config |
| `RetrieverConfig` | Per-case retrieval strategy: `dense`, `bm25`, or `hybrid`, plus `top_k` |
| `detect_failure_label()` | Assigns a heuristic label from retrieved document ids and gold document ids |
| `run_diagnosis()` | Runs baseline and intervention retrieval for every case |
| `DiagnosisReport` | Aggregates confirmed, fixed, moved, and per-case results |
| `format_diagnosis_report()` | Prints a readable terminal report |

The CLI entry point is:

```bash
rag diagnose --cases-file tests/fixtures/failure/cases.jsonl --index-dir .tiny-rag/index
```

Unlike `rag eval`, this command does not take `--retriever` or `--top-k`.
Those settings live inside each `FailureCase`, because the whole point is to
compare a baseline retrieval setup with a specific intervention.

---

## The Five Heuristic Labels

Phase 1.8 only labels failures that can be detected from retrieval results:

| Label | Meaning |
|---|---|
| `missing_evidence` | No retrieved document matches the gold document ids |
| `low_rank_evidence` | A gold document was retrieved, but its first hit is below the rank threshold |
| `distractor_evidence` | A gold document is well-ranked, but too much retrieved context is noise |
| `unanswerable_query` | The case intentionally has no gold documents in the corpus |
| `no_failure` | Retrieval satisfies the configured thresholds |

Two important failures are documented but not implemented as heuristics:
`unsupported_answer` and `citation_mismatch`. Those require answer judging,
usually by a human or an LLM-as-judge, because retrieval ids alone cannot prove
whether the generated answer was faithful.

---

## How Detection Reuses Evaluation Metrics

`detect_failure_label()` deliberately reuses the metric helpers from `eval.py`:

```python
hit_at_k(retrieved_doc_ids, gold_doc_ids)
reciprocal_rank(retrieved_doc_ids, gold_doc_ids)
context_precision_at_k(retrieved_doc_ids, gold_doc_ids)
```

This keeps one source of truth for retrieval math.

The detection order matters:

1. Empty gold list with `unanswerable_query` means unanswerable.
2. No hit means missing evidence.
3. A first hit below the rank threshold means low-rank evidence.
4. Low precision with an acceptable rank means distractor evidence.
5. Otherwise, there is no retrieval failure.

Low-rank detection happens before distractor detection. A gold document at rank
4 with many distractors should teach "the evidence is buried", not merely "the
context is noisy".

---

## Reading A Diagnosis Report

A report compares two runs per case:

```text
Diagnosis report  (n=6)
--------------------------------------------
  Confirmed  : 4
  Fixed      : 1
  Moved      : 1
--------------------------------------------
Case fc001  expected=missing_evidence
  baseline   : missing_evidence       hit=0.000  prec=0.000  recall=0.000  mrr=0.000
  interv.    : no_failure             hit=1.000  prec=0.500  recall=1.000  mrr=0.500
  FIXED
```

The outcome word is the main learning signal:

| Outcome | Meaning |
|---|---|
| `CONFIRMED` | The baseline reproduced the expected failure label |
| `FIXED` | The baseline failed, but the intervention returned `no_failure` |
| `MOVED` | The intervention changed the failure into a different failure |
| `UNCHANGED` | The result did not match the expected failure, and it was not fixed or moved |

The key habit is to compare labels and metrics together. `mrr` tells you whether
evidence moved up. `context_precision` tells you whether noise increased.
`context_recall` tells you whether all gold documents were covered.

---

## Why This Is Different From Evaluation

`rag eval` asks: "How well does this retriever perform over a dataset?"

`rag diagnose` asks: "For this known failure, did this intervention fix the
failure, move it, or leave it unchanged?"

Evaluation is broad measurement. Diagnosis is focused learning. Together they
make RAG work less mysterious: metrics reveal the pattern, traces show the run,
and failure cases turn the pattern into something repeatable.
