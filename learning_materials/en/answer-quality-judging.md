# Answer Quality Judging — Measuring The Generated Answer

Phase 2.0 adds answer-quality judging. Retrieval metrics answer "did we fetch
the right evidence?" Answer judging answers "did the final answer use that
evidence correctly?"

The judge is opt-in. Existing commands still behave the same when `--judge`
is omitted or set to `none`.

---

## Why Retrieval Metrics Are Not Enough

A retriever can find the right chunk and the answer can still be bad:

- it may invent a claim not present in the context
- it may answer a different question
- it may cite a source that does not support the sentence
- it may be factually wrong compared with a reference answer

Phase 1.6 measures retrieval. Phase 2.0 measures the generated answer after
retrieval and generation have already happened.

---

## The Judge Interface

The judge contracts live in `tiny_rag_lab/judge.py`.

```python
@dataclass
class JudgeVerdict:
    faithfulness: float
    answer_relevance: float
    citation_support: float
    answer_correctness: float | None
    judge_name: str
    latency: float
    notes: str = ""
```

The four scores mean:

| Score | Meaning |
|---|---|
| `faithfulness` | Is the answer grounded in retrieved context? |
| `answer_relevance` | Does the answer address the question? |
| `citation_support` | Do cited sources support the claims? |
| `answer_correctness` | Does the answer match a reference answer, when one exists? |

`answer_correctness` is `None` when no `reference_answer` is available. That
is different from `0.0`: missing reference data means "not measured", not
"wrong".

Two judge implementations are available:

| Judge | Purpose |
|---|---|
| `FakeJudge` | Deterministic offline tests and examples |
| `OpenAIJudge` | Real OpenAI-compatible JSON-mode judging |

`FakeJudge.verdict_map` is keyed by answer string. That matters for the
failure lab because baseline and intervention can use the same question but
different scripted answers.

---

## Answer Evaluation

`rag eval` can now run two reports in one command:

```bash
rag eval \
  --qa-file tests/fixtures/eval/qa.jsonl \
  --index-dir .tiny-rag/index \
  --judge fake \
  --generator fake
```

The retrieval section remains separate from the answer-quality section:

```text
Evaluation report  (...)
...

Answer quality report  (n=3, judge=fake)
--------------------------------------------
Faithfulness      :  1.000
Answer Relevance  :  1.000
Citation Support  :  1.000
```

This separation is intentional. Retrieval quality and answer quality can move
independently. A reranker might improve hit rate while the generator still
produces an unsupported answer, or a better prompt might improve faithfulness
without changing retrieval metrics.

Rows in `qa.jsonl` may include optional answer-side fields:

```json
{
  "question_id": "q1",
  "question": "What does the document cover?",
  "answer": "It covers retrieval testing.",
  "gold_doc_ids": ["with_h1.md"],
  "reference_answer": "The document covers retrieval testing topics.",
  "expected_facts": ["retrieval testing"]
}
```

Existing rows without those fields still load. `reference_answer` defaults to
`None`; `expected_facts` defaults to an empty list.

---

## Ask Trace Verdicts

`rag ask` can judge a single generated answer:

```bash
rag ask "sample document" \
  --index-dir .tiny-rag/index \
  --judge fake \
  --generator fake \
  --trace-out /tmp/ask.json
```

When judging is active, `AskTrace.verdict` is populated and the readable trace
adds a `Judge verdict` block. When judging is disabled, `verdict` is `null` in
JSON and no extra block appears.

This makes answer quality debuggable at the same level as retrieval: you can
see the chunks, prompt, answer, citations, latency, and judge verdict together.

---

## Answer-Side Failure Diagnosis

Phase 1.8 documented two failures that retrieval IDs cannot detect:

| Label | Meaning |
|---|---|
| `unsupported_answer` | The answer makes a claim not grounded in context |
| `citation_mismatch` | A citation does not support the claim it is attached to |

Phase 2.0 implements them with the judge:

```python
detect_answer_failure_label(verdict, thresholds)
```

Detection order is simple:

1. low `faithfulness` -> `unsupported_answer`
2. low `citation_support` -> `citation_mismatch`
3. otherwise -> `no_failure`

`rag diagnose --judge fake --generator fake` now prints the retrieval diagnosis
and then an answer diagnosis section for answer-side cases.

The fixture adds two cases:

| Case | Failure |
|---|---|
| `fc008` | unsupported answer |
| `fc009` | citation mismatch |

Those cases have `baseline_answer` and `intervention_answer` scripted directly
in the JSONL. When these fields are present, `run_answer_diagnosis` skips the
generator and judges the scripted answer. This keeps the failure lab fully
deterministic and offline.

---

## The Main Lesson

A production RAG system needs at least two measurement layers:

| Layer | Question |
|---|---|
| Retrieval evaluation | Did we retrieve the right evidence? |
| Answer judging | Did the answer use the evidence correctly? |

Phase 2.0 keeps both visible. Retrieval metrics, answer metrics, traces, and
failure cases remain separate objects so the reason for a regression stays
inspectable.
