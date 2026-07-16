# Reranking — From Candidates to Final Evidence

Retrieval and reranking solve different problems. A first-stage retriever must
search the whole index cheaply enough to produce a useful candidate pool. A
reranker spends more computation on that much smaller pool and decides which
candidates deserve the final evidence positions.

```text
all chunks -> first-stage retriever -> 20 candidates
           -> cross-encoder reranker -> final top 5
```

The first stage protects recall: the relevant chunk cannot be promoted if it
never enters the candidate pool. The second stage improves ordering and often
precision: fewer distractors reach context packing.

## Bi-encoder and cross-encoder scoring

Dense retrieval is a **bi-encoder** path. It embeds the query and every chunk
separately, then compares two stored vectors. Chunk vectors can be computed
once and reused, which makes whole-index search practical.

A cross-encoder instead reads the query and one candidate together:

```text
score = cross_encoder(query, candidate_text)
```

Joint attention can notice fine-grained query–passage relationships that a
single cosine similarity misses. The cost is that the model must run once per
candidate, so it is unsuitable as the first pass over hundreds or thousands of
chunks in this lab.

## The candidate-depth contract

`top_k` and `rerank_top_n` have different meanings:

- `rerank_top_n` is the number of first-stage candidates sent to the
  cross-encoder.
- `top_k` is the number of reranked results retained as final evidence.
- `rerank_top_n` must be greater than or equal to `top_k`.

Increasing candidate depth may recover evidence that the first stage ranked
too low, but it also increases cross-encoder work. Reranking cannot repair a
missing corpus document, a damaging chunk boundary, or a relevant chunk below
the candidate cutoff.

## Reading the rank-movement audit

The Studio's **Retrieval → Reranking** module shows every candidate before and
after the second pass:

| Field | Meaning |
|---|---|
| First rank and score | The candidate's position and score from dense, BM25, or hybrid retrieval |
| Reranker score | The cross-encoder's query–candidate relevance score |
| Final rank | Its position after reranking, if it remains in the final top-k |
| Movement | Promoted, demoted, unchanged, or dropped |

Do not compare a BM25, cosine, RRF, and cross-encoder score as if they shared a
scale. The useful evidence is the ordering each scorer produced and how that
ordering changed.

## One path for retrieval and generation

Explore applies the selected reranker to both **Retrieve** and **Live Ask**.
That consistency matters: the evidence a learner inspects must be the evidence
packed for generation. Otherwise the answer could be grounded in a different
candidate order from the one visible on screen.

The same idea is available through the CLI:

```bash
rag retrieve "What happens after queue delivery attempts are exhausted?" \
  --retriever hybrid --top-k 5 \
  --reranker cross-encoder --rerank-top-n 20

rag ask "What happens after queue delivery attempts are exhausted?" \
  --retriever hybrid --top-k 5 \
  --reranker cross-encoder --rerank-top-n 20
```

## What to inspect

1. Start with a curated Reranking question in the Studio.
2. Find the gold source in the first-stage pool.
3. Compare its first rank with its final rank.
4. Inspect which candidates were dropped from the final top five.
5. In Evaluation, compare Hybrid with Hybrid + cross-encoder across all 16
   reviewed questions. Check per-question evidence before interpreting the
   aggregate metrics.

Reranking is valuable when it produces a better final evidence set—not merely
when rows move.

## Related guides

- [Retrieval Mechanics](retrieval-mechanics.md) — BM25, dense, RRF, and the
  first-stage candidate lists.
- [Evaluating Retrieval](evaluating-retrieval.md) — metrics for checking
  whether rank changes improved retrieval quality.
- [Context Budget and Structured Answers](context-budget-and-structured-answers.md)
  — how final evidence is selected for the prompt.
