# Phase 3.4 Curated Retrieval Content Manifest

**Status:** Active content contract — owner accepted and independent
architecture review signed off 2026-07-16.

## Dataset Contract

- Corpus: bundled Cloudflare technical documentation selected in Phase 3.2.
- Index: bundled `cloudflare-state-structural-v1` NumPy index.
- Language: questions and source artifacts remain English; surrounding Studio
  teaching copy is English/Simplified Chinese.
- Size: exactly 16 reviewed questions, four per teaching category.
- Gold paths refer to corpus-relative Markdown documents and may contain more
  than one relevant document when the question intentionally crosses concepts.
- Expected observations guide lesson copy and regression assertions. They do
  not encode a permanent claim that one retrieval strategy always wins.

## Lexical Search

| ID | Question | Gold path(s) | Teaching purpose | Expected observation |
|---|---|---|---|---|
| `cf-lex-block-concurrency` | What does `blockConcurrencyWhile` prevent during Durable Object initialization? | `durable-objects/api/state.md` | Inspect how an exact API identifier becomes tokens and contributes to BM25. | Exact terms strongly identify the state API document; punctuation/token boundaries remain visible. |
| `cf-lex-batch-limits` | How do `max_batch_size` and `max_batch_timeout` trigger queue batch delivery? | `queues/configuration/batching-retries.md` | Compare two rare configuration terms and their individual BM25 contributions. | Both identifiers concentrate lexical evidence in the batching-and-retries document. |
| `cf-lex-retry-limit` | What does `max_retries` control for a queue consumer, and what happens after the limit? | `queues/configuration/batching-retries.md` | Connect an exact option to surrounding retry and dead-letter behavior. | The identifier contributes strongly while common explanatory words contribute less. |
| `cf-lex-alarm-method` | What does `setAlarm` schedule for a Durable Object? | `durable-objects/api/alarms.md` | Expose the strengths and limitations of the lab's intentionally simple tokenizer. | The explanation makes casing/backtick/token-boundary effects understandable instead of hiding a miss. |

## Dense Retrieval

| ID | Question | Gold path(s) | Teaching purpose | Expected observation |
|---|---|---|---|---|
| `cf-dense-kv-cache` | How does Workers KV trade immediate global updates for fast repeated reads? | `kv/concepts/how-kv-works.md` | Retrieve a paraphrase of the consistency/cache tradeoff without relying on exact wording. | Semantic similarity ranks the KV concepts document near the top. |
| `cf-dense-delivery-safety` | Why should a queue consumer make repeated message processing safe? | `queues/reference/delivery-guarantees.md` | Connect “safe repeated processing” with idempotency and at-least-once delivery. | Dense retrieval bridges the learner's paraphrase to the delivery-guarantee terminology. |
| `cf-dense-stale-edge` | Why can a key-value update be visible nearby but delayed elsewhere? | `kv/concepts/how-kv-works.md` | Inspect a natural-language description of eventual global propagation. | The relevant KV document ranks highly despite few product-specific tokens. |
| `cf-dense-workflow-recovery` | How can a long-running multi-step process pause and recover after failures? | `workflows/build/sleeping-and-retrying.md`; `workflows/build/rules-of-workflows.md` | Show semantic retrieval across two complementary workflow documents. | Both sleep/retry and durable-execution evidence should appear in the candidate set. |

## Hybrid Retrieval

| ID | Question | Gold path(s) | Teaching purpose | Expected observation |
|---|---|---|---|---|
| `cf-hybrid-r2-properties` | How do R2 consistency and durability differ? | `r2/reference/consistency.md`; `r2/reference/durability.md` | Combine exact R2 vocabulary with semantically related durability language. | Dense and BM25 favor different relevant documents; RRF retains evidence from both. |
| `cf-hybrid-queue-policy` | How should retry limits, dead-letter handling, and at-least-once delivery shape a queue consumer? | `queues/configuration/batching-retries.md`; `queues/reference/delivery-guarantees.md` | Join configuration terms with a broader reliability concept. | Each retriever contributes complementary queue evidence to the fused list. |
| `cf-hybrid-workflow-state` | How can durable steps sleep, retry, and preserve progress without relying on memory? | `workflows/build/sleeping-and-retrying.md`; `workflows/build/rules-of-workflows.md` | Mix exact workflow actions with a semantic durability requirement. | BM25 and dense surface complementary gold paths and their RRF contributions are inspectable. |
| `cf-hybrid-storage-choice` | When do KV caching semantics suit configuration while R2 consistency suits updates that must become globally visible immediately? | `kv/concepts/how-kv-works.md`; `r2/reference/consistency.md` | Compare concepts across two storage products in one query. | The fused list preserves relevant KV and R2 evidence that is split across source rankings. |

## Reranking

| ID | Question | Gold path(s) | Teaching purpose | Expected observation |
|---|---|---|---|---|
| `cf-rerank-named-stub` | What does `getByName` return for a Durable Object namespace? | `durable-objects/best-practices/create-durable-object-stubs-and-send-requests.md` | Observe a precise API answer move within a broader hybrid candidate pool. | The cross-encoder promotes the create-stubs document toward the final top results. |
| `cf-rerank-alarm-without-request` | How can a stateful coordinator schedule work even if no new request arrives? | `durable-objects/api/alarms.md` | Rerank a semantic alarm description that first-stage retrieval scatters. | The alarm document moves substantially upward from the hybrid candidate pool. |
| `cf-rerank-exhausted-delivery` | What happens to queue messages after they exhaust delivery attempts? | `queues/configuration/batching-retries.md` | Separate the final retry/dead-letter answer from generally related queue documents. | Reranking promotes the batching-and-retries evidence to or near first place. |
| `cf-rerank-workflow-sleep` | What do `step.sleep` and `step.sleepUntil` do in a Workflow? | `workflows/build/sleeping-and-retrying.md` | Compare first-stage lexical/semantic relevance with cross-encoder query-passage relevance. | The sleeping-and-retrying document moves to the first position or remains the strongest final evidence. |

## Preset Comparison Contract

All 16 questions participate in every browser evaluation preset. Category
labels are teaching annotations used to group results; they do not restrict a
question to one retriever.

| Preset | Configuration A | Configuration B | Primary lesson |
|---|---|---|---|
| BM25 vs Dense | BM25, top 5, no reranker | Dense, top 5, no reranker | Exact-token evidence versus semantic similarity. |
| Dense vs Hybrid | Dense, top 5, no reranker | Hybrid, top 5, no reranker | Whether lexical and semantic signals complement one another. |
| Hybrid vs Hybrid + cross-encoder | Hybrid, top 5, no reranker | Hybrid candidate depth 20, cross-encoder, final top 5 | Candidate generation versus final relevance ordering. |

## Content Acceptance Checks

1. Every stable question ID is unique and all gold paths exist in the pinned
   bundled corpus.
2. Each category contains exactly four questions and the evaluation set totals
   16.
3. The default cached embedding and cross-encoder models can reproduce the
   stated observations within deliberately non-brittle rank assertions.
4. Lesson text distinguishes an expected insight from a guaranteed model win.
5. Seed generation preserves the exact reviewed wording and gold paths.
