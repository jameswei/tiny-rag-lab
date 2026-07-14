# Phase 3.2 Content Manifest: Cloudflare State & Coordination

**Status:** Owner approved 2026-07-13. The reviewed corpus, indexes, and
saved-lesson package are bundled as immutable Phase 3.2 seed assets.

## Source and attribution

- Repository: [`cloudflare/cloudflare-docs`](https://github.com/cloudflare/cloudflare-docs)
- Pinned revision: `3dcb728cb29f4239e08ba894f0a40650d51ba4f6` (`production`,
  2026-07-13)
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Intended packaging: preserve each selected source path, the pinned revision,
  the original source URL, and attribution in the bundled corpus manifest.

## Proposed documents (40)

Every path below is relative to the pinned repository's `src/content/docs/`.

| Area | Count | Selected source paths |
|---|---:|---|
| Workers integration | 6 | `workers/index.mdx`; `workers/get-started/guide.mdx`; `workers/runtime-apis/bindings/durable-objects.mdx`; `workers/runtime-apis/bindings/queues.mdx`; `workers/runtime-apis/bindings/kv.mdx`; `workers/runtime-apis/bindings/R2.mdx` |
| Durable Objects | 8 | `durable-objects/concepts/what-are-durable-objects.mdx`; `durable-objects/concepts/durable-object-lifecycle.mdx`; `durable-objects/platform/storage-options.mdx`; `durable-objects/best-practices/rules-of-durable-objects.mdx`; `durable-objects/best-practices/create-durable-object-stubs-and-send-requests.mdx`; `durable-objects/api/state.mdx`; `durable-objects/api/alarms.mdx`; `durable-objects/examples/build-a-counter.mdx` |
| Queues | 7 | `queues/index.mdx`; `queues/get-started.mdx`; `queues/reference/how-queues-works.mdx`; `queues/reference/delivery-guarantees.mdx`; `queues/configuration/batching-retries.mdx`; `queues/configuration/consumer-concurrency.mdx`; `queues/examples/use-queues-with-durable-objects.mdx` |
| KV | 6 | `kv/index.mdx`; `kv/concepts/how-kv-works.mdx`; `kv/concepts/kv-bindings.mdx`; `kv/concepts/kv-namespaces.mdx`; `kv/examples/distributed-configuration-with-workers-kv.mdx`; `kv/examples/cache-data-with-workers-kv.mdx` |
| R2 | 6 | `r2/index.mdx`; `r2/how-r2-works.mdx`; `r2/reference/consistency.mdx`; `r2/reference/durability.mdx`; `r2/api/workers/workers-api-reference.mdx`; `r2/buckets/create-buckets.mdx` |
| Workflows | 7 | `workflows/index.mdx`; `workflows/get-started/guide.mdx`; `workflows/build/rules-of-workflows.mdx`; `workflows/build/sleeping-and-retrying.mdx`; `workflows/build/step-context.mdx`; `workflows/build/trigger-workflows.mdx`; `workflows/build/workers-api.mdx` |

The canonical guided index uses structural chunking with an 800-character
limit. The companion fixed-character index uses the existing 800-character,
120-character-overlap baseline. Both use the project default
`sentence-transformers/all-MiniLM-L6-v2` embedding model and NumPy storage.

The implementation-derived lesson package must record the pinned source
revision, the selected structural-index digest and ID, each lesson's retrieval
configuration, and `recorded_lesson_result` answer provenance. This makes a
saved answer inspectable as a captured teaching artifact rather than a live
generation claim.

## Proposed saved lessons (4)

Each lesson uses the structural index, dense retrieval, top-k 5, and a
captured context pack, prompt, answer, and citations. The saved answer is a
clearly-labelled recorded lesson result; it is not generated when a learner
replays it.

| ID | Order | Question | Learning focus | Expected primary evidence |
|---|---:|---|---|---|
| `cloudflare-do-coordinator-v1` | 1 | How can a Worker use a Durable Object namespace, stable ID, and stub to send requests for the same entity to one stateful coordinator? | Document-to-chunk provenance, Durable Object identity/stubs, and rank ordering. | Workers Durable Object binding; what Durable Objects are; create stubs/send requests. |
| `cloudflare-queues-retries-v1` | 2 | If a queue consumer fails while processing a message, how should retries and batching be considered? | Retrieval candidates versus context selection for delivery behavior. | How Queues work; delivery guarantees; batching and retries. |
| `cloudflare-kv-r2-choice-v1` | 3 | What is the tradeoff between Workers KV eventual consistency for global configuration and R2 object storage for mutable files? | Multi-document evidence and a constrained context pack. | How KV works; distributed configuration with KV; how R2 works; R2 consistency/durability. |
| `cloudflare-workflows-resume-v1` | 4 | How can a long-running multi-step process pause, retry, and resume safely? | Prompt construction, saved grounded answer, and citations. | Workflows guide; rules; sleeping/retrying; step context. |

## Review checklist

- The topic remains a neutral, transferable technical learning surface rather
  than a product tutorial.
- Every lesson has enough relevant and competing chunks to make retrieval and
  context decisions visible.
- The selected source paths and questions are useful to an English-reading
  learner even when the UI is shown in Chinese.
- The four lessons collectively demonstrate corpus, chunking, embeddings,
  retrieval, context packing, answer, and citations without requiring a live
  provider.
