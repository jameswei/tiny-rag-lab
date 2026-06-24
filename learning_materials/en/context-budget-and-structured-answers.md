# Context Budget And Structured Answers — Controlling What Reaches The Prompt

Phase 2.1 adds generation-side mechanics that were previously invisible:
which retrieved chunks actually enter the prompt, and why some are left out.
It also adds a machine-readable JSON output mode for `rag ask`.

---

## The Gap Between Retrieval and Prompt

After retrieval, you have a list of ranked chunks. Until Phase 2.1, all of
them entered the prompt unchanged. That works when `top_k` is small and
chunks are short. It breaks when context windows are finite:

- a 4 096-token window cannot hold ten 600-token chunks
- a reranker may return many candidates but only the top few matter
- you cannot tell from the output *which* chunks were included

Phase 2.1 makes the selection step explicit and inspectable.

---

## The Context Packing Step

`pack_context` in `tiny_rag_lab/context.py` does greedy selection:

```python
def pack_context(
    results: list[RetrievalResult],
    budget: int,
    counter: TokenCounter,
    question: str = "",
) -> ContextPackResult:
    ...
```

Algorithm:

1. Deduct `PROMPT_OVERHEAD` (100 tokens) plus the question token count from
   the budget. This reserves space for the prompt template wrapper.
2. Walk chunks in rank order.
3. For each chunk, format it the same way `assemble_prompt` would, count its
   tokens, and select it if it fits the remaining budget.
4. Return `ContextPackResult` with `selected`, `omitted`, and
   `estimated_tokens`.

The caller (`cmd_ask`, `run_answer_eval`, `run_answer_diagnosis`) then filters
the result list to `selected` chunk IDs before calling `assemble_prompt`.
`prompting.py` is unchanged.

---

## Token Counters

Two implementations are available:

| Counter | Accuracy | Requires |
|---|---|---|
| `FakeTokenCounter` | ~4 chars per token | nothing (always available) |
| `TiktokenCounter` | exact tiktoken encoding | `pip install tiktoken` |

`cli.py` auto-selects: `_make_token_counter()` tries `TiktokenCounter` and
falls back to `FakeTokenCounter` when tiktoken is not installed. Tests use
`FakeTokenCounter` directly; the fallback is transparent in production.

The key constraint: `FakeTokenCounter` is fast and offline but may over- or
under-select by a few tokens. `TiktokenCounter` is exact for OpenAI-family
models.

---

## What Appears In The Trace

When `--context-budget > 0`, `AskTrace.context_pack` is populated:

```json
{
  "context_pack": {
    "selected": ["chunk0000000001", "chunk0000000002"],
    "omitted":  ["chunk0000000003"],
    "estimated_tokens": 847,
    "budget": 8192,
    "counter_name": "tiktoken-gpt-4o-mini"
  }
}
```

The human-readable trace adds a block between the chunk list and the answer:

```text
Context packing  (budget=8192, counter=tiktoken-gpt-4o-mini)
  Selected  : 2 chunks   (~847 tokens used)
  Omitted   : 1 chunk
    - chunk0000000003
```

When `--context-budget 0` (the default), `context_pack` is `null` and the
block is absent. Phase 2.0 output is preserved exactly.

---

## CLI Usage

```bash
# Default: no budget, identical to Phase 2.0
rag ask "question" --index-dir .tiny-rag/index

# With budget: packing block appears in trace
rag ask "question" --index-dir .tiny-rag/index --context-budget 8192

# JSON output: full AskTrace as indented JSON to stdout
rag ask "question" --index-dir .tiny-rag/index --context-budget 8192 --output-format json

# eval and diagnose also accept --context-budget
rag eval --qa-file qa.jsonl --index-dir .tiny-rag/index --judge fake --generator fake --context-budget 8192
rag diagnose --cases-file cases.jsonl --index-dir .tiny-rag/index --judge fake --context-budget 8192
```

`--context-budget -1` raises a `ValueError` immediately. `--context-budget 0`
skips packing entirely.

---

## JSON Output Mode

`--output-format json` prints the full `AskTrace` dict to stdout instead of
the human-readable trace. This makes answer output consumable by downstream
scripts without parsing text:

```bash
rag ask "question" --index-dir .tiny-rag/index \
  --context-budget 8192 \
  --output-format json | jq '.context_pack.omitted'
```

`--trace-out PATH` continues to write JSON to a file regardless of
`--output-format`.

---

## The Main Lesson

Three things become visible that were previously hidden:

| Before Phase 2.1 | After Phase 2.1 |
|---|---|
| All retrieved chunks enter the prompt | Budget selects chunks greedily in rank order |
| No record of what was included or excluded | `context_pack.selected` and `omitted` lists |
| Text-only output from `rag ask` | Optional JSON output via `--output-format json` |

A tight budget forces you to see that ranked-first does not mean
fits-in-context. A chunk can rank highly but still be omitted if earlier
chunks consumed the remaining budget. That is a production reality that
Phase 2.1 makes observable.
