# Phase 2.1 Spec: Context Budget And Structured Answers

**Status:** Complete
**Authors:** Claude Code
**Based on:** `docs/phases/phase-1.9-2.2-final-roadmap.md`
**Taskboard:** `docs/phases/phase-2.1-taskboard.md`
**Date:** 2026-06-23

---

## Goal

Add generation-side production mechanics that control how much retrieved context
enters the prompt and optionally return machine-readable answers.

After Phase 2.0, we can measure whether an answer is faithful and relevant.
Phase 2.1 teaches the preceding constraint: which retrieved chunks actually reach
the generation step, and why some are left out. Context budgeting makes that
selection explicit and inspectable.

Budget enforcement defaults to **disabled** (`--context-budget 0` = unlimited).
Users opt-in by passing `--context-budget INT` where INT > 0. This preserves
existing command behaviour and test output. When a budget is set, context
packing is applied and a context packing block appears in the trace.

---

## Scope

### In Scope

- `tiny_rag_lab/context.py` (new): `TokenCounter` protocol, `FakeTokenCounter`,
  `TiktokenCounter` (lazy-loaded), `ContextPackResult` dataclass,
  `pack_context()` function, `PROMPT_OVERHEAD` constant
- `trace.py`: `AskTrace.context_pack: ContextPackResult | None = None`;
  `format_ask_trace` renders a context packing block when `context_pack` is
  present
- `cli.py`: `_make_token_counter() -> TokenCounter` factory; `--context-budget
  INT` flag on `rag ask`, `rag eval`, `rag diagnose` (default `0` = unlimited,
  must be >= 0); `--output-format text|json` on `rag ask` (default `text`)
- `eval.py`: `run_answer_eval` gains `counter: TokenCounter | None = None` and
  `context_budget: int | None = None` optional parameters
- `failure.py`: `run_answer_diagnosis` gains `counter: TokenCounter | None =
  None` and `context_budget: int | None = None` optional parameters
- `pyproject.toml`: `tiktoken` added as an optional dependency
  (`[project.optional-dependencies] tiktoken = ["tiktoken"]`)
- Tests: `tests/test_context.py` (new); updates to `tests/test_trace.py`,
  `tests/test_eval_runner.py`, `tests/test_cmd_ask.py`,
  `tests/test_cmd_eval.py`, `tests/test_cmd_diagnose.py`

### Out Of Scope For Phase 2.1

- Per-chunk token counts in `ChunkTrace` (only the aggregate packing summary is
  recorded)
- JSON output mode for `rag eval` or `rag diagnose`
- Streaming generation responses
- System-vs-user prompt restructuring
- Dynamic budget derived from the model's published context window
- Semantic or structural chunking changes (Phase 2.2)

---

## Design Decision 1: Budget Defaults To Disabled (0); Opt-In Via CLI Flag

`--context-budget` defaults to `0` (unlimited, no packing). This preserves
Phase 2.0 behaviour and test output. Users opt-in by passing `--context-budget
8192` or another positive integer. Chunks are then selected greedily in rank
order until the context token estimate reaches the budget.

When `--context-budget 0`: packing is skipped, all retrieved chunks pass to
`assemble_prompt`, and `AskTrace.context_pack` is `None`. When `--context-budget
> 0`: packing is applied, chunks may be omitted, and `AskTrace.context_pack`
is populated.

## Design Decision 2: Token Counter Is Auto-Selected, Not A CLI Flag

No `--counter` flag is added. `_make_token_counter()` in `cli.py` tries to
construct `TiktokenCounter`; if tiktoken is not installed it falls back to
`FakeTokenCounter`. Tests that need a specific counter inject it as a function
argument — the same pattern used for embedders, generators, and judges in prior
phases.

`FakeTokenCounter` uses `int(len(text) * 0.25)` (4 chars ≈ 1 token), which is
adequate for budget estimation and for all unit tests.

## Design Decision 3: Budget Applies To Context Blocks Using The Same Format As The Prompt

`pack_context()` receives the retrieval results and the query text. It first
deducts `PROMPT_OVERHEAD` (a module-level constant, default `100`) plus the
question token count from the budget. The remainder is available for context
blocks.

**Critical:** `pack_context` must count each block using the **exact same
formatting** that `assemble_prompt` uses. In `tiny_rag_lab/prompting.py`,
`_format_context_block` renders a chunk with `CONTEXT_BLOCK_TEMPLATE`. The
`pack_context` function must reuse that same template (by importing it or
calling a shared formatter) so token counts match the actual prompt. Otherwise
budget estimates can drift from what reaches the model.

This matches how production RAG systems define "context budget" and makes
`PROMPT_OVERHEAD` a visible, adjustable constant rather than a hidden
assumption.

## Design Decision 4: `pack_context` Returns A Result; The Caller Filters

`pack_context` does not call `assemble_prompt`. It returns `ContextPackResult`
with the selected and omitted `chunk_id` lists. The caller (`cmd_ask`,
`run_answer_eval`, etc.) filters the `RetrievalResult` list to only the
selected IDs and then calls `assemble_prompt(question, filtered_results)` as
before. This keeps `prompting.py` unchanged.

## Design Decision 5: `format_ask_trace` Renders A Context Packing Block

When `AskTrace.context_pack` is not `None`, `format_ask_trace` appends a
packing block after the retrieved sources section:

```
Context packing  (budget=8192, counter=tiktoken-gpt-4o-mini)
  Selected  : 3 chunks   (~847 tokens used)
  Omitted   : 1 chunk    (budget_exceeded_chunk_xxx)
```

When there are no omitted chunks the `Omitted` line still appears with `0
chunks`. When `context_pack` is `None` (budget=0), the block is absent.

## Design Decision 6: `--output-format json` Serializes The Full `AskTrace`

When `--output-format json`, `cmd_ask` calls `trace_to_dict(trace)` and
prints the result as indented JSON to stdout instead of calling
`format_ask_trace`. This reuses existing serialization and exposes all fields —
including verdict (when `--judge` is active) and context packing metadata.

`--trace-out PATH` continues to write indented JSON regardless of
`--output-format`.

## Design Decision 7: `run_answer_eval` And `run_answer_diagnosis` Are Opt-In

Both functions gain `counter: TokenCounter | None = None` and `context_budget:
int | None = None`. When `counter` is `None` or `context_budget` is `0` or
`None`, no packing is applied and behaviour is identical to Phase 2.0. All
existing tests continue to pass without modification. The CLI always passes a
non-`None` counter when `--context-budget > 0`.

---

## Data Contracts

### `tiny_rag_lab/context.py` (new)

```python
PROMPT_OVERHEAD = 100   # token reserve for question template wrapper


@dataclass
class ContextPackResult:
    """Records which chunks were selected or dropped by the context budget.

    All fields are JSON-native so dataclasses.asdict() serialises AskTrace
    without a custom encoder.
    """
    selected: list[str]       # chunk_ids in rank order that fit in budget
    omitted: list[str]        # chunk_ids dropped to stay within budget
    estimated_tokens: int     # tokens consumed by the selected context blocks
    budget: int               # the effective budget that was applied
    counter_name: str         # e.g. "tiktoken-gpt-4o-mini" or "char"


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


@dataclass
class FakeTokenCounter:
    name: str = "char"
    tokens_per_char: float = 0.25

    def count(self, text: str) -> int:
        return int(len(text) * self.tokens_per_char)


class TiktokenCounter:
    """Counts tokens with tiktoken. Lazy-imports tiktoken on construction."""

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        import tiktoken          # raises ImportError if not installed
        self._enc = tiktoken.encoding_for_model(model)
        self.name = f"tiktoken-{model}"

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


def pack_context(
    results: list[RetrievalResult],
    budget: int,
    counter: TokenCounter,
    question: str = "",
) -> ContextPackResult:
    """Greedily select chunks in rank order until the context budget is used.

    Deducts PROMPT_OVERHEAD + question tokens from budget before chunk
    selection. Token counts for each block must use the exact formatting of
    CONTEXT_BLOCK_TEMPLATE from tiny_rag_lab/prompting.py, so estimates match
    the actual prompt that assemble_prompt will produce.

    Returns ContextPackResult; caller filters results by
    pack_result.selected before calling assemble_prompt.

    Raises ValueError if budget < 0.
    """
```

### Update To `tiny_rag_lab/trace.py`

```python
@dataclass
class AskTrace:
    ...  # all existing fields unchanged
    context_pack: ContextPackResult | None = None   # NEW Phase 2.1
```

`ContextPackResult` is a dataclass with all JSON-native fields, so
`dataclasses.asdict()` continues to serialise the full trace without a custom
encoder. When `context_pack` is `None` it serialises as `null`.

`format_ask_trace` appends a context packing block after the sources section
when `trace.context_pack is not None`:

```
Context packing  (budget=8192, counter=tiktoken-gpt-4o-mini)
  Selected  : 3 chunks   (~847 tokens used)
  Omitted   : 0 chunks
```

When chunks were omitted, each omitted `chunk_id` is listed on its own line:

```
  Omitted   : 1 chunk
    - chunk_id_abc123
```

---

## Core Functions And Factories

```python
# tiny_rag_lab/context.py
def pack_context(
    results: list[RetrievalResult],
    budget: int,
    counter: TokenCounter,
    question: str = "",
) -> ContextPackResult:
    """
    1. overhead = PROMPT_OVERHEAD + counter.count(question)
    2. remaining = budget - overhead
    3. For each result in rank order:
         block_tokens = counter.count(formatted_block)
         if remaining >= block_tokens: select it, decrement remaining
         else: omit it
    4. Return ContextPackResult(selected, omitted, estimated_tokens, budget, counter_name)
    """
```

```python
# tiny_rag_lab/cli.py
def _make_token_counter() -> TokenCounter:
    try:
        from tiny_rag_lab.context import TiktokenCounter
        return TiktokenCounter()
    except ImportError:
        from tiny_rag_lab.context import FakeTokenCounter
        return FakeTokenCounter()
```

---

## CLI Changes

### `rag ask`

New flags:
- `--context-budget INT` (default `0` = unlimited; must be >= 0; raises error if negative)
- `--output-format text|json` (default `text`)

Pipeline change when `context_budget > 0`:
1. After retrieval (and optional reranking), call
   `pack_context(results, context_budget, counter, question=query)`
2. Filter `results` to `[r for r in results if r.chunk.chunk_id in pack_result.selected]`
3. Call `assemble_prompt(query, filtered_results)` as before
4. Attach `pack_result` to `AskTrace.context_pack`

When `context_budget == 0`: skip packing; `AskTrace.context_pack = None`.

When `--output-format json`: print `json.dumps(trace_to_dict(trace), indent=2)`
to stdout instead of `format_ask_trace(trace)`.

### `rag eval`

New flags:
- `--context-budget INT` (default `0` = unlimited; must be >= 0; raises error if negative)

Passes `counter=_make_token_counter()` and `context_budget=args.context_budget`
to `run_answer_eval` when `args.context_budget > 0`, else passes `counter=None`.

### `rag diagnose`

New flags:
- `--context-budget INT` (default `0` = unlimited; must be >= 0; raises error if negative)

Same pattern: passes counter + budget to `run_answer_diagnosis` when budget > 0.

---

## Required Tests

### `tests/test_context.py` (new)

- `FakeTokenCounter().count(text)` returns `int(len(text) * 0.25)`
- `FakeTokenCounter` is deterministic across repeated calls
- `pack_context` with `budget=0` or very large budget: `selected=all chunk_ids`,
  `omitted=[]`
- `pack_context` with tight budget: first N chunks selected, remainder omitted
- `pack_context` deducts question tokens from available budget (verified by
  checking with a long question vs short question against the same tight budget)
- `pack_context` uses the same block format as `CONTEXT_BLOCK_TEMPLATE` from
  `prompting.py` (token counts match formatted output)
- `pack_context(budget=-1, ...)` raises `ValueError` with clear message
- `ContextPackResult` round-trips through `dataclasses.asdict()` with all
  JSON-native types
- `TiktokenCounter` tests: `pytest.importorskip("tiktoken")`; `count` returns
  a positive integer for non-empty text; `name` starts with `"tiktoken-"`

### Updates To `tests/test_trace.py`

- `AskTrace(context_pack=None)` serialises `"context_pack": null`
- `AskTrace(context_pack=ContextPackResult(...))` serialises all pack fields
- `format_ask_trace` with `context_pack=None`: no `"Context packing"` line
- `format_ask_trace` with `context_pack` populated: contains `"Context packing"`,
  `"Selected"`, `"Omitted"` lines

### Updates To `tests/test_eval_runner.py`

- `run_answer_eval(counter=None)` or `run_answer_eval(context_budget=0)`: no
  packing applied, Phase 2.0 behaviour preserved
- `run_answer_eval(counter=FakeTokenCounter(), context_budget=8192, ...)`:
  returns `AnswerEvalReport`; budget not hit on small fixture, all chunks in
  context
- `run_answer_eval(counter=FakeTokenCounter(), context_budget=200, ...)` with
  typical multi-chunk retrieval: returns `AnswerEvalReport`; some chunks
  omitted; verify `FakeGenerator` answers from low-budget eval contain fewer
  source markers than high-budget eval on same sample
- `run_answer_eval(counter=FakeTokenCounter(), context_budget=-1, ...)`: raises
  `ValueError` about negative budget

### Updates To `tests/test_cmd_ask.py`

- `--context-budget 0 --output-format text` (default): exits 0; stdout has no
  `"Context packing"` block; identical to Phase 2.0 ask output
- `--context-budget 8192 --output-format text`: exits 0; stdout contains
  `"Context packing"` block showing selected/omitted counts
- `--context-budget 500 --output-format text --generator fake`: exits 0; with
  tight budget, `FakeGenerator` answer contains fewer source markers than
  available chunks, proving budget filtered them
- `--context-budget 8192 --output-format json`: exits 0; stdout is valid JSON;
  `json.loads(stdout)` has keys `"answer"` and `"context_pack"`
- `--context-budget -1 --output-format text`: exits non-zero with clear error
  message about negative budget
- `--output-format json --judge fake --generator fake --context-budget 8192`:
  verdict and context_pack fields both appear in JSON output

### Updates To `tests/test_cmd_eval.py`

- `--context-budget 0 --judge fake --generator fake` (default): exits 0; output
  identical to Phase 2.0
- `--context-budget 8192 --judge fake --generator fake`: exits 0
- `--context-budget 500 --judge fake --generator fake`: exits 0; answer text
  shows fewer source markers than `--context-budget 8192` on same eval set,
  proving budget filtered chunks
- `--context-budget -1 --judge fake`: exits non-zero with error about negative
  budget

### Updates To `tests/test_cmd_diagnose.py`

- `--context-budget 0 --judge fake` (default): exits 0; output identical to
  Phase 2.0
- `--context-budget 8192 --judge fake`: exits 0
- `--context-budget -1 --judge fake`: exits non-zero with error about negative
  budget

---

## Acceptance Criteria

Phase 2.1 is complete when:

1. `tiny_rag_lab/context.py` exists with `ContextPackResult`, `TokenCounter`
   protocol, `FakeTokenCounter`, `TiktokenCounter`, `pack_context`, and
   `PROMPT_OVERHEAD`.
2. `pack_context` uses the same block formatting as `CONTEXT_BLOCK_TEMPLATE`
   from `prompting.py` for token counting accuracy.
3. `pack_context(budget=-1, ...)` raises `ValueError` with clear message.
4. `AskTrace.context_pack` serialises via `dataclasses.asdict()` without a
   custom encoder.
5. Default `rag ask` (no flags) output is identical to Phase 2.0 ask output
   (no context packing block).
6. `rag ask --context-budget 8192 --output-format text` shows a context
   packing block.
7. `rag ask --context-budget 500 --output-format text --generator fake`
   produces fewer source markers in the answer than `--context-budget 8192` on
   the same query (proves budget actually omits chunks).
8. `rag ask --context-budget 8192 --output-format json` prints valid JSON with
   `"answer"` and `"context_pack"` keys.
9. `rag ask --context-budget -1` exits non-zero with clear error.
10. Default `rag eval` and `rag diagnose` output identical to Phase 2.0.
11. `rag eval --context-budget 500 --judge fake --generator fake` produces
    shorter answers than `--context-budget 8192` on same eval set.
12. `rag eval --context-budget -1` and `rag diagnose --context-budget -1` exit
    non-zero.
13. `uv run pytest --tb=short -q`: all passed with no regressions.
14. Default `uv run pytest` does not import tiktoken unless it is installed.

---

## Learning Notes

- Context budgeting teaches that "retrieved chunks" and "chunks in the prompt"
  are not the same set — reranking orders them, budgeting selects them. A
  learner can run `rag ask --context-budget 500` to deliberately trigger
  omissions and see which chunks are dropped.
- The `ContextPackResult.omitted` field makes budget pressure visible and
  debuggable. A chunk that was ranked well but omitted teaches that ranking
  alone is not enough — context windows are finite.
- `FakeTokenCounter` and `TiktokenCounter` show the same interface with
  different precision. The difference in token counts between char-estimation
  and tiktoken is itself a teaching moment about tokenisation.
- `--output-format json` extends the "inspectable intermediate data" philosophy
  from traces to the final answer, enabling downstream processing or
  comparison scripts without parsing text output.
