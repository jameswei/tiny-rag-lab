# Phase 1.9 Spec: Reranking

**Status:** Complete — closed 2026-06-17
**Authors:** Claude Code; based on Codex's final roadmap and Claude's review refinements
**Based on:** `docs/phases/phase-1.9-2.2-final-roadmap.md`,
`docs/phases/phase-1.9-2.1-quality-and-production-roadmap-codex-review-by-claude.md`,
`docs/phases/phase-1.8-failure-lab.md`
**Taskboard:** `docs/phases/phase-1.9-taskboard.md`
**Date:** 2026-06-16

---

## Goal

Add a second-pass reranking layer that reorders retrieved candidates by
query-document relevance before evaluation and answer generation. Introduce
the bi-encoder vs cross-encoder distinction as one teachable concept, give
the project a measurable upgrade path against the existing retrieval
metrics, and give the failure lab a demonstrable fix for
`low_rank_evidence` cases.

The reranker is opt-in: `--reranker none` (the default) preserves current
behavior bit-for-bit.

---

## Scope

### In Scope

- `Reranker` protocol and `RerankResult` dataclass in
  `tiny_rag_lab/reranker.py`
- `FakeReranker` for deterministic tests
- `CrossEncoderReranker` backed by `sentence-transformers`'s `CrossEncoder`
  for real runs, with lazy model loading and explicit model gating
- Two-stage retrieve flow: base retriever returns `rerank_top_n` candidates,
  reranker reorders and returns `top_k`
- `--reranker none|cross-encoder`, `--rerank-top-n INT`, and
  `--reranker-model NAME` CLI flags on `rag retrieve`, `rag eval`, and
  `rag ask`
- Trace updates: `RetrieveTrace` and `AskTrace` gain `reranker` and
  `rerank_top_n` fields; `ChunkTrace` gains optional `pre_rerank_rank` and
  `pre_rerank_score`; `latency_by_stage` gains a `rerank` key when rerank
  runs
- `EvalReport` gains `reranker` and `rerank_top_n` fields and the eval
  report formatter renders them
- `RetrieverConfig` in `failure.py` gains `reranker: str = "none"` and
  `rerank_top_n: int | None = None` fields (back-compat additions)
- New failure-fixture case `fc007` demonstrating cross-encoder reranking
  fixing a buried-evidence (`low_rank_evidence`) failure under hybrid
  retrieval, using `FakeReranker` in unit tests
- `run_diagnosis` accepts an optional `reranker: Reranker | None` argument;
  CLI builds it through a new `_make_reranker(name, model)` factory
- Updated tests across `test_reranker.py`, `test_eval.py`, `test_failure.py`,
  `test_cmd_retrieve.py`, `test_cmd_eval.py`, `test_cmd_ask.py`,
  `test_cmd_diagnose.py`

### Out Of Scope For Phase 1.9

- API-backed rerankers (Cohere Rerank, Voyage, Jina) — leave room behind the
  `--reranker` value space; not implemented now
- Bi-encoder reranking other than the existing dense path
- Multi-stage rerank chains
- Reranker score calibration or normalization across queries
- Token-budget interaction (deferred to Phase 2.1)
- Answer-quality measurement of rerank's effect on generation (deferred to
  Phase 2.0)
- A `cross-encoder` reranker default that requires downloading a large
  multilingual model — see Design Decision 4

---

## Design Decision 1: Reranker Is Opt-In, Default `none`

`--reranker none` preserves Phase 1 through 1.8 behavior exactly. The base
retriever returns `top_k` results directly; no rerank stage runs;
`pre_rerank_rank` and `pre_rerank_score` serialize as `null` on every
`ChunkTrace` (dataclass defaults are `None`, written as JSON `null`);
`RetrieveTrace.reranker` is `"none"`, `RetrieveTrace.rerank_top_n` is
`null`; no `"rerank"` key appears in `latency_by_stage` (the dict only
records stages that ran); no model download. This keeps the existing
test suite green by construction.

`--reranker cross-encoder` activates the two-stage flow: the base retriever
returns `rerank_top_n` candidates (default 20), the reranker reorders them,
and the top `top_k` post-rerank results are returned to the caller.

When `rerank_top_n < top_k`, the spec raises `ValueError` at the CLI layer
and the eval/diagnose runners — reranking fewer candidates than the final
slice would silently lose results.

## Design Decision 2: `RerankResult` Is Per-Chunk, Not Per-List

Each `RerankResult` carries one chunk's pre- and post-rerank ranks and
scores. The reranker returns `list[RerankResult]` ordered by `post_rank`
ascending. The retrieval-side caller maps these back to
`list[RetrievalResult]` so downstream code that consumes
`RetrievalResult` stays unchanged.

Keeping the rerank shape separate from `RetrievalResult` makes the rerank
mechanic visible at the interface and avoids overloading
`RetrievalResult.score` with two different semantic meanings.

## Design Decision 3: Trace Carries Post-Rerank Ranks As The Primary View

`ChunkTrace.rank` and `ChunkTrace.score` remain the final ranks and scores
after reranking (when rerank ran). The pre-rerank position is exposed as
optional fields:

```python
@dataclass
class ChunkTrace:
    ...
    score: float
    text_preview: str
    pre_rerank_rank: int | None = None    # 1-indexed; None when no rerank
    pre_rerank_score: float | None = None
```

This keeps existing trace consumers working — `rank` still means "what
position did this chunk end up at" — and adds the pre-rerank view as
inspectable diagnostic data.

## Design Decision 4: Default Model Is `cross-encoder/ms-marco-MiniLM-L-6-v2`

The default cross-encoder is `cross-encoder/ms-marco-MiniLM-L-6-v2`:

- ~80 MB model size — small enough that a learner can download it on a
  laptop without hesitating
- well-known MS MARCO-trained baseline cross-encoder
- English-focused — matches the primary `watsonxDocsQA` corpus

`--reranker-model NAME` lets the owner swap in `BAAI/bge-reranker-base` or
similar multilingual rerankers when the corpus needs them. The default is
chosen to minimize entry friction, not to be the best reranker on every
dataset.

## Design Decision 5: Lazy Load, Explicit Model Gating

`CrossEncoderReranker.__init__` records the model name but does not load
the model. The actual model load happens on the first `rerank()` call.
This means:

- importing `tiny_rag_lab.reranker` triggers no network or disk I/O
- `FakeReranker` paths never construct a `sentence-transformers.CrossEncoder`
- tests that target the real reranker explicitly gate themselves with
  `pytest.importorskip("sentence_transformers")` plus an env-flag check
  (`TINY_RAG_LAB_TEST_RERANKER=1`); they are skipped in default CI

CI runs `uv run pytest` with the env flag unset, so no model is ever
downloaded by the standard test path.

## Design Decision 6: Reranking Sits Outside The Base Retriever, Not Inside

`retrieve_by_vector`, `BM25Retriever.retrieve`, and `retrieve_hybrid` are
untouched. Reranking is composed at the call site (CLI, eval runner,
diagnose runner): retrieve `rerank_top_n`, then call
`reranker.rerank(query, results)`. This keeps the existing retrieval-plane
code unchanged, makes rerank's "second pass" nature visible in the call
sequence, and avoids threading rerank parameters through three retriever
signatures.

---

## Data Contracts

All new types live in `tiny_rag_lab/reranker.py`. All fields are JSON-native
so `dataclasses.asdict()` + `json.dumps()` serialize cleanly.

```python
@dataclass
class RerankResult:
    """One chunk's reranking outcome.

    pre_rank and post_rank are both 1-indexed, matching RetrievalResult.rank.
    pre_score is the base-retriever score (cosine, BM25, or RRF).
    post_score is the cross-encoder relevance score (typically unnormalized).
    """
    chunk_id: str
    pre_rank: int
    post_rank: int
    pre_score: float
    post_score: float
```

```python
class Reranker(Protocol):
    """Second-pass reranker over a pre-retrieved candidate set.

    Implementations must be deterministic for the same (query, candidates)
    pair, so trace and eval outputs are reproducible.
    """
    name: str

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
    ) -> list[RerankResult]:
        """Return RerankResults ordered by post_rank ascending.

        len(return) == len(candidates). Every input chunk appears once.
        """
        ...
```

```python
@dataclass
class FakeReranker:
    """Deterministic reranker for tests.

    name is the value reported in traces; tests can use any string.

    score_map maps chunk_id -> post_score. Chunks not in the map get
    post_score = 0.0. Ties break by original (pre-rerank) rank, so the
    rerank is fully deterministic.

    When score_map is None the reranker is a no-op: post_score == pre_score
    and post_rank == pre_rank.
    """
    name: str = "fake"
    score_map: dict[str, float] | None = None

    def rerank(self, query, candidates) -> list[RerankResult]: ...
```

```python
class CrossEncoderReranker:
    """Local cross-encoder reranker backed by sentence-transformers.

    The model is lazily loaded on the first rerank() call. Construction is
    free of side effects (no network, no disk read).
    """
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    name: str

    def __init__(self, model_name: str | None = None) -> None: ...
    def rerank(self, query, candidates) -> list[RerankResult]: ...
```

### Updates To Existing Types

`tiny_rag_lab/trace.py`:

```python
@dataclass
class ChunkTrace:
    rank: int
    chunk_id: str
    doc_id: str
    title: str
    path: str
    score: float
    text_preview: str
    pre_rerank_rank: int | None = None     # NEW
    pre_rerank_score: float | None = None  # NEW

@dataclass
class RetrieveTrace:
    query: str
    retriever: str
    top_k: int
    chunks: list[ChunkTrace] = field(default_factory=list)
    latency_by_stage: dict[str, float] = field(default_factory=dict)
    reranker: str = "none"                 # NEW
    rerank_top_n: int | None = None        # NEW

@dataclass
class AskTrace:
    # identical additions to RetrieveTrace
    reranker: str = "none"                 # NEW
    rerank_top_n: int | None = None        # NEW
```

`latency_by_stage` gains a `"rerank"` key when rerank ran. It is absent
when `reranker == "none"`.

`tiny_rag_lab/eval.py`:

```python
@dataclass
class EvalReport:
    n_questions: int
    top_k: int
    retriever: str = "dense"
    hit_rate: float = 0.0
    mrr: float = 0.0
    mean_context_precision: float = 0.0
    mean_context_recall: float = 0.0
    per_question: list[EvalResult] = field(default_factory=list)
    reranker: str = "none"                 # NEW
    rerank_top_n: int | None = None        # NEW
```

`tiny_rag_lab/failure.py`:

```python
@dataclass
class RetrieverConfig:
    retriever: str = "dense"
    top_k: int = 5
    reranker: str = "none"                 # NEW
    rerank_top_n: int | None = None        # NEW
```

`DiagnosisResult` and `DiagnosisReport` are unchanged. The diagnose report
formatter is unchanged — it continues to show only labels, metrics, and
outcome words per case, not retriever or reranker config. A learner who
wants to see which reranker fixed a case reads the case row in
`tests/fixtures/failure/cases.jsonl`. Surfacing retriever and reranker
config in the diagnose report is deferred; it would require widening
`DiagnosisResult` with `baseline_config` / `intervention_config` fields,
which is out of scope for Phase 1.9.

---

## Core Functions

```python
def apply_reranker(
    query: str,
    results: list[RetrievalResult],
    reranker: Reranker,
    top_k: int,
) -> tuple[list[RetrievalResult], list[RerankResult]]:
    """Run reranker over results and return (reordered, rerank_audit).

    reordered is the top min(top_k, len(results)) post-rerank slice as
    list[RetrievalResult] with rank reassigned to the post-rerank position
    and score replaced with the post-rerank score. Clipping matches
    retrieve_by_vector's behavior on small indexes so callers do not have
    to special-case sparse pools.

    rerank_audit is the full list[RerankResult] (length == len(results)),
    suitable for assembling trace fields.

    Returns ([], []) if results is empty.
    Raises ValueError if top_k < 0.

    Note: the invariant rerank_top_n >= top_k is enforced one layer up at
    the CLI / runner config check, before any base retrieval runs. This
    function does not re-check that invariant — it only deals with the
    candidate set it is actually handed.
    """
```

```python
def chunk_traces_from_rerank(
    results: list[RetrievalResult],
    rerank_audit: list[RerankResult] | None,
) -> list[ChunkTrace]:
    """Build ChunkTrace list with pre_rerank_* populated when rerank ran.

    When rerank_audit is None the returned ChunkTraces have pre_rerank_rank
    and pre_rerank_score == None.
    """
```

`run_retrieval_eval` gains two parameters:

```python
def run_retrieval_eval(
    samples: list[EvalSample],
    index: LoadedIndex,
    embedder: Embedder | None,
    top_k: int,
    retriever: str = "dense",
    reranker: Reranker | None = None,
    rerank_top_n: int | None = None,
) -> EvalReport:
    """When reranker is None the runner is identical to Phase 1.6/1.5.

    When reranker is provided, the base retriever is asked for rerank_top_n
    candidates per sample, reranker.rerank is called once per sample, and
    the top_k post-rerank slice feeds the metric functions.

    Raises ValueError if reranker is not None and rerank_top_n is None.
    Raises ValueError if rerank_top_n < top_k.
    """
```

`run_diagnosis` gains a `reranker` parameter:

```python
def run_diagnosis(
    cases: list[FailureCase],
    index: LoadedIndex,
    embedder: Embedder | None,
    reranker: Reranker | None = None,
    thresholds: DetectionThresholds | None = None,
) -> DiagnosisReport:
    """Cases with reranker == "none" run unchanged (Phase 1.8 behavior).
    Cases with reranker == "cross-encoder" use the supplied reranker.

    Raises ValueError if any case requires a non-none reranker and the
    reranker argument is None.
    """
```

---

## CLI Changes

### `rag retrieve`

```
rag retrieve "question" \
  --retriever {dense|bm25|hybrid} \
  --reranker {none|cross-encoder} \
  --rerank-top-n INT \
  --reranker-model NAME \
  --top-k INT \
  --index-dir PATH \
  --trace-out PATH
```

- `--reranker` default: `none`
- `--rerank-top-n` default: 20; ignored when `--reranker none`
- `--reranker-model` default: empty (use `CrossEncoderReranker.DEFAULT_MODEL`)
- `cmd_retrieve` builds the reranker through `_make_reranker(name, model)`;
  for `none` the factory returns `None` and the existing flow runs
- Trace `latency_by_stage` records `"rerank"` only when rerank ran

### `rag eval`

```
rag eval --qa-file PATH \
  --retriever {dense|bm25|hybrid} \
  --reranker {none|cross-encoder} \
  --rerank-top-n INT \
  --reranker-model NAME \
  --top-k INT \
  --index-dir PATH
```

- Same flag semantics as `rag retrieve`
- `EvalReport.reranker` and `EvalReport.rerank_top_n` populate the report
- The eval formatter prints a new line `Reranker         :  none` (or the
  active name + top-n) so config and metrics live together

### `rag ask`

```
rag ask "question" \
  --reranker {none|cross-encoder} \
  --rerank-top-n INT \
  --reranker-model NAME \
  --top-k INT \
  --index-dir PATH \
  --model NAME --api-key KEY --base-url URL \
  --trace-out PATH
```

- Same flag semantics as `rag retrieve`
- `AskTrace` carries the new fields; `format_ask_trace` shows the reranker
  name and per-stage latency including rerank when present
- `ask` still uses dense retrieval as the base; this phase does not add
  `--retriever` to `ask`

### `rag diagnose`

No new CLI flags. The reranker is implicit per case via `RetrieverConfig`.
`cmd_diagnose` builds a `CrossEncoderReranker` only when at least one case
uses `reranker == "cross-encoder"`. The factory is shared with `retrieve`/
`eval`/`ask`.

### Validation

`cmd_retrieve`, `cmd_eval`, `cmd_ask`, and the underlying runners raise
`ValueError` (CLI exits non-zero with a clear message) when:

- `--reranker cross-encoder` is set with `--rerank-top-n` < `--top-k`
- `--reranker-model` is set with `--reranker none`
- `--rerank-top-n` < 1 (invalid pool size)

---

## Failure Fixture Changes

Add `fc007` to `tests/fixtures/failure/cases.jsonl`:

| ID | Scenario | Expected Label | Baseline | Intervention |
|---|---|---|---|---|
| fc007 | Cross-encoder rerank fixes buried evidence | low_rank_evidence | bm25, top_k=6, rerank_top_n=null, reranker=none | bm25, top_k=1, rerank_top_n=6, reranker=cross-encoder |

The baseline retrieves six chunks under BM25 with no rerank; the gold
chunk lands at rank 4-6 because the question does not lexically match the
gold document, triggering `low_rank_evidence` (default
`low_rank_threshold = 3`, fires at rank > 3). The intervention retrieves
the same six-chunk candidate pool, reranks them with the cross-encoder,
and returns only the top one — the reranker pulls the gold chunk to rank
1, producing `no_failure`. `fixed = True`.

Existing cases `fc001` through `fc006` keep their current configs and
labels. Their JSONL rows do not include `reranker` or `rerank_top_n`
fields; the loader defaults them to `none` and `None` respectively, so the
existing tests remain valid byte-for-byte after the data-contract update.

Unit tests for `run_diagnosis` over the fixture use `FakeReranker` with a
scripted `score_map` that boosts the fc007 gold chunk into rank 1. The
real `CrossEncoderReranker` is not exercised in CI.

---

## Required Tests

### `tests/test_reranker.py` (new)

- `*_dataclass_*`: `RerankResult` round-trip; default values; field types
- `*_fake_noop_*`: `FakeReranker(score_map=None)` returns input order
  unchanged with `pre_score == post_score` and `pre_rank == post_rank`
- `*_fake_score_map_*`: a scripted map reorders results deterministically
  and missing chunks score 0.0
- `*_fake_ties_*`: ties break by pre-rerank rank
- `*_apply_reranker_*`: `top_k` slicing is correct; empty input returns
  `([], [])`; `top_k > len(results)` clips to `len(results)` (matches
  `retrieve_by_vector`); `top_k < 0` raises `ValueError`
- `*_chunk_traces_*`: `pre_rerank_*` populated when audit is given, both
  None when audit is None

### `tests/test_reranker_cross_encoder.py` (new, gated)

- `pytest.importorskip("sentence_transformers")`
- `pytest.mark.skipif` on env flag `TINY_RAG_LAB_TEST_RERANKER != "1"`
- Loads the default model, reranks a 3-chunk list against a simple query,
  asserts post_rank reorders relative to pre_rank

### Updates To Existing Tests

- `tests/test_eval_runner.py`:
  - `run_retrieval_eval` with `reranker=FakeReranker(...)` and
    `rerank_top_n=N` returns top_k results from the reranked candidates
  - `reranker=None` is identical to Phase 1.6 behavior
  - `rerank_top_n < top_k` raises `ValueError`
  - `EvalReport.reranker` and `EvalReport.rerank_top_n` populate correctly
- `tests/test_eval_metrics.py`:
  - `format_eval_report` prints the new `Reranker` and `Rerank Top-N`
    lines when reranker is active; lines are absent when `reranker == "none"`
- `tests/test_failure.py`:
  - `RetrieverConfig` default values; JSONL round-trip with and without
    reranker fields
  - `run_diagnosis` with `reranker=FakeReranker(...)` and a `fc007`-style
    case shows `low_rank_evidence` fixed (post-rerank gold at rank 1)
  - `run_diagnosis` raises `ValueError` when any case needs a reranker and
    none is supplied
  - all existing fc001-fc006 assertions still pass
- `tests/test_cmd_retrieve.py`:
  - `--reranker cross-encoder --rerank-top-n 20 --top-k 5` exits 0 when
    `_make_reranker` is patched to a `FakeReranker`
  - trace output contains the reranker name and per-stage rerank latency
  - `--reranker cross-encoder --rerank-top-n 3 --top-k 5` exits non-zero
  - default behavior (`--reranker none`) is identical to existing tests
- `tests/test_cmd_eval.py`: parallel additions
- `tests/test_cmd_ask.py`: parallel additions
- `tests/test_cmd_diagnose.py`:
  - fc007 case in the fixture runs end-to-end with a patched
    `FakeReranker`
  - diagnose output indicates fc007 was confirmed and fixed

### CLI Smoke (Documentation, Not CI)

`uv run rag retrieve "..." --reranker cross-encoder --rerank-top-n 20 --top-k 5`
runs end-to-end on the real corpus after a one-time model download. Not
run in CI.

---

## Acceptance Criteria

Phase 1.9 is complete when:

1. `tiny_rag_lab/reranker.py` exists with `RerankResult`, `Reranker`
   protocol, `FakeReranker`, `CrossEncoderReranker`, `apply_reranker`, and
   `chunk_traces_from_rerank`.
2. `tiny_rag_lab/trace.py`, `tiny_rag_lab/eval.py`, and
   `tiny_rag_lab/failure.py` have the new fields and parameters described
   above, with back-compat defaults.
3. `rag retrieve`, `rag eval`, `rag ask`, and `rag diagnose` accept the new
   reranker flags / fixture fields; default behavior is unchanged.
4. `tests/fixtures/failure/cases.jsonl` includes `fc007` with the
   `cross-encoder` intervention; fc001-fc006 are unchanged.
5. New tests use `FakeReranker` only. The default `uv run pytest` run does
   not import `sentence_transformers` or download any model.
6. `tests/test_reranker_cross_encoder.py` exists, is gated by the
   `TINY_RAG_LAB_TEST_RERANKER=1` env flag, and passes when enabled with
   the model available.
7. `uv run pytest --tb=short -q` is green with no regressions in
   pre-existing test counts (Phase 1.8 closed with 497 passed; Phase 1.9
   adds new tests).
8. `uv run rag retrieve --reranker none "anything" --index-dir PATH` and
   `uv run rag ask --reranker none "anything" --index-dir PATH` produce
   output identical to pre-1.9 (modulo new optional trace fields that are
   `None`).

---

## Learning Notes

- The reranker introduces the bi-encoder vs cross-encoder distinction
  visibly at the interface. A learner can compare `retrieve` results with
  and without `--reranker cross-encoder` on the same query and see
  candidate order change.
- The two-stage flow (`rerank_top_n` candidates → reranker → `top_k`) makes
  the precision-vs-recall trade-off concrete at the CLI surface.
- `low_rank_evidence` cases from the failure lab gain a clear remediation
  path. The pre/post rerank fields in the trace let a learner see exactly
  what moved.
- The opt-in design keeps Phase 1 through 1.8 behavior reachable for
  comparison and rollback, supporting the project's "make the mechanics
  visible" philosophy.
