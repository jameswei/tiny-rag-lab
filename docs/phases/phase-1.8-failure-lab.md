# Phase 1.8 Spec: RAG Failure Lab

**Status:** Active — activated by owner on 2026-06-11
**Authors:** Claude Code + owner decisions
**Based on:** `docs/roadmap.md`, `docs/phases/phase-1.7-observability.md`
**Taskboard:** `docs/phases/phase-1.8-taskboard.md`
**Date:** 2026-06-11

---

## Goal

Intentionally create and study common RAG failure modes using the eval and trace
infrastructure from Phases 1.6 and 1.7. Establish a failure taxonomy, a curated
JSONL fixture of failure scenarios with reproducible inputs, and a `rag diagnose`
command that runs baseline vs. intervention retrieval for each scenario and reports
which failures were confirmed, fixed, or moved.

---

## Scope

### In scope

- `LABEL_*` string constants for the five heuristic-detectable failure modes in
  `tiny_rag_lab/failure.py`
- `RetrieverConfig`, `FailureCase`, `DiagnosisResult`, `DiagnosisReport`,
  `DetectionThresholds` dataclasses in `tiny_rag_lab/failure.py`
- `load_failure_cases(path) -> list[FailureCase]` JSONL loader
- `detect_failure_label(retrieved_doc_ids, gold_doc_ids, expected_label, thresholds)`
  detection function reusing metric functions from `eval.py`
- `run_diagnosis(cases, index, embedder, thresholds) -> DiagnosisReport` runner
- `format_diagnosis_report(report) -> str` plain-text formatter
- Curated failure cases fixture: `tests/fixtures/failure/cases.jsonl` (6 scenarios)
- Two new small fixture Markdown docs for disambiguation scenarios:
  `tests/fixtures/corpus/section_alpha.md` and `tests/fixtures/corpus/section_beta.md`
- `rag diagnose --cases-file PATH [--index-dir PATH]` CLI command
- `cmd_diagnose` in `cli.py`
- Unit tests: `tests/test_failure.py`
- CLI tests: `tests/test_cmd_diagnose.py`

### Out of scope for Phase 1.8

- `unsupported_answer` and `citation_mismatch` detection — require LLM-as-judge;
  these labels are documented in `failure.py` comments but not implemented
- Bad chunking failure scenario — requires re-indexing with deliberate chunk_size
  parameters; deferred due to fixture management complexity
- Stale documents scenario — requires corpus-level mutation; documented as a known
  failure mode, not implemented
- `--trace-out` or trace recording on `rag diagnose` — deferred; formatter output
  serves the same human-readable purpose
- `--retriever` / `--top-k` CLI overrides for `rag diagnose` — all retrieval config
  comes from per-case JSONL fields; no CLI-level override in this phase
- Multi-run comparison reports — deferred to the later "Reporting" phase
- Answer quality metrics (faithfulness, answer correctness) — deferred

---

## Design Decision: Reuse `eval.py` Metric Functions

`detect_failure_label` imports and calls `hit_at_k`, `context_precision_at_k`, and
`reciprocal_rank` from `eval.py` rather than redefining them. This makes detection
explicitly grounded in the same math as evaluation and avoids two sources of truth
for retrieval metrics.

---

## Design Decision: Separate Loader for Unanswerable Cases

`load_eval_samples` in `eval.py` skips rows with empty `gold_doc_ids` (this matches
the eval runner's assumption that every sample has at least one gold document).
`load_failure_cases` does NOT skip empty `gold_doc_ids` — the `unanswerable_query`
scenario legitimately has no gold documents. A separate loader function is required.
The divergence is documented in the `load_failure_cases` docstring.

---

## Failure Taxonomy

Five heuristic-detectable failure labels (module-level string constants):

```python
LABEL_MISSING_EVIDENCE    = "missing_evidence"      # gold docs absent from retrieved set
LABEL_LOW_RANK_EVIDENCE   = "low_rank_evidence"     # gold present but first hit at rank > threshold
LABEL_DISTRACTOR_EVIDENCE = "distractor_evidence"   # context_precision below threshold
LABEL_UNANSWERABLE        = "unanswerable_query"    # query has no gold in corpus
LABEL_NO_FAILURE          = "no_failure"            # all thresholds met
```

Two additional failure modes are documented but not heuristically detectable:
- `unsupported_answer`: LLM-as-judge required
- `citation_mismatch`: LLM-as-judge required

---

## Data Contracts

All types live in `tiny_rag_lab/failure.py`. All fields are JSON-native so
`dataclasses.asdict()` + `json.dumps()` serializes without a custom encoder.

```python
@dataclass
class RetrieverConfig:
    retriever: str = "dense"    # "dense" | "bm25" | "hybrid"
    top_k: int = 5
```

```python
@dataclass
class FailureCase:
    case_id: str
    question: str
    gold_doc_ids: list[str]          # [] for unanswerable cases
    expected_label: str              # one of LABEL_* constants
    baseline: RetrieverConfig
    intervention: RetrieverConfig    # identical to baseline if no intervention designed
    notes: str                       # human explanation of failure and diagnostic fields
```

```python
@dataclass
class DetectionThresholds:
    low_rank_threshold: int = 3              # first gold hit at rank > this → low_rank_evidence
    distractor_precision_threshold: float = 0.5  # context_precision < this → distractor_evidence
```

```python
@dataclass
class DiagnosisResult:
    case_id: str
    question: str
    expected_label: str
    baseline_label: str
    intervention_label: str
    baseline_retrieved_doc_ids: list[str]   # doc_ids actually retrieved at baseline
    intervention_retrieved_doc_ids: list[str]  # doc_ids actually retrieved at intervention
    baseline_metrics: dict[str, float]      # keys: hit, reciprocal_rank, context_precision, context_recall
    intervention_metrics: dict[str, float]  # same keys
    fixed: bool = False    # baseline != no_failure and intervention == no_failure
    moved: bool = False    # both non-no_failure but different labels
```

```python
@dataclass
class DiagnosisReport:
    n_cases: int
    n_fixed: int = 0
    n_moved: int = 0
    n_confirmed: int = 0    # baseline_label == expected_label
    per_case: list[DiagnosisResult] = field(default_factory=list)
```

---

## Core Functions

```python
def load_failure_cases(path: Path) -> list[FailureCase]:
    """Load FailureCase objects from a cases.jsonl file.

    Does NOT skip rows with empty gold_doc_ids (unlike load_eval_samples).
    Skips rows with empty case_id or empty question.
    Silently skips malformed JSON rows.
    baseline and intervention default to RetrieverConfig() when absent.
    """
```

```python
def detect_failure_label(
    retrieved_doc_ids: list[str],
    gold_doc_ids: list[str],
    expected_label: str,
    thresholds: DetectionThresholds | None = None,
) -> str:
    """Assign a failure label from retrieved and gold doc IDs.

    Detection order (first match wins):
    1. gold_doc_ids empty + expected_label == LABEL_UNANSWERABLE → LABEL_UNANSWERABLE
    2. gold_doc_ids empty otherwise → LABEL_NO_FAILURE (cannot evaluate)
    3. hit_at_k False → LABEL_MISSING_EVIDENCE
    4. rank of first gold hit > low_rank_threshold → LABEL_LOW_RANK_EVIDENCE
    5. context_precision_at_k < threshold → LABEL_DISTRACTOR_EVIDENCE
    6. LABEL_NO_FAILURE

    Steps 4 and 5 are mutually exclusive by design: low_rank_evidence fires when the
    gold hit is buried (rank > threshold), regardless of precision. distractor_evidence
    fires only when the gold hit is well-ranked but the surrounding context is noisy
    (precision low despite an acceptable rank). Checking low_rank first prevents a
    large top_k from shadowing a rank-ordering failure with a precision signal.

    Calls hit_at_k, context_precision_at_k, reciprocal_rank from eval.py.
    """
```

```python
def run_diagnosis(
    cases: list[FailureCase],
    index: LoadedIndex,
    embedder: Embedder | None,
    thresholds: DetectionThresholds | None = None,
) -> DiagnosisReport:
    """Run baseline and intervention retrieval for every case.

    Builds BM25Retriever once before the case loop when any case uses
    bm25 or hybrid (same pattern as run_retrieval_eval).
    Raises ValueError if embedder is None and any case requires dense or hybrid.
    """
```

```python
def format_diagnosis_report(report: DiagnosisReport) -> str:
    """Return a plain-text summary. No ANSI escape codes.

    Example:
      Diagnosis report  (n=6)
      ────────────────────────────────────────────
        Confirmed  : 4
        Fixed      : 1
        Moved      : 1
      ────────────────────────────────────────────
      Case fc001  expected=missing_evidence
        baseline   : missing_evidence  hit=0.000  prec=0.000  recall=0.000  mrr=0.000
        interv.    : no_failure        hit=1.000  prec=0.500  recall=1.000  mrr=0.500
        FIXED
      ...
    """
```

---

## Curated Failure Cases Fixture

`tests/fixtures/failure/cases.jsonl` — 6 scenarios:

| ID | Scenario | Expected Label | Baseline | Intervention |
|---|---|---|---|---|
| fc001 | top_k too small misses evidence | missing_evidence | dense, top_k=1 | dense, top_k=4 |
| fc002 | top_k too large adds distractors | distractor_evidence | dense, top_k=6 | dense, top_k=1 |
| fc003 | ambiguous query retrieves wrong topic | missing_evidence | dense, top_k=2 | bm25, top_k=2 |
| fc004 | evidence present but low-ranked | low_rank_evidence | bm25, top_k=6 | hybrid, top_k=6 |
| fc005 | query is unanswerable from corpus | unanswerable_query | dense, top_k=3 | dense, top_k=3 |
| fc006 | retriever strategy comparison | missing_evidence | dense, top_k=2 | hybrid, top_k=2 |

fc005 has `gold_doc_ids: []`. All other cases reference doc_ids from the fixture
corpus. fc003/fc004/fc006 may reference `section_alpha.md` or `section_beta.md`.

Two new fixture docs added to `tests/fixtures/corpus/`:
- `section_alpha.md`: short doc about "topic alpha" — distinct content for disambiguation
- `section_beta.md`: short doc about "topic beta" — gives corpus two section-like docs

---

## CLI Changes

### `rag diagnose`

New subcommand:

```
rag diagnose --cases-file PATH [--index-dir PATH]
```

`cmd_diagnose`:
1. Load index from `--index-dir`.
2. Load failure cases from `--cases-file`.
3. Build embedder if any case uses dense or hybrid (same `_make_embedder` pattern).
4. Call `run_diagnosis(cases, index, embedder)`.
5. Print `format_diagnosis_report(report)`.

No `--trace-out`, no `--retriever` override, no `--top-k` override in Phase 1.8.

---

## Required Tests

**`tests/test_failure.py`** (grouped by task):

- `*_dataclass_*`: JSON round-trip for all five dataclasses; empty `gold_doc_ids`
  in `FailureCase` survives `dataclasses.asdict()`; default values correct
- `*_load_*`: returns 6 records from fixture; `fc005` unanswerable case not skipped;
  empty `case_id` row skipped; malformed JSON row skipped; `baseline` deserializes
  to `RetrieverConfig`
- `*_detect_*`: known-input tests for all label paths; custom thresholds respected;
  imports from `eval.py` verified (no reimplementation)
- `*_runner_*`: `DiagnosisReport.n_cases == 6`; `n_fixed` correct; fc005 gets
  `unanswerable_query` label; empty cases returns empty report; `None` embedder
  with dense case raises `ValueError`; BM25-only cases accept `None` embedder
- `*_format_*`: output contains n_cases count; contains confirmed/fixed/moved counts;
  per-case block has case_id and outcome word; no ANSI codes

**`tests/test_cmd_diagnose.py`**:

- `rag diagnose --help` exits 0, shows `--cases-file` flag
- Missing `--cases-file` exits non-zero
- End-to-end with FakeEmbedder + fixture index prints `Diagnosis report`
- Output contains at least one case block

---

## Acceptance Criteria

Phase 1.8 is complete when:

1. `tiny_rag_lab/failure.py` contains all five label constants, all five dataclasses,
   `load_failure_cases`, `detect_failure_label`, `run_diagnosis`, and
   `format_diagnosis_report`.
2. `tests/fixtures/failure/cases.jsonl` has 6 well-formed records including fc005
   with `gold_doc_ids: []`.
3. `rag diagnose --cases-file tests/fixtures/failure/cases.jsonl --index-dir PATH`
   runs end-to-end with FakeEmbedder and prints `Diagnosis report  (n=6)`.
4. All new types, functions, and the CLI command have passing tests using
   FakeEmbedder — no network access, model downloads, or API credentials required.
5. Full test suite passes without regression.
