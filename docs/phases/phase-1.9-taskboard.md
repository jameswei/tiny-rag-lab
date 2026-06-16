# Phase 1.9 Taskboard

This file tracks Phase 1.9 implementation tasks, dependencies, ownership, and
status in one lightweight table.

The implementation contract is `docs/phases/phase-1.9-reranking.md`. This
taskboard must stay aligned with that spec. Phase 1.9 was signed off by
Codex on 2026-06-16 and activated by the owner under "Current Phase" in
`docs/phases/README.md` the same day.

## Status Values

- `todo`: not started
- `in_progress`: actively being implemented
- `review`: implementation is ready for review and verification
- `blocked`: cannot proceed; blocker must be written in `Notes`
- `done`: reviewed, tested, and accepted

## Update Rules

- Set `Status` to `in_progress` before starting work.
- Set `Status` to `review` after implementation and local tests.
- Set `Status` to `done` only after review and required tests pass.
- The task owner must not mark their own task `done`; a different reviewing
  agent must sign off and make the `done` update.
- When marking `done`, record the reviewing agent and test result in `Notes`.
- Use `blocked` only with a concrete blocker in `Notes`.
- Keep `Owner` as an agent/person name or `unassigned`.
- Do not change task IDs after creation.
- Update `Notes` with skipped tests, setup limits, or follow-up work.

## Taskboard

| ID | Milestone | Task | Depends On | Status | Owner | Acceptance | Notes |
|---|---|---|---|---|---|---|---|
| P1.9-T01 | M1.9.0 | `reranker.py`: `RerankResult` dataclass, `Reranker` protocol, `FakeReranker`, `apply_reranker(query, results, reranker, top_k) -> (list[RetrievalResult], list[RerankResult])`, `chunk_traces_from_rerank(results, audit) -> list[ChunkTrace]` | — | done | claude | `RerankResult` round-trips through `dataclasses.asdict()`; `FakeReranker(score_map=None)` is a no-op (pre==post); `FakeReranker(score_map=...)` reorders deterministically with tie-break by `pre_rank`; `apply_reranker` slices to `min(top_k, len(results))` (clip semantics match `retrieve_by_vector`), raises `ValueError` only if `top_k < 0`, returns `([], [])` for empty input; `chunk_traces_from_rerank(audit=None)` leaves `pre_rerank_*` as `None`; `uv run pytest tests/test_reranker.py -k 'dataclass or fake or apply or chunk_traces' --tb=short -q`: N passed | Claimed by claude 2026-06-16; ready for review 2026-06-16. Reviewed and signed off by codex 2026-06-16; `uv run pytest tests/test_reranker.py --tb=short -q`: 22 passed; `uv run pytest tests/test_trace.py tests/test_cmd_retrieve.py tests/test_cmd_ask.py --tb=short -q`: 60 passed; `uv run pytest --tb=short -q`: 519 passed. No findings. |
| P1.9-T02 | M1.9.1 | `reranker.py`: `CrossEncoderReranker(model_name=None)`: lazy model load via `sentence-transformers.CrossEncoder`; class-level `DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"`; `__init__` does not load the model; `rerank()` loads on first call; new `tests/test_reranker_cross_encoder.py` gated by `pytest.importorskip("sentence_transformers")` and `TINY_RAG_LAB_TEST_RERANKER=1` env flag | P1.9-T01 | done | whale | Constructing `CrossEncoderReranker()` does no I/O (verified by patching `sentence_transformers.CrossEncoder` and asserting not called); name is `"cross-encoder"`; default model name exposed as class attribute; gated cross-encoder test passes when env flag is set and model is available; `uv run pytest tests/test_reranker_cross_encoder.py --tb=short -q` is skipped without the env flag and does not download the model | Reviewed and signed off by codex 2026-06-17; `uv run pytest tests/test_eval_runner.py tests/test_trace.py tests/test_cmd_retrieve.py tests/test_reranker.py tests/test_reranker_cross_encoder.py tests/test_eval_metrics.py --tb=short -q`: 159 passed, 8 skipped; `uv run pytest --tb=short -q`: 536 passed, 8 skipped. Verified `import tiny_rag_lab.reranker` does not import `sentence_transformers`. |
| P1.9-T03 | M1.9.2 | `trace.py`: `ChunkTrace.pre_rerank_rank` / `pre_rerank_score` (both `int|None` / `float|None`, default `None`); `RetrieveTrace.reranker` (default `"none"`) and `rerank_top_n` (default `None`); `cmd_retrieve` flags `--reranker {none,cross-encoder}` / `--rerank-top-n INT` / `--reranker-model NAME`; `_make_reranker(name, model)` factory in `cli.py`; two-stage retrieve flow when reranker is active; `latency_by_stage["rerank"]` populated; CLI validation errors for `rerank_top_n < top_k`, `--reranker-model` with `--reranker none`, and `rerank_top_n < 1` | P1.9-T01 | done | whale | `--reranker none` is a no-op: trace JSON has `reranker == "none"`, `rerank_top_n == null`, `pre_rerank_rank == null` for all chunks, no `rerank` key in `latency_by_stage`; `--reranker cross-encoder --rerank-top-n 20 --top-k 5` with patched `_make_reranker -> FakeReranker(...)` exits 0 and trace shows reranker name, top-n, and pre_rerank fields populated; invalid combinations exit non-zero with stderr explaining the violation; `uv run pytest tests/test_cmd_retrieve.py --tb=short -q`: N passed | Reviewed and signed off by codex 2026-06-17; `uv run pytest tests/test_eval_runner.py tests/test_trace.py tests/test_cmd_retrieve.py tests/test_reranker.py tests/test_reranker_cross_encoder.py tests/test_eval_metrics.py --tb=short -q`: 159 passed, 8 skipped; `uv run pytest --tb=short -q`: 536 passed, 8 skipped. Prior trace formatter question resolved. |
| P1.9-T04 | M1.9.3 | `eval.py`: `EvalReport.reranker` / `rerank_top_n` fields (defaults `"none"` / `None`); `run_retrieval_eval(..., reranker: Reranker \| None = None, rerank_top_n: int \| None = None)` two-stage flow; raises `ValueError` if reranker given and `rerank_top_n is None`, or `rerank_top_n < top_k`; `format_eval_report` prints reranker name and `rerank_top_n`; `cmd_eval` flags mirror `cmd_retrieve` | P1.9-T01 | done | whale | `run_retrieval_eval(reranker=None)` is identical to Phase 1.6 behavior on the existing fixture; `run_retrieval_eval(reranker=FakeReranker(...), rerank_top_n=10)` returns top_k results from the reranked pool; metrics computed over reranked slice; report formatter shows `Reranker         :  cross-encoder` (or `none`) and `Rerank Top-N     :  10` only when reranker is active; CLI smoke `uv run rag eval --qa-file FIXTURE --reranker none` matches pre-1.9 output; `uv run pytest tests/test_eval_runner.py tests/test_eval_metrics.py tests/test_cmd_eval.py --tb=short -q`: N passed | Reviewed and signed off by codex 2026-06-17; `uv run pytest tests/test_eval_runner.py tests/test_trace.py tests/test_cmd_retrieve.py tests/test_reranker.py tests/test_reranker_cross_encoder.py tests/test_eval_metrics.py --tb=short -q`: 159 passed, 8 skipped; `uv run pytest --tb=short -q`: 536 passed, 8 skipped. Prior empty-sample validation finding resolved with regression coverage. |
| P1.9-T05 | M1.9.4 | `trace.py`: `AskTrace.reranker` / `rerank_top_n` fields; `format_ask_trace` prints them and `rerank` latency when present; `cmd_ask` flags `--reranker` / `--rerank-top-n` / `--reranker-model`; two-stage retrieve before `assemble_prompt`; `pre_rerank_*` populated on `AskTrace.chunks` when rerank ran | P1.9-T01, P1.9-T03 | todo | unassigned | `--reranker none` on `rag ask` produces output identical to pre-1.9 (trace fields default); `--reranker cross-encoder --rerank-top-n 20 --top-k 5` with patched `_make_reranker -> FakeReranker(...)` and `_make_generator -> FakeGenerator` exits 0; `format_ask_trace` output contains reranker name and rerank latency line when active; ask still uses dense base retrieval (no `--retriever` flag added in this phase); `uv run pytest tests/test_cmd_ask.py --tb=short -q`: N passed |  |
| P1.9-T06 | M1.9.5 | `failure.py`: `RetrieverConfig.reranker` (default `"none"`) and `rerank_top_n` (default `None`); `run_diagnosis(..., reranker: Reranker \| None = None)` raises `ValueError` when any case needs a non-none reranker and `reranker is None`; `load_failure_cases` deserializes the new fields with defaults; fixture `tests/fixtures/failure/cases.jsonl` adds `fc007` — baseline hybrid top_k=6 reranker=none (so gold lands at rank 4-6 → `low_rank_evidence`); intervention hybrid top_k=2 rerank_top_n=6 reranker=cross-encoder (rerank pulls gold to rank 1-2 → `no_failure`); `cmd_diagnose` builds `CrossEncoderReranker` via `_make_reranker` when needed; tests use patched `FakeReranker`; `DiagnosisResult` and `format_diagnosis_report` are unchanged (config not surfaced in the diagnose report this phase) | P1.9-T01, P1.9-T03 | todo | unassigned | Existing fc001-fc006 JSONL rows still load (loader applies defaults for absent fields) and existing diagnose tests still pass; fc007 with `FakeReranker(score_map=...)` boosting the gold chunk to rank 1 produces `baseline_label == "low_rank_evidence"`, `intervention_label == "no_failure"`, `fixed == True`; `DiagnosisReport.n_cases == 7`; `run_diagnosis` raises `ValueError` if fc007 is included and `reranker is None`; `cmd_diagnose` end-to-end with patched `_make_reranker` exits 0 and report mentions fc007; `uv run pytest tests/test_failure.py tests/test_cmd_diagnose.py --tb=short -q`: N passed; full suite `uv run pytest --tb=short -q`: 497+ passed |  |
| P1.9-T07 | M1.9.6 | Phase close: update `docs/phases/README.md` to mark Phase 1.9 complete and set `Current Phase` to next or `No active phase`; update `docs/roadmap.md` Phase 1.9 section to "Complete; see ..."; run full test suite and CLI smoke; verify no `sentence_transformers` import in default test path | P1.9-T01–T06 | todo | unassigned | All P1.9-T01–T06 rows `done` with reviewer sign-off; `uv run pytest --tb=short -q`: all passed; `uv run rag retrieve --help` shows reranker flags; `uv run rag retrieve --reranker none "x" --index-dir PATH` matches pre-1.9 output on the fixture index; `python -c "import sys; import tiny_rag_lab.reranker; assert 'sentence_transformers' not in sys.modules"` succeeds; phase index updated |  |

## Review-Sensitive Tasks

These tasks require architecture or code review before being marked `done`:

- `P1.9-T01`: data contracts — `RerankResult` shape and `Reranker` protocol
  must remain JSON-native and free of numpy/torch imports at the
  `tiny_rag_lab.reranker` top level.
- `P1.9-T02`: gating — verify `import tiny_rag_lab.reranker` does not
  import `sentence_transformers`; verify the gated test correctly skips
  without the env flag.
- `P1.9-T03`: trace shape — `ChunkTrace.pre_rerank_*` defaults must be
  `None` so old trace consumers continue to read `rank`/`score` as the
  final values; back-compat verified by `--reranker none` byte-identical
  trace output (modulo new null fields).
- `P1.9-T04`: eval semantics — `rerank_top_n < top_k` validation must fire
  before any retrieval call to avoid wasted base-retrieve work; the
  report's two new fields must remain absent / default when `reranker is None`.
- `P1.9-T06`: failure fixture — fc007 must use `FakeReranker` in tests;
  the real `CrossEncoderReranker` must never run in CI. `RetrieverConfig`
  back-compat for fc001-fc006 must be explicit.

## Minimum Phase 1.9 Completion

Minimum Phase 1.9 completion requires `P1.9-T01` through `P1.9-T07`.
