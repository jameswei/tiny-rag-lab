"""Tests for tiny_rag_lab/reranker.py.

Test naming convention (P1.9-T01 scope):
  test_dataclass_*    — RerankResult round-trip and field types
  test_fake_*         — FakeReranker noop / score-map / ties / missing chunks
  test_apply_*        — apply_reranker slicing, clipping, errors
  test_chunk_traces_* — chunk_traces_from_rerank with audit on/off
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from tiny_rag_lab.models import Chunk, RetrievalResult
from tiny_rag_lab.reranker import (
    FakeReranker,
    RerankResult,
    apply_reranker,
    chunk_traces_from_rerank,
)
from tiny_rag_lab.trace import ChunkTrace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(chunk_id: str, doc_id: str = "doc.md", text: str = "body text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        char_start=0,
        char_end=len(text),
        metadata={"title": "Doc", "path": doc_id},
    )


def _result(chunk_id: str, rank: int, score: float, doc_id: str = "doc.md") -> RetrievalResult:
    return RetrievalResult(
        chunk=_chunk(chunk_id, doc_id=doc_id),
        score=score,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# RerankResult dataclass round-trip
# ---------------------------------------------------------------------------

def test_dataclass_rerank_result_round_trip_via_asdict():
    rr = RerankResult(
        chunk_id="abc123",
        pre_rank=2,
        post_rank=1,
        pre_score=0.4,
        post_score=0.9,
    )
    d = dataclasses.asdict(rr)
    assert d == {
        "chunk_id": "abc123",
        "pre_rank": 2,
        "post_rank": 1,
        "pre_score": 0.4,
        "post_score": 0.9,
    }


def test_dataclass_rerank_result_serializes_as_json():
    rr = RerankResult(chunk_id="x", pre_rank=1, post_rank=1, pre_score=0.0, post_score=0.0)
    payload = json.dumps(dataclasses.asdict(rr))
    parsed = json.loads(payload)
    assert parsed["chunk_id"] == "x"
    assert isinstance(parsed["pre_rank"], int)
    assert isinstance(parsed["post_score"], float)


# ---------------------------------------------------------------------------
# FakeReranker — noop mode (score_map is None)
# ---------------------------------------------------------------------------

def test_fake_noop_preserves_order():
    candidates = [
        _result("a", rank=1, score=0.9),
        _result("b", rank=2, score=0.8),
        _result("c", rank=3, score=0.7),
    ]
    audit = FakeReranker().rerank("q", candidates)
    assert [rr.chunk_id for rr in audit] == ["a", "b", "c"]
    for rr in audit:
        assert rr.pre_rank == rr.post_rank
        assert rr.pre_score == rr.post_score


def test_fake_noop_empty_input_returns_empty():
    assert FakeReranker().rerank("q", []) == []


def test_fake_noop_name_default():
    assert FakeReranker().name == "fake"


# ---------------------------------------------------------------------------
# FakeReranker — score_map mode
# ---------------------------------------------------------------------------

def test_fake_score_map_reorders_by_post_score():
    candidates = [
        _result("a", rank=1, score=0.5),
        _result("b", rank=2, score=0.5),
    ]
    score_map = {"a": 1.0, "b": 2.0}   # b should win
    audit = FakeReranker(score_map=score_map).rerank("q", candidates)
    assert [rr.chunk_id for rr in audit] == ["b", "a"]
    # post_rank is 1-indexed in returned order
    assert audit[0].post_rank == 1
    assert audit[1].post_rank == 2
    # pre_rank is preserved from input
    assert audit[0].pre_rank == 2     # b had pre_rank 2
    assert audit[1].pre_rank == 1     # a had pre_rank 1


def test_fake_score_map_missing_chunks_get_zero_score():
    candidates = [
        _result("a", rank=1, score=0.5),
        _result("b", rank=2, score=0.5),
    ]
    score_map = {"a": 1.0}   # b missing → score 0.0 → loses
    audit = FakeReranker(score_map=score_map).rerank("q", candidates)
    assert [rr.chunk_id for rr in audit] == ["a", "b"]
    assert audit[1].post_score == 0.0


def test_fake_score_map_ties_break_by_pre_rank():
    candidates = [
        _result("a", rank=1, score=0.0),
        _result("b", rank=2, score=0.0),
        _result("c", rank=3, score=0.0),
    ]
    score_map = {"a": 1.0, "b": 1.0, "c": 1.0}
    audit = FakeReranker(score_map=score_map).rerank("q", candidates)
    # All tied at 1.0 → pre_rank ascending wins → original order
    assert [rr.chunk_id for rr in audit] == ["a", "b", "c"]


def test_fake_score_map_returns_audit_for_every_candidate():
    candidates = [
        _result("a", rank=1, score=0.5),
        _result("b", rank=2, score=0.5),
        _result("c", rank=3, score=0.5),
        _result("d", rank=4, score=0.5),
    ]
    score_map = {"a": 0.1, "b": 0.5}
    audit = FakeReranker(score_map=score_map).rerank("q", candidates)
    assert len(audit) == len(candidates)
    assert {rr.chunk_id for rr in audit} == {"a", "b", "c", "d"}


def test_fake_score_map_pre_score_preserved_post_score_from_map():
    candidates = [_result("a", rank=1, score=0.42)]
    score_map = {"a": 0.99}
    audit = FakeReranker(score_map=score_map).rerank("q", candidates)
    assert audit[0].pre_score == 0.42
    assert audit[0].post_score == 0.99


# ---------------------------------------------------------------------------
# apply_reranker
# ---------------------------------------------------------------------------

def test_apply_reranker_slices_to_top_k():
    candidates = [_result(f"c{i}", rank=i, score=1.0 / i) for i in range(1, 6)]
    score_map = {f"c{i}": 1.0 / i for i in range(1, 6)}   # noop ordering
    reranker = FakeReranker(score_map=score_map)
    reordered, audit = apply_reranker("q", candidates, reranker, top_k=2)
    assert len(reordered) == 2
    assert len(audit) == 5
    assert [r.rank for r in reordered] == [1, 2]


def test_apply_reranker_clips_when_top_k_exceeds_candidates():
    candidates = [_result(f"c{i}", rank=i, score=1.0) for i in range(1, 4)]
    reranker = FakeReranker()   # noop
    reordered, audit = apply_reranker("q", candidates, reranker, top_k=10)
    # Clip semantics match retrieve_by_vector: no error, just return what we have
    assert len(reordered) == 3
    assert [r.rank for r in reordered] == [1, 2, 3]


def test_apply_reranker_empty_input_returns_empty_tuple():
    reranker = FakeReranker()
    reordered, audit = apply_reranker("q", [], reranker, top_k=5)
    assert reordered == []
    assert audit == []


def test_apply_reranker_negative_top_k_raises():
    candidates = [_result("a", rank=1, score=1.0)]
    reranker = FakeReranker()
    with pytest.raises(ValueError, match="top_k must be >= 0"):
        apply_reranker("q", candidates, reranker, top_k=-1)


def test_apply_reranker_post_score_replaces_pre_score_in_reordered():
    candidates = [
        _result("a", rank=1, score=0.5),
        _result("b", rank=2, score=0.5),
    ]
    score_map = {"a": 0.2, "b": 0.9}
    reranker = FakeReranker(score_map=score_map)
    reordered, audit = apply_reranker("q", candidates, reranker, top_k=2)
    assert reordered[0].chunk.chunk_id == "b"
    assert reordered[0].score == 0.9
    assert reordered[1].chunk.chunk_id == "a"
    assert reordered[1].score == 0.2


def test_apply_reranker_top_k_zero_returns_empty_reordered_but_full_audit():
    candidates = [
        _result("a", rank=1, score=0.5),
        _result("b", rank=2, score=0.5),
    ]
    reranker = FakeReranker()
    reordered, audit = apply_reranker("q", candidates, reranker, top_k=0)
    assert reordered == []
    assert len(audit) == 2


# ---------------------------------------------------------------------------
# chunk_traces_from_rerank
# ---------------------------------------------------------------------------

def test_chunk_traces_no_audit_leaves_pre_rerank_fields_none():
    results = [
        _result("a", rank=1, score=0.9),
        _result("b", rank=2, score=0.7),
    ]
    traces = chunk_traces_from_rerank(results, rerank_audit=None)
    assert len(traces) == 2
    for t in traces:
        assert t.pre_rerank_rank is None
        assert t.pre_rerank_score is None


def test_chunk_traces_with_audit_populates_pre_rerank_fields():
    # Suppose pre-rerank order was [a@1, b@2, c@3]; rerank moved b to rank 1.
    pre_a = RerankResult(chunk_id="a", pre_rank=1, post_rank=2, pre_score=0.5, post_score=0.3)
    pre_b = RerankResult(chunk_id="b", pre_rank=2, post_rank=1, pre_score=0.4, post_score=0.9)
    pre_c = RerankResult(chunk_id="c", pre_rank=3, post_rank=3, pre_score=0.2, post_score=0.1)
    audit = [pre_b, pre_a, pre_c]   # ordered by post_rank

    # Post-rerank top-2 slice that the trace formatter will see:
    reordered = [
        _result("b", rank=1, score=0.9),
        _result("a", rank=2, score=0.3),
    ]
    traces = chunk_traces_from_rerank(reordered, rerank_audit=audit)
    assert traces[0].chunk_id == "b"
    assert traces[0].pre_rerank_rank == 2
    assert traces[0].pre_rerank_score == 0.4
    assert traces[1].chunk_id == "a"
    assert traces[1].pre_rerank_rank == 1
    assert traces[1].pre_rerank_score == 0.5


def test_chunk_traces_empty_results():
    assert chunk_traces_from_rerank([], None) == []
    assert chunk_traces_from_rerank([], []) == []


def test_chunk_traces_returns_chunk_trace_instances():
    results = [_result("a", rank=1, score=0.5)]
    traces = chunk_traces_from_rerank(results, None)
    assert isinstance(traces[0], ChunkTrace)


def test_chunk_traces_carry_doc_metadata_from_results():
    results = [_result("a", rank=1, score=0.5, doc_id="path/to/doc.md")]
    traces = chunk_traces_from_rerank(results, None)
    assert traces[0].doc_id == "path/to/doc.md"
    assert traces[0].path == "path/to/doc.md"


def test_chunk_traces_text_preview_truncates_to_120_chars():
    long_text = "x" * 300
    chunk = Chunk(
        chunk_id="a", doc_id="d.md", text=long_text,
        char_start=0, char_end=300,
        metadata={"title": "T", "path": "d.md"},
    )
    results = [RetrievalResult(chunk=chunk, score=0.5, rank=1)]
    traces = chunk_traces_from_rerank(results, None)
    assert len(traces[0].text_preview) == 120
