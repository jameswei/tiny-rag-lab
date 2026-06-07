"""Tests for T12 — cosine retrieval.

Tests use known vectors so expected rankings are deterministic and do not
depend on any embedder or model download.
"""
from pathlib import Path

import numpy as np
import pytest

from tiny_rag_lab.embeddings import FakeEmbedder
from tiny_rag_lab.index_loader import LoadedIndex
from tiny_rag_lab.index_writer import write_index
from tiny_rag_lab.models import Chunk, RetrievalResult
from tiny_rag_lab.retrieval import DEFAULT_TOP_K, retrieve, retrieve_by_vector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str, doc_id: str = "docs/a.md") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=f"text for {chunk_id}",
        char_start=0,
        char_end=20,
        metadata={"title": "T", "path": "/corpus/docs/a.md",
                   "format": "markdown", "raw_hash": "x"},
    )


def _make_index(embeddings: np.ndarray) -> LoadedIndex:
    """Build a LoadedIndex from a raw embedding matrix (one chunk per row)."""
    n = embeddings.shape[0]
    chunks = [_make_chunk(f"chunk{i:04d}") for i in range(n)]
    return LoadedIndex(
        manifest={},
        chunks=chunks,
        embeddings=embeddings.astype(np.float32),
        chunk_ids=[c.chunk_id for c in chunks],
    )


def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return (a / np.linalg.norm(a)).astype(np.float32)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_returns_list_of_retrieval_result():
    index = _make_index(np.eye(3, dtype=np.float32))
    results = retrieve_by_vector(np.array([1, 0, 0], dtype=np.float32), index)
    assert isinstance(results, list)
    assert all(isinstance(r, RetrievalResult) for r in results)


# ---------------------------------------------------------------------------
# Ranking correctness with known vectors
# ---------------------------------------------------------------------------

def test_exact_match_ranks_first():
    # chunk0 is [1,0,0], chunk1 is [0,1,0], chunk2 is [0,0,1]
    # query [1,0,0] should rank chunk0 first with score 1.0
    index = _make_index(np.eye(3, dtype=np.float32))
    results = retrieve_by_vector(np.array([1, 0, 0], dtype=np.float32), index, top_k=3)
    assert results[0].chunk.chunk_id == "chunk0000"
    assert pytest.approx(results[0].score, abs=1e-5) == 1.0


def test_ranking_order_by_similarity():
    # construct three vectors with known cosine distances to query [1,1,0]
    # chunk0: [1,0,0] -> cos = 1/sqrt(2) ≈ 0.707
    # chunk1: [1,1,0] -> cos = 1.0 (normalized: [1/√2, 1/√2, 0])
    # chunk2: [0,0,1] -> cos = 0.0
    emb = np.array([
        _unit([1, 0, 0]),
        _unit([1, 1, 0]),
        _unit([0, 0, 1]),
    ], dtype=np.float32)
    index = _make_index(emb)
    query = _unit([1, 1, 0])
    results = retrieve_by_vector(query, index, top_k=3)
    ids = [r.chunk.chunk_id for r in results]
    assert ids == ["chunk0001", "chunk0000", "chunk0002"]


def test_scores_are_descending():
    emb = np.array([
        _unit([1, 0, 0]),
        _unit([1, 1, 0]),
        _unit([0, 0, 1]),
    ], dtype=np.float32)
    index = _make_index(emb)
    query = _unit([1, 1, 0])
    results = retrieve_by_vector(query, index, top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_ranks_are_1_indexed_and_sequential():
    index = _make_index(np.eye(4, dtype=np.float32))
    results = retrieve_by_vector(np.array([1, 0, 0, 0], dtype=np.float32), index, top_k=4)
    assert [r.rank for r in results] == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# top_k behaviour
# ---------------------------------------------------------------------------

def test_top_k_limits_results():
    index = _make_index(np.eye(10, dtype=np.float32))
    results = retrieve_by_vector(np.array([1] + [0] * 9, dtype=np.float32), index, top_k=3)
    assert len(results) == 3


def test_top_k_larger_than_index_returns_all():
    index = _make_index(np.eye(3, dtype=np.float32))
    results = retrieve_by_vector(np.array([1, 0, 0], dtype=np.float32), index, top_k=100)
    assert len(results) == 3


def test_default_top_k_is_five():
    assert DEFAULT_TOP_K == 5
    index = _make_index(np.eye(10, dtype=np.float32))
    results = retrieve_by_vector(np.array([1] + [0] * 9, dtype=np.float32), index)
    assert len(results) == DEFAULT_TOP_K


def test_negative_top_k_raises():
    index = _make_index(np.eye(3, dtype=np.float32))
    with pytest.raises(ValueError, match="top_k"):
        retrieve_by_vector(np.array([1, 0, 0], dtype=np.float32), index, top_k=-1)


def test_top_k_zero_returns_empty():
    index = _make_index(np.eye(3, dtype=np.float32))
    results = retrieve_by_vector(np.array([1, 0, 0], dtype=np.float32), index, top_k=0)
    assert results == []


# ---------------------------------------------------------------------------
# Zero-vector policy
# ---------------------------------------------------------------------------

def test_zero_query_returns_empty():
    index = _make_index(np.eye(3, dtype=np.float32))
    results = retrieve_by_vector(np.zeros(3, dtype=np.float32), index)
    assert results == []


def test_zero_index_row_scores_zero():
    emb = np.array([
        _unit([1, 0, 0]),
        np.zeros(3, dtype=np.float32),  # zero-norm row
        _unit([0, 1, 0]),
    ], dtype=np.float32)
    index = _make_index(emb)
    query = _unit([1, 0, 0])
    results = retrieve_by_vector(query, index, top_k=3)
    zero_result = next(r for r in results if r.chunk.chunk_id == "chunk0001")
    assert zero_result.score == 0.0


def test_zero_index_row_ranks_last():
    emb = np.array([
        _unit([1, 0, 0]),
        np.zeros(3, dtype=np.float32),  # should rank last
        _unit([0, 1, 0]),
    ], dtype=np.float32)
    index = _make_index(emb)
    query = _unit([1, 0, 0])
    results = retrieve_by_vector(query, index, top_k=3)
    assert results[-1].chunk.chunk_id == "chunk0001"


# ---------------------------------------------------------------------------
# Empty index
# ---------------------------------------------------------------------------

def test_empty_index_returns_empty():
    index = LoadedIndex(manifest={}, chunks=[], embeddings=np.empty((0, 4), dtype=np.float32), chunk_ids=[])
    results = retrieve_by_vector(np.array([1, 0, 0, 0], dtype=np.float32), index)
    assert results == []


# ---------------------------------------------------------------------------
# Score range
# ---------------------------------------------------------------------------

def test_scores_in_valid_range():
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((20, 8)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    emb = raw / norms
    index = _make_index(emb)
    query = _unit([1, 0, 0, 0, 0, 0, 0, 0])
    results = retrieve_by_vector(query, index, top_k=20)
    for r in results:
        assert -1.0 - 1e-5 <= r.score <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# Unnormalized input tolerance
# ---------------------------------------------------------------------------

def test_unnormalized_query_gives_same_ranking_as_normalized():
    emb = np.array([_unit([1, 0, 0]), _unit([1, 1, 0]), _unit([0, 1, 0])], dtype=np.float32)
    index = _make_index(emb)
    unit_q = _unit([1, 1, 0])
    scaled_q = unit_q * 42.0  # same direction, different magnitude
    r_unit = retrieve_by_vector(unit_q, index, top_k=3)
    r_scaled = retrieve_by_vector(scaled_q, index, top_k=3)
    assert [r.chunk.chunk_id for r in r_unit] == [r.chunk.chunk_id for r in r_scaled]


# ---------------------------------------------------------------------------
# retrieve() — high-level function with FakeEmbedder
# ---------------------------------------------------------------------------

def test_retrieve_with_fake_embedder(tmp_path):
    embedder = FakeEmbedder(dim=8)
    texts = ["alpha", "beta", "gamma"]
    embeddings = embedder.embed(texts)
    chunks = [_make_chunk(f"c{i:04d}") for i in range(3)]
    # manually override chunk text so embed([query]) matches chunks[0]
    index = LoadedIndex(
        manifest={},
        chunks=chunks,
        embeddings=embeddings,
        chunk_ids=[c.chunk_id for c in chunks],
    )
    # query "alpha" should embed identically to chunk 0's embedding
    results = retrieve("alpha", index, embedder, top_k=3)
    assert results[0].chunk.chunk_id == "c0000"
    assert pytest.approx(results[0].score, abs=1e-5) == 1.0


def test_retrieve_returns_retrieval_results(tmp_path):
    embedder = FakeEmbedder(dim=8)
    emb = embedder.embed(["a", "b", "c"])
    index = _make_index(emb)
    results = retrieve("a", index, embedder, top_k=2)
    assert len(results) == 2
    assert all(isinstance(r, RetrievalResult) for r in results)


# ---------------------------------------------------------------------------
# Round-trip with write_index / load_index
# ---------------------------------------------------------------------------

def test_retrieve_after_write_load_roundtrip(tmp_path):
    from tiny_rag_lab.index_loader import load_index
    from tiny_rag_lab.models import Document

    embedder = FakeEmbedder(dim=8)
    texts = ["watsonx overview", "retrieval augmented generation", "unrelated topic"]
    chunks = [
        Chunk(
            chunk_id=f"cid{i:04d}",
            doc_id="docs/a.md",
            text=t,
            char_start=0,
            char_end=len(t),
            metadata={"title": "T", "path": "/c/docs/a.md", "format": "markdown", "raw_hash": "x"},
        )
        for i, t in enumerate(texts)
    ]
    embeddings = embedder.embed(texts)
    docs = [Document(doc_id="docs/a.md", path="/c/docs/a.md", title="T",
                     format="markdown", raw_text="", normalized_text="", raw_hash="x")]
    index_dir = tmp_path / "index"
    write_index(
        index_dir, docs=docs, chunks=chunks, embeddings=embeddings,
        corpus_root=Path("/c"), embedding_backend="FakeEmbedder",
        embedding_model="fake", embedding_dim=8, chunk_size=800, chunk_overlap=120,
    )
    loaded = load_index(index_dir)
    results = retrieve("watsonx overview", loaded, embedder, top_k=1)
    assert results[0].chunk.chunk_id == "cid0000"
    assert pytest.approx(results[0].score, abs=1e-5) == 1.0
