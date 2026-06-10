"""Tests for tiny_rag_lab/hybrid.py — reciprocal_rank_fusion and retrieve_hybrid."""
import numpy as np

from tiny_rag_lab.bm25 import BM25Retriever
from tiny_rag_lab.embeddings import FakeEmbedder
from tiny_rag_lab.hybrid import reciprocal_rank_fusion, retrieve_hybrid
from tiny_rag_lab.index_loader import LoadedIndex
from tiny_rag_lab.models import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str, text: str = "") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        text=text or f"text for {chunk_id}",
        char_start=0,
        char_end=10,
        metadata={},
    )


def _make_result(chunk: Chunk, rank: int, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(chunk=chunk, score=score, rank=rank)


def _make_index(chunks: list[Chunk], dim: int = 8) -> LoadedIndex:
    embeddings = np.random.default_rng(42).random((len(chunks), dim)).astype(np.float32)
    return LoadedIndex(
        manifest={},
        chunks=chunks,
        embeddings=embeddings,
        chunk_ids=[c.chunk_id for c in chunks],
    )


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion — scoring
# ---------------------------------------------------------------------------

def test_rrf_chunk_in_both_lists_at_rank1_scores_highest():
    ca, cb, cc = _make_chunk("a"), _make_chunk("b"), _make_chunk("c")
    list1 = [_make_result(ca, 1), _make_result(cb, 2)]
    list2 = [_make_result(ca, 1), _make_result(cc, 2)]
    results = reciprocal_rank_fusion([list1, list2], top_k=3)
    assert results[0].chunk.chunk_id == "a"
    assert results[0].rank == 1


def test_rrf_chunk_in_both_lists_scores_higher_than_single_list():
    ca, cb = _make_chunk("a"), _make_chunk("b")
    # ca appears in both at rank 1; cb appears in only one at rank 1
    list1 = [_make_result(ca, 1)]
    list2 = [_make_result(ca, 1), _make_result(cb, 2)]
    results = reciprocal_rank_fusion([list1, list2], top_k=2)
    scores = {r.chunk.chunk_id: r.score for r in results}
    assert scores["a"] > scores["b"]


def test_rrf_chunk_in_one_list_still_appears():
    ca, cb = _make_chunk("a"), _make_chunk("b")
    list1 = [_make_result(ca, 1)]
    list2 = [_make_result(cb, 1)]
    results = reciprocal_rank_fusion([list1, list2], top_k=2)
    chunk_ids = {r.chunk.chunk_id for r in results}
    assert "a" in chunk_ids
    assert "b" in chunk_ids


def test_rrf_score_formula():
    ca = _make_chunk("a")
    k = 60
    list1 = [_make_result(ca, 1)]
    list2 = [_make_result(ca, 2)]
    results = reciprocal_rank_fusion([list1, list2], top_k=1, k=k)
    expected = 1 / (k + 1) + 1 / (k + 2)
    assert abs(results[0].score - expected) < 1e-9


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion — output shape and rank numbering
# ---------------------------------------------------------------------------

def test_rrf_returns_top_k():
    chunks = [_make_chunk(f"c{i}") for i in range(6)]
    list1 = [_make_result(c, i + 1) for i, c in enumerate(chunks[:3])]
    list2 = [_make_result(c, i + 1) for i, c in enumerate(chunks[3:])]
    results = reciprocal_rank_fusion([list1, list2], top_k=4)
    assert len(results) == 4


def test_rrf_rank_values_1_indexed_and_contiguous():
    chunks = [_make_chunk(f"c{i}") for i in range(4)]
    list1 = [_make_result(c, i + 1) for i, c in enumerate(chunks)]
    results = reciprocal_rank_fusion([list1], top_k=4)
    assert [r.rank for r in results] == [1, 2, 3, 4]


def test_rrf_clips_to_available_chunks():
    ca, cb = _make_chunk("a"), _make_chunk("b")
    results = reciprocal_rank_fusion([[_make_result(ca, 1), _make_result(cb, 2)]], top_k=10)
    assert len(results) == 2


def test_rrf_empty_lists_returns_empty():
    results = reciprocal_rank_fusion([[], []], top_k=5)
    assert results == []


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion — tie-breaking (dense wins)
# ---------------------------------------------------------------------------

def test_rrf_tiebreak_dense_first():
    ca, cb = _make_chunk("a"), _make_chunk("b")
    # ca and cb each appear in exactly one list at rank 1 — equal RRF score.
    # ca is in the first (dense) list, so it should rank first after stable sort.
    dense = [_make_result(ca, 1)]
    bm25 = [_make_result(cb, 1)]
    results = reciprocal_rank_fusion([dense, bm25], top_k=2)
    assert results[0].chunk.chunk_id == "a"


# ---------------------------------------------------------------------------
# retrieve_hybrid
# ---------------------------------------------------------------------------

def test_retrieve_hybrid_returns_top_k():
    chunks = [_make_chunk(f"c{i}", f"word{i} content") for i in range(10)]
    index = _make_index(chunks)
    embedder = FakeEmbedder(dim=8)
    results = retrieve_hybrid("word1", index, embedder, top_k=5)
    assert len(results) == 5


def test_retrieve_hybrid_rank_1_indexed_contiguous():
    chunks = [_make_chunk(f"c{i}", f"word{i} content") for i in range(6)]
    index = _make_index(chunks)
    embedder = FakeEmbedder(dim=8)
    results = retrieve_hybrid("word1", index, embedder, top_k=4)
    assert [r.rank for r in results] == [1, 2, 3, 4]


def test_retrieve_hybrid_accepts_injected_bm25():
    chunks = [_make_chunk(f"c{i}", f"word{i} content") for i in range(6)]
    index = _make_index(chunks)
    embedder = FakeEmbedder(dim=8)
    bm25 = BM25Retriever(chunks)
    results = retrieve_hybrid("word1", index, embedder, top_k=3, bm25_retriever=bm25)
    assert len(results) == 3


def test_retrieve_hybrid_builds_bm25_internally_when_none():
    chunks = [_make_chunk(f"c{i}", f"word{i} content") for i in range(6)]
    index = _make_index(chunks)
    embedder = FakeEmbedder(dim=8)
    results = retrieve_hybrid("word1", index, embedder, top_k=3, bm25_retriever=None)
    assert len(results) == 3


def test_retrieve_hybrid_returns_retrieval_results():
    chunks = [_make_chunk(f"c{i}", f"word{i} content") for i in range(4)]
    index = _make_index(chunks)
    embedder = FakeEmbedder(dim=8)
    results = retrieve_hybrid("word0", index, embedder, top_k=4)
    assert all(isinstance(r, RetrievalResult) for r in results)
