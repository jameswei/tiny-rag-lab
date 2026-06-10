"""Tests for tiny_rag_lab/bm25.py — BM25Retriever and _tokenize helper."""
import pytest

from tiny_rag_lab.bm25 import BM25Retriever, _tokenize
from tiny_rag_lab.models import Chunk


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        text=text,
        char_start=0,
        char_end=len(text),
        metadata={},
    )


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

def test_tokenize_lowercases_and_splits():
    assert _tokenize("Hello World") == ["hello", "world"]


def test_tokenize_empty_string():
    assert _tokenize("") == []


def test_tokenize_single_word():
    assert _tokenize("RAG") == ["rag"]


# ---------------------------------------------------------------------------
# BM25Retriever.retrieve — basic ranking
# ---------------------------------------------------------------------------

def test_unique_term_ranks_first():
    chunks = [
        _make_chunk("c1", "machine learning is great"),
        _make_chunk("c2", "python is a programming language"),
        _make_chunk("c3", "zephyr is a unique keyword here"),
    ]
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("zephyr", top_k=3)
    assert results[0].chunk.chunk_id == "c3"
    assert results[0].rank == 1


def test_rank_values_are_1_indexed_and_contiguous():
    chunks = [_make_chunk(f"c{i}", f"word{i} content here") for i in range(4)]
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("word1", top_k=4)
    ranks = [r.rank for r in results]
    assert ranks == list(range(1, len(ranks) + 1))


def test_top_k_clips_to_corpus_size():
    chunks = [_make_chunk(f"c{i}", f"text {i}") for i in range(3)]
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("text", top_k=10)
    assert len(results) == 3


def test_top_k_respected_when_smaller_than_corpus():
    chunks = [_make_chunk(f"c{i}", f"text {i}") for i in range(5)]
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("text", top_k=2)
    assert len(results) == 2


def test_negative_top_k_raises():
    chunks = [_make_chunk("c1", "some text")]
    retriever = BM25Retriever(chunks)
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("some", top_k=-1)


# ---------------------------------------------------------------------------
# BM25Retriever.retrieve — empty cases
# ---------------------------------------------------------------------------

def test_empty_corpus_returns_empty():
    retriever = BM25Retriever([])
    assert retriever.retrieve("anything") == []


def test_empty_query_returns_empty():
    chunks = [_make_chunk("c1", "some text here")]
    retriever = BM25Retriever(chunks)
    assert retriever.retrieve("") == []


def test_whitespace_only_query_returns_empty():
    chunks = [_make_chunk("c1", "some text here")]
    retriever = BM25Retriever(chunks)
    assert retriever.retrieve("   ") == []


def test_all_empty_token_corpus_returns_empty():
    # Chunks whose text tokenizes to [] — guard against BM25Okapi ZeroDivisionError
    chunks = [_make_chunk("c1", "   "), _make_chunk("c2", "")]
    retriever = BM25Retriever(chunks)
    assert retriever.retrieve("anything") == []


# ---------------------------------------------------------------------------
# score field
# ---------------------------------------------------------------------------

def test_score_is_float():
    chunks = [_make_chunk("c1", "some text"), _make_chunk("c2", "other stuff")]
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("text", top_k=2)
    for r in results:
        assert isinstance(r.score, float)
