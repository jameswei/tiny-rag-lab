"""Gated tests for CrossEncoderReranker (P1.9-T02).

These tests are skipped by default in CI because they require:
1. The sentence_transformers package to be installed.
2. The TINY_RAG_LAB_TEST_RERANKER=1 env flag to be set (prevents accidental
   model downloads in test suites).

When both conditions are met and the default cross-encoder model is cached
locally, the tests exercise the real CrossEncoder path. Otherwise they are
silently skipped — no model download is ever triggered by the test suite.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

# Gate 1: skip unless the env flag is explicitly set (no import of sentence_transformers yet).
if os.environ.get("TINY_RAG_LAB_TEST_RERANKER") != "1":
    pytest.skip(
        "TINY_RAG_LAB_TEST_RERANKER not set; set to 1 to enable real cross-encoder tests",
        allow_module_level=True,
    )

# Gate 2: skip if sentence_transformers is not installed.
pytest.importorskip("sentence_transformers")

from tiny_rag_lab.models import Chunk, RetrievalResult
from tiny_rag_lab.reranker import CrossEncoderReranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(chunk_id: str, text: str, doc_id: str = "doc.md") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        char_start=0,
        char_end=len(text),
        metadata={"title": "Doc", "path": doc_id},
    )


def _result(chunk_id: str, rank: int, score: float, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk=_chunk(chunk_id, text=text),
        score=score,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# Construction — no I/O
# ---------------------------------------------------------------------------

def test_cross_encoder_constructor_does_not_load_model():
    """CrossEncoder is not imported or constructed during __init__."""
    with patch("sentence_transformers.CrossEncoder") as mock_ce:
        reranker = CrossEncoderReranker()
        mock_ce.assert_not_called()
    assert reranker.name == "cross-encoder"


def test_cross_encoder_name_is_cross_encoder():
    assert CrossEncoderReranker().name == "cross-encoder"


def test_cross_encoder_default_model_exposed():
    assert CrossEncoderReranker.DEFAULT_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_cross_encoder_custom_model_name_stored():
    reranker = CrossEncoderReranker(model_name="custom/model")
    assert reranker._model_name == "custom/model"


# ---------------------------------------------------------------------------
# Gated real-model tests
# ---------------------------------------------------------------------------

def test_cross_encoder_rerank_reorders_candidates():
    """Real model reorders a query-relevant chunk above an irrelevant one."""
    reranker = CrossEncoderReranker()

    candidates = [
        _result("a", rank=1, score=0.0, text="The sky is blue and the sun is bright."),
        _result("b", rank=2, score=0.0, text="Python is a high-level programming language."),
    ]
    audit = reranker.rerank("What color is the sky?", candidates)

    assert len(audit) == 2
    assert audit[0].chunk_id == "a"  # "sky is blue" should rank higher
    assert audit[0].post_rank == 1
    assert audit[1].chunk_id == "b"
    assert audit[1].post_rank == 2


def test_cross_encoder_rerank_empty_input():
    reranker = CrossEncoderReranker()
    assert reranker.rerank("query", []) == []


def test_cross_encoder_rerank_preserves_pre_rank_and_pre_score():
    """Each audit record carries the original pre-rerank rank and score."""
    reranker = CrossEncoderReranker()

    candidates = [
        _result("a", rank=1, score=0.9, text="The capital of France is Paris."),
        _result("b", rank=2, score=0.8, text="Berlin is the capital of Germany."),
    ]
    audit = reranker.rerank("What is the capital of France?", candidates)

    for rr in audit:
        assert rr.pre_rank in (1, 2)
        assert rr.pre_score in (0.9, 0.8)
        assert rr.post_rank in (1, 2)


def test_cross_encoder_rerank_single_candidate():
    """Single candidate should remain at rank 1."""
    reranker = CrossEncoderReranker()

    candidates = [
        _result("a", rank=1, score=0.5, text="Machine learning is a subset of AI."),
    ]
    audit = reranker.rerank("What is machine learning?", candidates)

    assert len(audit) == 1
    assert audit[0].post_rank == 1
    assert audit[0].chunk_id == "a"
