"""Calculation-level contracts used by the Phase 3.4 Retrieval UI."""

import json

import numpy as np
import pytest

from tiny_rag_lab.bm25 import BM25Retriever
from tiny_rag_lab.hybrid import reciprocal_rank_fusion_with_explanation
from tiny_rag_lab.index_loader import LoadedIndex
from tiny_rag_lab.models import Chunk, RetrievalResult
from tiny_rag_lab.reranker import RerankResult, explain_rerank
from tiny_rag_lab.retrieval import explain_dense_results, retrieve_by_vector


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"{chunk_id}.md",
        text=text,
        char_start=0,
        char_end=len(text),
        metadata={},
    )


def _result(chunk: Chunk, rank: int, score: float) -> RetrievalResult:
    return RetrievalResult(chunk=chunk, rank=rank, score=score)


def test_bm25_explanation_reproduces_ranked_scores():
    retriever = BM25Retriever([
        _chunk("first", "rare term appears twice rare"),
        _chunk("second", "common words only"),
        _chunk("third", "rare once"),
    ])

    results, explanation = retriever.retrieve_with_explanation("rare rare term", top_k=3)

    assert explanation.query_tokens == ["rare", "rare", "term"]
    assert explanation.corpus_size == 3
    assert [candidate.chunk_id for candidate in explanation.candidates] == [
        result.chunk.chunk_id for result in results
    ]
    for candidate in explanation.candidates:
        assert sum(term.contribution for term in candidate.terms) == pytest.approx(
            candidate.score, abs=1e-12,
        )
    rare = next(term for term in explanation.candidates[0].terms if term.term == "rare")
    assert rare.query_frequency == 2
    assert rare.term_frequency == 2
    assert rare.document_frequency == 2


def test_bm25_empty_explanation_is_still_typed():
    results, explanation = BM25Retriever([]).retrieve_with_explanation("term")
    assert results == []
    assert explanation.query_tokens == ["term"]
    assert explanation.candidates == []


def test_dense_explanation_reproduces_cosine_and_preserves_sign():
    chunks = [_chunk("positive", "positive"), _chunk("negative", "negative")]
    index = LoadedIndex(
        manifest={},
        chunks=chunks,
        embeddings=np.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32),
        chunk_ids=[chunk.chunk_id for chunk in chunks],
    )
    query = np.asarray([2.0, -2.0], dtype=np.float32)
    results = retrieve_by_vector(query, index, top_k=2)

    explanation = explain_dense_results(query, index, results, preview_dimensions=2)

    assert explanation[0].chunk_id == "positive"
    assert explanation[0].dot_product == pytest.approx(4.0)
    assert explanation[0].cosine_similarity == pytest.approx(results[0].score)
    assert explanation[0].query_vector_preview == [2.0, -2.0]
    assert explanation[0].chunk_vector_preview == [1.0, -1.0]
    assert explanation[1].cosine_similarity == pytest.approx(-1.0)


def test_dense_explanation_rejects_wrong_dimension():
    chunk = _chunk("one", "one")
    index = LoadedIndex(
        manifest={}, chunks=[chunk],
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        chunk_ids=["one"],
    )
    with pytest.raises(ValueError, match="dimension"):
        explain_dense_results(
            np.asarray([1.0], dtype=np.float32), index,
            [_result(chunk, 1, 1.0)],
        )


def test_rrf_explanation_reproduces_fused_score():
    first, second = _chunk("first", "first"), _chunk("second", "second")
    dense = [_result(first, 1, 0.8), _result(second, 2, 0.7)]
    bm25 = [_result(second, 1, 4.0), _result(first, 2, 3.0)]

    fused, explanation = reciprocal_rank_fusion_with_explanation(
        [dense, bm25], top_k=2, source_names=["dense", "bm25"],
    )

    assert [item.chunk_id for item in explanation] == [item.chunk.chunk_id for item in fused]
    for candidate in explanation:
        assert sum(source.contribution for source in candidate.sources) == pytest.approx(
            candidate.score,
        )
        assert {source.source for source in candidate.sources} == {"dense", "bm25"}
    assert explanation[0].sources[0].contribution == pytest.approx(1 / 61)


def test_rrf_explanation_requires_one_name_per_list():
    with pytest.raises(ValueError, match="source_names"):
        reciprocal_rank_fusion_with_explanation([[]], top_k=1, source_names=[])


def test_rerank_explanation_marks_movement_and_dropped_candidates():
    audit = [
        RerankResult("a", pre_rank=1, post_rank=3, pre_score=0.9, post_score=0.1),
        RerankResult("b", pre_rank=2, post_rank=1, pre_score=0.8, post_score=0.9),
        RerankResult("c", pre_rank=3, post_rank=2, pre_score=0.7, post_score=0.8),
        RerankResult("d", pre_rank=4, post_rank=4, pre_score=0.6, post_score=0.0),
    ]

    explanation = explain_rerank(audit, final_top_k=3)

    assert [item.outcome for item in explanation] == [
        "moved_down", "moved_up", "moved_up", "dropped",
    ]
    assert explanation[0].rank_delta == -2
    assert explanation[1].rank_delta == 1
    assert explanation[3].final_rank is None
    assert explanation[3].rank_delta is None
    # The explanation remains directly JSON-safe for API artifacts.
    json.dumps([item.__dict__ for item in explanation])
