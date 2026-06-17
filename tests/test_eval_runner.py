"""Tests for tiny_rag_lab/eval.py — loader and runner.

T02: loader tests (marked with 'load' in name)
T04: runner tests
"""
import json
from pathlib import Path

import pytest

from tiny_rag_lab.chunking import chunk_documents
from tiny_rag_lab.documents import load_documents
from tiny_rag_lab.embeddings import FakeEmbedder
from tiny_rag_lab.eval import EvalReport, EvalResult, EvalSample, load_eval_samples, run_retrieval_eval
from tiny_rag_lab.reranker import FakeReranker
from tiny_rag_lab.index_loader import LoadedIndex

FIXTURE_QA = Path(__file__).parent / "fixtures" / "eval" / "qa.jsonl"


# ---------------------------------------------------------------------------
# T02 — load_eval_samples
# ---------------------------------------------------------------------------

def test_load_eval_samples_returns_list():
    samples = load_eval_samples(FIXTURE_QA)
    assert isinstance(samples, list)


def test_load_eval_samples_fixture_count():
    samples = load_eval_samples(FIXTURE_QA)
    assert len(samples) == 3


def test_load_eval_samples_returns_eval_sample_objects():
    samples = load_eval_samples(FIXTURE_QA)
    for s in samples:
        assert isinstance(s, EvalSample)


def test_load_eval_samples_field_mapping():
    samples = load_eval_samples(FIXTURE_QA)
    s = samples[0]
    assert s.question_id == "q001"
    assert "sample document" in s.question.lower()
    assert len(s.answer) > 0
    assert s.gold_doc_ids == ["with_h1.md"]


def test_load_eval_samples_all_gold_doc_ids_populated():
    samples = load_eval_samples(FIXTURE_QA)
    for s in samples:
        assert len(s.gold_doc_ids) > 0


def test_load_eval_samples_skips_empty_question(tmp_path):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        json.dumps({"question_id": "q1", "question": "", "answer": "A", "gold_doc_ids": ["a.md"]}) + "\n" +
        json.dumps({"question_id": "q2", "question": "Real Q?", "answer": "A", "gold_doc_ids": ["b.md"]}) + "\n"
    )
    samples = load_eval_samples(qa)
    assert len(samples) == 1
    assert samples[0].question_id == "q2"


def test_load_eval_samples_skips_empty_gold_doc_ids(tmp_path):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        json.dumps({"question_id": "q1", "question": "Q?", "answer": "A", "gold_doc_ids": []}) + "\n" +
        json.dumps({"question_id": "q2", "question": "Q2?", "answer": "A", "gold_doc_ids": ["b.md"]}) + "\n"
    )
    samples = load_eval_samples(qa)
    assert len(samples) == 1
    assert samples[0].question_id == "q2"


def test_load_eval_samples_skips_whitespace_only_question(tmp_path):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        json.dumps({"question_id": "q1", "question": "   ", "answer": "A", "gold_doc_ids": ["a.md"]}) + "\n"
    )
    samples = load_eval_samples(qa)
    assert len(samples) == 0


def test_load_eval_samples_skips_blank_lines(tmp_path):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        "\n" +
        json.dumps({"question_id": "q1", "question": "Q?", "answer": "A", "gold_doc_ids": ["a.md"]}) + "\n" +
        "\n"
    )
    samples = load_eval_samples(qa)
    assert len(samples) == 1


def test_load_eval_samples_skips_malformed_json(tmp_path):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        "not valid json\n" +
        json.dumps({"question_id": "q1", "question": "Q?", "answer": "A", "gold_doc_ids": ["a.md"]}) + "\n"
    )
    samples = load_eval_samples(qa)
    assert len(samples) == 1


def test_load_eval_samples_skips_wrong_type_gold_doc_ids(tmp_path):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        json.dumps({"question_id": "q1", "question": "Q?", "answer": "A", "gold_doc_ids": 42}) + "\n" +
        json.dumps({"question_id": "q2", "question": "Q2?", "answer": "A", "gold_doc_ids": ["b.md"]}) + "\n"
    )
    samples = load_eval_samples(qa)
    assert len(samples) == 1
    assert samples[0].question_id == "q2"


def test_load_eval_samples_skips_non_dict_json_rows(tmp_path):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        "[]\n" +
        json.dumps({"question_id": "q1", "question": "Q?", "answer": "A", "gold_doc_ids": ["a.md"]}) + "\n"
    )
    samples = load_eval_samples(qa)
    assert len(samples) == 1


def test_load_eval_samples_reference_answer_and_expected_facts_populated(tmp_path):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(json.dumps({
        "question_id": "q1", "question": "Q?", "answer": "A",
        "gold_doc_ids": ["a.md"],
        "reference_answer": "The correct answer.",
        "expected_facts": ["fact one", "fact two"],
    }) + "\n")
    samples = load_eval_samples(qa)
    assert len(samples) == 1
    assert samples[0].reference_answer == "The correct answer."
    assert samples[0].expected_facts == ["fact one", "fact two"]


def test_load_eval_samples_reference_answer_defaults_to_none(tmp_path):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(json.dumps({
        "question_id": "q1", "question": "Q?", "answer": "A",
        "gold_doc_ids": ["a.md"],
    }) + "\n")
    samples = load_eval_samples(qa)
    assert samples[0].reference_answer is None


def test_load_eval_samples_expected_facts_defaults_to_empty_list(tmp_path):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(json.dumps({
        "question_id": "q1", "question": "Q?", "answer": "A",
        "gold_doc_ids": ["a.md"],
    }) + "\n")
    samples = load_eval_samples(qa)
    assert samples[0].expected_facts == []


def test_load_eval_samples_existing_fixture_still_loads(tmp_path):
    """Existing qa.jsonl rows without the new fields load with back-compat defaults."""
    samples = load_eval_samples(FIXTURE_QA)
    assert len(samples) > 0
    for s in samples:
        assert s.reference_answer is None
        assert s.expected_facts == []


# ---------------------------------------------------------------------------
# T04 — run_retrieval_eval helpers
# ---------------------------------------------------------------------------

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"
FIXTURE_QA = Path(__file__).parent / "fixtures" / "eval" / "qa.jsonl"


def _build_index(dim: int = 8) -> LoadedIndex:
    """Build an in-memory LoadedIndex from the fixture corpus using FakeEmbedder."""
    docs = load_documents(FIXTURE_CORPUS)
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    embedder = FakeEmbedder(dim=dim)
    embeddings = embedder.embed([c.text for c in chunks])
    return LoadedIndex(
        manifest={},
        chunks=chunks,
        embeddings=embeddings,
        chunk_ids=[c.chunk_id for c in chunks],
    )


# ---------------------------------------------------------------------------
# T04 — run_retrieval_eval
# ---------------------------------------------------------------------------

def test_runner_returns_eval_report():
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    report = run_retrieval_eval(samples, index, embedder, top_k=3)
    assert isinstance(report, EvalReport)


def test_runner_n_questions_matches_sample_count():
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    report = run_retrieval_eval(samples, index, embedder, top_k=3)
    assert report.n_questions == 3


def test_runner_per_question_has_one_result_per_sample():
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    report = run_retrieval_eval(samples, index, embedder, top_k=3)
    assert len(report.per_question) == 3


def test_runner_per_question_results_are_eval_result():
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    report = run_retrieval_eval(samples, index, embedder, top_k=3)
    for r in report.per_question:
        assert isinstance(r, EvalResult)


def test_runner_top_k_stored_in_report():
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    report = run_retrieval_eval(samples, index, embedder, top_k=7)
    assert report.top_k == 7


def test_runner_hit_rate_is_mean_of_per_question_hits():
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    report = run_retrieval_eval(samples, index, embedder, top_k=3)
    expected = sum(r.hit for r in report.per_question) / len(report.per_question)
    assert report.hit_rate == pytest.approx(expected)


def test_runner_mrr_is_mean_of_per_question_rr():
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    report = run_retrieval_eval(samples, index, embedder, top_k=3)
    expected = sum(r.reciprocal_rank for r in report.per_question) / len(report.per_question)
    assert report.mrr == pytest.approx(expected)


def test_runner_mean_context_precision_is_arithmetic_mean():
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    report = run_retrieval_eval(samples, index, embedder, top_k=3)
    expected = sum(r.context_precision for r in report.per_question) / len(report.per_question)
    assert report.mean_context_precision == pytest.approx(expected)


def test_runner_mean_context_recall_is_arithmetic_mean():
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    report = run_retrieval_eval(samples, index, embedder, top_k=3)
    expected = sum(r.context_recall for r in report.per_question) / len(report.per_question)
    assert report.mean_context_recall == pytest.approx(expected)


def test_runner_per_question_question_ids_match_samples():
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    report = run_retrieval_eval(samples, index, embedder, top_k=3)
    assert [r.question_id for r in report.per_question] == [s.question_id for s in samples]


def test_runner_retrieved_doc_ids_length_at_most_top_k():
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    top_k = 2
    report = run_retrieval_eval(samples, index, embedder, top_k=top_k)
    for r in report.per_question:
        assert len(r.retrieved_doc_ids) <= top_k


def test_runner_empty_samples_returns_empty_report():
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    report = run_retrieval_eval([], index, embedder, top_k=5)
    assert report.n_questions == 0
    assert report.per_question == []
    assert report.hit_rate == 0.0


# ---------------------------------------------------------------------------
# T04 (Phase 1.5) — retriever field and multi-strategy paths
# ---------------------------------------------------------------------------

def test_runner_retriever_field_defaults_to_dense():
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    report = run_retrieval_eval(samples, index, embedder, top_k=3)
    assert report.retriever == "dense"


def test_runner_retriever_field_stored_in_report():
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    report = run_retrieval_eval(samples, index, embedder, top_k=3, retriever="dense")
    assert report.retriever == "dense"


def test_runner_bm25_path_works_with_none_embedder():
    index = _build_index()
    samples = load_eval_samples(FIXTURE_QA)
    report = run_retrieval_eval(samples, index, None, top_k=3, retriever="bm25")
    assert isinstance(report, EvalReport)
    assert report.retriever == "bm25"
    assert report.n_questions == 3


def test_runner_hybrid_path_returns_report():
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    report = run_retrieval_eval(samples, index, embedder, top_k=3, retriever="hybrid")
    assert report.retriever == "hybrid"
    assert report.n_questions == 3


def test_runner_dense_with_none_embedder_raises():
    index = _build_index()
    samples = load_eval_samples(FIXTURE_QA)
    with pytest.raises(ValueError, match="embedder"):
        run_retrieval_eval(samples, index, None, top_k=3, retriever="dense")


def test_runner_hybrid_with_none_embedder_raises():
    index = _build_index()
    samples = load_eval_samples(FIXTURE_QA)
    with pytest.raises(ValueError, match="embedder"):
        run_retrieval_eval(samples, index, None, top_k=3, retriever="hybrid")


def test_runner_empty_samples_bm25_returns_empty_report():
    index = _build_index()
    report = run_retrieval_eval([], index, None, top_k=5, retriever="bm25")
    assert report.n_questions == 0
    assert report.retriever == "bm25"


def test_runner_invalid_retriever_raises():
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    with pytest.raises(ValueError, match="retriever"):
        run_retrieval_eval(samples, index, embedder, top_k=3, retriever="bogus")


# ---------------------------------------------------------------------------
# P1.9-T04 — reranker integration
# ---------------------------------------------------------------------------

def test_runner_reranker_default_fields_are_none():
    """Without reranker, EvalReport fields default to 'none'/None."""
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    report = run_retrieval_eval(samples, index, embedder, top_k=3)
    assert report.reranker == "none"
    assert report.rerank_top_n is None


def test_runner_with_reranker_sets_report_fields():
    """Reranker fields populate correctly in EvalReport."""
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    reranker = FakeReranker()
    report = run_retrieval_eval(
        samples, index, embedder, top_k=2, reranker=reranker, rerank_top_n=5,
    )
    assert report.reranker == "fake"
    assert report.rerank_top_n == 5
    assert report.n_questions == 3
    assert len(report.per_question) == 3


def test_runner_reranker_none_is_identical_to_no_reranker():
    """run_retrieval_eval with reranker=None matches pre-1.9 behavior."""
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)

    report_no_rerank = run_retrieval_eval(samples, index, embedder, top_k=3)
    report_none = run_retrieval_eval(samples, index, embedder, top_k=3, reranker=None)

    assert report_no_rerank.hit_rate == report_none.hit_rate
    assert report_no_rerank.mrr == report_none.mrr
    assert report_no_rerank.mean_context_precision == report_none.mean_context_precision
    assert report_no_rerank.mean_context_recall == report_none.mean_context_recall
    assert report_no_rerank.reranker == report_none.reranker == "none"


def test_runner_rerank_top_n_none_with_reranker_raises():
    """reranker provided but rerank_top_n is None raises ValueError."""
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    with pytest.raises(ValueError, match="rerank_top_n"):
        run_retrieval_eval(
            samples, index, embedder, top_k=3,
            reranker=FakeReranker(), rerank_top_n=None,
        )


def test_runner_rerank_top_n_lt_top_k_raises():
    """rerank_top_n < top_k raises ValueError."""
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    with pytest.raises(ValueError, match="rerank_top_n"):
        run_retrieval_eval(
            samples, index, embedder, top_k=5,
            reranker=FakeReranker(), rerank_top_n=3,
        )


def test_runner_reranker_retrieved_at_most_top_k():
    """Post-rerank slices to top_k even when rerank_top_n is larger."""
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    samples = load_eval_samples(FIXTURE_QA)
    top_k = 2
    report = run_retrieval_eval(
        samples, index, embedder, top_k=top_k,
        reranker=FakeReranker(), rerank_top_n=10,
    )
    for r in report.per_question:
        assert len(r.retrieved_doc_ids) <= top_k


# ---------------------------------------------------------------------------
# P1.9-T04 — empty-sample validation regression (reviewer finding)
# ---------------------------------------------------------------------------

def test_runner_empty_samples_with_reranker_top_n_none_raises():
    """Validation fires before the empty-sample early return."""
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    with pytest.raises(ValueError, match="rerank_top_n"):
        run_retrieval_eval(
            [], index, embedder, top_k=3,
            reranker=FakeReranker(), rerank_top_n=None,
        )


def test_runner_empty_samples_with_rerank_top_n_lt_top_k_raises():
    """Validation fires before the empty-sample early return."""
    index = _build_index()
    embedder = FakeEmbedder(dim=8)
    with pytest.raises(ValueError, match="rerank_top_n"):
        run_retrieval_eval(
            [], index, embedder, top_k=5,
            reranker=FakeReranker(), rerank_top_n=3,
        )