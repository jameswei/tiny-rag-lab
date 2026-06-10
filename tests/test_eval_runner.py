"""Tests for tiny_rag_lab/eval.py — loader and runner.

T02: loader tests (marked with 'load' in name)
T04: runner tests                               — added in T04
"""
import json
from pathlib import Path

import pytest

from tiny_rag_lab.eval import EvalSample, load_eval_samples

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
