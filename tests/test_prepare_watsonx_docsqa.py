"""Tests for scripts/prepare_watsonx_docsqa.py.

All tests use synthetic rows matching the real ibm-research/watsonxDocsQA
schema. No network access, no HuggingFace downloads.

Real corpus split fields: doc_id, title, md_document, document, url
Real QA split fields:     question_id, question, correct_answer,
                          correct_answer_document_ids (string or list[str])
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from prepare_watsonx_docsqa import (
    DATASET_NAME,
    build_manifest,
    doc_to_markdown,
    extract_documents,
    extract_qa_pairs,
    write_corpus,
    write_manifest,
    write_qa_jsonl,
)

# ---------------------------------------------------------------------------
# Fixtures — real schema
# ---------------------------------------------------------------------------

CORPUS_ROWS = [
    {
        "doc_id": "doc-001",
        "title": "What is watsonx.ai",
        "md_document": "watsonx.ai is IBM's AI and data platform.",
        "document": "watsonx.ai is IBM's AI and data platform (plain).",
        "url": "https://ibm.com/watsonx/ai",
    },
    {
        "doc_id": "doc-001",   # duplicate — should be deduplicated
        "title": "What is watsonx.ai",
        "md_document": "watsonx.ai is IBM's AI and data platform.",
        "document": "duplicate",
        "url": "https://ibm.com/watsonx/ai",
    },
    {
        "doc_id": "doc-002",
        "title": "watsonx.data overview",
        "md_document": "watsonx.data is a data store for AI workloads.",
        "document": "plain fallback",
        "url": "https://ibm.com/watsonx/data",
    },
]

QA_ROWS = [
    {
        "question_id": "q-001",
        "question": "What is watsonx.ai?",
        "correct_answer": "IBM's AI and data platform.",
        "correct_answer_document_ids": "doc-001",
    },
    {
        "question_id": "q-002",
        "question": "Who makes watsonx.ai?",
        "correct_answer": "IBM.",
        "correct_answer_document_ids": "doc-001",
    },
    {
        "question_id": "q-003",
        "question": "What is watsonx.data?",
        "correct_answer": "A data store for AI workloads.",
        "correct_answer_document_ids": "doc-002",
    },
]

# ---------------------------------------------------------------------------
# DATASET_NAME constant
# ---------------------------------------------------------------------------

def test_default_dataset_name():
    assert DATASET_NAME == "ibm-research/watsonxDocsQA"


# ---------------------------------------------------------------------------
# doc_to_markdown
# ---------------------------------------------------------------------------

def test_doc_to_markdown_has_h1():
    md = doc_to_markdown("My Title", "Some content.")
    assert md.startswith("# My Title\n")


def test_doc_to_markdown_contains_text():
    md = doc_to_markdown("Title", "Body content here.")
    assert "Body content here." in md


def test_doc_to_markdown_empty_title_fallback():
    md = doc_to_markdown("", "Content.")
    assert "# Untitled" in md


# ---------------------------------------------------------------------------
# extract_documents
# ---------------------------------------------------------------------------

def test_extract_documents_deduplicates():
    docs, id_map = extract_documents(CORPUS_ROWS)
    assert len(docs) == 2  # doc-001 appears twice, stored once


def test_extract_documents_returns_id_map():
    docs, id_map = extract_documents(CORPUS_ROWS)
    assert set(id_map.keys()) == {"doc-001", "doc-002"}


def test_extract_documents_prepared_id_format():
    docs, id_map = extract_documents(CORPUS_ROWS)
    assert id_map["doc-001"].startswith("docs/")
    assert id_map["doc-001"].endswith(".md")


def test_extract_documents_prefers_md_document():
    # md_document should be used over document when both present
    docs, _ = extract_documents(CORPUS_ROWS)
    assert "AI and data platform." in docs["doc-001"]["text"]
    assert "plain" not in docs["doc-001"]["text"]


def test_extract_documents_preserves_title():
    docs, _ = extract_documents(CORPUS_ROWS)
    assert docs["doc-001"]["title"] == "What is watsonx.ai"


def test_extract_documents_skips_non_corpus_rows():
    # QA rows have no md_document or document field — should be skipped
    docs, id_map = extract_documents(QA_ROWS)
    assert len(docs) == 0
    assert len(id_map) == 0


# ---------------------------------------------------------------------------
# extract_qa_pairs — id_map linkage (the critical fix)
# ---------------------------------------------------------------------------

def test_extract_qa_pairs_count():
    _, id_map = extract_documents(CORPUS_ROWS)
    pairs = extract_qa_pairs(QA_ROWS, id_map)
    assert len(pairs) == 3


def test_extract_qa_pairs_gold_doc_ids_translated():
    _, id_map = extract_documents(CORPUS_ROWS)
    pairs = extract_qa_pairs(QA_ROWS, id_map)
    # gold_doc_ids must use prepared_id (same as manifest doc_id), not original dataset id
    assert pairs[0]["gold_doc_ids"] == [id_map["doc-001"]]
    assert pairs[2]["gold_doc_ids"] == [id_map["doc-002"]]


def test_extract_qa_pairs_supports_legacy_gold_field():
    _, id_map = extract_documents(CORPUS_ROWS)
    rows = [
        {
            "question_id": "q-old",
            "question": "Legacy schema?",
            "correct_answer": "Yes.",
            "ground_truths_contexts_ids": ["doc-001"],
        }
    ]
    pairs = extract_qa_pairs(rows, id_map)
    assert pairs[0]["gold_doc_ids"] == [id_map["doc-001"]]


def test_extract_qa_pairs_splits_comma_separated_gold_ids():
    _, id_map = extract_documents(CORPUS_ROWS)
    rows = [
        {
            "question_id": "q-multi",
            "question": "Which docs?",
            "correct_answer": "Both.",
            "correct_answer_document_ids": "doc-001, doc-002",
        }
    ]
    pairs = extract_qa_pairs(rows, id_map)
    assert pairs[0]["gold_doc_ids"] == [id_map["doc-001"], id_map["doc-002"]]


def test_extract_qa_pairs_handles_scalar_gold_id():
    _, id_map = extract_documents(CORPUS_ROWS)
    rows = [
        {
            "question_id": "q-scalar",
            "question": "Scalar schema?",
            "correct_answer": "Yes.",
            "correct_answer_document_ids": 123,
        }
    ]
    pairs = extract_qa_pairs(rows, id_map)
    assert pairs[0]["gold_doc_ids"] == ["123"]


def test_extract_qa_pairs_gold_ids_match_manifest():
    """Core linkage test: qa.jsonl gold_doc_ids must equal manifest doc_ids."""
    docs, id_map = extract_documents(CORPUS_ROWS)
    pairs = extract_qa_pairs(QA_ROWS, id_map)
    manifest_doc_ids = {info["prepared_id"] for info in docs.values()}
    for pair in pairs:
        for gid in pair["gold_doc_ids"]:
            assert gid in manifest_doc_ids, (
                f"gold_doc_id {gid!r} not in manifest; linkage is broken"
            )


def test_extract_qa_pairs_answer_field():
    _, id_map = extract_documents(CORPUS_ROWS)
    pairs = extract_qa_pairs(QA_ROWS, id_map)
    assert pairs[0]["answer"] == "IBM's AI and data platform."


def test_extract_qa_pairs_question_id():
    _, id_map = extract_documents(CORPUS_ROWS)
    pairs = extract_qa_pairs(QA_ROWS, id_map)
    assert pairs[0]["question_id"] == "q-001"


def test_extract_qa_pairs_skips_missing_question():
    _, id_map = extract_documents(CORPUS_ROWS)
    rows = [{"document_id": "x", "md_document": "text"}]  # no question
    pairs = extract_qa_pairs(rows, id_map)
    assert pairs == []


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def test_write_corpus_creates_files(tmp_path):
    docs, _ = extract_documents(CORPUS_ROWS)
    records = write_corpus(tmp_path, docs)
    assert len(records) == 2
    assert len(list((tmp_path / "docs").glob("*.md"))) == 2


def test_write_corpus_record_doc_id_matches_prepared_id(tmp_path):
    docs, id_map = extract_documents(CORPUS_ROWS)
    records = write_corpus(tmp_path, docs)
    record_doc_ids = {r["doc_id"] for r in records}
    assert record_doc_ids == set(id_map.values())


def test_write_corpus_includes_original_doc_id(tmp_path):
    docs, _ = extract_documents(CORPUS_ROWS)
    records = write_corpus(tmp_path, docs)
    original_ids = {r["original_doc_id"] for r in records}
    assert "doc-001" in original_ids
    assert "doc-002" in original_ids


def test_write_manifest(tmp_path):
    docs, _ = extract_documents(CORPUS_ROWS)
    records = write_corpus(tmp_path, docs)
    manifest = build_manifest("test/dataset", records)
    write_manifest(tmp_path, manifest)
    data = json.loads((tmp_path / "dataset-manifest.json").read_text())
    assert data["dataset_name"] == "test/dataset"
    assert data["document_count"] == 2


def test_write_qa_jsonl(tmp_path):
    docs, id_map = extract_documents(CORPUS_ROWS)
    pairs = extract_qa_pairs(QA_ROWS, id_map)
    write_qa_jsonl(tmp_path, pairs)
    lines = (tmp_path / "qa.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert "question" in first
    assert "answer" in first
    assert "gold_doc_ids" in first
    assert isinstance(first["gold_doc_ids"], list)
