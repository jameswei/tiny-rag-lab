"""Tests for T05 (document loader) and T06 (text normalization)."""

import hashlib
from pathlib import Path

import pytest

from tiny_rag_lab.documents import load_document, load_documents, normalize_text

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"


# ---------------------------------------------------------------------------
# T06 — normalize_text
# ---------------------------------------------------------------------------

def test_crlf_normalized_to_lf():
    assert normalize_text("line1\r\nline2") == "line1\nline2"


def test_cr_normalized_to_lf():
    assert normalize_text("line1\rline2") == "line1\nline2"


def test_trailing_whitespace_stripped():
    assert normalize_text("hello   \nworld  ") == "hello\nworld"


def test_two_blank_lines_preserved():
    text = "a\n\n\nb"
    result = normalize_text(text)
    assert result == "a\n\n\nb"


def test_more_than_two_blank_lines_collapsed():
    # 4 blank lines between a and b → exactly 2
    text = "a\n\n\n\n\nb"
    result = normalize_text(text)
    assert result == "a\n\n\nb"


def test_headings_preserved():
    text = "# Title\n\n## Section\n\nContent."
    assert normalize_text(text) == text


def test_already_normalized_is_unchanged():
    text = "# Title\n\nParagraph one.\n\nParagraph two.\n"
    assert normalize_text(text) == text


def test_empty_string():
    assert normalize_text("") == ""


def test_mixed_line_endings():
    result = normalize_text("a\r\nb\rc\nd")
    assert result == "a\nb\nc\nd"


# ---------------------------------------------------------------------------
# T05 — load_document
# ---------------------------------------------------------------------------

def test_load_md_with_h1_title():
    doc = load_document(FIXTURES / "with_h1.md", FIXTURES)
    assert doc.title == "Sample Document Title"


def test_load_md_without_h1_uses_filename():
    doc = load_document(FIXTURES / "no_h1.md", FIXTURES)
    assert doc.title == "no_h1"


def test_load_txt_uses_filename():
    doc = load_document(FIXTURES / "plain.txt", FIXTURES)
    assert doc.title == "plain"


def test_load_md_format():
    doc = load_document(FIXTURES / "with_h1.md", FIXTURES)
    assert doc.format == "markdown"


def test_load_txt_format():
    doc = load_document(FIXTURES / "plain.txt", FIXTURES)
    assert doc.format == "text"


def test_load_doc_id_is_corpus_relative_posix():
    doc = load_document(FIXTURES / "with_h1.md", FIXTURES)
    assert doc.doc_id == "with_h1.md"


def test_load_nested_doc_id():
    doc = load_document(FIXTURES / "subdir" / "nested.md", FIXTURES)
    assert doc.doc_id == "subdir/nested.md"


def test_load_raw_hash_is_sha256():
    doc = load_document(FIXTURES / "with_h1.md", FIXTURES)
    raw = (FIXTURES / "with_h1.md").read_text(encoding="utf-8")
    expected = hashlib.sha256(raw.encode()).hexdigest()
    assert doc.raw_hash == expected


def test_load_normalized_text_populated():
    doc = load_document(FIXTURES / "with_h1.md", FIXTURES)
    assert doc.normalized_text != ""


def test_load_raw_text_unchanged():
    doc = load_document(FIXTURES / "with_h1.md", FIXTURES)
    raw = (FIXTURES / "with_h1.md").read_text(encoding="utf-8")
    assert doc.raw_text == raw


def test_load_path_is_absolute_string():
    doc = load_document(FIXTURES / "with_h1.md", FIXTURES)
    assert Path(doc.path).is_absolute()


# ---------------------------------------------------------------------------
# T05 — load_documents
# ---------------------------------------------------------------------------

def test_load_documents_finds_all_files():
    docs = load_documents(FIXTURES)
    names = {Path(d.path).name for d in docs}
    assert "with_h1.md" in names
    assert "no_h1.md" in names
    assert "plain.txt" in names
    assert "nested.md" in names


def test_load_documents_excludes_gitkeep():
    docs = load_documents(FIXTURES)
    names = {Path(d.path).name for d in docs}
    assert ".gitkeep" not in names


def test_load_documents_sorted_order():
    docs = load_documents(FIXTURES)
    paths = [d.doc_id for d in docs]
    assert paths == sorted(paths)


def test_load_documents_all_have_normalized_text():
    docs = load_documents(FIXTURES)
    for doc in docs:
        assert isinstance(doc.normalized_text, str)
