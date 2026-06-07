"""Tests for T07 — character chunker.

The slice invariant is the central correctness property:
    document.normalized_text[chunk.char_start:chunk.char_end] == chunk.text
Every test that produces chunks verifies this invariant.
"""
import pytest

from tiny_rag_lab.chunking import chunk_document, chunk_documents
from tiny_rag_lab.models import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(text: str, doc_id: str = "docs/test.md") -> Document:
    return Document(
        doc_id=doc_id,
        path=f"/corpus/{doc_id}",
        title="Test",
        format="markdown",
        raw_text=text,
        normalized_text=text,
        raw_hash="deadbeef",
    )


def _assert_slice_invariant(doc: Document, chunks: list) -> None:
    for chunk in chunks:
        assert doc.normalized_text[chunk.char_start:chunk.char_end] == chunk.text, (
            f"Slice invariant violated for chunk {chunk.chunk_id}: "
            f"text[{chunk.char_start}:{chunk.char_end}] != chunk.text"
        )


# ---------------------------------------------------------------------------
# Slice invariant
# ---------------------------------------------------------------------------

def test_slice_invariant_single_chunk():
    doc = _make_doc("Short text that fits in one chunk.")
    chunks = chunk_document(doc, chunk_size=800)
    _assert_slice_invariant(doc, chunks)


def test_slice_invariant_multiple_chunks():
    doc = _make_doc("A" * 200, "docs/a.md")
    chunks = chunk_document(doc, chunk_size=80, chunk_overlap=20)
    assert len(chunks) > 1
    _assert_slice_invariant(doc, chunks)


def test_slice_invariant_with_real_text():
    text = "# Title\n\nFirst paragraph.\n\nSecond paragraph with more content.\n\nThird."
    doc = _make_doc(text)
    chunks = chunk_document(doc, chunk_size=30, chunk_overlap=5)
    _assert_slice_invariant(doc, chunks)


def test_slice_invariant_final_partial_chunk():
    # Text length not a multiple of (chunk_size - chunk_overlap)
    doc = _make_doc("A" * 95)
    chunks = chunk_document(doc, chunk_size=40, chunk_overlap=10)
    _assert_slice_invariant(doc, chunks)
    # Last chunk must end at exactly len(text)
    assert chunks[-1].char_end == 95


# ---------------------------------------------------------------------------
# Chunk count and overlap
# ---------------------------------------------------------------------------

def test_text_shorter_than_chunk_size_gives_one_chunk():
    doc = _make_doc("Hello world.")
    chunks = chunk_document(doc, chunk_size=800)
    assert len(chunks) == 1


# Boundary cases for redundant-tail-chunk regression (default chunk_size=800, overlap=120)
def test_no_redundant_tail_chunk_length_799():
    # text shorter than chunk_size: single chunk, no overlap tail
    doc = _make_doc("A" * 799)
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].char_start == 0 and chunks[0].char_end == 799


def test_no_redundant_tail_chunk_length_800():
    # text exactly chunk_size: single chunk, no overlap tail
    doc = _make_doc("A" * 800)
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].char_end == 800


def test_no_redundant_tail_chunk_length_1480():
    # 1480 = 800 + 1*(800-120): two chunks, second reaches end, no third tail
    doc = _make_doc("A" * 1480)
    chunks = chunk_document(doc)
    assert len(chunks) == 2
    assert chunks[0].char_start == 0 and chunks[0].char_end == 800
    assert chunks[1].char_start == 680 and chunks[1].char_end == 1480


def test_text_exactly_chunk_size_gives_one_chunk():
    doc = _make_doc("A" * 800)
    chunks = chunk_document(doc, chunk_size=800, chunk_overlap=0)
    assert len(chunks) == 1


def test_overlap_produces_shared_characters():
    text = "A" * 100
    doc = _make_doc(text)
    chunks = chunk_document(doc, chunk_size=40, chunk_overlap=10)
    # Consecutive chunks share chunk_overlap characters
    for i in range(len(chunks) - 1):
        shared = chunks[i].text[-(10):]
        next_start = chunks[i + 1].text[:10]
        assert shared == next_start


def test_no_gap_between_consecutive_chunks():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 4
    doc = _make_doc(text)
    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=3)
    for i in range(len(chunks) - 1):
        # The end of chunk i and start of chunk i+1 must not have a gap
        assert chunks[i + 1].char_start < chunks[i].char_end


# ---------------------------------------------------------------------------
# Empty / whitespace filtering
# ---------------------------------------------------------------------------

def test_empty_text_gives_no_chunks():
    doc = _make_doc("")
    assert chunk_document(doc) == []


def test_whitespace_only_text_gives_no_chunks():
    doc = _make_doc("   \n\n   \n")
    assert chunk_document(doc) == []


# ---------------------------------------------------------------------------
# Stable chunk IDs
# ---------------------------------------------------------------------------

def test_chunk_ids_stable_across_runs():
    doc = _make_doc("Stable text for ID testing.", "docs/stable.md")
    chunks_a = chunk_document(doc, chunk_size=50, chunk_overlap=5)
    chunks_b = chunk_document(doc, chunk_size=50, chunk_overlap=5)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]


def test_chunk_ids_differ_by_doc():
    text = "Same text in different documents."
    doc_a = _make_doc(text, "docs/a.md")
    doc_b = _make_doc(text, "docs/b.md")
    ids_a = {c.chunk_id for c in chunk_document(doc_a)}
    ids_b = {c.chunk_id for c in chunk_document(doc_b)}
    assert ids_a.isdisjoint(ids_b)


def test_chunk_ids_are_16_hex_chars():
    doc = _make_doc("Some content for ID length check.")
    for chunk in chunk_document(doc):
        assert len(chunk.chunk_id) == 16
        assert all(c in "0123456789abcdef" for c in chunk.chunk_id)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_chunk_metadata_has_required_fields():
    doc = _make_doc("Content for metadata test.")
    for chunk in chunk_document(doc):
        assert "title" in chunk.metadata
        assert "path" in chunk.metadata
        assert "format" in chunk.metadata
        assert "raw_hash" in chunk.metadata


def test_chunk_doc_id_matches_document():
    doc = _make_doc("Content.", "docs/myfile.md")
    for chunk in chunk_document(doc):
        assert chunk.doc_id == "docs/myfile.md"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_chunk_size_zero_raises():
    doc = _make_doc("text")
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_document(doc, chunk_size=0)


def test_overlap_gte_chunk_size_raises():
    doc = _make_doc("text")
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document(doc, chunk_size=10, chunk_overlap=10)


def test_negative_overlap_raises():
    doc = _make_doc("text")
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document(doc, chunk_size=10, chunk_overlap=-1)


# ---------------------------------------------------------------------------
# chunk_documents
# ---------------------------------------------------------------------------

def test_chunk_documents_combines_all():
    docs = [_make_doc("Doc one content.", f"docs/doc{i}.md") for i in range(3)]
    chunks = chunk_documents(docs, chunk_size=800)
    assert len(chunks) == 3
    doc_ids = [c.doc_id for c in chunks]
    assert doc_ids == ["docs/doc0.md", "docs/doc1.md", "docs/doc2.md"]
