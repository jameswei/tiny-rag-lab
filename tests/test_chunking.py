"""Tests for T07 — character chunker.

The slice invariant is the central correctness property:
    document.normalized_text[chunk.char_start:chunk.char_end] == chunk.text
Every test that produces chunks verifies this invariant.
"""
import pytest

from tiny_rag_lab.chunking import (
    chunk_document,
    chunk_document_semantic,
    chunk_document_structural,
    chunk_documents,
)
from tiny_rag_lab.embeddings import FakeEmbedder
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


# ---------------------------------------------------------------------------
# Phase 2.2 (P2.2-T01): chunk_document_structural
# ---------------------------------------------------------------------------

_MARKDOWN_FIXTURE = (
    "# Title\n\n"
    "First paragraph here.\n\n"
    "Second paragraph with more words in it.\n\n"
    "- item one\n- item two\n"
)


def test_structural_slice_invariant_across_sizes():
    doc = _make_doc(_MARKDOWN_FIXTURE)
    for chunk_size in (20, 40, 60, 800):
        chunks = chunk_document_structural(doc, chunk_size=chunk_size, chunk_overlap=5)
        _assert_slice_invariant(doc, chunks)


def test_structural_heading_only_block_never_stands_alone():
    doc = _make_doc(_MARKDOWN_FIXTURE)
    chunks = chunk_document_structural(doc, chunk_size=800, chunk_overlap=5)
    # The heading and its body must be packed into the same chunk, not split
    # into a heading-only chunk followed by a separate body chunk.
    assert any("# Title" in c.text and "First paragraph" in c.text for c in chunks)


def test_structural_consecutive_headings_never_form_a_heading_only_chunk():
    # Regression: "# H1" merging with the immediately following "## H2"
    # block must not stop there if "## H2" is itself heading-only too — the
    # whole run of consecutive headings must merge through to the next body
    # block, not surface as a heading-only chunk on its own.
    text = "# H1\n\n## H2\n\nBody paragraph with enough words."
    doc = _make_doc(text)
    chunks = chunk_document_structural(doc, chunk_size=20, chunk_overlap=3)
    _assert_slice_invariant(doc, chunks)
    assert not any(
        c.text.strip() in ("# H1", "## H2", "# H1\n\n## H2") for c in chunks
    )


def test_structural_trailing_consecutive_headings_with_no_body():
    # Edge case: a run of consecutive headings at the very end of a
    # document has nothing to merge into. Must not crash and must still
    # satisfy the slice invariant.
    doc = _make_doc("Body.\n\n# H1\n\n## H2")
    chunks = chunk_document_structural(doc, chunk_size=10, chunk_overlap=2)
    _assert_slice_invariant(doc, chunks)


def test_structural_heading_with_no_body_stays_standalone():
    # Edge case: a heading at the very end of a document has nothing to
    # merge with at block-split time. Use a chunk_size tight enough that
    # packing can't combine it with the preceding block either, so it
    # surfaces as its own chunk rather than being silently absorbed.
    doc = _make_doc("Intro paragraph.\n\n# Trailing Heading")
    chunks = chunk_document_structural(doc, chunk_size=20, chunk_overlap=5)
    _assert_slice_invariant(doc, chunks)
    assert any(c.text.strip() == "# Trailing Heading" for c in chunks)


def test_structural_oversized_block_packs_by_sentence_not_character_window():
    # A single block (no blank line inside) made of several short sentences
    # that together exceed chunk_size. Tier 2 (sentence packing) must produce
    # non-overlapping chunks aligned to sentence boundaries — not tier 3
    # (character windowing), which would overlap and ignore sentence breaks.
    text = (
        "Sentence one is here. Sentence two is here. "
        "Sentence three is here. Sentence four is here."
    )
    doc = _make_doc(text)
    chunks = chunk_document_structural(doc, chunk_size=50, chunk_overlap=5)
    _assert_slice_invariant(doc, chunks)
    assert len(chunks) > 1
    # Tier 1/2 chunks are exactly contiguous (no overlap).
    for i in range(len(chunks) - 1):
        assert chunks[i].char_end == chunks[i + 1].char_start
    # Every chunk boundary falls on a sentence boundary (ends right after
    # ". "), not mid-sentence.
    for c in chunks[:-1]:
        assert c.text.endswith(". ") or c.text.endswith(".")


def test_structural_oversized_single_sentence_falls_back_to_character_window():
    # One run-on sentence with no internal punctuation, longer than
    # chunk_size. Only this case may use chunk_overlap.
    text = "word " * 40  # 200 chars, one giant "sentence" (no terminal punctuation)
    doc = _make_doc(text.strip())
    chunks = chunk_document_structural(doc, chunk_size=50, chunk_overlap=10)
    _assert_slice_invariant(doc, chunks)
    assert len(chunks) > 1
    # The character-window fallback overlaps consecutive chunks.
    shared = chunks[0].text[-10:]
    next_start = chunks[1].text[:10]
    assert shared == next_start


def test_structural_metadata_and_chunk_id_match_chunk_document_contract():
    doc = _make_doc(_MARKDOWN_FIXTURE, "docs/struct.md")
    chunks = chunk_document_structural(doc, chunk_size=800, chunk_overlap=5)
    for c in chunks:
        assert c.doc_id == "docs/struct.md"
        assert "title" in c.metadata
        assert "path" in c.metadata
        assert "format" in c.metadata
        assert "raw_hash" in c.metadata
        assert len(c.chunk_id) == 16
        assert all(ch in "0123456789abcdef" for ch in c.chunk_id)


def test_structural_empty_text_gives_no_chunks():
    doc = _make_doc("")
    assert chunk_document_structural(doc) == []


def test_structural_whitespace_only_text_gives_no_chunks():
    doc = _make_doc("   \n\n   \n")
    assert chunk_document_structural(doc) == []


def test_structural_validation_errors_match_chunk_document():
    doc = _make_doc("text")
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_document_structural(doc, chunk_size=0)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document_structural(doc, chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document_structural(doc, chunk_size=10, chunk_overlap=-1)


def test_structural_large_chunk_size_yields_single_chunk():
    doc = _make_doc(_MARKDOWN_FIXTURE)
    chunks = chunk_document_structural(doc, chunk_size=800, chunk_overlap=5)
    assert len(chunks) == 1
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(_MARKDOWN_FIXTURE)


# ---------------------------------------------------------------------------
# Phase 2.2 (P2.2-T02): chunk_document_semantic
# ---------------------------------------------------------------------------

_SENTENCES_FIXTURE = (
    "Sentence one is here. Sentence two is here. "
    "Sentence three is here. Sentence four is here."
)


def test_semantic_slice_invariant():
    doc = _make_doc(_SENTENCES_FIXTURE)
    embedder = FakeEmbedder(dim=8)
    chunks = chunk_document_semantic(doc, embedder, chunk_size=50, chunk_overlap=5)
    _assert_slice_invariant(doc, chunks)


def test_semantic_low_threshold_driven_purely_by_chunk_size():
    # similarity_threshold below any achievable cosine value (>= -1.0) means
    # the topic-shift condition can never fire; splitting is driven only by
    # chunk_size, identical to greedy size-based packing.
    doc = _make_doc(_SENTENCES_FIXTURE)
    embedder = FakeEmbedder(dim=8)
    chunks = chunk_document_semantic(
        doc, embedder, chunk_size=50, chunk_overlap=5, similarity_threshold=-2.0
    )
    _assert_slice_invariant(doc, chunks)
    assert len(chunks) == 2
    for i in range(len(chunks) - 1):
        assert chunks[i].char_end == chunks[i + 1].char_start


def test_semantic_high_threshold_splits_every_sentence():
    # similarity_threshold above any achievable cosine value (<= 1.0) forces
    # a topic shift on every sentence pair, so each sentence becomes its own
    # chunk when chunk_size is not the binding constraint.
    doc = _make_doc(_SENTENCES_FIXTURE)
    embedder = FakeEmbedder(dim=8)
    chunks = chunk_document_semantic(
        doc, embedder, chunk_size=800, chunk_overlap=5, similarity_threshold=2.0
    )
    _assert_slice_invariant(doc, chunks)
    assert len(chunks) == 4


def test_semantic_oversized_single_sentence_falls_back_to_character_window():
    text = ("word " * 40).strip()  # one run-on "sentence", no punctuation
    doc = _make_doc(text)
    embedder = FakeEmbedder(dim=8)
    chunks = chunk_document_semantic(doc, embedder, chunk_size=50, chunk_overlap=10)
    _assert_slice_invariant(doc, chunks)
    assert len(chunks) > 1
    # The character-window fallback overlaps consecutive chunks.
    shared = chunks[0].text[-10:]
    next_start = chunks[1].text[:10]
    assert shared == next_start


def test_semantic_embedder_called_exactly_once_per_document():
    # Review-sensitive: embedding must be one batch call for all sentences,
    # not one call per sentence, or indexing becomes far slower than
    # documented.
    calls = []

    class CountingEmbedder(FakeEmbedder):
        def embed(self, texts):
            calls.append(len(texts))
            return super().embed(texts)

    doc = _make_doc(_SENTENCES_FIXTURE)
    chunk_document_semantic(doc, CountingEmbedder(dim=8), chunk_size=50, chunk_overlap=5)
    assert calls == [4]  # one call, batch of all 4 sentences


def test_semantic_deterministic_with_same_fake_embedder_seed():
    doc = _make_doc(_SENTENCES_FIXTURE)
    chunks_a = chunk_document_semantic(doc, FakeEmbedder(dim=8), chunk_size=50, chunk_overlap=5)
    chunks_b = chunk_document_semantic(doc, FakeEmbedder(dim=8), chunk_size=50, chunk_overlap=5)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]


def test_semantic_metadata_and_chunk_id_match_chunk_document_contract():
    doc = _make_doc(_SENTENCES_FIXTURE, "docs/semantic.md")
    embedder = FakeEmbedder(dim=8)
    chunks = chunk_document_semantic(doc, embedder, chunk_size=50, chunk_overlap=5)
    for c in chunks:
        assert c.doc_id == "docs/semantic.md"
        assert "title" in c.metadata
        assert "path" in c.metadata
        assert "format" in c.metadata
        assert "raw_hash" in c.metadata
        assert len(c.chunk_id) == 16
        assert all(ch in "0123456789abcdef" for ch in c.chunk_id)


def test_semantic_empty_text_gives_no_chunks():
    doc = _make_doc("")
    assert chunk_document_semantic(doc, FakeEmbedder(dim=8)) == []


def test_semantic_whitespace_only_text_gives_no_chunks():
    doc = _make_doc("   \n\n   \n")
    assert chunk_document_semantic(doc, FakeEmbedder(dim=8)) == []


def test_semantic_validation_errors_match_chunk_document():
    doc = _make_doc("text")
    embedder = FakeEmbedder(dim=8)
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_document_semantic(doc, embedder, chunk_size=0)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document_semantic(doc, embedder, chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document_semantic(doc, embedder, chunk_size=10, chunk_overlap=-1)
