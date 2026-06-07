import dataclasses
import json

from tiny_rag_lab.models import (
    Chunk,
    Document,
    RagTrace,
    RetrievalResult,
    make_chunk_id,
)

# ---------------------------------------------------------------------------
# make_chunk_id
# ---------------------------------------------------------------------------

def test_chunk_id_is_16_hex_chars():
    cid = make_chunk_id("docs/example.md", 0, "hello world")
    assert len(cid) == 16
    assert all(c in "0123456789abcdef" for c in cid)


def test_chunk_id_is_deterministic():
    a = make_chunk_id("docs/example.md", 0, "hello world")
    b = make_chunk_id("docs/example.md", 0, "hello world")
    assert a == b


def test_chunk_id_differs_by_offset():
    a = make_chunk_id("docs/example.md", 0, "same text")
    b = make_chunk_id("docs/example.md", 10, "same text")
    assert a != b


def test_chunk_id_differs_by_doc():
    a = make_chunk_id("docs/a.md", 0, "same text")
    b = make_chunk_id("docs/b.md", 0, "same text")
    assert a != b


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

def _sample_document() -> Document:
    raw = "# Hello\n\nWorld."
    return Document(
        doc_id="docs/example.md",
        path="/corpus/docs/example.md",
        title="Hello",
        format="markdown",
        raw_text=raw,
        normalized_text=raw,
        raw_hash="abc123",
    )


def test_document_fields():
    doc = _sample_document()
    assert doc.doc_id == "docs/example.md"
    assert doc.format == "markdown"


def test_document_serializes():
    doc = _sample_document()
    d = dataclasses.asdict(doc)
    assert json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# Chunk — slice invariant
# ---------------------------------------------------------------------------

def _sample_chunk(normalized_text: str, start: int, end: int) -> Chunk:
    text = normalized_text[start:end]
    return Chunk(
        chunk_id=make_chunk_id("docs/example.md", start, text),
        doc_id="docs/example.md",
        text=text,
        char_start=start,
        char_end=end,
        metadata={"title": "Hello", "path": "/corpus/docs/example.md",
                   "format": "markdown", "raw_hash": "abc123"},
    )


def test_chunk_slice_invariant():
    normalized = "Hello world, this is a test document with some content."
    chunk = _sample_chunk(normalized, 6, 20)
    assert normalized[chunk.char_start:chunk.char_end] == chunk.text


def test_chunk_id_stable():
    normalized = "Hello world, this is a test document."
    c1 = _sample_chunk(normalized, 0, 10)
    c2 = _sample_chunk(normalized, 0, 10)
    assert c1.chunk_id == c2.chunk_id


def test_chunk_serializes():
    normalized = "Hello world."
    chunk = _sample_chunk(normalized, 0, 5)
    d = dataclasses.asdict(chunk)
    assert json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------

def test_retrieval_result_fields():
    normalized = "Some text for retrieval."
    chunk = _sample_chunk(normalized, 0, 9)
    result = RetrievalResult(chunk=chunk, score=0.87, rank=1)
    assert result.rank == 1
    assert 0.0 <= result.score <= 1.0


def test_retrieval_result_serializes():
    normalized = "Some text."
    chunk = _sample_chunk(normalized, 0, 4)
    result = RetrievalResult(chunk=chunk, score=0.5, rank=1)
    d = dataclasses.asdict(result)
    assert json.dumps(d)


# ---------------------------------------------------------------------------
# RagTrace
# ---------------------------------------------------------------------------

def test_rag_trace_serializes():
    normalized = "Trace text."
    chunk = _sample_chunk(normalized, 0, 5)
    result = RetrievalResult(chunk=chunk, score=0.9, rank=1)
    trace = RagTrace(
        query="What is this?",
        retrieved_chunks=[result],
        prompt="Answer from context...",
        answer="It is a test.",
        citations=["[Source: abc123456789abcd]"],
        latency_by_stage={"embed": 0.01, "retrieve": 0.005, "generate": 0.8},
    )
    d = dataclasses.asdict(trace)
    assert json.dumps(d)
    assert d["query"] == "What is this?"
    assert len(d["retrieved_chunks"]) == 1
