"""Tests for T11 — index loader.

Uses write_index to build real on-disk fixtures so the loader is tested
against the exact bytes the writer produces.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from tiny_rag_lab.index_loader import LoadedIndex, load_index
from tiny_rag_lab.index_writer import write_index
from tiny_rag_lab.models import Chunk, Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(doc_id: str = "docs/a.md") -> Document:
    return Document(
        doc_id=doc_id,
        path=f"/corpus/{doc_id}",
        title="Test Doc",
        format="markdown",
        raw_text="Hello world.",
        normalized_text="Hello world.",
        raw_hash="abc123",
    )


def _make_chunk(chunk_id: str, doc_id: str = "docs/a.md") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text="Hello world.",
        char_start=0,
        char_end=12,
        metadata={"title": "Test Doc", "path": f"/corpus/{doc_id}",
                   "format": "markdown", "raw_hash": "abc123"},
    )


@pytest.fixture()
def written_index(tmp_path):
    """Writes a 2-doc, 3-chunk index and returns (index_dir, docs, chunks, embeddings)."""
    docs = [_make_doc("docs/a.md"), _make_doc("docs/b.md")]
    chunks = [
        _make_chunk("aaa0000000000001", "docs/a.md"),
        _make_chunk("aaa0000000000002", "docs/a.md"),
        _make_chunk("bbb0000000000001", "docs/b.md"),
    ]
    dim = 8
    rng = np.random.default_rng(99)
    embeddings = rng.standard_normal((3, dim)).astype(np.float32)

    index_dir = tmp_path / "index"
    write_index(
        index_dir,
        docs=docs,
        chunks=chunks,
        embeddings=embeddings,
        corpus_root=Path("/corpus"),
        embedding_backend="FakeEmbedder",
        embedding_model="fake",
        embedding_dim=dim,
        chunk_size=800,
        chunk_overlap=120,
    )
    return index_dir, docs, chunks, embeddings


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_returns_loaded_index(written_index):
    index_dir, *_ = written_index
    result = load_index(index_dir)
    assert isinstance(result, LoadedIndex)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def test_manifest_is_dict(written_index):
    index_dir, *_ = written_index
    result = load_index(index_dir)
    assert isinstance(result.manifest, dict)


def test_manifest_document_count(written_index):
    index_dir, docs, *_ = written_index
    result = load_index(index_dir)
    assert result.manifest["document_count"] == len(docs)


def test_manifest_chunk_count(written_index):
    index_dir, _, chunks, _ = written_index
    result = load_index(index_dir)
    assert result.manifest["chunk_count"] == len(chunks)


def test_manifest_embedding_backend(written_index):
    index_dir, *_ = written_index
    result = load_index(index_dir)
    assert result.manifest["embedding_backend"] == "FakeEmbedder"


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------

def test_chunks_count(written_index):
    index_dir, _, chunks, _ = written_index
    result = load_index(index_dir)
    assert len(result.chunks) == len(chunks)


def test_chunks_roundtrip(written_index):
    index_dir, _, chunks, _ = written_index
    result = load_index(index_dir)
    for original, loaded in zip(chunks, result.chunks):
        assert dataclasses.asdict(loaded) == dataclasses.asdict(original)


def test_chunks_are_chunk_instances(written_index):
    index_dir, *_ = written_index
    result = load_index(index_dir)
    for chunk in result.chunks:
        assert isinstance(chunk, Chunk)


def test_chunk_ids_match_chunk_objects(written_index):
    index_dir, *_ = written_index
    result = load_index(index_dir)
    assert result.chunk_ids == [c.chunk_id for c in result.chunks]


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def test_embeddings_shape(written_index):
    index_dir, _, chunks, embeddings = written_index
    result = load_index(index_dir)
    assert result.embeddings.shape == embeddings.shape


def test_embeddings_dtype(written_index):
    index_dir, *_ = written_index
    result = load_index(index_dir)
    assert result.embeddings.dtype == np.float32


def test_embeddings_values(written_index):
    index_dir, _, _, embeddings = written_index
    result = load_index(index_dir)
    np.testing.assert_array_equal(result.embeddings, embeddings)


def test_embedding_row_order_matches_chunks(written_index):
    index_dir, _, chunks, _ = written_index
    result = load_index(index_dir)
    assert [c.chunk_id for c in result.chunks] == result.chunk_ids


# ---------------------------------------------------------------------------
# chunk_ids parallel array
# ---------------------------------------------------------------------------

def test_chunk_ids_list_length(written_index):
    index_dir, _, chunks, _ = written_index
    result = load_index(index_dir)
    assert len(result.chunk_ids) == len(chunks)


def test_chunk_ids_values(written_index):
    index_dir, _, chunks, _ = written_index
    result = load_index(index_dir)
    assert result.chunk_ids == [c.chunk_id for c in chunks]


# ---------------------------------------------------------------------------
# Missing file errors
# ---------------------------------------------------------------------------

def test_missing_manifest_raises(written_index):
    index_dir, *_ = written_index
    (index_dir / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_index(index_dir)


def test_missing_chunks_raises(written_index):
    index_dir, *_ = written_index
    (index_dir / "chunks.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        load_index(index_dir)


def test_missing_embeddings_raises(written_index):
    index_dir, *_ = written_index
    (index_dir / "embeddings.npz").unlink()
    with pytest.raises(FileNotFoundError):
        load_index(index_dir)


def test_nonexistent_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path / "no_such_dir")


# ---------------------------------------------------------------------------
# chunk_id mismatch validation
# ---------------------------------------------------------------------------

def test_chunk_id_mismatch_raises(written_index):
    index_dir, _, chunks, embeddings = written_index
    # Corrupt embeddings.npz with wrong chunk_ids
    bad_ids = np.array(["wrongid1", "wrongid2", "wrongid3"])
    np.savez(
        index_dir / "embeddings.npz",
        embeddings=embeddings,
        chunk_ids=bad_ids,
    )
    with pytest.raises(ValueError, match="chunk_ids mismatch"):
        load_index(index_dir)


# ---------------------------------------------------------------------------
# Empty index
# ---------------------------------------------------------------------------

def test_empty_index_roundtrip(tmp_path):
    index_dir = tmp_path / "empty"
    write_index(
        index_dir,
        docs=[],
        chunks=[],
        embeddings=np.empty((0, 8), dtype=np.float32),
        corpus_root=Path("/corpus"),
        embedding_backend="FakeEmbedder",
        embedding_model="fake",
        embedding_dim=8,
        chunk_size=800,
        chunk_overlap=120,
    )
    result = load_index(index_dir)
    assert result.chunks == []
    assert result.chunk_ids == []
    assert result.embeddings.shape == (0, 8)
