"""Tests for T10 — index writer.

Verifies that write_index produces correctly structured manifest.json,
chunks.jsonl, and embeddings.npz.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from tiny_rag_lab.index_writer import SCHEMA_VERSION, write_index
from tiny_rag_lab.models import Chunk, Document


# ---------------------------------------------------------------------------
# Fixtures
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


def _make_chunk(chunk_id: str = "chunk0001", doc_id: str = "docs/a.md") -> Chunk:
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
def small_index(tmp_path):
    """Writes a minimal 2-doc, 3-chunk index; yields (index_dir, docs, chunks, embeddings)."""
    docs = [_make_doc("docs/a.md"), _make_doc("docs/b.md")]
    chunks = [
        _make_chunk("aaa0000000000001", "docs/a.md"),
        _make_chunk("aaa0000000000002", "docs/a.md"),
        _make_chunk("bbb0000000000001", "docs/b.md"),
    ]
    dim = 8
    rng = np.random.default_rng(42)
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
# File existence
# ---------------------------------------------------------------------------

def test_creates_index_dir(tmp_path):
    index_dir = tmp_path / "deep" / "nested" / "index"
    write_index(
        index_dir,
        docs=[_make_doc()],
        chunks=[_make_chunk()],
        embeddings=np.zeros((1, 8), dtype=np.float32),
        corpus_root=Path("/corpus"),
        embedding_backend="FakeEmbedder",
        embedding_model="fake",
        embedding_dim=8,
        chunk_size=800,
        chunk_overlap=120,
    )
    assert index_dir.is_dir()


def test_all_three_files_exist(small_index):
    index_dir, *_ = small_index
    assert (index_dir / "manifest.json").exists()
    assert (index_dir / "chunks.jsonl").exists()
    assert (index_dir / "embeddings.npz").exists()


# ---------------------------------------------------------------------------
# manifest.json
# ---------------------------------------------------------------------------

def test_manifest_schema_version(small_index):
    index_dir, *_ = small_index
    manifest = json.loads((index_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == SCHEMA_VERSION


def test_manifest_counts(small_index):
    index_dir, docs, chunks, _ = small_index
    manifest = json.loads((index_dir / "manifest.json").read_text())
    assert manifest["document_count"] == len(docs)
    assert manifest["chunk_count"] == len(chunks)


def test_manifest_chunking_params(small_index):
    index_dir, *_ = small_index
    manifest = json.loads((index_dir / "manifest.json").read_text())
    assert manifest["chunk_size"] == 800
    assert manifest["chunk_overlap"] == 120
    assert manifest["chunking_strategy"] == "fixed_character"
    assert manifest["chunking_params"] == {}


def test_manifest_round_trips_semantic_chunking_fields(tmp_path):
    index_dir = tmp_path / "index"
    write_index(
        index_dir,
        docs=[_make_doc()],
        chunks=[_make_chunk()],
        embeddings=np.zeros((1, 8), dtype=np.float32),
        corpus_root=Path("/corpus"),
        embedding_backend="FakeEmbedder",
        embedding_model="fake",
        embedding_dim=8,
        chunk_size=800,
        chunk_overlap=120,
        chunking_strategy="semantic",
        chunking_params={"similarity_threshold": 0.7},
    )
    manifest = json.loads((index_dir / "manifest.json").read_text())
    assert manifest["chunking_strategy"] == "semantic"
    assert manifest["chunking_params"] == {"similarity_threshold": 0.7}


def test_manifest_embedding_metadata(small_index):
    index_dir, *_ = small_index
    manifest = json.loads((index_dir / "manifest.json").read_text())
    assert manifest["embedding_backend"] == "FakeEmbedder"
    assert manifest["embedding_model"] == "fake"
    assert manifest["embedding_dim"] == 8


def test_manifest_corpus_root(small_index):
    index_dir, *_ = small_index
    manifest = json.loads((index_dir / "manifest.json").read_text())
    assert manifest["corpus_root"] == "/corpus"


def test_manifest_corpus_files_count(small_index):
    index_dir, docs, *_ = small_index
    manifest = json.loads((index_dir / "manifest.json").read_text())
    assert len(manifest["corpus_files"]) == len(docs)


def test_manifest_corpus_files_fields(small_index):
    index_dir, *_ = small_index
    manifest = json.loads((index_dir / "manifest.json").read_text())
    for entry in manifest["corpus_files"]:
        assert "doc_id" in entry
        assert "path" in entry
        assert "raw_hash" in entry


def test_manifest_corpus_files_hashes(small_index):
    index_dir, docs, *_ = small_index
    manifest = json.loads((index_dir / "manifest.json").read_text())
    recorded = {e["doc_id"]: e["raw_hash"] for e in manifest["corpus_files"]}
    for doc in docs:
        assert recorded[doc.doc_id] == doc.raw_hash


def test_manifest_created_at_is_iso8601(small_index):
    from datetime import datetime
    index_dir, *_ = small_index
    manifest = json.loads((index_dir / "manifest.json").read_text())
    # Must parse without error
    dt = datetime.fromisoformat(manifest["created_at"])
    assert dt.tzinfo is not None  # timezone-aware


# ---------------------------------------------------------------------------
# chunks.jsonl
# ---------------------------------------------------------------------------

def test_chunks_jsonl_line_count(small_index):
    index_dir, _, chunks, _ = small_index
    lines = (index_dir / "chunks.jsonl").read_text().strip().splitlines()
    assert len(lines) == len(chunks)


def test_chunks_jsonl_valid_json(small_index):
    index_dir, *_ = small_index
    for line in (index_dir / "chunks.jsonl").read_text().strip().splitlines():
        obj = json.loads(line)
        assert "chunk_id" in obj


def test_chunks_jsonl_roundtrip(small_index):
    index_dir, _, chunks, _ = small_index
    lines = (index_dir / "chunks.jsonl").read_text().strip().splitlines()
    for original, line in zip(chunks, lines):
        loaded = json.loads(line)
        assert loaded == dataclasses.asdict(original)


def test_chunks_jsonl_no_embedding_field(small_index):
    index_dir, *_ = small_index
    for line in (index_dir / "chunks.jsonl").read_text().strip().splitlines():
        obj = json.loads(line)
        assert "embedding" not in obj


# ---------------------------------------------------------------------------
# embeddings.npz
# ---------------------------------------------------------------------------

def test_embeddings_npz_keys(small_index):
    index_dir, *_ = small_index
    data = np.load(index_dir / "embeddings.npz", allow_pickle=True)
    assert "embeddings" in data
    assert "chunk_ids" in data


def test_embeddings_shape(small_index):
    index_dir, _, chunks, embeddings = small_index
    data = np.load(index_dir / "embeddings.npz", allow_pickle=True)
    assert data["embeddings"].shape == embeddings.shape


def test_embeddings_dtype(small_index):
    index_dir, *_ = small_index
    data = np.load(index_dir / "embeddings.npz", allow_pickle=True)
    assert data["embeddings"].dtype == np.float32


def test_embeddings_values_match(small_index):
    index_dir, _, _, embeddings = small_index
    data = np.load(index_dir / "embeddings.npz", allow_pickle=True)
    np.testing.assert_array_equal(data["embeddings"], embeddings)


def test_chunk_ids_order(small_index):
    index_dir, _, chunks, _ = small_index
    data = np.load(index_dir / "embeddings.npz", allow_pickle=True)
    assert list(data["chunk_ids"]) == [c.chunk_id for c in chunks]


def test_chunk_ids_count(small_index):
    index_dir, _, chunks, _ = small_index
    data = np.load(index_dir / "embeddings.npz", allow_pickle=True)
    assert len(data["chunk_ids"]) == len(chunks)


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------

def test_shape_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        write_index(
            tmp_path / "index",
            docs=[_make_doc()],
            chunks=[_make_chunk()],
            embeddings=np.zeros((3, 8), dtype=np.float32),  # wrong row count
            corpus_root=Path("/corpus"),
            embedding_backend="FakeEmbedder",
            embedding_model="fake",
            embedding_dim=8,
            chunk_size=800,
            chunk_overlap=120,
        )


# ---------------------------------------------------------------------------
# Empty corpus
# ---------------------------------------------------------------------------

def test_empty_corpus(tmp_path):
    index_dir = tmp_path / "index"
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
    manifest = json.loads((index_dir / "manifest.json").read_text())
    assert manifest["document_count"] == 0
    assert manifest["chunk_count"] == 0
    lines = (index_dir / "chunks.jsonl").read_text().strip()
    assert lines == ""
    data = np.load(index_dir / "embeddings.npz", allow_pickle=True)
    assert data["embeddings"].shape == (0, 8)
