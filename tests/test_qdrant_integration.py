"""Optional local Qdrant smoke; run only when its Compose profile is active."""
from __future__ import annotations

import os

import numpy as np
import pytest

if not os.environ.get("QDRANT_URL"):
    pytest.skip("set QDRANT_URL after starting the optional Compose profile", allow_module_level=True)

from tiny_rag_lab.index_loader import load_index
from tiny_rag_lab.index_backend import NumpyIndexBackend
from tiny_rag_lab.index_writer import write_index
from tiny_rag_lab.models import Chunk, Document
from tiny_rag_lab.qdrant_backend import QdrantIndexBackend


def test_qdrant_profile_searches_same_local_chunk_artifact(tmp_path):
    doc = Document("alpha.md", "/corpus/alpha.md", "Alpha", "markdown", "alpha", "alpha", "hash")
    chunk = Chunk("alpha", "alpha.md", "alpha", 0, 5, {"title": "Alpha", "path": "/corpus/alpha.md", "format": "markdown", "raw_hash": "hash"})
    index_dir = tmp_path / "qdrant-index"
    collection = f"tiny_rag_test_{tmp_path.name.replace('-', '_')}"
    write_index(
        index_dir, [doc], [chunk], np.array([[1.0, 0.0]], dtype=np.float32),
        corpus_root=tmp_path, embedding_backend="fake", embedding_model="fake", embedding_dim=2,
        chunk_size=800, chunk_overlap=120, index_backend="qdrant", backend_identity=collection,
    )
    backend = QdrantIndexBackend(os.environ["QDRANT_URL"])
    index = load_index(index_dir)
    backend.build(collection, index)
    hits = backend.search(np.array([1.0, 0.0], dtype=np.float32), index, top_k=1)
    numpy_hits = NumpyIndexBackend().search(np.array([1.0, 0.0], dtype=np.float32), index, top_k=1)
    assert hits[0].result.chunk.chunk_id == "alpha"
    assert hits[0].result.chunk.chunk_id == numpy_hits[0].result.chunk.chunk_id
    assert hits[0].result.rank == numpy_hits[0].result.rank == 1
    assert hits[0].score_semantics == "qdrant_cosine_similarity"
    assert numpy_hits[0].score_semantics == "cosine_similarity[-1,1]"
    assert hits[0].backend == "qdrant"
