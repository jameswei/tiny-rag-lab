import numpy as np
import pytest

from tiny_rag_lab.index_loader import LoadedIndex
from tiny_rag_lab.models import Chunk
from tiny_rag_lab.qdrant_backend import QdrantBackendError, QdrantIndexBackend


def _index() -> LoadedIndex:
    chunk = Chunk(
        chunk_id="known", doc_id="doc.md", text="evidence", char_start=0, char_end=8,
        metadata={"title": "Doc", "path": "doc.md", "format": "markdown", "raw_hash": "hash"},
    )
    return LoadedIndex(
        manifest={"backend_identity": "collection"}, chunks=[chunk],
        embeddings=np.array([[1.0, 0.0]], dtype=np.float32), chunk_ids=["known"],
    )


def test_qdrant_connection_failure_becomes_actionable_backend_error():
    class _UnavailableClient:
        def query_points(self, **_kwargs):
            raise ConnectionError("connection refused")

    backend = object.__new__(QdrantIndexBackend)
    backend._client = _UnavailableClient()

    with pytest.raises(QdrantBackendError, match="unavailable"):
        backend.search(np.array([1.0, 0.0]), _index(), top_k=1)


def test_qdrant_point_missing_from_local_index_becomes_actionable_error():
    class _Point:
        score = 0.9
        payload = {"chunk_id": "stale"}

    class _Client:
        def query_points(self, **_kwargs):
            return type("Response", (), {"points": [_Point()]})()

    backend = object.__new__(QdrantIndexBackend)
    backend._client = _Client()

    with pytest.raises(QdrantBackendError, match="does not match"):
        backend.search(np.array([1.0, 0.0]), _index(), top_k=1)
