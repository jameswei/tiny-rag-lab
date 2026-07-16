import numpy as np
import pytest
from qdrant_client import QdrantClient

from tiny_rag_lab.index_loader import LoadedIndex
from tiny_rag_lab.models import Chunk, RetrievalResult
from tiny_rag_lab.qdrant_backend import (
    QdrantBackendError,
    QdrantIndexBackend,
    compare_exact_rankings,
    source_vector_fingerprint,
)


def _index() -> LoadedIndex:
    chunk = Chunk(
        chunk_id="known", doc_id="doc.md", text="evidence", char_start=0, char_end=8,
        metadata={"title": "Doc", "path": "doc.md", "format": "markdown", "raw_hash": "hash"},
    )
    return LoadedIndex(
        manifest={"backend_identity": "collection"}, chunks=[chunk],
        embeddings=np.array([[1.0, 0.0]], dtype=np.float32), chunk_ids=["known"],
    )


def _teaching_index() -> LoadedIndex:
    chunks = [
        Chunk(
            chunk_id="queues", doc_id="queues/retries.md", text="queue retries",
            char_start=0, char_end=13,
            metadata={"title": "Queue retries", "path": "/source/queues/retries.md"},
        ),
        Chunk(
            chunk_id="kv", doc_id="kv/cache.md", text="kv cache",
            char_start=0, char_end=8,
            metadata={"title": "KV cache", "path": "/source/kv/cache.md"},
        ),
    ]
    return LoadedIndex(
        manifest={"backend_identity": "teaching"}, chunks=chunks,
        embeddings=np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        chunk_ids=[chunk.chunk_id for chunk in chunks],
    )


def _memory_backend() -> QdrantIndexBackend:
    backend = object.__new__(QdrantIndexBackend)
    backend._client = QdrantClient(":memory:")
    return backend


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


def test_source_vector_fingerprint_covers_ids_order_and_float32_vectors():
    index = _teaching_index()
    original = source_vector_fingerprint(index)
    assert original == source_vector_fingerprint(index)

    changed = _teaching_index()
    changed.embeddings[0, 0] += np.float32(0.01)
    assert source_vector_fingerprint(changed) != original

    reordered = _teaching_index()
    reordered.chunk_ids.reverse()
    assert source_vector_fingerprint(reordered) != original


def test_prepare_teaching_collection_is_verified_and_idempotent():
    backend = _memory_backend()
    index = _teaching_index()

    first = backend.prepare_teaching_collection("course", index)
    second = backend.prepare_teaching_collection("course", index)

    assert first.verified is True and first.reused is False
    assert second.verified is True and second.reused is True
    assert first.collection == second.collection
    assert first.source_fingerprint == source_vector_fingerprint(index)
    records = backend._client.retrieve(
        "course", ids=[0, 1], with_payload=True, with_vectors=True,
    )
    assert records[0].vector == pytest.approx([1.0, 0.0])
    assert records[0].payload == {
        "chunk_id": "queues",
        "doc_id": "queues/retries.md",
        "title": "Queue retries",
        "path": "queues/retries.md",
        "source_group": "queues",
        "source_fingerprint": first.source_fingerprint,
    }


def test_prepare_repairs_mismatched_payload_before_reuse():
    backend = _memory_backend()
    index = _teaching_index()
    first = backend.prepare_teaching_collection("course", index)
    backend._client.set_payload(
        first.collection, payload={"source_fingerprint": "stale"}, points=[0],
    )

    repaired = backend.prepare_teaching_collection("course", index)

    assert repaired.reused is False
    assert backend.teaching_status("course", index) is not None


def test_prepare_rejects_extra_points_and_keeps_alias_live_until_cutover():
    from qdrant_client.models import PointStruct

    backend = _memory_backend()
    index = _teaching_index()
    first = backend.prepare_teaching_collection("course", index)
    backend._client.upsert(
        first.collection,
        points=[PointStruct(id=99, vector=[1.0, 0.0], payload={"chunk_id": "unexpected"})],
        wait=True,
    )
    assert backend.teaching_status("course", index) is None

    original_create = backend._client.create_collection
    original_upsert = backend._client.upsert
    observed_alias_targets = []

    def assert_old_alias_then(callable_, *args, **kwargs):
        aliases = {item.alias_name: item.collection_name for item in backend._client.get_aliases().aliases}
        observed_alias_targets.append(aliases.get("course"))
        assert aliases.get("course") == first.collection
        return callable_(*args, **kwargs)

    backend._client.create_collection = lambda *args, **kwargs: assert_old_alias_then(original_create, *args, **kwargs)
    backend._client.upsert = lambda *args, **kwargs: assert_old_alias_then(original_upsert, *args, **kwargs)
    repaired = backend.prepare_teaching_collection("course", index)

    assert repaired.reused is False
    assert repaired.collection != first.collection
    assert observed_alias_targets == [first.collection, first.collection]
    assert backend.teaching_status("course", index) is not None
    assert backend._client.get_collection("course").points_count == len(index.chunks)
    assert backend.prepare_teaching_collection("course", index).reused is True


def test_prepare_repairs_every_field_used_by_payload_inspection_and_filters():
    backend = _memory_backend()
    index = _teaching_index()
    first = backend.prepare_teaching_collection("course", index)
    backend._client.set_payload(
        first.collection,
        payload={"doc_id": "wrong.md", "path": "wrong.md", "source_group": "wrong"},
        points=[1],
    )

    assert backend.teaching_status("course", index) is None
    repaired = backend.prepare_teaching_collection("course", index)
    filtered = backend.search_exact(
        "course", np.array([0.0, 1.0], dtype=np.float32), index, top_k=2,
        source_group="kv",
    )

    assert repaired.reused is False
    assert [hit.result.chunk.chunk_id for hit in filtered] == ["kv"]
    assert filtered[0].payload["doc_id"] == "kv/cache.md"
    assert filtered[0].payload["path"] == "kv/cache.md"


def test_exact_search_supports_separate_source_group_filter():
    backend = _memory_backend()
    index = _teaching_index()
    backend.prepare_teaching_collection("course", index)

    unfiltered = backend.search_exact(
        "course", np.array([1.0, 0.0], dtype=np.float32), index, top_k=2,
    )
    filtered = backend.search_exact(
        "course", np.array([1.0, 0.0], dtype=np.float32), index, top_k=2,
        source_group="kv",
    )

    assert [hit.result.chunk.chunk_id for hit in unfiltered] == ["queues", "kv"]
    assert [hit.result.chunk.chunk_id for hit in filtered] == ["kv"]
    assert filtered[0].payload["source_group"] == "kv"


def test_legacy_minimal_payload_remains_searchable_but_disables_filters():
    from qdrant_client.models import Distance, PointStruct, VectorParams

    backend = _memory_backend()
    index = _index()
    backend._client.create_collection(
        "collection", vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    backend._client.upsert(
        "collection", points=[PointStruct(id=0, vector=[1.0, 0.0], payload={"chunk_id": "known"})],
        wait=True,
    )

    results = backend.search(np.array([1.0, 0.0], dtype=np.float32), index, top_k=1)

    assert results[0].result.chunk.chunk_id == "known"
    assert backend.payload_filters_available("collection", index) is False


def test_exact_parity_treats_near_equal_score_reordering_as_tied():
    first, second = _teaching_index().chunks
    numpy_results = [
        RetrievalResult(first, score=0.500000, rank=1),
        RetrievalResult(second, score=0.499996, rank=2),
    ]
    qdrant_results = [
        RetrievalResult(second, score=0.500001, rank=1),
        RetrievalResult(first, score=0.499999, rank=2),
    ]

    report = compare_exact_rankings(numpy_results, qdrant_results, tolerance=1e-5)

    assert report.equivalent is True
    assert all(item.equivalent for item in report.items)


def test_exact_parity_reports_real_rank_mismatch():
    first, second = _teaching_index().chunks
    numpy_results = [
        RetrievalResult(first, score=0.9, rank=1),
        RetrievalResult(second, score=0.2, rank=2),
    ]
    qdrant_results = [
        RetrievalResult(second, score=0.8, rank=1),
        RetrievalResult(first, score=0.3, rank=2),
    ]

    report = compare_exact_rankings(numpy_results, qdrant_results)

    assert report.equivalent is False
    assert not all(item.equivalent for item in report.items)
