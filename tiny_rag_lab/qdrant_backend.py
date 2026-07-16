"""Optional local Qdrant adapter for Phase 3.0.

It deliberately stores only vector-search points in Qdrant. The local index
directory remains the canonical source for chunks and inspectable vectors.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4

import numpy as np

from tiny_rag_lab.index_backend import VectorSearchHit
from tiny_rag_lab.index_loader import LoadedIndex, load_index
from tiny_rag_lab.models import RetrievalResult


class QdrantBackendError(RuntimeError):
    """A non-secret failure that a local-lab learner can act on."""


@dataclass(frozen=True)
class QdrantPreparationReport:
    alias: str
    collection: str
    source_fingerprint: str
    point_count: int
    dimension: int
    reused: bool
    verified: bool


@dataclass(frozen=True)
class QdrantExactHit:
    result: RetrievalResult
    payload: dict


@dataclass(frozen=True)
class ParityItem:
    chunk_id: str
    numpy_rank: int | None
    qdrant_rank: int | None
    numpy_score: float | None
    qdrant_score: float | None
    equivalent: bool


@dataclass(frozen=True)
class ParityReport:
    equivalent: bool
    score_tolerance: float
    items: list[ParityItem]


def source_vector_fingerprint(index: LoadedIndex) -> str:
    """Fingerprint ordered IDs and canonical source float32 vector bytes."""
    embeddings = np.asarray(index.embeddings, dtype="<f4", order="C")
    digest = hashlib.sha256()
    digest.update(b"tiny-rag-lab-source-vectors-v1\0cosine\0")
    digest.update(str(embeddings.shape).encode("ascii"))
    for chunk_id in index.chunk_ids:
        encoded = chunk_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    digest.update(embeddings.tobytes(order="C"))
    return digest.hexdigest()


def chunk_source_group(chunk) -> str:
    """Return the curated top-level Cloudflare source group."""
    return chunk.doc_id.split("/", 1)[0]


def compare_exact_rankings(
    numpy_results: list[RetrievalResult],
    qdrant_results: list[RetrievalResult],
    *,
    tolerance: float = 1e-5,
) -> ParityReport:
    """Compare exact results while treating near-equal score ties as groups."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    def tie_groups(results: list[RetrievalResult]) -> list[set[str]]:
        groups: list[set[str]] = []
        anchor: float | None = None
        for result in results:
            if anchor is None or abs(result.score - anchor) > tolerance:
                groups.append(set())
                anchor = result.score
            groups[-1].add(result.chunk.chunk_id)
        return groups

    numpy_groups = tie_groups(numpy_results)
    qdrant_groups = tie_groups(qdrant_results)
    numpy_group_by_id = {
        chunk_id: position for position, group in enumerate(numpy_groups) for chunk_id in group
    }
    qdrant_group_by_id = {
        chunk_id: position for position, group in enumerate(qdrant_groups) for chunk_id in group
    }
    numpy_by_id = {result.chunk.chunk_id: result for result in numpy_results}
    qdrant_by_id = {result.chunk.chunk_id: result for result in qdrant_results}
    ordered_ids = list(numpy_by_id)
    ordered_ids.extend(chunk_id for chunk_id in qdrant_by_id if chunk_id not in numpy_by_id)
    items = []
    for chunk_id in ordered_ids:
        numpy_result = numpy_by_id.get(chunk_id)
        qdrant_result = qdrant_by_id.get(chunk_id)
        equivalent = (
            numpy_result is not None
            and qdrant_result is not None
            and numpy_group_by_id[chunk_id] == qdrant_group_by_id[chunk_id]
            and abs(numpy_result.score - qdrant_result.score) <= tolerance
        )
        items.append(ParityItem(
            chunk_id=chunk_id,
            numpy_rank=numpy_result.rank if numpy_result else None,
            qdrant_rank=qdrant_result.rank if qdrant_result else None,
            numpy_score=numpy_result.score if numpy_result else None,
            qdrant_score=qdrant_result.score if qdrant_result else None,
            equivalent=equivalent,
        ))
    return ParityReport(
        equivalent=(numpy_groups == qdrant_groups and all(item.equivalent for item in items)),
        score_tolerance=tolerance,
        items=items,
    )


def qdrant_is_available(url: str) -> bool:
    """Return whether the optional local service can answer a small request.

    This is deliberately a readiness check, not an attempt to hide Qdrant
    behind a managed abstraction.  The browser uses it to avoid offering an
    unavailable backend, while the actual build and search still use the same
    visible ``QdrantIndexBackend`` below.
    """
    try:
        with urlopen(f"{url.rstrip('/')}/healthz", timeout=0.75) as response:  # noqa: S310 - local configured service
            return 200 <= response.status < 300
    except OSError:
        return False


class QdrantIndexBackend:
    name = "qdrant"
    score_semantics = "qdrant_cosine_similarity"

    def __init__(self, url: str) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - optional install boundary
            raise RuntimeError("Install tiny-rag-lab[qdrant] to use the Qdrant backend") from exc
        self._client = QdrantClient(url=url)

    def open(self, index_dir: Path) -> LoadedIndex:
        return load_index(index_dir)

    def build(self, collection: str, index: LoadedIndex) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        fingerprint = source_vector_fingerprint(index)
        if self._client.collection_exists(collection):
            self._client.delete_collection(collection)
        self._client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=index.embeddings.shape[1], distance=Distance.COSINE),
        )
        self._client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=position,
                    vector=[float(value) for value in index.embeddings[position]],
                    payload=self._teaching_payload(
                        chunk, source_fingerprint=fingerprint,
                    ),
                )
                for position, chunk in enumerate(index.chunks)
            ],
            wait=True,
        )

    def delete(self, collection: str) -> None:
        """Remove an unpublished collection during cooperative cancellation."""
        if self._client.collection_exists(collection):
            self._client.delete_collection(collection)

    @staticmethod
    def _teaching_payload(chunk, *, source_fingerprint: str) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title": chunk.metadata.get("title", ""),
            "path": chunk.doc_id,
            "source_group": chunk_source_group(chunk),
            "source_fingerprint": source_fingerprint,
        }

    def _verify_teaching_collection(
        self,
        collection: str,
        index: LoadedIndex,
        source_fingerprint: str,
        *,
        vector_tolerance: float = 1e-6,
    ) -> bool:
        from qdrant_client.models import Distance

        try:
            info = self._client.get_collection(collection)
            vectors = info.config.params.vectors
            if isinstance(vectors, dict):
                return False
            if vectors.size != index.embeddings.shape[1] or vectors.distance != Distance.COSINE:
                return False
            if info.points_count != len(index.chunks):
                return False
            records = self._client.retrieve(
                collection_name=collection,
                ids=list(range(len(index.chunks))),
                with_payload=True,
                with_vectors=True,
            )
        except Exception:
            return False
        if len(records) != len(index.chunks):
            return False
        by_id = {int(record.id): record for record in records}
        if set(by_id) != set(range(len(index.chunks))):
            return False
        for position, chunk in enumerate(index.chunks):
            record = by_id[position]
            payload = record.payload or {}
            expected_payload = self._teaching_payload(
                chunk, source_fingerprint=source_fingerprint,
            )
            if any(payload.get(key) != value for key, value in expected_payload.items()):
                return False
            remote = np.asarray(record.vector, dtype=np.float32)
            source = np.asarray(index.embeddings[position], dtype=np.float32)
            norm = float(np.linalg.norm(source))
            normalized = source / norm if norm else source
            if remote.shape != normalized.shape:
                return False
            if not np.allclose(remote, normalized, atol=vector_tolerance, rtol=0.0):
                return False
        return True

    def prepare_teaching_collection(
        self,
        alias: str,
        index: LoadedIndex,
    ) -> QdrantPreparationReport:
        """Idempotently publish a verified exact-vector teaching collection."""
        from qdrant_client.models import (
            CreateAlias,
            CreateAliasOperation,
            DeleteAlias,
            DeleteAliasOperation,
            Distance,
            PointStruct,
            VectorParams,
        )

        fingerprint = source_vector_fingerprint(index)
        canonical = f"{alias}__{fingerprint[:16]}"
        aliases = {
            item.alias_name: item.collection_name
            for item in self._client.get_aliases().aliases
        }
        previous = aliases.get(alias)

        # Reuse the published collection even when it carries a repair suffix.
        # A repair must remain idempotent on the next call rather than being
        # rebuilt only to recover the canonical name.
        physical = previous or canonical
        reused = bool(previous and self._verify_teaching_collection(
            previous, index, fingerprint,
        ))
        if not reused and canonical != previous and self._client.collection_exists(canonical):
            if self._verify_teaching_collection(canonical, index, fingerprint):
                physical = canonical
                reused = True
            else:
                self._client.delete_collection(canonical)
        if not reused:
            # If the corrupted collection is currently published, build under
            # a distinct name. The alias continues serving the old collection
            # until the replacement has been fully populated and verified.
            physical = (
                f"{canonical}__repair_{uuid4().hex[:8]}"
                if previous == canonical
                else canonical
            )
            self._client.create_collection(
                collection_name=physical,
                vectors_config=VectorParams(
                    size=index.embeddings.shape[1], distance=Distance.COSINE,
                ),
            )
            self._client.upsert(
                collection_name=physical,
                points=[
                    PointStruct(
                        id=position,
                        vector=[float(value) for value in index.embeddings[position]],
                        payload=self._teaching_payload(
                            chunk, source_fingerprint=fingerprint,
                        ),
                    )
                    for position, chunk in enumerate(index.chunks)
                ],
                wait=True,
            )
            if not self._verify_teaching_collection(physical, index, fingerprint):
                self._client.delete_collection(physical)
                raise QdrantBackendError(
                    "Qdrant did not preserve the complete teaching collection. Try preparing it again."
                )

        if previous != physical:
            operations = []
            if previous:
                operations.append(DeleteAliasOperation(
                    delete_alias=DeleteAlias(alias_name=alias),
                ))
            operations.append(CreateAliasOperation(
                create_alias=CreateAlias(collection_name=physical, alias_name=alias),
            ))
            self._client.update_collection_aliases(operations)
            if previous and previous.startswith(f"{alias}__") and previous != physical:
                self._client.delete_collection(previous)
        return QdrantPreparationReport(
            alias=alias,
            collection=physical,
            source_fingerprint=fingerprint,
            point_count=len(index.chunks),
            dimension=index.embeddings.shape[1],
            reused=reused,
            verified=True,
        )

    def payload_filters_available(
        self, collection: str, index: LoadedIndex,
    ) -> bool:
        """Report whether every point has the source-group teaching payload.

        Older Phase 3.0 Qdrant collections stored only ``chunk_id``. They stay
        searchable, but callers must not offer payload filtering for them.
        """
        try:
            records = self._client.retrieve(
                collection_name=collection,
                ids=list(range(len(index.chunks))),
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return False
        if len(records) != len(index.chunks):
            return False
        return all(
            isinstance((record.payload or {}).get("source_group"), str)
            and bool((record.payload or {}).get("source_group"))
            for record in records
        )

    def teaching_status(
        self, alias: str, index: LoadedIndex, *, raise_on_error: bool = False,
    ) -> QdrantPreparationReport | None:
        fingerprint = source_vector_fingerprint(index)
        try:
            aliases = {
                item.alias_name: item.collection_name
                for item in self._client.get_aliases().aliases
            }
            physical = aliases.get(alias)
            if not physical or not self._verify_teaching_collection(
                physical, index, fingerprint,
            ):
                return None
        except Exception as exc:
            if raise_on_error:
                raise QdrantBackendError(
                    "Qdrant status is unavailable. Check the local service and try again."
                ) from exc
            return None
        return QdrantPreparationReport(
            alias=alias,
            collection=physical,
            source_fingerprint=fingerprint,
            point_count=len(index.chunks),
            dimension=index.embeddings.shape[1],
            reused=True,
            verified=True,
        )

    def search_exact(
        self,
        collection: str,
        query_vector: np.ndarray,
        index: LoadedIndex,
        top_k: int,
        *,
        source_group: str | None = None,
    ) -> list[QdrantExactHit]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue, SearchParams

        query_filter = None
        if source_group:
            query_filter = Filter(must=[FieldCondition(
                key="source_group", match=MatchValue(value=source_group),
            )])
        try:
            response = self._client.query_points(
                collection_name=collection,
                query=[float(value) for value in query_vector],
                query_filter=query_filter,
                search_params=SearchParams(exact=True),
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise QdrantBackendError(
                "Exact Qdrant search is unavailable. Prepare the teaching collection again."
            ) from exc
        by_id = {chunk.chunk_id: chunk for chunk in index.chunks}
        hits = []
        for rank, point in enumerate(response.points, start=1):
            payload = dict(point.payload or {})
            chunk_id = str(payload.get("chunk_id", ""))
            if chunk_id not in by_id:
                raise QdrantBackendError(
                    "Qdrant collection does not match the local teaching index."
                )
            hits.append(QdrantExactHit(
                result=RetrievalResult(
                    chunk=by_id[chunk_id], score=float(point.score), rank=rank,
                ),
                payload=payload,
            ))
        return hits

    def search(self, query_vector: np.ndarray, index: LoadedIndex, top_k: int) -> list[VectorSearchHit]:
        collection = index.manifest.get("backend_identity")
        if not collection:
            raise ValueError("Qdrant index manifest has no collection identity")
        try:
            response = self._client.query_points(
                collection_name=collection,
                query=[float(value) for value in query_vector],
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise QdrantBackendError(
                "Qdrant search is unavailable. Start the optional Qdrant profile or rebuild this index."
            ) from exc
        by_id = {chunk.chunk_id: chunk for chunk in index.chunks}
        hits = []
        for rank, point in enumerate(response.points, start=1):
            chunk_id = str((point.payload or {}).get("chunk_id", ""))
            if chunk_id not in by_id:
                raise QdrantBackendError(
                    "Qdrant collection does not match this local index. Rebuild the index before searching."
                )
            hits.append(VectorSearchHit(
                result=RetrievalResult(chunk=by_id[chunk_id], score=float(point.score), rank=rank),
                backend=self.name,
                score_semantics=self.score_semantics,
            ))
        return hits
