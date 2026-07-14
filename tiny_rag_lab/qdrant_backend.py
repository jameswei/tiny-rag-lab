"""Optional local Qdrant adapter for Phase 3.0.

It deliberately stores only vector-search points in Qdrant. The local index
directory remains the canonical source for chunks and inspectable vectors.
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import numpy as np

from tiny_rag_lab.index_backend import VectorSearchHit
from tiny_rag_lab.index_loader import LoadedIndex, load_index
from tiny_rag_lab.models import RetrievalResult


class QdrantBackendError(RuntimeError):
    """A non-secret failure that a local-lab learner can act on."""


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
                    payload={"chunk_id": chunk.chunk_id},
                )
                for position, chunk in enumerate(index.chunks)
            ],
            wait=True,
        )

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
