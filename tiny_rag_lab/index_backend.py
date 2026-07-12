"""Small, visible vector-index boundary used by the CLI and local visual lab.

The default implementation deliberately remains the NumPy files already used
by tiny-rag-lab.  A backend only owns vector persistence/search: chunks,
BM25, hybrid fusion, prompt assembly, and teaching artifacts stay project
owned.  Optional adapters (such as Qdrant) are added without changing that
conceptual pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from tiny_rag_lab.index_loader import LoadedIndex, load_index
from tiny_rag_lab.models import RetrievalResult
from tiny_rag_lab.retrieval import retrieve_by_vector


@dataclass(frozen=True)
class VectorSearchHit:
    """One backend-neutral dense-search result.

    `score` is deliberately accompanied by `score_semantics`; consumers must
    not assume scores from different engines are interchangeable.
    """

    result: RetrievalResult
    backend: str
    score_semantics: str


class VectorIndexBackend(Protocol):
    """The narrow vector-storage/search seam.

    Build/open concerns stay in the existing writer/loader for NumPy during
    Phase 3.0's first slice.  Adapters expose the same `search` output so the
    rest of the retrieval plane remains independent of the storage choice.
    """

    name: str
    score_semantics: str

    def open(self, index_dir: Path) -> LoadedIndex: ...

    def search(
        self, query_vector: np.ndarray, index: LoadedIndex, top_k: int
    ) -> list[VectorSearchHit]: ...


class NumpyIndexBackend:
    """The original exact local cosine-similarity index."""

    name = "numpy"
    score_semantics = "cosine_similarity[-1,1]"

    def open(self, index_dir: Path) -> LoadedIndex:
        return load_index(index_dir)

    def search(
        self, query_vector: np.ndarray, index: LoadedIndex, top_k: int
    ) -> list[VectorSearchHit]:
        return [
            VectorSearchHit(
                result=result,
                backend=self.name,
                score_semantics=self.score_semantics,
            )
            for result in retrieve_by_vector(query_vector, index, top_k=top_k)
        ]


def backend_from_manifest(manifest: dict) -> VectorIndexBackend:
    """Return the default backend for a persisted index.

    Qdrant is intentionally not silently selected here: callers must opt in
    and provide its connection configuration.  This protects old CLI flows
    from gaining an implicit network/service dependency.
    """

    backend = manifest.get("index_backend", "numpy")
    if backend == "numpy":
        return NumpyIndexBackend()
    raise ValueError(
        f"Index backend {backend!r} needs an explicitly configured adapter"
    )
