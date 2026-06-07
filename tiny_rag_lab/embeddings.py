"""Embedding interface, fake embedder (T08), and real embedder (T09).

Embedder is the single interface the retrieval layer depends on.
FakeEmbedder is deterministic and self-contained — safe for all tests.
SentenceTransformerEmbedder is the real Phase 1 backend; it downloads
model weights from HuggingFace on first use (requires network once).
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Interface contract for all embedding backends.

    embed() takes a list of strings and returns a float32 matrix of shape
    (len(texts), dim). Retrieval code must not depend on which backend
    produced the vectors.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts. Returns shape (len(texts), dim), dtype float32."""
        ...


class FakeEmbedder(Embedder):
    """Deterministic unit-vector embedder for tests.

    Each text is hashed with SHA-256; the first 4 bytes seed a NumPy RNG
    that produces a standard-normal vector, which is then L2-normalised.
    Same text always produces the same vector. No model download needed.
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = []
        for text in texts:
            seed = int.from_bytes(
                hashlib.sha256(text.encode()).digest()[:4], byteorder="little"
            )
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


class SentenceTransformerEmbedder(Embedder):
    """Real embedding backend using sentence-transformers (T09).

    Default model: sentence-transformers/all-MiniLM-L6-v2 (384-dim).
    Weights are downloaded from HuggingFace on the first call and cached
    locally. A fresh machine needs network access once; subsequent runs
    are fully offline.

    normalize_embeddings=True produces L2-unit vectors, consistent with
    FakeEmbedder and with cosine retrieval via dot product.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, local_files_only: bool = False) -> None:
        from sentence_transformers import SentenceTransformer  # deferred import
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, local_files_only=local_files_only)

    @property
    def dim(self) -> int:
        """Embedding dimension reported by the loaded model."""
        # get_embedding_dimension() is the current name; fall back to the
        # deprecated get_sentence_embedding_dimension() for older installs.
        getter = getattr(
            self._model,
            "get_embedding_dimension",
            None,
        ) or self._model.get_sentence_embedding_dimension
        return getter()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)
