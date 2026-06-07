"""Embedding interface and fake embedder (T08).

Embedder is the single interface the retrieval layer depends on.
FakeEmbedder is deterministic and self-contained — safe for all tests.
SentenceTransformerEmbedder (T09) is the real Phase 1 backend.
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
