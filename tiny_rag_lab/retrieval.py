"""Cosine retrieval for T12.

Two public functions:

  retrieve_by_vector(query_vec, index, top_k) -> list[RetrievalResult]
    Core ranking logic. Accepts a pre-computed query vector so it can be
    called directly in tests with known vectors.

  retrieve(query_text, index, embedder, top_k) -> list[RetrievalResult]
    Embeds query_text with embedder, then delegates to retrieve_by_vector.

Zero-vector policy:
  - A zero query vector returns an empty list (no meaningful ranking possible).
  - Zero-norm index rows receive a score of 0.0 and rank last among ties.

Vectors are re-normalized inside retrieve_by_vector so callers can pass either
raw or pre-normalized vectors without affecting results.
"""
from __future__ import annotations

import numpy as np

from tiny_rag_lab.embeddings import Embedder
from tiny_rag_lab.index_loader import LoadedIndex
from tiny_rag_lab.models import RetrievalResult

DEFAULT_TOP_K = 5


def retrieve(
    query_text: str,
    index: LoadedIndex,
    embedder: Embedder,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievalResult]:
    """Embed query_text and return the top_k most similar chunks."""
    query_vec = embedder.embed([query_text])[0]
    return retrieve_by_vector(query_vec, index, top_k=top_k)


def retrieve_by_vector(
    query_vec: np.ndarray,
    index: LoadedIndex,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievalResult]:
    """Return the top_k chunks ranked by cosine similarity to query_vec.

    query_vec must be 1-D with length matching the index embedding dimension.
    Returns an empty list if the index has no chunks or the query is a zero vector.
    """
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")

    if len(index.chunks) == 0:
        return []

    query_vec = np.asarray(query_vec, dtype=np.float32)
    q_norm = float(np.linalg.norm(query_vec))
    if q_norm == 0.0:
        return []
    query_unit = query_vec / q_norm

    emb = index.embeddings.astype(np.float32)           # (N, dim)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)  # (N, 1)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    emb_unit = emb / safe_norms                          # zero rows stay zero

    scores = emb_unit @ query_unit                       # (N,)

    # zero-norm index rows should score 0, not whatever emb/1.0 @ query gives
    scores[norms[:, 0] == 0.0] = 0.0

    actual_k = min(top_k, len(scores))
    top_indices = np.argsort(scores)[::-1][:actual_k]

    return [
        RetrievalResult(
            chunk=index.chunks[int(idx)],
            score=float(scores[idx]),
            rank=rank,
        )
        for rank, idx in enumerate(top_indices, start=1)
    ]
