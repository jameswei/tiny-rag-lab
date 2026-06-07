"""Index writer for T10 — persists indexed artifacts to disk.

Writes three files under index_dir:
  manifest.json   — metadata about the index (schema, embedder, corpus files)
  chunks.jsonl    — one serialized Chunk per line (no vectors)
  embeddings.npz  — float32 matrix + parallel chunk_ids array

Row order in embeddings.npz matches the order of chunks passed to write_index.
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tiny_rag_lab.models import Chunk, Document

SCHEMA_VERSION = "1.0"


def write_index(
    index_dir: Path,
    docs: list[Document],
    chunks: list[Chunk],
    embeddings: np.ndarray,
    *,
    corpus_root: Path,
    embedding_backend: str,
    embedding_model: str,
    embedding_dim: int,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Write manifest.json, chunks.jsonl, and embeddings.npz to index_dir.

    embeddings must have shape (len(chunks), embedding_dim) and dtype float32.
    Row i of embeddings corresponds to chunks[i].
    """
    if embeddings.shape != (len(chunks), embedding_dim):
        raise ValueError(
            f"embeddings shape {embeddings.shape} does not match "
            f"({len(chunks)}, {embedding_dim})"
        )

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    _write_manifest(
        index_dir,
        docs=docs,
        chunks=chunks,
        corpus_root=corpus_root,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    _write_chunks(index_dir, chunks)
    _write_embeddings(index_dir, chunks, embeddings)


def _write_manifest(
    index_dir: Path,
    *,
    docs: list[Document],
    chunks: list[Chunk],
    corpus_root: Path,
    embedding_backend: str,
    embedding_model: str,
    embedding_dim: int,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    corpus_files = [
        {"doc_id": doc.doc_id, "path": doc.path, "raw_hash": doc.raw_hash}
        for doc in docs
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus_root": str(corpus_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(docs),
        "chunk_count": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_backend": embedding_backend,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "corpus_files": corpus_files,
    }
    manifest_path = index_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_chunks(index_dir: Path, chunks: list[Chunk]) -> None:
    chunks_path = index_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(dataclasses.asdict(chunk)) + "\n")


def _write_embeddings(
    index_dir: Path, chunks: list[Chunk], embeddings: np.ndarray
) -> None:
    chunk_ids = np.array([c.chunk_id for c in chunks])
    np.savez(
        index_dir / "embeddings.npz",
        embeddings=embeddings.astype(np.float32),
        chunk_ids=chunk_ids,
    )
