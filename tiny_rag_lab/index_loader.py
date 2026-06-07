"""Index loader for T11 — reads persisted index artifacts from disk.

load_index(index_dir) reads the three files written by write_index and
returns a LoadedIndex with:
  - manifest: the parsed manifest dict
  - chunks: list[Chunk] in original write order
  - embeddings: float32 ndarray of shape (chunk_count, embedding_dim)
  - chunk_ids: list[str] parallel to both chunks and embeddings rows

Raises FileNotFoundError if any required file is missing.
Raises ValueError if chunk IDs in chunks.jsonl and embeddings.npz diverge.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tiny_rag_lab.models import Chunk


@dataclass
class LoadedIndex:
    """The in-memory representation of a persisted index."""

    manifest: dict[str, Any]
    chunks: list[Chunk]
    embeddings: np.ndarray   # shape (chunk_count, embedding_dim), float32
    chunk_ids: list[str]     # parallel to chunks and embeddings rows


def load_index(index_dir: Path) -> LoadedIndex:
    """Load manifest, chunks, and embeddings from index_dir.

    Validates that chunk IDs in chunks.jsonl and embeddings.npz agree and
    that the embedding matrix row count matches the chunk list length.
    """
    index_dir = Path(index_dir)

    manifest_path = index_dir / "manifest.json"
    chunks_path = index_dir / "chunks.jsonl"
    embeddings_path = index_dir / "embeddings.npz"

    for p in (manifest_path, chunks_path, embeddings_path):
        if not p.exists():
            raise FileNotFoundError(f"Index file not found: {p}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    chunks = _load_chunks(chunks_path)

    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    chunk_ids_npz = [str(cid) for cid in data["chunk_ids"]]

    chunk_ids_jsonl = [c.chunk_id for c in chunks]
    if chunk_ids_jsonl != chunk_ids_npz:
        raise ValueError(
            "chunk_ids mismatch between chunks.jsonl and embeddings.npz"
        )

    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            f"embeddings row count {embeddings.shape[0]} != chunk count {len(chunks)}"
        )

    return LoadedIndex(
        manifest=manifest,
        chunks=chunks,
        embeddings=embeddings,
        chunk_ids=chunk_ids_jsonl,
    )


def _load_chunks(chunks_path: Path) -> list[Chunk]:
    chunks = []
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        chunks.append(Chunk(**obj))
    return chunks
