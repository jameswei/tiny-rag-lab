"""Core data contracts for the RAG pipeline.

All four types are plain dataclasses so they serialize cleanly with
dataclasses.asdict() and json.dumps(). numpy arrays are intentionally
absent — embeddings live in the index layer, not here.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


def make_chunk_id(doc_id: str, char_start: int, chunk_text: str) -> str:
    """Return a 16-hex-char deterministic chunk ID.

    Formula: sha256(doc_id + ":" + str(char_start) + ":" + chunk_text)[:16]
    Stable across re-indexing as long as the source text and offsets are
    unchanged.
    """
    raw = f"{doc_id}:{char_start}:{chunk_text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class Document:
    """A source file loaded from the corpus.

    doc_id is the POSIX-style path relative to the --corpus root, e.g.
    "docs/example.md". path is the absolute filesystem path used for loading.
    char offsets in Chunk refer to normalized_text, not raw_text.
    """

    doc_id: str
    path: str
    title: str
    format: str          # "markdown" | "text"
    raw_text: str
    normalized_text: str
    raw_hash: str        # sha256 of raw_text


@dataclass
class Chunk:
    """The atomic retrieval unit produced from a Document.

    Invariant: document.normalized_text[char_start:char_end] == text
    chunk_id is produced by make_chunk_id(doc_id, char_start, text).
    metadata must include at least title, path, format, and raw_hash.
    """

    chunk_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int
    metadata: dict[str, Any]


@dataclass
class RetrievalResult:
    """A single ranked result from the retrieval plane.

    score is cosine similarity in [-1, 1]. rank is 1-indexed.
    """

    chunk: Chunk
    score: float
    rank: int            # 1-indexed


