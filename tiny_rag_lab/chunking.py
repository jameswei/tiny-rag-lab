"""Character-based document chunking (T07).

chunk_document splits Document.normalized_text into fixed-size overlapping
windows. All offsets are Python string character indices into normalized_text.

Spec invariant (must hold for every produced Chunk):
    document.normalized_text[chunk.char_start:chunk.char_end] == chunk.text
"""
from __future__ import annotations

from tiny_rag_lab.models import Chunk, Document, make_chunk_id


def chunk_document(
    doc: Document,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Split doc.normalized_text into overlapping character windows.

    Args:
        doc:          source document (normalized_text must be populated)
        chunk_size:   window size in characters
        chunk_overlap: number of characters shared between consecutive windows

    Returns a list of Chunk objects in document order. Empty or
    whitespace-only windows are skipped.

    chunk_overlap must be strictly less than chunk_size so the sliding
    window always advances.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})"
        )

    text = doc.normalized_text
    step = chunk_size - chunk_overlap
    metadata = {
        "title": doc.title,
        "path": doc.path,
        "format": doc.format,
        "raw_hash": doc.raw_hash,
    }

    chunks: list[Chunk] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc.doc_id, start, chunk_text),
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    char_start=start,
                    char_end=end,
                    metadata=metadata,
                )
            )

        if end == len(text):
            # Reached the end of the document — stop here to avoid emitting
            # a redundant tail chunk wholly contained in the current window.
            break

        start += step

    return chunks


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Chunk a list of documents and return all chunks in document order."""
    result: list[Chunk] = []
    for doc in docs:
        result.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return result
