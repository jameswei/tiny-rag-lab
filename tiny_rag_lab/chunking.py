"""Character-based document chunking (T07) plus structural chunking (Phase 2.2).

chunk_document splits Document.normalized_text into fixed-size overlapping
windows. All offsets are Python string character indices into normalized_text.

chunk_document_structural (Phase 2.2) packs Markdown-aware blocks instead of
sliding a fixed window: blocks first, then sentences for a block that alone
exceeds chunk_size, then the same character window chunk_document uses for a
single sentence that still exceeds chunk_size. See Design Decision 2 in
docs/phases/phase-2.2-structural-semantic-chunking.md.

Spec invariant (must hold for every produced Chunk, regardless of strategy):
    document.normalized_text[chunk.char_start:chunk.char_end] == chunk.text
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

from tiny_rag_lab.models import Chunk, Document, make_chunk_id

if TYPE_CHECKING:
    from tiny_rag_lab.embeddings import Embedder

_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t].*$")


def _validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """Shared validation for every chunking strategy.

    Raises the exact ValueError text chunk_document has always raised, so
    existing tests asserting on that text keep passing unmodified.
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


def _chunk_metadata(doc: Document) -> dict:
    return {
        "title": doc.title,
        "path": doc.path,
        "format": doc.format,
        "raw_hash": doc.raw_hash,
    }


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
    _validate_chunk_params(chunk_size, chunk_overlap)

    text = doc.normalized_text
    step = chunk_size - chunk_overlap
    metadata = _chunk_metadata(doc)

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


# ---------------------------------------------------------------------------
# Phase 2.2: structural chunking
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[tuple[int, int]]:
    """Split text into contiguous (start, end) sentence spans.

    A span ends after one or more [.!?] characters plus any immediately
    following whitespace, so spans are contiguous and cover the whole
    string with no gaps — concatenating them reproduces the original text
    exactly, which is what lets packed chunks satisfy the slice invariant.
    Text with no terminal punctuation is a single span (the whole string).
    """
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    n = len(text)
    i = 0
    while i < n:
        if text[i] in ".!?":
            j = i + 1
            while j < n and text[j] in ".!?":
                j += 1
            while j < n and text[j] in " \t\n\r":
                j += 1
            spans.append((start, j))
            start = j
            i = j
        else:
            i += 1
    if start < n:
        spans.append((start, n))
    return spans


def _split_blocks(text: str) -> list[tuple[int, int]]:
    """Split text into contiguous (start, end) blocks at blank-line boundaries.

    A run of two or more consecutive newlines is attached to the end of the
    preceding block, so blocks are contiguous and cover the whole string
    with no gaps. A block whose only content is a single Markdown ATX
    heading line is merged with the following block, so a heading never
    forms a chunk with no body.
    """
    if not text:
        return []
    raw: list[tuple[int, int]] = []
    start = 0
    n = len(text)
    i = 0
    while i < n:
        if text[i] == "\n":
            j = i
            newline_run = 0
            while j < n and text[j] == "\n":
                newline_run += 1
                j += 1
            if newline_run >= 2:
                raw.append((start, j))
                start = j
                i = j
                continue
        i += 1
    if start < n:
        raw.append((start, n))

    merged: list[tuple[int, int]] = []
    idx = 0
    n_raw = len(raw)
    while idx < n_raw:
        b_start, b_end = raw[idx]
        if not _is_heading_only_block(text, b_start, b_end):
            merged.append((b_start, b_end))
            idx += 1
            continue
        # Absorb a run of consecutive heading-only blocks (e.g. "# H1\n\n##
        # H2\n\n") and merge the whole run with the next block so a heading
        # never ends a chunk without a body when one is available. If no
        # body follows, the whole run becomes one block (nothing to merge
        # into, same as a single trailing heading).
        j = idx + 1
        while j < n_raw and _is_heading_only_block(text, raw[j][0], raw[j][1]):
            j += 1
        end_idx = j if j < n_raw else n_raw - 1
        merged.append((b_start, raw[end_idx][1]))
        idx = end_idx + 1
    return merged


def _is_heading_only_block(text: str, start: int, end: int) -> bool:
    non_blank_lines = [line for line in text[start:end].splitlines() if line.strip()]
    return len(non_blank_lines) == 1 and bool(_HEADING_RE.match(non_blank_lines[0]))


def _chunk_oversized_span(
    doc: Document,
    start: int,
    end: int,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Tier-3 fallback: slide a character window over text[start:end].

    Identical algorithm to chunk_document, restricted to a span — the only
    place either new chunking strategy consumes chunk_overlap.
    """
    text = doc.normalized_text
    metadata = _chunk_metadata(doc)
    step = chunk_size - chunk_overlap

    chunks: list[Chunk] = []
    pos = start
    while pos < end:
        window_end = min(pos + chunk_size, end)
        chunk_text = text[pos:window_end]
        if chunk_text.strip():
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc.doc_id, pos, chunk_text),
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    char_start=pos,
                    char_end=window_end,
                    metadata=metadata,
                )
            )
        if window_end == end:
            break
        pos += step
    return chunks


_OversizedHandler = Callable[[Document, int, int, int, int], list[Chunk]]


def _pack_units(
    doc: Document,
    units: list[tuple[int, int]],
    chunk_size: int,
    chunk_overlap: int,
    metadata: dict,
    oversized_handler: _OversizedHandler,
) -> list[Chunk]:
    """Greedily pack contiguous (start, end) unit spans into chunks.

    Units are merged in rank order while the running span stays within
    chunk_size; packed chunks are exactly contiguous (no overlap). A unit
    that alone exceeds chunk_size is expanded by oversized_handler instead
    of being packed directly — this is how tier 1 falls back to tier 2, and
    tier 2 falls back to tier 3.
    """
    chunks: list[Chunk] = []
    cur_start: int | None = None
    cur_end: int | None = None

    def flush() -> None:
        if cur_start is None:
            return
        chunk_text = doc.normalized_text[cur_start:cur_end]
        if chunk_text.strip():
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc.doc_id, cur_start, chunk_text),
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    char_start=cur_start,
                    char_end=cur_end,
                    metadata=metadata,
                )
            )

    for u_start, u_end in units:
        u_len = u_end - u_start
        if u_len > chunk_size:
            flush()
            cur_start = cur_end = None
            chunks.extend(oversized_handler(doc, u_start, u_end, chunk_size, chunk_overlap))
            continue
        if cur_start is None:
            cur_start, cur_end = u_start, u_end
        elif (cur_end - cur_start) + u_len <= chunk_size:
            cur_end = u_end
        else:
            flush()
            cur_start, cur_end = u_start, u_end

    flush()
    return chunks


def chunk_document_structural(
    doc: Document,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Pack Markdown-aware blocks into chunks up to chunk_size.

    Three-tier packing (Design Decision 2 in the Phase 2.2 spec):
      1. Pack whole blocks (_split_blocks) up to chunk_size, non-overlapping.
      2. A block that alone exceeds chunk_size is split into sentences
         (_split_sentences) and packed at sentence granularity instead of
         going straight to character windowing.
      3. Only a single sentence that itself exceeds chunk_size falls back
         to _chunk_oversized_span (the only place chunk_overlap is used).
    """
    _validate_chunk_params(chunk_size, chunk_overlap)
    text = doc.normalized_text
    if not text.strip():
        return []

    metadata = _chunk_metadata(doc)

    def _pack_oversized_block(
        doc: Document, start: int, end: int, chunk_size: int, chunk_overlap: int
    ) -> list[Chunk]:
        sentence_units = [
            (start + s, start + e) for s, e in _split_sentences(text[start:end])
        ]
        return _pack_units(
            doc, sentence_units, chunk_size, chunk_overlap, metadata, _chunk_oversized_span
        )

    blocks = _split_blocks(text)
    return _pack_units(doc, blocks, chunk_size, chunk_overlap, metadata, _pack_oversized_block)


# ---------------------------------------------------------------------------
# Phase 2.2: semantic chunking
# ---------------------------------------------------------------------------

def chunk_document_semantic(
    doc: Document,
    embedder: Embedder,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    similarity_threshold: float = 0.5,
) -> list[Chunk]:
    """Pack sentences into chunks, splitting at topic shifts.

    Splits normalized_text into sentences (_split_sentences), embeds all of
    them in a single batch call, then packs sentences in order: a chunk
    closes (and a new one starts) when the next sentence would exceed
    chunk_size, or the cosine similarity between consecutive sentences'
    embeddings drops below similarity_threshold. Embedder vectors are
    L2-normalized, so cosine similarity is a plain dot product. A single
    sentence that itself exceeds chunk_size falls back to
    _chunk_oversized_span (the only place chunk_overlap is used).
    """
    _validate_chunk_params(chunk_size, chunk_overlap)
    text = doc.normalized_text
    if not text.strip():
        return []

    metadata = _chunk_metadata(doc)
    sentence_spans = _split_sentences(text)
    vectors = embedder.embed([text[s:e] for s, e in sentence_spans])

    chunks: list[Chunk] = []
    cur_start: int | None = None
    cur_end: int | None = None

    def flush() -> None:
        if cur_start is None:
            return
        chunk_text = text[cur_start:cur_end]
        if chunk_text.strip():
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc.doc_id, cur_start, chunk_text),
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    char_start=cur_start,
                    char_end=cur_end,
                    metadata=metadata,
                )
            )

    for i, (s_start, s_end) in enumerate(sentence_spans):
        s_len = s_end - s_start
        if s_len > chunk_size:
            flush()
            cur_start = cur_end = None
            chunks.extend(_chunk_oversized_span(doc, s_start, s_end, chunk_size, chunk_overlap))
            continue

        if cur_start is None:
            cur_start, cur_end = s_start, s_end
            continue

        similarity = float(vectors[i - 1] @ vectors[i])
        fits_budget = (cur_end - cur_start) + s_len <= chunk_size
        topic_shift = similarity < similarity_threshold
        if fits_budget and not topic_shift:
            cur_end = s_end
        else:
            flush()
            cur_start, cur_end = s_start, s_end

    flush()
    return chunks
