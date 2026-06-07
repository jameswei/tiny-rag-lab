"""Document loading and text normalization.

T05: load_document, load_documents
T06: normalize_text

Normalization is called inside load_document so that Document.normalized_text
is always populated. Keeping both functions here avoids a circular import and
makes the text-preparation pipeline visible in one place.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from tiny_rag_lab.models import Document

_SUPPORTED_SUFFIXES = {".md", ".txt"}


# ---------------------------------------------------------------------------
# T06 — text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Return a normalized copy of text.

    Rules (from the phase spec):
    1. Normalize line endings to \\n.
    2. Strip trailing whitespace from each line.
    3. Collapse runs of more than two consecutive blank lines to two.
    4. Preserve Markdown headings and punctuation.
    """
    # 1. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split("\n")]

    # 3. Collapse runs of >2 blank lines to exactly 2
    result: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                result.append(line)
        else:
            blank_run = 0
            result.append(line)

    return "\n".join(result)


# ---------------------------------------------------------------------------
# T05 — document loading
# ---------------------------------------------------------------------------

def _extract_title(raw_text: str, path: Path, fmt: str) -> str:
    """Return the first H1 heading for Markdown, or the filename stem."""
    if fmt == "markdown":
        for line in raw_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
    return path.stem


def load_document(path: Path, corpus_root: Path) -> Document:
    """Load a single .md or .txt file and return a Document.

    doc_id is the POSIX-style path relative to corpus_root, e.g.
    "docs/example.md". normalized_text is populated by normalize_text.
    raw_hash is the SHA-256 of raw_text (before normalization).
    """
    raw_text = path.read_text(encoding="utf-8")
    fmt = "markdown" if path.suffix.lower() == ".md" else "text"
    return Document(
        doc_id=path.relative_to(corpus_root).as_posix(),
        path=str(path),
        title=_extract_title(raw_text, path, fmt),
        format=fmt,
        raw_text=raw_text,
        normalized_text=normalize_text(raw_text),
        raw_hash=hashlib.sha256(raw_text.encode()).hexdigest(),
    )


def load_documents(corpus_root: Path) -> list[Document]:
    """Recursively load all .md and .txt files under corpus_root.

    Files are returned in sorted order (by path) for determinism.
    """
    paths = sorted(
        p for p in corpus_root.rglob("*")
        if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    )
    return [load_document(p, corpus_root) for p in paths]
