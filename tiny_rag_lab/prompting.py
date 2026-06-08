"""Prompt assembly for T15.

Three public functions:

  assemble_prompt(question, results) -> str
    Builds the full LLM prompt from the spec template. The template is a
    module-level string so it is easy to read, edit, and audit.

  format_source_table(results) -> str
    Formats a human-readable source table for the CLI (chunk_id → title/path).

  PROMPT_TEMPLATE and CONTEXT_BLOCK_TEMPLATE are exposed so callers and tests
  can inspect the exact format without parsing assembled output.
"""
from __future__ import annotations

from tiny_rag_lab.models import RetrievalResult

# ---------------------------------------------------------------------------
# Templates — kept as visible module-level strings per spec.
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You are a retrieval-augmented assistant. Answer the question using only the
provided context.

If the context is insufficient, say that the provided context does not contain
enough information to answer. Do not use outside knowledge.

Cite every factual claim with the source marker for the context block that
supports it.

Question:
{question}

Context:
{context_blocks}

Answer:"""

CONTEXT_BLOCK_TEMPLATE = """\
[Source: {chunk_id}]
Title: {title}
Path: {path}

{chunk_text}"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assemble_prompt(question: str, results: list[RetrievalResult]) -> str:
    """Return the full LLM prompt for question given retrieved results.

    Each result becomes one context block in ranked order. An empty results
    list produces a valid prompt with no context blocks so the model can
    still reply that context is insufficient.
    """
    context_blocks = "\n\n".join(
        _format_context_block(r) for r in results
    )
    return PROMPT_TEMPLATE.format(
        question=question,
        context_blocks=context_blocks,
    )


def format_source_table(results: list[RetrievalResult]) -> str:
    """Return a CLI-printable source table mapping chunk IDs to titles/paths.

    Example output:
        Sources:
          [Source: abc1234567890123]  Sample Title  (docs/example.md)
          [Source: def9876543210987]  Other Title   (docs/other.md)
    """
    if not results:
        return "Sources: (none)"

    lines = ["Sources:"]
    for r in results:
        title = r.chunk.metadata.get("title", "")
        path = r.chunk.metadata.get("path", r.chunk.doc_id)
        lines.append(f"  [Source: {r.chunk.chunk_id}]  {title}  ({path})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_context_block(result: RetrievalResult) -> str:
    title = result.chunk.metadata.get("title", "")
    path = result.chunk.metadata.get("path", result.chunk.doc_id)
    return CONTEXT_BLOCK_TEMPLATE.format(
        chunk_id=result.chunk.chunk_id,
        title=title,
        path=path,
        chunk_text=result.chunk.text,
    )
