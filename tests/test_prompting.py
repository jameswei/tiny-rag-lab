"""Tests for T15 — prompt assembly.

Verifies that assemble_prompt produces a prompt matching the spec template:
question, context blocks with [Source: chunk_id] / Title / Path / text, and
all required instruction lines. Also tests format_source_table output.
"""
import pytest

from tiny_rag_lab.models import Chunk, RetrievalResult
from tiny_rag_lab.prompting import (
    CONTEXT_BLOCK_TEMPLATE,
    PROMPT_TEMPLATE,
    assemble_prompt,
    format_source_table,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(chunk_id: str, text: str, rank: int = 1, score: float = 0.9,
                 title: str = "Test Title", path: str = "docs/test.md") -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="docs/test.md",
        text=text,
        char_start=0,
        char_end=len(text),
        metadata={"title": title, "path": path, "format": "markdown", "raw_hash": "x"},
    )
    return RetrievalResult(chunk=chunk, score=score, rank=rank)


# ---------------------------------------------------------------------------
# Template exposure
# ---------------------------------------------------------------------------

def test_prompt_template_is_string():
    assert isinstance(PROMPT_TEMPLATE, str)


def test_context_block_template_is_string():
    assert isinstance(CONTEXT_BLOCK_TEMPLATE, str)


def test_prompt_template_has_question_placeholder():
    assert "{question}" in PROMPT_TEMPLATE


def test_prompt_template_has_context_blocks_placeholder():
    assert "{context_blocks}" in PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# assemble_prompt — required instructions
# ---------------------------------------------------------------------------

def test_prompt_contains_question():
    r = _make_result("aaa0000000000001", "Some context text.")
    prompt = assemble_prompt("What is watsonx?", [r])
    assert "What is watsonx?" in prompt


def test_prompt_instructs_context_only():
    r = _make_result("aaa0000000000001", "context")
    prompt = assemble_prompt("Q?", [r])
    assert "only" in prompt.lower()
    assert "context" in prompt.lower()


def test_prompt_instructs_insufficient_context():
    r = _make_result("aaa0000000000001", "context")
    prompt = assemble_prompt("Q?", [r])
    # Must tell the model to say when context is not enough
    assert "insufficient" in prompt.lower() or "does not contain" in prompt.lower()


def test_prompt_instructs_cite_sources():
    r = _make_result("aaa0000000000001", "context")
    prompt = assemble_prompt("Q?", [r])
    assert "cite" in prompt.lower() or "source" in prompt.lower()


def test_prompt_ends_with_answer_label():
    r = _make_result("aaa0000000000001", "context")
    prompt = assemble_prompt("Q?", [r])
    assert prompt.strip().endswith("Answer:")


# ---------------------------------------------------------------------------
# assemble_prompt — context block content
# ---------------------------------------------------------------------------

def test_prompt_contains_source_marker():
    r = _make_result("abc1234567890123", "context text")
    prompt = assemble_prompt("Q?", [r])
    assert "[Source: abc1234567890123]" in prompt


def test_prompt_contains_chunk_text():
    r = _make_result("aaa0000000000001", "watsonx is an AI platform")
    prompt = assemble_prompt("Q?", [r])
    assert "watsonx is an AI platform" in prompt


def test_prompt_contains_title():
    r = _make_result("aaa0000000000001", "text", title="IBM watsonx Overview")
    prompt = assemble_prompt("Q?", [r])
    assert "IBM watsonx Overview" in prompt


def test_prompt_contains_path():
    r = _make_result("aaa0000000000001", "text", path="docs/overview.md")
    prompt = assemble_prompt("Q?", [r])
    assert "docs/overview.md" in prompt


def test_prompt_contains_all_chunk_ids():
    results = [
        _make_result("chunk0000000001", "first context", rank=1),
        _make_result("chunk0000000002", "second context", rank=2),
        _make_result("chunk0000000003", "third context", rank=3),
    ]
    prompt = assemble_prompt("Multi-chunk question?", results)
    for r in results:
        assert f"[Source: {r.chunk.chunk_id}]" in prompt


def test_prompt_contains_all_chunk_texts():
    results = [
        _make_result("cid1", "alpha context", rank=1),
        _make_result("cid2", "beta context", rank=2),
    ]
    prompt = assemble_prompt("Q?", results)
    assert "alpha context" in prompt
    assert "beta context" in prompt


def test_prompt_preserves_rank_order():
    results = [
        _make_result("first_chunk_id__", "first chunk text", rank=1),
        _make_result("second_chunk_id_", "second chunk text", rank=2),
    ]
    prompt = assemble_prompt("Q?", results)
    assert prompt.index("first_chunk_id__") < prompt.index("second_chunk_id_")


# ---------------------------------------------------------------------------
# assemble_prompt — empty results
# ---------------------------------------------------------------------------

def test_empty_results_still_produces_prompt():
    prompt = assemble_prompt("What is watsonx?", [])
    assert "What is watsonx?" in prompt
    assert "Answer:" in prompt


def test_empty_results_has_no_source_marker():
    prompt = assemble_prompt("Q?", [])
    assert "[Source:" not in prompt


# ---------------------------------------------------------------------------
# format_source_table
# ---------------------------------------------------------------------------

def test_source_table_contains_chunk_id():
    r = _make_result("abc1234567890123", "text")
    table = format_source_table([r])
    assert "abc1234567890123" in table


def test_source_table_contains_title():
    r = _make_result("aaa0000000000001", "text", title="IBM watsonx Overview")
    table = format_source_table([r])
    assert "IBM watsonx Overview" in table


def test_source_table_contains_path():
    r = _make_result("aaa0000000000001", "text", path="docs/overview.md")
    table = format_source_table([r])
    assert "docs/overview.md" in table


def test_source_table_has_sources_header():
    r = _make_result("aaa0000000000001", "text")
    table = format_source_table([r])
    assert table.startswith("Sources:")


def test_source_table_lists_all_results():
    results = [
        _make_result("cid1", "t", rank=1, title="Doc A", path="docs/a.md"),
        _make_result("cid2", "t", rank=2, title="Doc B", path="docs/b.md"),
        _make_result("cid3", "t", rank=3, title="Doc C", path="docs/c.md"),
    ]
    table = format_source_table(results)
    for r in results:
        assert r.chunk.chunk_id in table


def test_source_table_empty_results():
    table = format_source_table([])
    assert "none" in table.lower()
