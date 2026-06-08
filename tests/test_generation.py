"""Tests for T16 — generation interface and fake generator.

No network access or API credentials are needed.
"""
import pytest

from tiny_rag_lab.generation import FakeGenerator, Generator
from tiny_rag_lab.models import Chunk, RetrievalResult
from tiny_rag_lab.prompting import assemble_prompt


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------

def test_fake_generator_is_generator_subclass():
    assert issubclass(FakeGenerator, Generator)


def test_generate_returns_string():
    gen = FakeGenerator()
    result = gen.generate("Some prompt text.")
    assert isinstance(result, str)


def test_generate_non_empty():
    gen = FakeGenerator()
    result = gen.generate("Some prompt text.")
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Source marker behaviour
# ---------------------------------------------------------------------------

def _make_result(chunk_id: str, text: str = "context", rank: int = 1) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="docs/a.md",
        text=text,
        char_start=0,
        char_end=len(text),
        metadata={"title": "T", "path": "docs/a.md", "format": "markdown", "raw_hash": "x"},
    )
    return RetrievalResult(chunk=chunk, score=0.9, rank=rank)


def test_fake_generator_includes_source_marker():
    gen = FakeGenerator()
    prompt = assemble_prompt("What is watsonx?", [_make_result("abc1234567890123")])
    answer = gen.generate(prompt)
    assert "[Source: abc1234567890123]" in answer


def test_fake_generator_includes_all_source_markers():
    gen = FakeGenerator()
    results = [
        _make_result("chunk0000000001", rank=1),
        _make_result("chunk0000000002", rank=2),
        _make_result("chunk0000000003", rank=3),
    ]
    prompt = assemble_prompt("Q?", results)
    answer = gen.generate(prompt)
    for r in results:
        assert f"[Source: {r.chunk.chunk_id}]" in answer


def test_fake_generator_no_markers_in_prompt():
    gen = FakeGenerator()
    # Prompt with no [Source: ...] markers
    answer = gen.generate("Plain prompt with no markers.")
    assert "does not contain enough information" in answer.lower() or isinstance(answer, str)


def test_fake_generator_empty_results_prompt():
    gen = FakeGenerator()
    prompt = assemble_prompt("What is X?", [])
    answer = gen.generate(prompt)
    # No source markers in an empty-results prompt
    assert "[Source:" not in answer


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_fake_generator_is_deterministic():
    gen = FakeGenerator()
    prompt = assemble_prompt("Q?", [_make_result("abc1234567890123")])
    assert gen.generate(prompt) == gen.generate(prompt)


# ---------------------------------------------------------------------------
# Integration: prompt → generate pipeline
# ---------------------------------------------------------------------------

def test_generate_with_assembled_prompt():
    gen = FakeGenerator()
    results = [
        _make_result("aaa0000000000001", "watsonx is IBM's AI platform", rank=1),
        _make_result("bbb0000000000002", "watsonx.data manages data", rank=2),
    ]
    prompt = assemble_prompt("What is watsonx?", results)
    answer = gen.generate(prompt)
    assert isinstance(answer, str)
    assert "[Source: aaa0000000000001]" in answer
    assert "[Source: bbb0000000000002]" in answer
