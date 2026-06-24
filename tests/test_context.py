"""Tests for tiny_rag_lab/context.py — P2.1-T01 contracts.

Covers: FakeTokenCounter, TiktokenCounter, ContextPackResult serialization,
and pack_context budget enforcement and block-format alignment.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from tiny_rag_lab.models import Chunk, RetrievalResult
from tiny_rag_lab.context import (
    PROMPT_OVERHEAD,
    ContextPackResult,
    FakeTokenCounter,
    pack_context,
)
from tiny_rag_lab.prompting import _format_context_block, CONTEXT_BLOCK_TEMPLATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    chunk_id: str,
    text: str,
    rank: int = 1,
    score: float = 0.9,
    title: str = "Test Title",
    path: str = "docs/test.md",
) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="docs/test.md",
        text=text,
        char_start=0,
        char_end=len(text),
        metadata={"title": title, "path": path, "format": "markdown", "raw_hash": "x"},
    )
    return RetrievalResult(chunk=chunk, score=score, rank=rank)


def _three_results() -> list[RetrievalResult]:
    return [
        _make_result("chunk0000000001", "Alpha context text for testing." * 4, rank=1),
        _make_result("chunk0000000002", "Beta context text for testing." * 4, rank=2),
        _make_result("chunk0000000003", "Gamma context text for testing." * 4, rank=3),
    ]


# ---------------------------------------------------------------------------
# FakeTokenCounter
# ---------------------------------------------------------------------------

def test_fake_counter_count_returns_floor_of_quarter_length():
    c = FakeTokenCounter()
    assert c.count("1234") == 1        # 4 * 0.25 = 1
    assert c.count("12345678") == 2    # 8 * 0.25 = 2
    assert c.count("abc") == 0         # 3 * 0.25 = 0 (floor)


def test_fake_counter_default_name():
    assert FakeTokenCounter().name == "char"


def test_fake_counter_is_deterministic():
    c = FakeTokenCounter()
    text = "repeatability check"
    assert c.count(text) == c.count(text)


def test_fake_counter_empty_string():
    assert FakeTokenCounter().count("") == 0


def test_fake_counter_custom_ratio():
    c = FakeTokenCounter(tokens_per_char=0.5)
    assert c.count("1234") == 2


# ---------------------------------------------------------------------------
# ContextPackResult serialization
# ---------------------------------------------------------------------------

def test_context_pack_result_asdict_all_json_native():
    result = ContextPackResult(
        selected=["abc", "def"],
        omitted=["ghi"],
        estimated_tokens=120,
        budget=500,
        counter_name="char",
    )
    d = dataclasses.asdict(result)
    serialized = json.dumps(d)
    parsed = json.loads(serialized)
    assert parsed["selected"] == ["abc", "def"]
    assert parsed["omitted"] == ["ghi"]
    assert parsed["estimated_tokens"] == 120
    assert parsed["budget"] == 500
    assert parsed["counter_name"] == "char"


def test_context_pack_result_empty_lists_serialize():
    result = ContextPackResult(
        selected=[],
        omitted=[],
        estimated_tokens=0,
        budget=8192,
        counter_name="char",
    )
    d = dataclasses.asdict(result)
    serialized = json.dumps(d)
    parsed = json.loads(serialized)
    assert parsed["selected"] == []
    assert parsed["omitted"] == []


# ---------------------------------------------------------------------------
# pack_context — budget validation
# ---------------------------------------------------------------------------

def test_pack_context_raises_on_negative_budget():
    c = FakeTokenCounter()
    results = _three_results()
    with pytest.raises(ValueError, match="context_budget"):
        pack_context(results, budget=-1, counter=c)


def test_pack_context_raises_on_negative_budget_minus_large():
    c = FakeTokenCounter()
    with pytest.raises(ValueError):
        pack_context([], budget=-100, counter=c)


# ---------------------------------------------------------------------------
# pack_context — budget=0 means unlimited
# ---------------------------------------------------------------------------

def test_pack_context_zero_budget_selects_all():
    c = FakeTokenCounter()
    results = _three_results()
    pack = pack_context(results, budget=0, counter=c)
    assert pack.selected == [r.chunk.chunk_id for r in results]
    assert pack.omitted == []
    assert pack.budget == 0


def test_pack_context_very_large_budget_selects_all():
    c = FakeTokenCounter()
    results = _three_results()
    pack = pack_context(results, budget=999_999, counter=c)
    assert pack.selected == [r.chunk.chunk_id for r in results]
    assert pack.omitted == []


# ---------------------------------------------------------------------------
# pack_context — tight budget omits lower-ranked chunks
# ---------------------------------------------------------------------------

def test_pack_context_tight_budget_omits_last_chunks():
    c = FakeTokenCounter()
    results = _three_results()

    # Compute how many tokens each block costs
    block_tokens = [c.count(_format_context_block(r)) for r in results]
    # Budget allows only the first block after overhead
    question = "short question"
    overhead = PROMPT_OVERHEAD + c.count(question)
    budget = overhead + block_tokens[0]  # exactly fits first chunk only

    pack = pack_context(results, budget=budget, counter=c, question=question)

    assert pack.selected == [results[0].chunk.chunk_id]
    assert pack.omitted == [results[1].chunk.chunk_id, results[2].chunk.chunk_id]
    assert pack.estimated_tokens == block_tokens[0]


def test_pack_context_tight_budget_omits_chunk_ids_in_order():
    c = FakeTokenCounter()
    results = _three_results()
    block_tokens = [c.count(_format_context_block(r)) for r in results]
    question = "q"
    overhead = PROMPT_OVERHEAD + c.count(question)
    # Budget fits first two chunks
    budget = overhead + block_tokens[0] + block_tokens[1]

    pack = pack_context(results, budget=budget, counter=c, question=question)

    assert pack.selected == [results[0].chunk.chunk_id, results[1].chunk.chunk_id]
    assert pack.omitted == [results[2].chunk.chunk_id]


def test_pack_context_empty_results():
    c = FakeTokenCounter()
    pack = pack_context([], budget=500, counter=c)
    assert pack.selected == []
    assert pack.omitted == []
    assert pack.estimated_tokens == 0


# ---------------------------------------------------------------------------
# pack_context — question token deduction
# ---------------------------------------------------------------------------

def test_pack_context_long_question_leaves_less_budget():
    c = FakeTokenCounter()
    results = _three_results()
    block_tokens = [c.count(_format_context_block(r)) for r in results]

    short_q = "short"
    long_q = "x" * 400  # 400 chars → 100 tokens with FakeTokenCounter

    # Budget that fits the first chunk with a short question but not with a
    # long question (long_q adds 100 tokens of overhead that eats into space)
    budget = PROMPT_OVERHEAD + c.count(short_q) + block_tokens[0]

    pack_short = pack_context(results, budget=budget, counter=c, question=short_q)
    pack_long = pack_context(results, budget=budget, counter=c, question=long_q)

    # Short question: first chunk fits
    assert results[0].chunk.chunk_id in pack_short.selected
    # Long question: no chunks fit (overhead alone exceeds remaining budget)
    assert pack_long.selected == []
    assert len(pack_long.omitted) == len(results)


# ---------------------------------------------------------------------------
# pack_context — block format matches CONTEXT_BLOCK_TEMPLATE
# ---------------------------------------------------------------------------

def test_pack_context_block_format_matches_context_block_template():
    """Token count in ContextPackResult must match what CONTEXT_BLOCK_TEMPLATE
    produces, so budget estimates align with the actual assembled prompt."""
    c = FakeTokenCounter()
    r = _make_result("abc1234567890123", "some chunk text here", title="Doc A", path="docs/a.md")

    # What pack_context will count (via _format_context_block)
    expected_block = _format_context_block(r)
    expected_tokens = c.count(expected_block)

    # Verify it matches the raw template expansion
    manual_block = CONTEXT_BLOCK_TEMPLATE.format(
        chunk_id="abc1234567890123",
        title="Doc A",
        path="docs/a.md",
        chunk_text="some chunk text here",
    )
    assert expected_block == manual_block, "block format must match CONTEXT_BLOCK_TEMPLATE"

    # pack_context with a budget that fits exactly this one chunk
    question = ""
    overhead = PROMPT_OVERHEAD + c.count(question)
    budget = overhead + expected_tokens

    pack = pack_context([r], budget=budget, counter=c, question=question)
    assert pack.selected == ["abc1234567890123"]
    assert pack.estimated_tokens == expected_tokens


# ---------------------------------------------------------------------------
# pack_context — counter_name in result
# ---------------------------------------------------------------------------

def test_pack_context_result_carries_counter_name():
    c = FakeTokenCounter()
    pack = pack_context([], budget=500, counter=c)
    assert pack.counter_name == "char"


def test_pack_context_result_carries_budget():
    c = FakeTokenCounter()
    pack = pack_context([], budget=1234, counter=c)
    assert pack.budget == 1234


# ---------------------------------------------------------------------------
# TiktokenCounter (gated — only runs when tiktoken is installed)
# ---------------------------------------------------------------------------

def test_tiktoken_counter_name_starts_with_tiktoken():
    pytest.importorskip("tiktoken")
    from tiny_rag_lab.context import TiktokenCounter
    c = TiktokenCounter()
    assert c.name.startswith("tiktoken-")


def test_tiktoken_counter_count_returns_positive_int_for_nonempty():
    pytest.importorskip("tiktoken")
    from tiny_rag_lab.context import TiktokenCounter
    c = TiktokenCounter()
    result = c.count("hello world")
    assert isinstance(result, int)
    assert result > 0


def test_tiktoken_counter_count_empty_string():
    pytest.importorskip("tiktoken")
    from tiny_rag_lab.context import TiktokenCounter
    c = TiktokenCounter()
    assert c.count("") == 0


def test_tiktoken_counter_default_model():
    pytest.importorskip("tiktoken")
    from tiny_rag_lab.context import TiktokenCounter
    c = TiktokenCounter()
    assert "gpt-4o-mini" in c.name


def test_tiktoken_counter_no_api_call_on_construction():
    pytest.importorskip("tiktoken")
    from tiny_rag_lab.context import TiktokenCounter
    c = TiktokenCounter()
    assert c is not None
