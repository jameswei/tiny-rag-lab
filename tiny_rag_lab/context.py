"""Context packing for Phase 2.1.

Controls which retrieved chunks enter the prompt by enforcing a token budget.
Token counting uses a small protocol so tests can inject FakeTokenCounter
without loading tiktoken.

Public API:

  PROMPT_OVERHEAD         module-level int; tokens reserved for prompt wrapper
  ContextPackResult       dataclass; records selected/omitted chunk_ids
  TokenCounter            Protocol; count(text) -> int
  FakeTokenCounter        char-based estimator; 4 chars ≈ 1 token
  TiktokenCounter         tiktoken-backed counter; lazy-imports on construction
  pack_context(results, budget, counter, question="") -> ContextPackResult
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tiny_rag_lab.models import RetrievalResult
from tiny_rag_lab.prompting import _format_context_block

PROMPT_OVERHEAD = 100   # token reserve for the question template wrapper


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class ContextPackResult:
    """Records which chunks were selected or dropped by the context budget.

    All fields are JSON-native so dataclasses.asdict() serialises AskTrace
    without a custom encoder.
    """
    selected: list[str]       # chunk_ids in rank order that fit the budget
    omitted: list[str]        # chunk_ids dropped to stay within budget
    estimated_tokens: int     # tokens consumed by the selected context blocks
    budget: int               # the effective budget that was applied
    counter_name: str         # e.g. "tiktoken-gpt-4o-mini" or "char"


# ---------------------------------------------------------------------------
# Token counter protocol and implementations
# ---------------------------------------------------------------------------

class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


@dataclass
class FakeTokenCounter:
    """Char-based token estimator. 4 chars ≈ 1 token (tokens_per_char=0.25)."""
    name: str = "char"
    tokens_per_char: float = 0.25

    def count(self, text: str) -> int:
        return int(len(text) * self.tokens_per_char)


class TiktokenCounter:
    """Token counter backed by tiktoken. Lazy-imports tiktoken on construction.

    Raises ImportError if tiktoken is not installed.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        import tiktoken  # raises ImportError if not installed
        self._enc = tiktoken.encoding_for_model(model)
        self.name = f"tiktoken-{model}"

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


# ---------------------------------------------------------------------------
# Core packing function
# ---------------------------------------------------------------------------

def pack_context(
    results: list[RetrievalResult],
    budget: int,
    counter: TokenCounter,
    question: str = "",
) -> ContextPackResult:
    """Greedily select chunks in rank order until the context budget is used.

    Deducts PROMPT_OVERHEAD + question tokens from budget before chunk
    selection. Each chunk block is formatted with _format_context_block (the
    same function used by assemble_prompt) so token estimates match the actual
    prompt.

    When budget == 0, all chunks are selected (unlimited).
    Raises ValueError if budget < 0.
    """
    if budget < 0:
        raise ValueError(f"context_budget must be >= 0, got {budget!r}")

    if budget == 0:
        selected = [r.chunk.chunk_id for r in results]
        total = sum(counter.count(_format_context_block(r)) for r in results)
        return ContextPackResult(
            selected=selected,
            omitted=[],
            estimated_tokens=total,
            budget=budget,
            counter_name=counter.name,
        )

    overhead = PROMPT_OVERHEAD + counter.count(question)
    remaining = budget - overhead

    selected: list[str] = []
    omitted: list[str] = []
    estimated_tokens = 0

    for result in results:
        block = _format_context_block(result)
        block_tokens = counter.count(block)
        if remaining >= block_tokens:
            selected.append(result.chunk.chunk_id)
            estimated_tokens += block_tokens
            remaining -= block_tokens
        else:
            omitted.append(result.chunk.chunk_id)

    return ContextPackResult(
        selected=selected,
        omitted=omitted,
        estimated_tokens=estimated_tokens,
        budget=budget,
        counter_name=counter.name,
    )
