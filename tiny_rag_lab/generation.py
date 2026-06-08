"""Generation interface and fake generator for T16.

Generator is the single interface the ask pipeline depends on.
FakeGenerator is self-contained and deterministic — safe for all tests.

T17 adds the real OpenAI-compatible generator.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod


class Generator(ABC):
    """Interface contract for all generation backends.

    generate() takes a fully assembled prompt string and returns the model's
    answer as a string. The pipeline must not depend on which backend is used.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate an answer for prompt. Returns a plain text string."""
        ...


class FakeGenerator(Generator):
    """Deterministic test generator that echoes source markers from the prompt.

    Scans the prompt for all [Source: <chunk_id>] markers and echoes each one
    in the answer. This lets tests verify that the pipeline correctly passes
    source markers through without requiring any network or API credentials.

    Example answer:
        Based on the provided context: [Source: abc1234567890123]
        The answer is derived from the retrieved documents. [Source: def9876543210987]
    """

    # Pattern that matches [Source: <any 16-char hex chunk_id>]
    _SOURCE_RE = re.compile(r"\[Source: ([^\]]+)\]")

    def generate(self, prompt: str) -> str:
        markers = self._SOURCE_RE.findall(prompt)
        if not markers:
            return (
                "Based on the provided context, "
                "the context does not contain enough information to answer."
            )
        cited = " ".join(f"[Source: {m}]" for m in markers)
        return f"Based on the provided context: {cited}"
