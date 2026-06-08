"""Generation interface, fake generator (T16), and real OpenAI backend (T17).

Generator is the single interface the ask pipeline depends on.
FakeGenerator is self-contained and deterministic — safe for all tests.
OpenAIGenerator calls any OpenAI-compatible chat completions endpoint.
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


class OpenAIGenerator(Generator):
    """Real generation backend using any OpenAI-compatible chat API (T17).

    Credentials and endpoint are injected at construction time so the same
    class works with OpenAI, Azure OpenAI, Ollama, or any other compatible
    service. Tests mock the underlying client — no real credentials needed.

    Configuration priority (highest first):
      1. Constructor arguments
      2. Standard environment variables read by the openai SDK:
         OPENAI_API_KEY, OPENAI_BASE_URL
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI  # deferred import — fails gracefully if not installed

        self.model = model or self.DEFAULT_MODEL
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
