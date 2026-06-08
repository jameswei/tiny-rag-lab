"""Tests for T17 — OpenAI-compatible generator.

All tests mock the OpenAI client — no real credentials or network needed.
"""
from unittest.mock import MagicMock, patch

import pytest

from tiny_rag_lab.generation import Generator, OpenAIGenerator


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------

def test_openai_generator_is_generator_subclass():
    assert issubclass(OpenAIGenerator, Generator)


def test_default_model():
    assert OpenAIGenerator.DEFAULT_MODEL == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# generate() calls the OpenAI API correctly
# ---------------------------------------------------------------------------

def _mock_openai_response(content: str):
    """Build a minimal mock matching openai.ChatCompletion response shape."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def test_generate_returns_model_content():
    with patch("tiny_rag_lab.generation.OpenAIGenerator.__init__", return_value=None):
        gen = OpenAIGenerator.__new__(OpenAIGenerator)
        gen.model = "gpt-4o-mini"
        gen._client = MagicMock()
        gen._client.chat.completions.create.return_value = _mock_openai_response(
            "The answer is watsonx."
        )
    result = gen.generate("What is watsonx?")
    assert result == "The answer is watsonx."


def test_generate_passes_prompt_as_user_message():
    with patch("tiny_rag_lab.generation.OpenAIGenerator.__init__", return_value=None):
        gen = OpenAIGenerator.__new__(OpenAIGenerator)
        gen.model = "gpt-4o-mini"
        gen._client = MagicMock()
        gen._client.chat.completions.create.return_value = _mock_openai_response("ok")
    gen.generate("my prompt")
    call_kwargs = gen._client.chat.completions.create.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs["messages"]
    assert any(m["role"] == "user" and m["content"] == "my prompt" for m in messages)


def test_generate_passes_model_name():
    with patch("tiny_rag_lab.generation.OpenAIGenerator.__init__", return_value=None):
        gen = OpenAIGenerator.__new__(OpenAIGenerator)
        gen.model = "my-custom-model"
        gen._client = MagicMock()
        gen._client.chat.completions.create.return_value = _mock_openai_response("ok")
    gen.generate("prompt")
    call_kwargs = gen._client.chat.completions.create.call_args
    model_used = call_kwargs.kwargs.get("model") or call_kwargs.args[0]
    assert model_used == "my-custom-model"


# ---------------------------------------------------------------------------
# Constructor — credentials forwarding
# ---------------------------------------------------------------------------

def test_constructor_passes_api_key():
    with patch.dict("sys.modules", {"openai": MagicMock()}):
        import sys
        mock_openai = sys.modules["openai"]
        mock_openai.OpenAI = MagicMock(return_value=MagicMock())
        gen = OpenAIGenerator(api_key="test-key", model="gpt-4o-mini")
        mock_openai.OpenAI.assert_called_once()
        call_kwargs = mock_openai.OpenAI.call_args.kwargs
        assert call_kwargs.get("api_key") == "test-key"


def test_constructor_passes_base_url():
    with patch.dict("sys.modules", {"openai": MagicMock()}):
        import sys
        mock_openai = sys.modules["openai"]
        mock_openai.OpenAI = MagicMock(return_value=MagicMock())
        gen = OpenAIGenerator(base_url="http://localhost:11434/v1", model="llama3")
        call_kwargs = mock_openai.OpenAI.call_args.kwargs
        assert call_kwargs.get("base_url") == "http://localhost:11434/v1"


def test_constructor_uses_default_model_when_none():
    with patch.dict("sys.modules", {"openai": MagicMock()}):
        import sys
        sys.modules["openai"].OpenAI = MagicMock(return_value=MagicMock())
        gen = OpenAIGenerator()
        assert gen.model == OpenAIGenerator.DEFAULT_MODEL
