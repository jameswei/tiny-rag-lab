"""Gated tests for OpenAIJudge and CLI factories (P2.0-T02).

These tests are skipped by default because they require:
1. The TINY_RAG_LAB_TEST_JUDGE=1 env flag to be set (prevents accidental
   LLM calls in the default test suite).
2. The openai package to be installed.

When both conditions are met and OPENAI_API_KEY is set, the real OpenAI path
is exercised. Otherwise the tests are silently skipped — no LLM call is ever
triggered by the default suite.
"""
from __future__ import annotations

import os

import pytest

# Gate 1: skip unless the env flag is explicitly set (no openai import yet).
if os.environ.get("TINY_RAG_LAB_TEST_JUDGE") != "1":
    pytest.skip(
        "TINY_RAG_LAB_TEST_JUDGE not set; set to 1 to enable real OpenAI judge tests",
        allow_module_level=True,
    )

# Gate 2: skip if the openai package is not installed.
pytest.importorskip("openai")

from tiny_rag_lab.judge import FakeJudge, OpenAIJudge  # noqa: E402
from tiny_rag_lab.cli import _make_judge, _make_generator_from_flag  # noqa: E402


# ---------------------------------------------------------------------------
# OpenAIJudge construction
# ---------------------------------------------------------------------------

def test_openai_judge_construction_has_no_side_effects():
    """__init__ must not trigger any network call or import openai at module level."""
    judge = OpenAIJudge(model="gpt-4o-mini", api_key="sk-test")
    assert judge._model == "gpt-4o-mini"
    assert judge._api_key == "sk-test"
    assert judge._base_url is None


def test_openai_judge_construction_with_base_url():
    judge = OpenAIJudge(model="gpt-4o-mini", api_key="sk-test", base_url="http://localhost:11434")
    assert judge._base_url == "http://localhost:11434"


def test_openai_judge_name():
    judge = OpenAIJudge(model="gpt-4o-mini", api_key="sk-test")
    assert judge.name == "openai"


def test_openai_judge_default_model_constant():
    assert OpenAIJudge.DEFAULT_MODEL == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# _make_judge factory
# ---------------------------------------------------------------------------

def test_make_judge_none_returns_none():
    result = _make_judge("none", None, None, None)
    assert result is None


def test_make_judge_fake_returns_fake_judge():
    result = _make_judge("fake", None, None, None)
    assert isinstance(result, FakeJudge)


def test_make_judge_openai_raises_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _make_judge("openai", None, None, None)


def test_make_judge_openai_uses_env_var_when_no_flag(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test")
    result = _make_judge("openai", None, None, None)
    assert isinstance(result, OpenAIJudge)
    assert result._api_key == "sk-env-test"


def test_make_judge_openai_flag_takes_priority_over_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test")
    result = _make_judge("openai", None, "sk-flag-test", None)
    assert result._api_key == "sk-flag-test"


def test_make_judge_openai_defaults_model_to_default_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    result = _make_judge("openai", None, None, None)
    assert result._model == OpenAIJudge.DEFAULT_MODEL


def test_make_judge_openai_uses_explicit_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    result = _make_judge("openai", "gpt-4o", None, None)
    assert result._model == "gpt-4o"


def test_make_judge_unknown_raises():
    with pytest.raises(ValueError, match="Unknown"):
        _make_judge("unknown", None, None, None)


# ---------------------------------------------------------------------------
# _make_generator_from_flag factory
# ---------------------------------------------------------------------------

def test_make_generator_from_flag_fake_returns_fake_generator():
    from tiny_rag_lab.generation import FakeGenerator
    result = _make_generator_from_flag("fake", None)
    assert isinstance(result, FakeGenerator)


# ---------------------------------------------------------------------------
# openai not imported at module level in judge.py
# ---------------------------------------------------------------------------

def test_import_judge_does_not_import_openai():
    import sys
    # openai may be loaded from above tests — we verify the module-level import
    # guard by checking that a fresh import of judge would not trigger openai
    # at module level. Since we can't easily unload modules, we confirm
    # that openai is NOT in the top-level imports of judge.py by inspection.
    import importlib, inspect
    import tiny_rag_lab.judge as judge_mod
    src = inspect.getsource(judge_mod)
    # Top-level (non-indented) import of openai would look like "^import openai"
    import re
    assert not re.search(r"^import openai", src, re.MULTILINE), \
        "openai must not be imported at module level in judge.py"
    assert not re.search(r"^from openai", src, re.MULTILINE), \
        "openai must not be imported at module level in judge.py"


# ---------------------------------------------------------------------------
# Live OpenAI call (only runs when OPENAI_API_KEY is set)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping live API call",
)
def test_openai_judge_live_call_returns_valid_verdict():
    judge = OpenAIJudge(
        model=os.environ.get("OPENAI_MODEL", OpenAIJudge.DEFAULT_MODEL),
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    verdict = judge.judge(
        query="What is RAG?",
        context=["RAG stands for Retrieval-Augmented Generation."],
        answer="RAG is Retrieval-Augmented Generation.",
    )
    assert 0.0 <= verdict.faithfulness <= 1.0
    assert 0.0 <= verdict.answer_relevance <= 1.0
    assert 0.0 <= verdict.citation_support <= 1.0
    assert verdict.answer_correctness is None  # no reference_answer supplied
    assert verdict.judge_name == "openai"
    assert verdict.latency > 0.0
