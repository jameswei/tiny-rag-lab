"""Tests for T18 — rag ask CLI command.

Patches both _make_embedder and _make_generator with fake backends so no
model download or API credentials are needed.
"""
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from tiny_rag_lab.cli import build_parser, cmd_ask, cmd_index
from tiny_rag_lab.embeddings import FakeEmbedder
from tiny_rag_lab.generation import FakeGenerator

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embedder_factory(dim: int = 8):
    def _make(model_name=None):
        return FakeEmbedder(dim=dim)
    return _make


def _fake_generator_factory():
    def _make(args):
        return FakeGenerator()
    return _make


def _index_args(corpus, index_dir):
    return build_parser().parse_args([
        "index", "--corpus", str(corpus),
        "--index-dir", str(index_dir),
        "--chunk-size", "200", "--chunk-overlap", "20",
    ])


def _ask_args(query, index_dir, top_k=3):
    return build_parser().parse_args([
        "ask", query,
        "--index-dir", str(index_dir),
        "--top-k", str(top_k),
    ])


@pytest.fixture()
def ask_setup(tmp_path):
    """Build an index with fake embedder, return index_dir."""
    index_dir = tmp_path / "index"
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_index(_index_args(FIXTURE_CORPUS, index_dir))
    return index_dir


# ---------------------------------------------------------------------------
# T18 — cmd_ask output
# ---------------------------------------------------------------------------

def test_ask_prints_answer(ask_setup, capsys):
    args = _ask_args("sample document", ask_setup)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()):
        cmd_ask(args)
    out = capsys.readouterr().out
    assert len(out.strip()) > 0


def test_ask_prints_source_table(ask_setup, capsys):
    args = _ask_args("sample document", ask_setup)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()):
        cmd_ask(args)
    out = capsys.readouterr().out
    assert "Sources:" in out


def test_ask_prints_timings(ask_setup, capsys):
    args = _ask_args("sample document", ask_setup)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()):
        cmd_ask(args)
    out = capsys.readouterr().out
    assert "Timings:" in out
    assert "embed=" in out
    assert "retrieve=" in out
    assert "generate=" in out


def test_ask_output_contains_source_markers(ask_setup, capsys):
    args = _ask_args("sample document", ask_setup)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()):
        cmd_ask(args)
    out = capsys.readouterr().out
    assert "[Source:" in out


def test_ask_respects_top_k(ask_setup, capsys):
    # top_k=1 means one source marker in the answer and one row in the source table.
    args = _ask_args("sample document", ask_setup, top_k=1)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()):
        cmd_ask(args)
    out = capsys.readouterr().out
    answer_section, rest = out.split("\n\nSources:", maxsplit=1)
    source_table, _timings = rest.split("\n\nTimings:", maxsplit=1)
    assert len(re.findall(r"\[Source:", answer_section)) == 1
    assert len(re.findall(r"\[Source:", source_table)) == 1


# ---------------------------------------------------------------------------
# Parser — new flags on rag ask
# ---------------------------------------------------------------------------

def test_ask_parser_has_model_flag():
    args = build_parser().parse_args(["ask", "Q?", "--model", "gpt-4o"])
    assert args.model == "gpt-4o"


def test_ask_parser_has_api_key_flag():
    args = build_parser().parse_args(["ask", "Q?", "--api-key", "sk-test"])
    assert args.api_key == "sk-test"


def test_ask_parser_has_base_url_flag():
    args = build_parser().parse_args(["ask", "Q?", "--base-url", "http://localhost/v1"])
    assert args.base_url == "http://localhost/v1"


def test_ask_parser_defaults_are_none():
    args = build_parser().parse_args(["ask", "Q?"])
    assert args.model is None
    assert args.api_key is None
    assert args.base_url is None


# ---------------------------------------------------------------------------
# Existing test still passes (rag ask --help)
# ---------------------------------------------------------------------------

def test_ask_help():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["ask", "--help"])
    assert exc.value.code == 0
