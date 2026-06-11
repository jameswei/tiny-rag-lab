"""Tests for rag retrieve --trace-out (P1.7-T04).

All tests patch _make_embedder with FakeEmbedder — no model downloads or
network access required.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tiny_rag_lab.cli import build_parser, cmd_index, cmd_retrieve
from tiny_rag_lab.embeddings import FakeEmbedder

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embedder_factory(dim: int = 8):
    def _make(model_name=None):
        return FakeEmbedder(dim=dim)
    return _make


def _build_index(tmp_path):
    index_dir = tmp_path / "index"
    parser = build_parser()
    args = parser.parse_args([
        "index",
        "--corpus", str(FIXTURE_CORPUS),
        "--index-dir", str(index_dir),
        "--chunk-size", "200",
        "--chunk-overlap", "20",
    ])
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_index(args)
    return index_dir


def _retrieve_args(query, index_dir, top_k=3, retriever="dense", trace_out=None):
    parser = build_parser()
    argv = ["retrieve", query, "--index-dir", str(index_dir), "--top-k", str(top_k),
            "--retriever", retriever]
    if trace_out is not None:
        argv += ["--trace-out", str(trace_out)]
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def index_dir(tmp_path):
    return _build_index(tmp_path)


# ---------------------------------------------------------------------------
# --trace-out flag presence
# ---------------------------------------------------------------------------

def test_retrieve_help_shows_trace_out_flag(capsys):
    parser = build_parser()
    try:
        parser.parse_args(["retrieve", "--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "--trace-out" in out


def test_retrieve_without_trace_out_still_prints_output(index_dir, capsys):
    args = _retrieve_args("sample document", index_dir, top_k=2)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_retrieve(args)
    out = capsys.readouterr().out
    assert "Rank 1" in out


# ---------------------------------------------------------------------------
# --trace-out dense retriever
# ---------------------------------------------------------------------------

def test_retrieve_trace_out_creates_file(tmp_path, index_dir):
    out_path = tmp_path / "trace.json"
    args = _retrieve_args("sample document", index_dir, trace_out=out_path)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_retrieve(args)
    assert out_path.exists()


def test_retrieve_trace_out_valid_json(tmp_path, index_dir):
    out_path = tmp_path / "trace.json"
    args = _retrieve_args("sample document", index_dir, trace_out=out_path)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_retrieve(args)
    loaded = json.loads(out_path.read_text())
    assert isinstance(loaded, dict)


def test_retrieve_trace_out_top_level_fields(tmp_path, index_dir):
    out_path = tmp_path / "trace.json"
    args = _retrieve_args("sample document", index_dir, top_k=2, trace_out=out_path)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_retrieve(args)
    loaded = json.loads(out_path.read_text())
    assert loaded["retriever"] == "dense"
    assert loaded["top_k"] == 2
    assert loaded["query"] == "sample document"
    assert isinstance(loaded["chunks"], list)


def test_retrieve_trace_out_chunk_fields(tmp_path, index_dir):
    out_path = tmp_path / "trace.json"
    args = _retrieve_args("sample document", index_dir, top_k=1, trace_out=out_path)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_retrieve(args)
    loaded = json.loads(out_path.read_text())
    chunk = loaded["chunks"][0]
    for key in ("rank", "score", "chunk_id", "doc_id", "title", "path"):
        assert key in chunk, f"missing key in chunk: {key}"


def test_retrieve_trace_out_dense_latency_keys(tmp_path, index_dir):
    out_path = tmp_path / "trace.json"
    args = _retrieve_args("sample document", index_dir, trace_out=out_path)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_retrieve(args)
    loaded = json.loads(out_path.read_text())
    latency = loaded["latency_by_stage"]
    assert "load" in latency
    assert "embed" in latency
    assert "retrieve" in latency


def test_retrieve_trace_out_creates_parent_dirs(tmp_path, index_dir):
    out_path = tmp_path / "nested" / "dir" / "trace.json"
    args = _retrieve_args("sample document", index_dir, trace_out=out_path)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_retrieve(args)
    assert out_path.exists()


# ---------------------------------------------------------------------------
# --trace-out BM25 retriever
# ---------------------------------------------------------------------------

def test_retrieve_trace_out_bm25_omits_embed_key(tmp_path, index_dir):
    out_path = tmp_path / "bm25_trace.json"
    args = _retrieve_args("sample document", index_dir, retriever="bm25",
                          trace_out=out_path)
    # BM25 must not call _make_embedder
    with patch("tiny_rag_lab.cli._make_embedder",
               side_effect=RuntimeError("embedder must not be loaded for bm25")):
        cmd_retrieve(args)
    loaded = json.loads(out_path.read_text())
    assert loaded["retriever"] == "bm25"
    assert "embed" not in loaded["latency_by_stage"]
    assert "load" in loaded["latency_by_stage"]
    assert "retrieve" in loaded["latency_by_stage"]


# ---------------------------------------------------------------------------
# --trace-out hybrid retriever
# ---------------------------------------------------------------------------

def test_retrieve_trace_out_hybrid_latency_keys(tmp_path, index_dir):
    out_path = tmp_path / "hybrid_trace.json"
    args = _retrieve_args("sample document", index_dir, retriever="hybrid",
                          trace_out=out_path)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_retrieve(args)
    loaded = json.loads(out_path.read_text())
    assert loaded["retriever"] == "hybrid"
    assert "load" in loaded["latency_by_stage"]
    assert "embed" in loaded["latency_by_stage"]
    assert "retrieve" in loaded["latency_by_stage"]
