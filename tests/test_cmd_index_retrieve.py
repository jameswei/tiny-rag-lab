"""Tests for T13 (rag index) and T14 (rag retrieve).

All tests patch _make_embedder to return a FakeEmbedder so no model
download or network access is required. The fixture corpus at
tests/fixtures/corpus/ provides real documents.
"""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tiny_rag_lab.cli import build_parser, cmd_index, cmd_retrieve
from tiny_rag_lab.embeddings import FakeEmbedder
from tiny_rag_lab.models import Chunk, Document

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embedder_factory(dim: int = 8):
    """Returns a callable that ignores model_name and yields a FakeEmbedder."""
    def _make(model_name=None):
        return FakeEmbedder(dim=dim)
    return _make


def _index_args(
    corpus,
    index_dir,
    chunk_size=200,
    chunk_overlap=20,
    chunking_strategy="fixed_character",
    semantic_similarity_threshold=0.5,
):
    parser = build_parser()
    return parser.parse_args([
        "index",
        "--corpus", str(corpus),
        "--index-dir", str(index_dir),
        "--chunk-size", str(chunk_size),
        "--chunk-overlap", str(chunk_overlap),
        "--chunking-strategy", chunking_strategy,
        "--semantic-similarity-threshold", str(semantic_similarity_threshold),
    ])


def _retrieve_args(query, index_dir, top_k=3, retriever="dense"):
    parser = build_parser()
    return parser.parse_args([
        "retrieve", query,
        "--index-dir", str(index_dir),
        "--top-k", str(top_k),
        "--retriever", retriever,
    ])


# ---------------------------------------------------------------------------
# T13 — rag index
# ---------------------------------------------------------------------------

@pytest.fixture()
def built_index(tmp_path):
    """Run cmd_index with FakeEmbedder; return (index_dir, args)."""
    index_dir = tmp_path / "index"
    args = _index_args(FIXTURE_CORPUS, index_dir)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_index(args)
    return index_dir, args


def test_index_creates_manifest(built_index):
    index_dir, _ = built_index
    assert (index_dir / "manifest.json").exists()


def test_index_creates_chunks_jsonl(built_index):
    index_dir, _ = built_index
    assert (index_dir / "chunks.jsonl").exists()


def test_index_creates_embeddings_npz(built_index):
    index_dir, _ = built_index
    assert (index_dir / "embeddings.npz").exists()


def test_index_manifest_document_count(built_index):
    index_dir, _ = built_index
    manifest = json.loads((index_dir / "manifest.json").read_text())
    assert manifest["document_count"] > 0


def test_index_manifest_chunk_count_matches_jsonl(built_index):
    index_dir, _ = built_index
    manifest = json.loads((index_dir / "manifest.json").read_text())
    lines = [l for l in (index_dir / "chunks.jsonl").read_text().splitlines() if l.strip()]
    assert manifest["chunk_count"] == len(lines)


def test_index_manifest_chunk_params(built_index):
    index_dir, args = built_index
    manifest = json.loads((index_dir / "manifest.json").read_text())
    assert manifest["chunk_size"] == args.chunk_size
    assert manifest["chunk_overlap"] == args.chunk_overlap
    assert manifest["chunking_strategy"] == "fixed_character"
    assert manifest["chunking_params"] == {}


def test_index_embeddings_shape(built_index):
    index_dir, _ = built_index
    manifest = json.loads((index_dir / "manifest.json").read_text())
    data = np.load(index_dir / "embeddings.npz", allow_pickle=True)
    assert data["embeddings"].shape[0] == manifest["chunk_count"]


def test_index_embedding_backend_in_manifest(built_index):
    index_dir, _ = built_index
    manifest = json.loads((index_dir / "manifest.json").read_text())
    # FakeEmbedder is used in tests; real runs use SentenceTransformerEmbedder
    assert manifest["embedding_backend"] in ("FakeEmbedder", "SentenceTransformerEmbedder")


def test_index_prints_summary(tmp_path, capsys):
    index_dir = tmp_path / "index"
    args = _index_args(FIXTURE_CORPUS, index_dir)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_index(args)
    out = capsys.readouterr().out
    assert "Documents" in out
    assert "Chunks" in out
    assert "Model" in out
    assert "strategy=fixed_character" in out


def test_index_creates_parent_dirs(tmp_path):
    index_dir = tmp_path / "deep" / "nested" / "index"
    args = _index_args(FIXTURE_CORPUS, index_dir)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_index(args)
    assert (index_dir / "manifest.json").exists()


def test_index_structural_manifest_records_strategy(tmp_path):
    index_dir = tmp_path / "index"
    args = _index_args(FIXTURE_CORPUS, index_dir, chunking_strategy="structural")
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_index(args)
    manifest = json.loads((index_dir / "manifest.json").read_text())
    assert manifest["chunking_strategy"] == "structural"
    assert manifest["chunking_params"] == {}


def test_index_semantic_manifest_records_strategy_and_threshold(tmp_path):
    index_dir = tmp_path / "index"
    args = _index_args(
        FIXTURE_CORPUS,
        index_dir,
        chunking_strategy="semantic",
        semantic_similarity_threshold=0.7,
    )
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_index(args)
    manifest = json.loads((index_dir / "manifest.json").read_text())
    assert manifest["chunking_strategy"] == "semantic"
    assert manifest["chunking_params"] == {"similarity_threshold": 0.7}


def test_index_help_shows_chunking_flags(capsys):
    parser = build_parser()
    try:
        parser.parse_args(["index", "--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "--chunking-strategy" in out
    assert "--semantic-similarity-threshold" in out


def _make_doc(text: str, doc_id: str = "docs/test.md") -> Document:
    return Document(
        doc_id=doc_id,
        path=f"/corpus/{doc_id}",
        title="Test",
        format="markdown",
        raw_text=text,
        normalized_text=text,
        raw_hash="deadbeef",
    )


def _make_chunk(doc: Document) -> Chunk:
    return Chunk(
        chunk_id="abc123abc123abcd",
        doc_id=doc.doc_id,
        text=doc.normalized_text,
        char_start=0,
        char_end=len(doc.normalized_text),
        metadata={
            "title": doc.title,
            "path": doc.path,
            "format": doc.format,
            "raw_hash": doc.raw_hash,
        },
    )


def test_index_constructs_embedder_after_chunking_for_fixed_character(tmp_path):
    index_dir = tmp_path / "index"
    args = _index_args(FIXTURE_CORPUS, index_dir, chunking_strategy="fixed_character")
    events = []
    doc = _make_doc("Sentence one. Sentence two.")

    def fake_chunk_documents_with_strategy(docs, **kwargs):
        events.append(("chunking", kwargs["strategy"], kwargs["embedder"]))
        return [_make_chunk(doc)]

    def fake_make_embedder(model_name=None):
        events.append(("embedder", model_name))
        return FakeEmbedder(dim=8)

    with (
        patch("tiny_rag_lab.documents.load_documents", return_value=[doc]),
        patch(
            "tiny_rag_lab.chunking.chunk_documents_with_strategy",
            side_effect=fake_chunk_documents_with_strategy,
        ),
        patch("tiny_rag_lab.cli._make_embedder", side_effect=fake_make_embedder),
    ):
        cmd_index(args)

    assert events[0] == ("chunking", "fixed_character", None)
    assert events[1] == ("embedder", None)


def test_index_constructs_embedder_after_chunking_for_structural(tmp_path):
    index_dir = tmp_path / "index"
    args = _index_args(FIXTURE_CORPUS, index_dir, chunking_strategy="structural")
    events = []
    doc = _make_doc("Sentence one. Sentence two.")

    def fake_chunk_documents_with_strategy(docs, **kwargs):
        events.append(("chunking", kwargs["strategy"], kwargs["embedder"]))
        return [_make_chunk(doc)]

    def fake_make_embedder(model_name=None):
        events.append(("embedder", model_name))
        return FakeEmbedder(dim=8)

    with (
        patch("tiny_rag_lab.documents.load_documents", return_value=[doc]),
        patch(
            "tiny_rag_lab.chunking.chunk_documents_with_strategy",
            side_effect=fake_chunk_documents_with_strategy,
        ),
        patch("tiny_rag_lab.cli._make_embedder", side_effect=fake_make_embedder),
    ):
        cmd_index(args)

    assert events[0] == ("chunking", "structural", None)
    assert events[1] == ("embedder", None)


def test_index_constructs_embedder_before_chunking_for_semantic(tmp_path):
    index_dir = tmp_path / "index"
    args = _index_args(
        FIXTURE_CORPUS,
        index_dir,
        chunking_strategy="semantic",
        semantic_similarity_threshold=0.7,
    )
    events = []
    doc = _make_doc("Sentence one. Sentence two.")

    def fake_chunk_documents_with_strategy(docs, **kwargs):
        events.append(("chunking", kwargs["strategy"], type(kwargs["embedder"]).__name__))
        assert kwargs["similarity_threshold"] == 0.7
        return [_make_chunk(doc)]

    def fake_make_embedder(model_name=None):
        events.append(("embedder", model_name))
        return FakeEmbedder(dim=8)

    with (
        patch("tiny_rag_lab.documents.load_documents", return_value=[doc]),
        patch(
            "tiny_rag_lab.chunking.chunk_documents_with_strategy",
            side_effect=fake_chunk_documents_with_strategy,
        ),
        patch("tiny_rag_lab.cli._make_embedder", side_effect=fake_make_embedder),
    ):
        cmd_index(args)

    assert events[0] == ("embedder", None)
    assert events[1] == ("chunking", "semantic", "FakeEmbedder")


# ---------------------------------------------------------------------------
# T14 — rag retrieve
# ---------------------------------------------------------------------------

@pytest.fixture()
def retrieve_setup(tmp_path):
    """Build an index then return (index_dir, embedder) for retrieve tests."""
    index_dir = tmp_path / "index"
    args = _index_args(FIXTURE_CORPUS, index_dir)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_index(args)
    return index_dir


def test_retrieve_prints_results(retrieve_setup, capsys):
    args = _retrieve_args("sample document", retrieve_setup, top_k=2)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_retrieve(args)
    out = capsys.readouterr().out
    assert "Rank 1" in out


def test_retrieve_output_contains_chunk_id(retrieve_setup, capsys):
    args = _retrieve_args("sample document", retrieve_setup, top_k=1)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_retrieve(args)
    out = capsys.readouterr().out
    assert "chunk_id=" in out


def test_retrieve_output_contains_score(retrieve_setup, capsys):
    args = _retrieve_args("sample document", retrieve_setup, top_k=1)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_retrieve(args)
    out = capsys.readouterr().out
    assert "score=" in out


def test_retrieve_output_contains_title(retrieve_setup, capsys):
    args = _retrieve_args("sample document", retrieve_setup, top_k=1)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_retrieve(args)
    out = capsys.readouterr().out
    assert "title" in out


def test_retrieve_output_contains_path(retrieve_setup, capsys):
    args = _retrieve_args("sample document", retrieve_setup, top_k=1)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_retrieve(args)
    out = capsys.readouterr().out
    assert "path" in out


def test_retrieve_respects_top_k(retrieve_setup, capsys):
    args = _retrieve_args("sample document", retrieve_setup, top_k=2)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_retrieve(args)
    out = capsys.readouterr().out
    assert "Rank 1" in out
    assert "Rank 2" in out
    assert "Rank 3" not in out


def test_retrieve_no_results_message(tmp_path, capsys):
    # Build an empty index (no docs → no chunks → no results possible)
    empty_corpus = tmp_path / "empty_corpus"
    empty_corpus.mkdir()
    index_dir = tmp_path / "index"
    args_idx = _index_args(empty_corpus, index_dir)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_index(args_idx)
    args_ret = _retrieve_args("anything", index_dir, top_k=5)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_retrieve(args_ret)
    out = capsys.readouterr().out
    assert "No results" in out


def test_retrieve_returns_results_from_semantic_chunked_index(tmp_path, capsys):
    index_dir = tmp_path / "index"
    args_idx = _index_args(
        FIXTURE_CORPUS,
        index_dir,
        chunking_strategy="semantic",
        semantic_similarity_threshold=0.7,
    )
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_index(args_idx)

    args_ret = _retrieve_args("sample document", index_dir, top_k=2)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_retrieve(args_ret)

    out = capsys.readouterr().out
    assert "Rank 1" in out


# ---------------------------------------------------------------------------
# T03 — --retriever flag
# ---------------------------------------------------------------------------

def test_retrieve_help_shows_retriever_flag(capsys):
    parser = build_parser()
    try:
        parser.parse_args(["retrieve", "--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "--retriever" in out


def test_retrieve_bm25_returns_ranked_results(retrieve_setup, capsys):
    args = _retrieve_args("sample document", retrieve_setup, top_k=2, retriever="bm25")
    # Patch _make_embedder to raise — BM25 path must never call it.
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=RuntimeError("embedder must not be loaded for bm25")):
        cmd_retrieve(args)
    out = capsys.readouterr().out
    assert "Rank 1" in out
    assert "chunk_id=" in out


def test_retrieve_hybrid_returns_ranked_results(retrieve_setup, capsys):
    args = _retrieve_args("sample document", retrieve_setup, top_k=2, retriever="hybrid")
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_retrieve(args)
    out = capsys.readouterr().out
    assert "Rank 1" in out
    assert "chunk_id=" in out


def test_retrieve_invalid_retriever_exits_nonzero():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["retrieve", "query", "--retriever", "foo"])
    assert exc_info.value.code != 0


def test_retrieve_default_retriever_is_dense():
    parser = build_parser()
    args = parser.parse_args(["retrieve", "query"])
    assert args.retriever == "dense"
