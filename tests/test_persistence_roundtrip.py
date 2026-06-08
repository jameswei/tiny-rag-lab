"""Tests for T19 — persisted index retrieval round-trip.

This test exercises the normal fixture-corpus path:

  load documents -> chunk -> embed -> write index -> load index -> retrieve

The key assertion is that loading saved artifacts preserves the expected
top-ranked chunk, proving the persisted index is enough to reproduce retrieval.
"""
from pathlib import Path

from tiny_rag_lab.chunking import chunk_documents
from tiny_rag_lab.documents import load_documents
from tiny_rag_lab.embeddings import FakeEmbedder
from tiny_rag_lab.index_loader import LoadedIndex, load_index
from tiny_rag_lab.index_writer import write_index
from tiny_rag_lab.retrieval import retrieve


FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"


def test_loaded_index_preserves_expected_top_retrieval(tmp_path):
    docs = load_documents(FIXTURE_CORPUS)
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=100)
    target = next(c for c in chunks if c.doc_id == "with_h1.md")

    embedder = FakeEmbedder(dim=8)
    embeddings = embedder.embed([c.text for c in chunks])

    in_memory_index = LoadedIndex(
        manifest={},
        chunks=chunks,
        embeddings=embeddings,
        chunk_ids=[c.chunk_id for c in chunks],
    )
    before = retrieve(target.text, in_memory_index, embedder, top_k=1)

    index_dir = tmp_path / "index"
    write_index(
        index_dir,
        docs=docs,
        chunks=chunks,
        embeddings=embeddings,
        corpus_root=FIXTURE_CORPUS,
        embedding_backend="FakeEmbedder",
        embedding_model="fake",
        embedding_dim=8,
        chunk_size=1000,
        chunk_overlap=100,
    )

    loaded = load_index(index_dir)
    after = retrieve(target.text, loaded, embedder, top_k=1)

    assert before[0].chunk.chunk_id == target.chunk_id
    assert after[0].chunk.chunk_id == target.chunk_id
    assert after[0].chunk.doc_id == "with_h1.md"
    assert after[0].chunk.text == target.text
