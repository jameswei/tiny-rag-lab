"""Tests for T08 — embedding interface and fake embedder."""
import numpy as np
import pytest

from tiny_rag_lab.embeddings import Embedder, FakeEmbedder


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

def test_fake_embedder_is_embedder_subclass():
    assert issubclass(FakeEmbedder, Embedder)


def test_embedder_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Embedder()


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------

def test_embed_returns_correct_shape():
    embedder = FakeEmbedder(dim=16)
    result = embedder.embed(["hello", "world", "foo"])
    assert result.shape == (3, 16)


def test_embed_returns_float32():
    embedder = FakeEmbedder(dim=8)
    result = embedder.embed(["test"])
    assert result.dtype == np.float32


def test_embed_single_text():
    embedder = FakeEmbedder(dim=8)
    result = embedder.embed(["single"])
    assert result.shape == (1, 8)


def test_embed_empty_list():
    embedder = FakeEmbedder(dim=8)
    result = embedder.embed([])
    assert result.shape == (0, 8)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_embed_is_deterministic():
    embedder = FakeEmbedder(dim=16)
    texts = ["retrieval augmented generation", "vector similarity search"]
    a = embedder.embed(texts)
    b = embedder.embed(texts)
    np.testing.assert_array_equal(a, b)


def test_embed_single_consistent_with_batch():
    embedder = FakeEmbedder(dim=16)
    texts = ["alpha", "beta", "gamma"]
    batch = embedder.embed(texts)
    for i, text in enumerate(texts):
        single = embedder.embed([text])
        np.testing.assert_array_equal(single[0], batch[i])


# ---------------------------------------------------------------------------
# Unit vectors
# ---------------------------------------------------------------------------

def test_embed_produces_unit_vectors():
    embedder = FakeEmbedder(dim=32)
    texts = ["a", "b", "c", "longer text with more words"]
    result = embedder.embed(texts)
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Distinctness — different texts produce different vectors
# ---------------------------------------------------------------------------

def test_different_texts_produce_different_vectors():
    embedder = FakeEmbedder(dim=16)
    a = embedder.embed(["watsonx.ai overview"])
    b = embedder.embed(["getting started with watsonx.data"])
    assert not np.allclose(a, b)


# ---------------------------------------------------------------------------
# Fixture retrieval: closest text retrieves correctly
# ---------------------------------------------------------------------------

def test_fake_embedder_enables_fixture_retrieval():
    """Show that the fake embedder supports the deterministic fixture retrieval
    acceptance criterion: embed a query, find nearest neighbour by dot product.
    """
    embedder = FakeEmbedder(dim=32)

    corpus = [
        "watsonx.ai is IBM's AI platform",
        "watsonx.data stores data for AI workloads",
        "watsonx.governance manages AI lifecycle",
    ]
    query = "watsonx.ai is IBM's AI platform"  # exact match

    corpus_vecs = embedder.embed(corpus)   # (3, 32)
    query_vec = embedder.embed([query])    # (1, 32)

    scores = corpus_vecs @ query_vec[0]    # dot product = cosine (unit vecs)
    top = int(np.argmax(scores))
    assert top == 0  # exact match should score highest
