"""Tests for T09 — SentenceTransformerEmbedder.

These tests require the sentence-transformers model weights to be cached
locally. Per agent-guidelines, Phase 1 tests must not require HuggingFace
model downloads, so the entire module is skipped when the model is
unavailable (no network, no cache).

To run these tests manually after downloading the model:
    uv run pytest tests/test_sentence_transformer_embedder.py -v
"""
import numpy as np
import pytest

from tiny_rag_lab.embeddings import SentenceTransformerEmbedder


@pytest.fixture(scope="module")
def embedder():
    """Load the real embedder once per test module; skip if model not cached locally."""
    try:
        return SentenceTransformerEmbedder(local_files_only=True)
    except Exception as exc:
        pytest.skip(f"SentenceTransformerEmbedder not available locally: {exc}")


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_is_embedder_subclass():
    from tiny_rag_lab.embeddings import Embedder
    assert issubclass(SentenceTransformerEmbedder, Embedder)


def test_default_model_name():
    assert SentenceTransformerEmbedder.DEFAULT_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Shape, dtype, dimension
# ---------------------------------------------------------------------------

def test_embed_shape(embedder):
    result = embedder.embed(["hello world", "watsonx overview"])
    assert result.shape == (2, embedder.dim)


def test_embed_dtype_float32(embedder):
    result = embedder.embed(["test sentence"])
    assert result.dtype == np.float32


def test_dim_is_384(embedder):
    # all-MiniLM-L6-v2 produces 384-dimensional embeddings
    assert embedder.dim == 384


def test_embed_single_text(embedder):
    result = embedder.embed(["single"])
    assert result.shape == (1, 384)


def test_embed_empty_list(embedder):
    result = embedder.embed([])
    assert result.shape == (0, embedder.dim)


# ---------------------------------------------------------------------------
# Unit vectors
# ---------------------------------------------------------------------------

def test_embed_produces_unit_vectors(embedder):
    texts = [
        "What is watsonx.ai?",
        "IBM data platform overview",
        "retrieval augmented generation",
    ]
    result = embedder.embed(texts)
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_embed_is_deterministic(embedder):
    texts = ["determinism check one", "determinism check two"]
    a = embedder.embed(texts)
    b = embedder.embed(texts)
    np.testing.assert_array_equal(a, b)


def test_single_consistent_with_batch(embedder):
    texts = ["alpha sentence", "beta sentence", "gamma sentence"]
    batch = embedder.embed(texts)
    for i, text in enumerate(texts):
        single = embedder.embed([text])
        np.testing.assert_allclose(single[0], batch[i], atol=1e-6)


# ---------------------------------------------------------------------------
# Semantic retrieval smoke test
# ---------------------------------------------------------------------------

def test_similar_texts_score_higher_than_unrelated(embedder):
    """Semantically related texts should have higher cosine similarity
    than unrelated ones — basic sanity check for the real model."""
    query = embedder.embed(["What is IBM watsonx?"])
    related = embedder.embed(["IBM watsonx is an AI platform"])
    unrelated = embedder.embed(["The weather in Paris today"])

    score_related = float(np.dot(query[0], related[0]))
    score_unrelated = float(np.dot(query[0], unrelated[0]))

    assert score_related > score_unrelated, (
        f"Expected related ({score_related:.3f}) > unrelated ({score_unrelated:.3f})"
    )
