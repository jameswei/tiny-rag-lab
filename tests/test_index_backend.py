import numpy as np

from tiny_rag_lab.embeddings import FakeEmbedder
from tiny_rag_lab.index_backend import NumpyIndexBackend, backend_from_manifest
from tiny_rag_lab.index_writer import write_index
from tiny_rag_lab.models import Chunk, Document


def _index(tmp_path):
    doc = Document(
        doc_id="a.md", path="/corpus/a.md", title="A", format="markdown",
        raw_text="alpha", normalized_text="alpha", raw_hash="hash",
    )
    chunk = Chunk(
        chunk_id="a", doc_id="a.md", text="alpha", char_start=0, char_end=5,
        metadata={"title": "A", "path": "/corpus/a.md", "format": "markdown", "raw_hash": "hash"},
    )
    embedder = FakeEmbedder(dim=2)
    index_dir = tmp_path / "index"
    write_index(
        index_dir, [doc], [chunk], np.array([[1.0, 0.0]], dtype=np.float32),
        corpus_root=tmp_path, embedding_backend="fake", embedding_model="fake",
        embedding_dim=2, chunk_size=800, chunk_overlap=120,
    )
    return index_dir


def test_numpy_backend_search_has_explicit_score_semantics(tmp_path):
    index = NumpyIndexBackend().open(_index(tmp_path))
    hits = NumpyIndexBackend().search(np.array([1.0, 0.0]), index, top_k=1)
    assert hits[0].backend == "numpy"
    assert hits[0].score_semantics == "cosine_similarity[-1,1]"
    assert hits[0].result.chunk.chunk_id == "a"


def test_old_manifest_defaults_to_numpy_backend():
    assert type(backend_from_manifest({})).__name__ == "NumpyIndexBackend"
