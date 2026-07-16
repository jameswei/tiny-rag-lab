import hashlib
import json
from pathlib import Path
import threading

from fastapi.testclient import TestClient
import numpy as np
import pytest

from tiny_rag_lab.browser_eval import (
    BUNDLED_EVALUATION_IDENTITY,
    BrowserEvaluationCancelled,
    BrowserEvaluationError,
    RetrievalConfiguration,
    run_browser_comparison,
    sha256_file,
    validate_evaluation_bundle,
)
from tiny_rag_lab.index_loader import load_index
from tiny_rag_lab.index_writer import write_index
from tiny_rag_lab.models import Chunk, Document, make_chunk_id
from tiny_rag_lab.qdrant_backend import source_vector_fingerprint
from tiny_rag_lab.web_api import create_app


class _Embedder:
    calls = []

    def __init__(self, *_args, **_kwargs):
        self.calls.append((_args, _kwargs))
        self.dim = 2

    def embed(self, texts):
        return np.asarray([
            [1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0]
            for text in texts
        ], dtype=np.float32)


def test_release_identity_matches_reviewed_repository_assets():
    index_dir = Path("assets/seed/v1/indexes/cloudflare-state-structural-v1")
    questions = Path("assets/seed/v2/corpora/cloudflare-state-v1/retrieval-questions.jsonl")
    index = load_index(index_dir)

    assert sha256_file(questions) == BUNDLED_EVALUATION_IDENTITY["questions_sha256"]
    assert sha256_file(index_dir / "chunks.jsonl") == BUNDLED_EVALUATION_IDENTITY["chunks_sha256"]
    assert sha256_file(index_dir / "embeddings.npz") == BUNDLED_EVALUATION_IDENTITY["embeddings_sha256"]
    assert source_vector_fingerprint(index) == BUNDLED_EVALUATION_IDENTITY["source_vector_fingerprint"]


def _bundle(root: Path):
    index_dir = root / "indexes" / "cloudflare-state-structural-v1"
    corpus_dir = root / "corpora" / "cloudflare-state-v1"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    documents = []
    chunks = []
    vectors = []
    for name, text, vector in [
        ("alpha.md", "alpha queue delivery details", [1.0, 0.0]),
        ("beta.md", "beta object storage details", [0.0, 1.0]),
        ("mixed.md", "alpha and beta coordination", [0.7, 0.7]),
        ("other.md", "unrelated worker runtime", [-1.0, 0.0]),
    ]:
        raw_hash = hashlib.sha256(text.encode()).hexdigest()
        document = Document(name, name, name, "markdown", text, text, raw_hash)
        chunk = Chunk(
            make_chunk_id(name, 0, text), name, text, 0, len(text),
            {"title": name, "path": name, "format": "markdown", "raw_hash": raw_hash},
        )
        documents.append(document)
        chunks.append(chunk)
        vectors.append(vector)
    write_index(
        index_dir, documents, chunks, np.asarray(vectors, dtype=np.float32),
        corpus_root=corpus_dir / "files", embedding_backend="test",
        embedding_model="test-embedder", embedding_dim=2,
        chunk_size=800, chunk_overlap=120, chunking_strategy="structural",
        source_corpus_id="cloudflare-state-v1",
    )
    index_manifest = json.loads((index_dir / "manifest.json").read_text())
    index_manifest["embedding_revision"] = "test-revision"
    (index_dir / "manifest.json").write_text(json.dumps(index_manifest))

    questions = []
    for category in ("lexical", "dense", "hybrid", "reranking"):
        for number in range(4):
            alpha = number % 2 == 0
            questions.append({
                "question_id": f"{category}-{number}",
                "category": category,
                "question": f"{'alpha' if alpha else 'beta'} details {number}",
                "gold_doc_ids": ["alpha.md" if alpha else "beta.md"],
            })
    questions_path = corpus_dir / "retrieval-questions.jsonl"
    questions_path.write_text("".join(json.dumps(item) + "\n" for item in questions))
    index = load_index(index_dir)
    manifest = {
        "questions_sha256": sha256_file(questions_path),
        "chunks_sha256": sha256_file(index_dir / "chunks.jsonl"),
        "embeddings_sha256": sha256_file(index_dir / "embeddings.npz"),
        "source_vector_fingerprint": source_vector_fingerprint(index),
        "document_count": 4,
        "chunk_count": 4,
        "distance_metric": "cosine",
        "embedding_dimension": 2,
        "embedding_model": "test-embedder",
        "embedding_revision": "test-revision",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "reranker_revision": "c5ee24cb16019beea0893ab7796b1df96625c6b8",
        "question_count": 16,
    }
    manifest_path = corpus_dir / "evaluation-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return index_dir, questions_path, manifest_path


def _validate(paths):
    manifest = json.loads(paths[2].read_text())
    return validate_evaluation_bundle(*paths, trusted_identity=manifest)


def test_bundle_validation_accepts_exact_reviewed_assets(tmp_path):
    index, questions, manifest = _validate(_bundle(tmp_path))

    assert len(index.chunks) == 4
    assert len(questions) == 16
    assert manifest["question_count"] == 16


def test_bundle_validation_rejects_fingerprint_drift(tmp_path):
    index_dir, questions_path, manifest_path = _bundle(tmp_path)
    questions_path.write_text(questions_path.read_text() + "\n")

    with pytest.raises(BrowserEvaluationError, match="questions_sha256"):
        validate_evaluation_bundle(
            index_dir, questions_path, manifest_path,
            trusted_identity=json.loads(manifest_path.read_text()),
        )


def test_bundle_validation_rejects_rewritten_artifact_and_manifest(tmp_path):
    index_dir, questions_path, manifest_path = _bundle(tmp_path)
    trusted = json.loads(manifest_path.read_text())
    chunks_path = index_dir / "chunks.jsonl"
    chunks_path.write_text(chunks_path.read_text() + "\n")
    rewritten = json.loads(manifest_path.read_text())
    rewritten["chunks_sha256"] = sha256_file(chunks_path)
    manifest_path.write_text(json.dumps(rewritten))

    with pytest.raises(BrowserEvaluationError, match="chunks_sha256"):
        validate_evaluation_bundle(
            index_dir, questions_path, manifest_path, trusted_identity=trusted,
        )


def test_comparison_keeps_metrics_and_full_evidence_separate(tmp_path):
    index, questions, _manifest = _validate(_bundle(tmp_path))

    result = run_browser_comparison(
        questions, index,
        RetrievalConfiguration("bm25", top_k=2),
        RetrievalConfiguration("dense", top_k=2),
        embedder=_Embedder(),
    )

    assert result["question_count"] == 16
    assert set(result["left"]["metrics"]) == {
        "n_questions", "hit_rate", "mrr", "context_precision", "context_recall",
    }
    assert len(result["right"]["questions"]) == 16
    assert len(result["right"]["questions"][0]["evidence"]) == 2
    assert result["right"]["questions"][0]["evidence"][0]["doc_id"] == "alpha.md"


def test_comparison_rejects_effectively_identical_sides(tmp_path):
    index, questions, _manifest = _validate(_bundle(tmp_path))

    with pytest.raises(BrowserEvaluationError, match="different"):
        run_browser_comparison(
            questions, index,
            RetrievalConfiguration("bm25", rerank_top_n=10),
            RetrievalConfiguration("bm25", rerank_top_n=40),
            embedder=_Embedder(),
        )


def test_comparison_cooperatively_cancels_between_sides(tmp_path):
    index, questions, _manifest = _validate(_bundle(tmp_path))
    boundaries = []

    with pytest.raises(BrowserEvaluationCancelled):
        run_browser_comparison(
            questions, index,
            RetrievalConfiguration("bm25"), RetrievalConfiguration("dense"),
            embedder=_Embedder(),
            checkpoint=lambda current, total, side: boundaries.append((current, total, side)) or False,
        )

    assert boundaries == [(0, 16, "left")]


def test_evaluation_api_runs_job_and_publishes_result(tmp_path, monkeypatch):
    _Embedder.calls.clear()
    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    client = TestClient(create_app(tmp_path))
    paths = _bundle(tmp_path)
    monkeypatch.setattr(
        "tiny_rag_lab.browser_eval.BUNDLED_EVALUATION_IDENTITY",
        json.loads(paths[2].read_text()),
    )

    status = client.get("/api/evaluations/status")
    response = client.post("/api/evaluations", json={
        "left": {"retriever": "bm25", "top_k": 2},
        "right": {"retriever": "dense", "top_k": 2},
    })

    assert status.json()["ready"] is True
    assert response.status_code == 202
    job_id = response.json()["id"]
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "complete"
    result = client.get(f"/api/jobs/{job_id}/result").json()
    assert result["question_count"] == 16
    assert result["bundle"]["index_id"] == "cloudflare-state-structural-v1"
    assert any(options.get("revision") == "test-revision" for _args, options in _Embedder.calls)


def test_evaluation_api_rejects_identical_configs(tmp_path, monkeypatch):
    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    client = TestClient(create_app(tmp_path))
    paths = _bundle(tmp_path)
    monkeypatch.setattr(
        "tiny_rag_lab.browser_eval.BUNDLED_EVALUATION_IDENTITY",
        json.loads(paths[2].read_text()),
    )

    response = client.post("/api/evaluations", json={
        "left": {"retriever": "bm25", "top_k": 5, "rerank_top_n": 10},
        "right": {"retriever": "bm25", "top_k": 5, "rerank_top_n": 30},
    })

    assert response.status_code == 422
    assert "different" in response.json()["detail"]


def test_evaluation_api_persists_progress_and_cancels_without_result(tmp_path, monkeypatch):
    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    scheduled = []
    monkeypatch.setattr(
        "tiny_rag_lab.web_api.BackgroundTasks.add_task",
        lambda _self, function, *args, **kwargs: scheduled.append((function, args, kwargs)),
    )
    client = TestClient(create_app(tmp_path))
    paths = _bundle(tmp_path)
    monkeypatch.setattr(
        "tiny_rag_lab.browser_eval.BUNDLED_EVALUATION_IDENTITY",
        json.loads(paths[2].read_text()),
    )
    checkpoint_reached = threading.Event()
    release = threading.Event()

    def controlled_comparison(*_args, checkpoint, **_kwargs):
        assert checkpoint(3, 16, "right") is True
        checkpoint_reached.set()
        assert release.wait(timeout=2)
        if not checkpoint(3, 16, "left"):
            raise BrowserEvaluationCancelled()
        raise AssertionError("cancellation was not accepted")

    monkeypatch.setattr("tiny_rag_lab.web_api.run_browser_comparison", controlled_comparison)
    response = client.post("/api/evaluations", json={
        "left": {"retriever": "bm25", "top_k": 2},
        "right": {"retriever": "dense", "top_k": 2},
    })
    job_id = response.json()["id"]
    active = client.get("/api/jobs/active?kind=evaluation").json()["items"]
    assert active[0]["left"] == {
        "retriever": "bm25", "top_k": 2, "reranker": "none", "rerank_top_n": 20,
    }
    assert active[0]["right"]["retriever"] == "dense"
    function, args, kwargs = scheduled.pop()
    worker = threading.Thread(target=function, args=args, kwargs=kwargs)
    worker.start()
    assert checkpoint_reached.wait(timeout=2)
    state = client.get(f"/api/jobs/{job_id}").json()
    assert state["status"] == "running"
    assert state["progress"]["current"] == 3

    assert client.post(f"/api/jobs/{job_id}/cancel").json()["status"] == "cancel_requested"
    release.set()
    worker.join(timeout=2)

    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "cancelled"
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 404


def test_evaluation_api_failure_is_terminal_and_has_no_result(tmp_path, monkeypatch):
    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    scheduled = []
    monkeypatch.setattr(
        "tiny_rag_lab.web_api.BackgroundTasks.add_task",
        lambda _self, function, *args, **kwargs: scheduled.append((function, args, kwargs)),
    )
    client = TestClient(create_app(tmp_path))
    paths = _bundle(tmp_path)
    monkeypatch.setattr(
        "tiny_rag_lab.browser_eval.BUNDLED_EVALUATION_IDENTITY",
        json.loads(paths[2].read_text()),
    )
    monkeypatch.setattr(
        "tiny_rag_lab.web_api.run_browser_comparison",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("secret detail")),
    )
    response = client.post("/api/evaluations", json={
        "left": {"retriever": "bm25"}, "right": {"retriever": "dense"},
    })
    job_id = response.json()["id"]
    function, args, kwargs = scheduled.pop()
    function(*args, **kwargs)

    state = client.get(f"/api/jobs/{job_id}").json()
    assert state["status"] == "failed"
    assert state["error"] == "Evaluation failed. Check local model readiness and the server logs, then try again."
    assert "secret detail" not in state["error"]
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 404
