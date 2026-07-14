import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np

from tiny_rag_lab.index_writer import write_index
from tiny_rag_lab.models import Chunk, Document, make_chunk_id
from tiny_rag_lab.web_api import create_app


class _Embedder:
    def __init__(self, model_name="test-embedder", **_kwargs):
        self.model_name = model_name
        self.dim = 2

    def embed(self, texts):
        return np.array([[1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0] for text in texts], dtype=np.float32)


def _seeded_client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "TINY_RAG_LAB_SEED_DIR",
        str(Path("assets/seed/v1").resolve()),
    )
    return TestClient(create_app(tmp_path))


def _write_catalog_index(root, *, source_corpus_id="watsonxdocsqa-v1", index_id="catalog-index"):
    question = json.loads((root / "corpora" / "watsonxdocsqa-v1" / "questions.jsonl").read_text().splitlines()[0])
    text = "catalog evidence"
    doc_id = question["gold_doc_ids"][0]
    document = Document(
        doc_id=doc_id, path=f"/data/corpora/watsonxdocsqa-v1/files/{doc_id}",
        title="Catalog evidence", format="markdown", raw_text=text,
        normalized_text=text, raw_hash=hashlib.sha256(text.encode()).hexdigest(),
    )
    chunk = Chunk(
        chunk_id=make_chunk_id(doc_id, 0, text), doc_id=doc_id, text=text,
        char_start=0, char_end=len(text),
        metadata={"title": document.title, "path": document.path, "format": "markdown", "raw_hash": document.raw_hash},
    )
    write_index(
        root / "indexes" / index_id, [document], [chunk], np.array([[0.0, 1.0]], dtype=np.float32),
        corpus_root=root / "corpora" / "watsonxdocsqa-v1" / "files",
        embedding_backend="test", embedding_model="test-embedder", embedding_dim=2,
        chunk_size=800, chunk_overlap=120, source_corpus_id=source_corpus_id,
    )


def test_health_and_provider_status_are_non_secret(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/health").json()["status"] == "ok"
    status = client.get("/api/provider-status").json()
    assert "api_key" not in status


def test_backend_status_reports_numpy_and_optional_qdrant_readiness(tmp_path, monkeypatch):
    monkeypatch.setattr("tiny_rag_lab.web_api.qdrant_is_available", lambda _url: True)

    items = TestClient(create_app(tmp_path)).get("/api/backends").json()["items"]

    assert items == [
        {"id": "numpy", "available": True},
        {"id": "qdrant", "available": True},
    ]


def test_starter_run_is_an_offline_replay_artifact(tmp_path):
    run = TestClient(create_app(tmp_path)).get("/api/starter-run").json()
    assert run["mode"] == "replay"
    assert run["trace"]["query"]
    assert run["evidence"][0]["text"]


def test_saved_lessons_are_complete_offline_artifacts(tmp_path, monkeypatch):
    client = _seeded_client(tmp_path, monkeypatch)

    listing = client.get("/api/lessons").json()["items"]
    assert [item["id"] for item in listing] == [
        "cloudflare-do-coordinator-v1", "cloudflare-queues-retries-v1",
        "cloudflare-kv-r2-choice-v1", "cloudflare-workflows-resume-v1",
    ]
    lesson = client.get("/api/lessons/cloudflare-do-coordinator-v1").json()
    assert lesson["lesson"]["answer_provenance"] == "recorded_lesson_result"
    assert lesson["lesson"]["source_snapshot"]["source_revision"] == "3dcb728cb29f4239e08ba894f0a40650d51ba4f6"
    assert len(lesson["lesson"]["source_snapshot"]["corpus_digest"]) == 64
    assert len(lesson["lesson"]["source_snapshot"]["index_digest"]) == 64
    run = lesson["run"]
    assert run["mode"] == "saved_lesson"
    assert run["query_vector"] and run["evidence"]
    assert run["trace"]["prompt"] and run["trace"]["context_pack"]
    assert run["config"]["answer_provenance"] == "recorded_lesson_result"
    for item in listing:
        saved = client.get(f"/api/lessons/{item['id']}").json()
        lesson = saved["lesson"]
        run = saved["run"]
        selected = {entry["chunk_id"]: entry for entry in run["evidence"] if entry["selected_for_context"]}
        assert selected and run["trace"]["context_pack"]["omitted"]
        supporting = set(lesson["answer_supporting_chunk_ids"])
        assert supporting <= set(selected)
        assert {selected[chunk_id]["doc_id"] for chunk_id in supporting} >= set(lesson["required_supporting_document_ids"])
        assert set(run["trace"]["citations"]) <= set(selected)


def test_provider_test_accepts_environment_configuration_and_disables_sdk_retries(tmp_path, monkeypatch):
    captured = {}

    class _Generator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def generate(self, _prompt, **_kwargs):
            return "OK"

    monkeypatch.setenv("OPENAI_BASE_URL", "http://local-provider/v1")
    monkeypatch.setattr("tiny_rag_lab.web_api.OpenAIGenerator", _Generator)
    response = TestClient(create_app(tmp_path)).post("/api/provider/test")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "message": "Provider connection verified"}
    assert captured["timeout"] == 10.0
    assert captured["max_retries"] == 0


def test_catalog_hides_gold_until_validated_run_and_persists_check(tmp_path, monkeypatch):
    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    client = _seeded_client(tmp_path, monkeypatch)
    listing = client.get("/api/corpora/watsonxdocsqa-v1/questions")
    assert listing.status_code == 200
    questions = listing.json()["items"]
    assert len(questions) == 75
    assert all(set(item) == {"id", "question", "featured"} for item in questions)

    _write_catalog_index(tmp_path)
    response = client.post("/api/runs/retrieve", json={
        "index_id": "catalog-index", "catalog_question_id": "train_1", "query": "browser text is ignored",
    })
    assert response.status_code == 201
    run = response.json()
    assert run["trace"]["query"] == questions[0]["question"]
    assert run["catalog_check"]["question_id"] == "train_1"
    assert run["catalog_check"]["hit"] is True
    assert client.get(f"/api/runs/{run['run_id']}").json()["catalog_check"] == run["catalog_check"]
    # Restart reads the durable source_corpus_id from manifest rather than a
    # browser-only association.
    restarted = TestClient(create_app(tmp_path))
    assert restarted.get(f"/api/runs/{run['run_id']}").json()["catalog_check"]["hit"] is True


def test_catalog_question_rejects_wrong_or_legacy_index_but_free_query_remains_valid(tmp_path, monkeypatch):
    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    client = _seeded_client(tmp_path, monkeypatch)
    _write_catalog_index(tmp_path, source_corpus_id="another-corpus", index_id="wrong-index")
    _write_catalog_index(tmp_path, source_corpus_id=None, index_id="legacy-index")

    assert client.post("/api/runs/retrieve", json={"index_id": "wrong-index", "catalog_question_id": "train_1"}).status_code == 409
    assert client.post("/api/runs/retrieve", json={"index_id": "legacy-index", "catalog_question_id": "train_1"}).status_code == 409
    free = client.post("/api/runs/retrieve", json={"index_id": "legacy-index", "query": "alpha"})
    assert free.status_code == 201
    assert free.json()["catalog_check"] is None


def test_slim_model_gate_keeps_bm25_available_but_blocks_dense_and_hybrid(tmp_path, monkeypatch):
    class _MissingEmbedder:
        def __init__(self, **_kwargs):
            raise OSError("model is not cached")

    client = _seeded_client(tmp_path, monkeypatch)
    _write_catalog_index(tmp_path, index_id="slim-index")
    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _MissingEmbedder)

    assert client.post("/api/runs/retrieve", json={"index_id": "slim-index", "query": "catalog", "retriever": "bm25"}).status_code == 201
    for retriever in ("dense", "hybrid"):
        response = client.post("/api/runs/retrieve", json={"index_id": "slim-index", "query": "catalog", "retriever": retriever})
        assert response.status_code == 409
        assert "Download the default embedding model" in response.json()["detail"]


def test_failure_lessons_include_localized_explanation_and_trace_artifacts(tmp_path):
    lessons = TestClient(create_app(tmp_path)).get("/api/failure-lessons").json()["items"]
    lesson = next(item for item in lessons if item["id"] == "missing-evidence")
    assert lesson["explanation"]["en"]
    assert lesson["explanation"]["zh"]
    for side in (lesson["baseline"], lesson["intervention"]):
        assert side["trace"]["evidence"]
        assert "context_pack" in side["trace"]
        assert "answer" in side["trace"]
        assert "outcome_label" in side["trace"]


def test_restart_marks_inflight_job_failed_and_retryable(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "index-stale.json").write_text('{"id":"index-stale","status":"running","kind":"index"}')
    client = TestClient(create_app(tmp_path))
    job = client.get("/api/jobs/index-stale").json()
    assert job["status"] == "failed"
    assert "restarted" in job["error"]


def test_live_ask_requires_an_effective_provider_not_an_empty_override(tmp_path):
    client = TestClient(create_app(tmp_path))
    payload = {"index_id": "unused", "query": "question"}
    assert client.post("/api/runs/ask", json=payload).status_code == 409
    assert client.post("/api/runs/ask", json={**payload, "provider": {}}).status_code == 409


def test_upload_accepts_small_markdown_corpus(tmp_path):
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/corpora/upload",
        data={"name": "Notes"},
        files=[("files", ("notes.md", b"# Notes\nhello", "text/markdown"))],
    )
    assert response.status_code == 201
    corpus = response.json()
    assert corpus["kind"] == "custom"
    assert client.get("/api/corpora").json()["items"] == [corpus]


def test_upload_rejects_unsupported_file_type(tmp_path):
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/corpora/upload",
        files=[("files", ("notes.pdf", b"not really pdf", "application/pdf"))],
    )
    assert response.status_code == 422


def test_upload_rejects_duplicate_filenames(tmp_path):
    response = TestClient(create_app(tmp_path)).post(
        "/api/corpora/upload",
        files=[
            ("files", ("same.md", b"one", "text/markdown")),
            ("files", ("same.md", b"two", "text/markdown")),
        ],
    )
    assert response.status_code == 422


def test_upload_rejects_oversized_corpus_without_preserving_partial_files(tmp_path, monkeypatch):
    monkeypatch.setattr("tiny_rag_lab.web_api.MAX_UPLOAD_BYTES", 5)
    client = TestClient(create_app(tmp_path))

    response = client.post(
        "/api/corpora/upload",
        files=[("files", ("notes.md", b"six-bytes", "text/markdown"))],
    )

    assert response.status_code == 422
    assert list((tmp_path / "corpora").iterdir()) == []


def test_job_admission_rejects_a_visible_queued_job(tmp_path, monkeypatch):
    class _MissingEmbedder:
        def __init__(self, **_kwargs):
            raise OSError("model is not cached")

    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _MissingEmbedder)
    client = TestClient(create_app(tmp_path))
    (tmp_path / "jobs" / "already-queued.json").write_text(
        '{"id":"already-queued","status":"queued","kind":"index"}'
    )

    response = client.post("/api/models/default/download")

    assert response.status_code == 409
    assert "already-queued" in response.json()["detail"]


def test_index_and_retrieve_persist_a_replayable_run(tmp_path, monkeypatch):
    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    client = TestClient(create_app(tmp_path))
    corpus = client.post(
        "/api/corpora/upload",
        files=[("files", ("alpha.md", b"# Alpha\nalpha evidence", "text/markdown"))],
    ).json()

    job = client.post("/api/indexes", json={"corpus_id": corpus["id"]}).json()
    completed = client.get(f"/api/jobs/{job['id']}").json()
    assert completed["status"] == "complete"

    run = client.post("/api/runs/retrieve", json={
        "index_id": completed["index_id"], "query": "alpha", "top_k": 1,
    }).json()
    assert run["index"]["manifest"]["index_backend"] == "numpy"
    assert run["index"]["manifest"]["source_corpus_id"] == corpus["id"]
    assert run["evidence"][0]["text"] == "# Alpha\nalpha evidence"
    assert client.get(f"/api/runs/{run['run_id']}").json()["run_id"] == run["run_id"]
    inspection = client.get(f"/api/indexes/{completed['index_id']}").json()
    assert inspection["document_count"] == 1
    assert inspection["chunk_count"] == 1


def test_qdrant_search_failure_is_a_non_secret_service_error(tmp_path, monkeypatch):
    from tiny_rag_lab.qdrant_backend import QdrantBackendError

    class _UnavailableBackend:
        name = "qdrant"
        score_semantics = "qdrant_cosine_similarity"

        def search(self, *_args, **_kwargs):
            raise QdrantBackendError("Qdrant search is unavailable. Start the optional Qdrant profile or rebuild this index.")

    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    monkeypatch.setattr("tiny_rag_lab.web_api.backend_from_manifest", lambda *_args, **_kwargs: _UnavailableBackend())
    client = TestClient(create_app(tmp_path))
    corpus = client.post(
        "/api/corpora/upload",
        files=[("files", ("alpha.md", b"alpha evidence", "text/markdown"))],
    ).json()
    index_job = client.post("/api/indexes", json={"corpus_id": corpus["id"]}).json()
    index_id = client.get(f"/api/jobs/{index_job['id']}").json()["index_id"]

    response = client.post("/api/runs/retrieve", json={"index_id": index_id, "query": "alpha"})

    assert response.status_code == 503
    assert "Qdrant search is unavailable" in response.json()["detail"]


def test_index_requires_explicit_model_download(tmp_path, monkeypatch):
    class _MissingEmbedder:
        def __init__(self, **_kwargs):
            raise OSError("model is not cached")

    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _MissingEmbedder)
    client = TestClient(create_app(tmp_path))
    corpus = client.post(
        "/api/corpora/upload",
        files=[("files", ("notes.md", b"# Notes", "text/markdown"))],
    ).json()
    assert client.get("/api/models/default/status").json()["ready"] is False
    assert client.post("/api/indexes", json={"corpus_id": corpus["id"]}).status_code == 409


def test_qdrant_index_requires_a_ready_local_service(tmp_path, monkeypatch):
    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    monkeypatch.setattr("tiny_rag_lab.web_api.qdrant_is_available", lambda _url: False)
    client = TestClient(create_app(tmp_path))
    corpus = client.post(
        "/api/corpora/upload",
        files=[("files", ("notes.md", b"# Notes", "text/markdown"))],
    ).json()

    response = client.post("/api/indexes", json={"corpus_id": corpus["id"], "index_backend": "qdrant"})

    assert response.status_code == 409
    assert response.json()["detail"] == "Qdrant is not ready. Start the optional Qdrant service, then try building again."


def test_ask_trace_preserves_context_omissions_for_replay(tmp_path, monkeypatch):
    class _Generator:
        def __init__(self, **_kwargs):
            pass

        def generate(self, _prompt):
            return "Grounded answer [Source: alpha.md]"

    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    monkeypatch.setattr("tiny_rag_lab.web_api.OpenAIGenerator", _Generator)
    client = TestClient(create_app(tmp_path))
    corpus = client.post(
        "/api/corpora/upload",
        files=[
            ("files", ("alpha.md", b"alpha " * 100, "text/markdown")),
            ("files", ("beta.md", b"beta " * 100, "text/markdown")),
        ],
    ).json()
    index_job = client.post("/api/indexes", json={"corpus_id": corpus["id"]}).json()
    index_id = client.get(f"/api/jobs/{index_job['id']}").json()["index_id"]

    run = client.post("/api/runs/ask", json={
        "index_id": index_id, "query": "alpha", "top_k": 2, "context_budget": 300,
        "provider": {"base_url": "http://local-provider/v1"},
    }).json()
    assert run["created_at"]
    assert run["config"] == {"retriever": "dense", "top_k": 2, "context_budget": 300}
    assert len(run["evidence"]) == 2
    assert {item["selected_for_context"] for item in run["evidence"]} == {True, False}
    assert run["trace"]["context_pack"]["omitted"]


def test_live_ask_generation_failure_is_saved_as_a_replayable_error(tmp_path, monkeypatch):
    class _FailingGenerator:
        def __init__(self, **_kwargs):
            pass

        def generate(self, _prompt):
            raise ConnectionError("provider unavailable")

    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    monkeypatch.setattr("tiny_rag_lab.web_api.OpenAIGenerator", _FailingGenerator)
    client = TestClient(create_app(tmp_path))
    corpus = client.post(
        "/api/corpora/upload",
        files=[("files", ("alpha.md", b"alpha evidence", "text/markdown"))],
    ).json()
    index_job = client.post("/api/indexes", json={"corpus_id": corpus["id"]}).json()
    index_id = client.get(f"/api/jobs/{index_job['id']}").json()["index_id"]

    response = client.post("/api/runs/ask", json={
        "index_id": index_id, "query": "alpha",
        "provider": {"base_url": "http://local-provider/v1"},
    })

    assert response.status_code == 201
    run = response.json()
    assert run["error"] == "Live generation failed. Check your provider settings and try again."
    assert run["trace"]["prompt"]
    assert client.get(f"/api/runs/{run['run_id']}").json()["error"] == run["error"]


def test_index_failure_is_logged_and_exposed_as_a_non_secret_job_error(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    monkeypatch.setattr("tiny_rag_lab.web_api.load_documents", lambda _path: (_ for _ in ()).throw(ValueError("bad corpus")))
    client = TestClient(create_app(tmp_path))
    corpus = client.post(
        "/api/corpora/upload",
        files=[("files", ("alpha.md", b"alpha evidence", "text/markdown"))],
    ).json()

    job = client.post("/api/indexes", json={"corpus_id": corpus["id"]}).json()
    state = client.get(f"/api/jobs/{job['id']}").json()

    assert state["status"] == "failed"
    assert state["error"] == "Indexing failed. Check the local server logs and try again."
    assert "Index job" in caplog.text
