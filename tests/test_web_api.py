from fastapi.testclient import TestClient
import numpy as np

from tiny_rag_lab.web_api import create_app


class _Embedder:
    def __init__(self, model_name="test-embedder", **_kwargs):
        self.model_name = model_name
        self.dim = 2

    def embed(self, texts):
        return np.array([[1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0] for text in texts], dtype=np.float32)


def test_health_and_provider_status_are_non_secret(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/health").json()["status"] == "ok"
    status = client.get("/api/provider-status").json()
    assert "api_key" not in status


def test_starter_run_is_an_offline_replay_artifact(tmp_path):
    run = TestClient(create_app(tmp_path)).get("/api/starter-run").json()
    assert run["mode"] == "replay"
    assert run["trace"]["query"]
    assert run["evidence"][0]["text"]


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
    assert run["evidence"][0]["text"] == "# Alpha\nalpha evidence"
    assert client.get(f"/api/runs/{run['run_id']}").json()["run_id"] == run["run_id"]


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
