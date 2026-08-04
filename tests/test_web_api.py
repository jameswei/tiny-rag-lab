import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from tiny_rag_lab.index_writer import write_index
from tiny_rag_lab.jobs import LocalJobStore
from tiny_rag_lab.models import Chunk, Document, make_chunk_id
from tiny_rag_lab.qdrant_backend import QdrantBackendError, QdrantIndexBackend
from tiny_rag_lab.reranker import FakeReranker
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


def _write_retrieval_materials(
    root, *, category="lexical", question="What does catalog evidence explain?",
):
    target = root / "corpora" / "cloudflare-state-v1"
    target.mkdir(parents=True, exist_ok=True)
    item = {
        "question_id": "cf-lex-test",
        "category": category,
        "question": question,
        "gold_doc_ids": ["doc.md"],
        "teaching_note": {"en": "Inspect the term.", "zh": "检查词项。"},
        "expected_observation": {"en": "Exact terms contribute.", "zh": "精确词项会贡献分数。"},
    }
    (target / "retrieval-questions.jsonl").write_text(json.dumps(item) + "\n")
    return item


def _write_teaching_index(root, *, index_backend="numpy", backend_identity=None):
    documents = []
    chunks = []
    vectors = []
    for path, text, vector in [
        ("r2/how-r2-works.md", "catalog evidence for object storage", [0.0, 1.0]),
        ("queues/retries.md", "alpha queue retry evidence", [1.0, 0.0]),
    ]:
        raw_hash = hashlib.sha256(text.encode()).hexdigest()
        document = Document(
            doc_id=path, path=path, title=Path(path).stem.replace("-", " ").title(),
            format="markdown", raw_text=text, normalized_text=text, raw_hash=raw_hash,
        )
        chunk = Chunk(
            chunk_id=make_chunk_id(path, 0, text), doc_id=path, text=text,
            char_start=0, char_end=len(text),
            metadata={"title": document.title, "path": path, "format": "markdown", "raw_hash": raw_hash},
        )
        documents.append(document)
        chunks.append(chunk)
        vectors.append(vector)
    write_index(
        root / "indexes" / "cloudflare-state-structural-v1",
        documents, chunks, np.asarray(vectors, dtype=np.float32),
        corpus_root=root / "corpora" / "cloudflare-state-v1" / "files",
        embedding_backend="test", embedding_model="test-embedder", embedding_dim=2,
        chunk_size=800, chunk_overlap=120, source_corpus_id="cloudflare-state-v1",
        index_backend=index_backend, backend_identity=backend_identity,
    )


def test_health_and_provider_status_are_non_secret(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/health").json()["status"] == "ok"
    status = client.get("/api/provider-status").json()
    assert "api_key" not in status


def test_provider_status_reports_env_base_url_and_model_without_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-never-appear")
    client = TestClient(create_app(tmp_path))
    status = client.get("/api/provider-status").json()
    assert status["base_url"] == "https://provider.example/v1"
    assert status["model"] == "test-model"
    assert status["api_key_configured"] is True
    assert "api_key" not in status
    assert "sk-should-never-appear" not in json.dumps(status)


def test_backend_status_reports_numpy_and_optional_qdrant_readiness(tmp_path, monkeypatch):
    monkeypatch.setattr("tiny_rag_lab.web_api.qdrant_is_available", lambda _url: True)

    items = TestClient(create_app(tmp_path)).get("/api/backends").json()["items"]

    assert items == [
        {"id": "numpy", "available": True},
        {"id": "qdrant", "available": True},
    ]


def test_qdrant_course_reports_optional_service_without_hiding_launch_step(tmp_path, monkeypatch):
    _write_teaching_index(tmp_path)
    monkeypatch.setattr("tiny_rag_lab.web_api.qdrant_is_available", lambda _url: False)
    client = TestClient(create_app(tmp_path))

    status = client.get("/api/retrieval/qdrant/status")
    prepare = client.post("/api/retrieval/qdrant/prepare")

    assert status.status_code == 200
    assert status.json()["available"] is False
    assert status.json()["prepared"] is False
    assert status.json()["launch_command"] == "docker compose --profile qdrant up -d"
    assert prepare.status_code == 409
    assert "profile qdrant" in prepare.json()["detail"]


def test_qdrant_course_prepares_exact_copy_and_compares_payload_filter(tmp_path, monkeypatch):
    from qdrant_client import QdrantClient

    _write_teaching_index(tmp_path)
    material = _write_retrieval_materials(tmp_path)
    backend = QdrantIndexBackend.__new__(QdrantIndexBackend)
    backend._client = QdrantClient(location=":memory:")
    monkeypatch.setattr("tiny_rag_lab.web_api.qdrant_is_available", lambda _url: True)
    monkeypatch.setattr("tiny_rag_lab.web_api.QdrantIndexBackend", lambda _url: backend)
    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    client = TestClient(create_app(tmp_path))

    before = client.get("/api/retrieval/qdrant/status").json()
    first = client.post("/api/retrieval/qdrant/prepare")
    second = client.post("/api/retrieval/qdrant/prepare")
    comparison = client.post("/api/retrieval/qdrant/compare", json={
        "retrieval_material_id": material["question_id"], "top_k": 2,
        "source_group": "r2",
    })

    assert before["prepared"] is False
    assert first.status_code == 201
    assert first.json()["verified"] is True
    assert first.json()["reused"] is False
    assert second.json()["reused"] is True
    assert comparison.status_code == 200
    body = comparison.json()
    assert body["parity"]["equivalent"] is True
    assert [item["chunk_id"] for item in body["numpy"]] == [
        item["chunk_id"] for item in body["qdrant"]
    ]
    assert body["qdrant"][0]["payload"]["source_fingerprint"] == first.json()["source_fingerprint"]
    assert {item["payload"]["source_group"] for item in body["filtered_qdrant"]} == {"r2"}

    backend.search_exact = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        QdrantBackendError(
            "Exact Qdrant search is unavailable. Prepare the teaching collection again."
        )
    )
    failed = client.post("/api/retrieval/qdrant/compare", json={
        "retrieval_material_id": material["question_id"], "top_k": 2,
    })
    assert failed.status_code == 503
    assert failed.json()["detail"] == "Exact Qdrant search is unavailable. Prepare the teaching collection again."


def test_legacy_qdrant_index_reports_payload_filters_unavailable(tmp_path, monkeypatch):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    _write_teaching_index(tmp_path, index_backend="qdrant", backend_identity="legacy")
    backend = QdrantIndexBackend.__new__(QdrantIndexBackend)
    backend._client = QdrantClient(location=":memory:")
    backend._client.create_collection(
        "legacy", vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    backend._client.upsert(
        "legacy",
        points=[
            PointStruct(id=0, vector=[0.0, 1.0], payload={"chunk_id": make_chunk_id("r2/how-r2-works.md", 0, "catalog evidence for object storage")}),
            PointStruct(id=1, vector=[1.0, 0.0], payload={"chunk_id": make_chunk_id("queues/retries.md", 0, "alpha queue retry evidence")}),
        ],
        wait=True,
    )
    monkeypatch.setattr("tiny_rag_lab.web_api.backend_from_manifest", lambda *_args, **_kwargs: backend)

    detail = TestClient(create_app(tmp_path)).get(
        "/api/indexes/cloudflare-state-structural-v1"
    )

    assert detail.status_code == 200
    assert detail.json()["capabilities"] == {"payload_filters": False}


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


def test_retrieval_materials_drive_server_resolved_explained_run(tmp_path, monkeypatch):
    client = _seeded_client(tmp_path, monkeypatch)
    material = _write_retrieval_materials(tmp_path)
    _write_catalog_index(
        tmp_path, source_corpus_id="cloudflare-state-v1",
        index_id="retrieval-course-index",
    )

    listing = client.get("/api/retrieval/materials")
    assert listing.status_code == 200
    assert listing.json()["items"] == [material]

    run = client.post("/api/runs/retrieve", json={
        "index_id": "retrieval-course-index",
        "retrieval_material_id": material["question_id"],
        "query": "browser text is ignored",
        "retriever": "bm25",
        "top_k": 1,
        "explain": True,
    })

    assert run.status_code == 201
    payload = run.json()
    assert payload["trace"]["query"] == material["question"]
    assert payload["schema_version"] == "1.1"
    assert payload["explanations"]["kind"] == "bm25"
    candidate = payload["explanations"]["bm25"]["candidates"][0]
    assert sum(term["contribution"] for term in candidate["terms"]) == pytest.approx(candidate["score"])


def test_retrieval_material_rejects_non_cloudflare_index(tmp_path, monkeypatch):
    client = _seeded_client(tmp_path, monkeypatch)
    material = _write_retrieval_materials(tmp_path)
    _write_catalog_index(tmp_path, index_id="wrong-course-index")

    response = client.post("/api/runs/retrieve", json={
        "index_id": "wrong-course-index",
        "retrieval_material_id": material["question_id"],
        "retriever": "bm25",
    })

    assert response.status_code == 409
    assert "bundled Cloudflare index" in response.json()["detail"]


def test_hybrid_explanation_exposes_source_lists_and_exact_rrf_contributions(tmp_path, monkeypatch):
    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    material = _write_retrieval_materials(
        tmp_path, category="hybrid", question="How does alpha evidence relate to catalog evidence?",
    )
    _write_teaching_index(tmp_path)
    response = TestClient(create_app(tmp_path)).post("/api/runs/retrieve", json={
        "index_id": "cloudflare-state-structural-v1",
        "retrieval_material_id": material["question_id"],
        "retriever": "hybrid", "top_k": 2, "explain": True,
    })

    assert response.status_code == 201
    explanation = response.json()["explanations"]
    assert explanation["kind"] == "hybrid"
    assert len(explanation["hybrid"]["dense"]) == 2
    assert len(explanation["hybrid"]["bm25"]) == 2
    for candidate in explanation["hybrid"]["candidates"]:
        assert candidate["score"] == pytest.approx(
            sum(source["contribution"] for source in candidate["sources"])
        )


def test_reranker_model_status_reports_the_pinned_local_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tiny_rag_lab.web_api.CrossEncoderReranker.default_model_available",
        lambda: True,
    )

    status = TestClient(create_app(tmp_path)).get("/api/models/reranker/status")

    assert status.status_code == 200
    assert status.json() == {
        "ready": True,
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "revision": "c5ee24cb16019beea0893ab7796b1df96625c6b8",
    }


def test_web_reranking_keeps_candidate_pool_separate_from_final_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    monkeypatch.setattr(
        "tiny_rag_lab.web_api.CrossEncoderReranker",
        lambda **_kwargs: FakeReranker(name="cross-encoder"),
    )
    client = TestClient(create_app(tmp_path))
    corpus = client.post(
        "/api/corpora/upload",
        files=[
            ("files", ("alpha.md", b"alpha evidence", "text/markdown")),
            ("files", ("beta.md", b"beta evidence", "text/markdown")),
        ],
    ).json()
    job = client.post("/api/indexes", json={"corpus_id": corpus["id"]}).json()
    index_id = client.get(f"/api/jobs/{job['id']}").json()["index_id"]

    response = client.post("/api/runs/retrieve", json={
        "index_id": index_id,
        "query": "alpha",
        "retriever": "dense",
        "top_k": 1,
        "reranker": "cross-encoder",
        "rerank_top_n": 2,
        "explain": True,
    })

    assert response.status_code == 201
    run = response.json()
    assert len(run["evidence"]) == 1
    assert len(run["candidates"]) == 2
    assert run["explanations"]["reranking"]["candidate_count"] == 2
    assert run["trace"]["chunks"][0]["pre_rerank_rank"] == 1
    assert run["config"]["reranker"] == "cross-encoder"
    assert run["config"]["rerank_top_n"] == 2
    assert run["trace"]["reranker"] == "cross-encoder"
    assert run["trace"]["rerank_top_n"] == 2


def test_web_reranking_rejects_candidate_depth_below_final_top_k(tmp_path, monkeypatch):
    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    client = _seeded_client(tmp_path, monkeypatch)
    _write_catalog_index(tmp_path, index_id="rerank-index")

    response = client.post("/api/runs/retrieve", json={
        "index_id": "rerank-index",
        "query": "alpha",
        "top_k": 5,
        "reranker": "cross-encoder",
        "rerank_top_n": 2,
    })

    assert response.status_code == 422
    assert "rerank_top_n" in response.json()["detail"]


@pytest.mark.parametrize("generation_fails", [False, True])
def test_reranked_ask_preserves_pre_rerank_trace_on_success_and_failure(
    tmp_path, monkeypatch, generation_fails,
):
    class _Generator:
        def __init__(self, **_kwargs):
            pass

        def generate(self, _prompt):
            if generation_fails:
                raise RuntimeError("provider failed")
            return "Grounded answer"

    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    monkeypatch.setattr("tiny_rag_lab.web_api.OpenAIGenerator", _Generator)
    monkeypatch.setattr(
        "tiny_rag_lab.web_api.CrossEncoderReranker",
        lambda **_kwargs: FakeReranker(name="cross-encoder"),
    )
    client = TestClient(create_app(tmp_path))
    corpus = client.post(
        "/api/corpora/upload",
        files=[
            ("files", ("alpha.md", b"alpha evidence", "text/markdown")),
            ("files", ("beta.md", b"beta evidence", "text/markdown")),
        ],
    ).json()
    job = client.post("/api/indexes", json={"corpus_id": corpus["id"]}).json()
    index_id = client.get(f"/api/jobs/{job['id']}").json()["index_id"]

    response = client.post("/api/runs/ask", json={
        "index_id": index_id,
        "query": "alpha",
        "top_k": 1,
        "reranker": "cross-encoder",
        "rerank_top_n": 2,
        "provider": {"base_url": "http://local-provider/v1"},
    })

    assert response.status_code == 201
    run = response.json()
    assert run["trace"]["chunks"][0]["pre_rerank_rank"] == 1
    assert run["trace"]["chunks"][0]["pre_rerank_score"] is not None
    assert run["trace"]["reranker"] == "cross-encoder"
    assert run["trace"]["rerank_top_n"] == 2
    assert bool(run["error"]) is generation_fails


@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_text"),
    [
        (OSError("not cached"), 409, "Download the default reranker model"),
        (RuntimeError("inference exploded"), 500, "Cross-encoder reranking failed"),
    ],
)
def test_web_reranker_distinguishes_missing_model_from_runtime_failure(
    tmp_path, monkeypatch, failure, expected_status, expected_text,
):
    class _FailingReranker:
        def __init__(self, **_kwargs):
            pass

        def rerank(self, _query, _candidates):
            raise failure

    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    monkeypatch.setattr("tiny_rag_lab.web_api.CrossEncoderReranker", _FailingReranker)
    client = TestClient(create_app(tmp_path))
    corpus = client.post(
        "/api/corpora/upload",
        files=[("files", ("alpha.md", b"alpha evidence", "text/markdown"))],
    ).json()
    job = client.post("/api/indexes", json={"corpus_id": corpus["id"]}).json()
    index_id = client.get(f"/api/jobs/{job['id']}").json()["index_id"]

    response = client.post("/api/runs/retrieve", json={
        "index_id": index_id,
        "query": "alpha",
        "top_k": 1,
        "reranker": "cross-encoder",
        "rerank_top_n": 1,
    })

    assert response.status_code == expected_status
    assert expected_text in response.json()["detail"]


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


def test_job_api_discovers_cancels_and_reads_only_complete_results(tmp_path):
    client = TestClient(create_app(tmp_path))
    store = LocalJobStore(tmp_path / "jobs")
    job = store.admit("evaluation", preset="dense-vs-hybrid")
    store.start(job["id"], total=16)

    active = client.get("/api/jobs/active", params={"kind": "evaluation"})
    cancelled = client.post(f"/api/jobs/{job['id']}/cancel")

    assert active.status_code == 200
    assert [item["id"] for item in active.json()["items"]] == [job["id"]]
    assert cancelled.status_code == 202
    assert cancelled.json()["status"] == "cancel_requested"
    assert client.get(f"/api/jobs/{job['id']}/result").status_code == 404

    assert store.progress(job["id"], 1, total=16, message="Checkpoint") is False
    complete_job = store.admit("evaluation")
    store.start(complete_job["id"], total=1)
    store.complete(complete_job["id"], result={"questions": []})
    assert client.get(f"/api/jobs/{complete_job['id']}/result").json() == {"questions": []}


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


def test_reranker_download_verifies_exact_snapshot_before_ready(tmp_path, monkeypatch):
    readiness = iter([False, True])
    calls = []
    monkeypatch.setattr(
        "tiny_rag_lab.web_api.CrossEncoderReranker.default_model_available",
        lambda: next(readiness),
    )
    monkeypatch.setattr(
        "tiny_rag_lab.web_api.CrossEncoderReranker.ensure_default_model",
        lambda *, local_files_only: calls.append(local_files_only) or "/cached/model",
    )
    client = TestClient(create_app(tmp_path))

    queued = client.post("/api/models/reranker/download")
    state = client.get(f"/api/jobs/{queued.json()['id']}").json()

    assert queued.status_code == 202
    assert calls == [False]
    assert state["status"] == "complete"


def test_reranker_download_never_reports_partial_snapshot_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tiny_rag_lab.web_api.CrossEncoderReranker.default_model_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "tiny_rag_lab.web_api.CrossEncoderReranker.ensure_default_model",
        lambda *, local_files_only: "/partial/model",
    )
    client = TestClient(create_app(tmp_path))

    queued = client.post("/api/models/reranker/download")
    state = client.get(f"/api/jobs/{queued.json()['id']}").json()

    assert state["status"] == "failed"
    assert "network" in state["error"]
    assert client.get("/api/models/reranker/status").json()["ready"] is False


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


def test_index_job_honors_cancellation_after_inflight_embedding_call(tmp_path, monkeypatch):
    store = LocalJobStore(tmp_path / "jobs")

    class _CancellingEmbedder(_Embedder):
        def embed(self, texts):
            job = store.active(kind="index")[0]
            store.request_cancel(job["id"])
            return super().embed(texts)

    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _CancellingEmbedder)
    client = TestClient(create_app(tmp_path))
    corpus = client.post(
        "/api/corpora/upload",
        files=[("files", ("alpha.md", b"alpha evidence", "text/markdown"))],
    ).json()

    queued = client.post("/api/indexes", json={"corpus_id": corpus["id"]}).json()
    job = client.get(f"/api/jobs/{queued['id']}").json()

    assert job["status"] == "cancelled"
    assert list((tmp_path / "indexes").iterdir()) == []


def test_index_job_does_not_publish_when_cancel_arrives_after_final_checkpoint(tmp_path, monkeypatch):
    original_progress = LocalJobStore.progress

    def cancel_after_checkpoint(self, job_id, current, **kwargs):
        accepted = original_progress(self, job_id, current, **kwargs)
        if accepted and current == 5:
            self.request_cancel(job_id)
        return accepted

    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    monkeypatch.setattr(LocalJobStore, "progress", cancel_after_checkpoint)
    client = TestClient(create_app(tmp_path))
    corpus = client.post(
        "/api/corpora/upload",
        files=[("files", ("alpha.md", b"alpha evidence", "text/markdown"))],
    ).json()

    queued = client.post("/api/indexes", json={"corpus_id": corpus["id"]}).json()
    job = client.get(f"/api/jobs/{queued['id']}").json()

    assert job["status"] == "cancelled"
    assert list((tmp_path / "indexes").iterdir()) == []


def test_qdrant_index_failure_after_build_deletes_unpublished_collection(tmp_path, monkeypatch):
    built = []
    deleted = []

    class _Backend:
        def build(self, collection, _index):
            built.append(collection)

        def delete(self, collection):
            deleted.append(collection)

    original_progress = LocalJobStore.progress

    def fail_after_build(self, job_id, current, **kwargs):
        if current == 5:
            raise RuntimeError("publication checkpoint failed")
        return original_progress(self, job_id, current, **kwargs)

    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    monkeypatch.setattr("tiny_rag_lab.web_api.qdrant_is_available", lambda _url: True)
    monkeypatch.setattr("tiny_rag_lab.web_api.backend_from_manifest", lambda *_args, **_kwargs: _Backend())
    monkeypatch.setattr(LocalJobStore, "progress", fail_after_build)
    client = TestClient(create_app(tmp_path))
    corpus = client.post(
        "/api/corpora/upload",
        files=[("files", ("alpha.md", b"alpha evidence", "text/markdown"))],
    ).json()

    queued = client.post("/api/indexes", json={
        "corpus_id": corpus["id"], "index_backend": "qdrant",
    }).json()
    job = client.get(f"/api/jobs/{queued['id']}").json()

    assert job["status"] == "failed"
    assert deleted == built and len(deleted) == 1
    assert list((tmp_path / "indexes").iterdir()) == []


def test_partial_qdrant_build_failure_still_deletes_owned_collection(tmp_path, monkeypatch):
    built = []
    deleted = []

    class _Backend:
        def build(self, collection, _index):
            built.append(collection)
            raise RuntimeError("failed after remote collection creation")

        def delete(self, collection):
            deleted.append(collection)

    monkeypatch.setattr("tiny_rag_lab.web_api.SentenceTransformerEmbedder", _Embedder)
    monkeypatch.setattr("tiny_rag_lab.web_api.qdrant_is_available", lambda _url: True)
    monkeypatch.setattr("tiny_rag_lab.web_api.backend_from_manifest", lambda *_args, **_kwargs: _Backend())
    client = TestClient(create_app(tmp_path))
    corpus = client.post(
        "/api/corpora/upload",
        files=[("files", ("alpha.md", b"alpha evidence", "text/markdown"))],
    ).json()

    queued = client.post("/api/indexes", json={
        "corpus_id": corpus["id"], "index_backend": "qdrant",
    }).json()
    job = client.get(f"/api/jobs/{queued['id']}").json()

    assert job["status"] == "failed"
    assert deleted == built and len(deleted) == 1


def test_restart_recovers_published_artifact_and_cleans_unpublished_ownership(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir(parents=True)
    (tmp_path / "indexes" / ".dead.staging").mkdir(parents=True)
    (tmp_path / "corpora" / ".watsonxdocsqa.staging").mkdir(parents=True)
    (tmp_path / "indexes" / "published-index").mkdir(parents=True)
    (jobs / "index-dead.json").write_text(json.dumps({
        "id": "index-dead", "kind": "index", "status": "running",
        "artifact": {"kind": "index", "id": "missing-index", "staging_name": ".dead.staging", "qdrant_collection": "orphan"},
    }))
    (jobs / "index-published.json").write_text(json.dumps({
        "id": "index-published", "kind": "index", "status": "publishing",
        "artifact": {"kind": "index", "id": "published-index", "staging_name": ".published.staging", "qdrant_collection": "owned"},
    }))
    deleted = []
    monkeypatch.setattr("tiny_rag_lab.web_api.qdrant_is_available", lambda _url: True)
    monkeypatch.setattr(
        "tiny_rag_lab.web_api.QdrantIndexBackend",
        lambda _url: type("Backend", (), {"delete": lambda _self, collection: deleted.append(collection)})(),
    )

    client = TestClient(create_app(tmp_path))

    assert client.get("/api/jobs/index-dead").json()["status"] == "failed"
    assert client.get("/api/jobs/index-dead").json()["cleanup_pending"] is False
    assert client.get("/api/jobs/index-published").json()["status"] == "complete"
    assert deleted == ["orphan"]
    assert not (tmp_path / "indexes" / ".dead.staging").exists()
    assert not (tmp_path / "corpora" / ".watsonxdocsqa.staging").exists()


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
