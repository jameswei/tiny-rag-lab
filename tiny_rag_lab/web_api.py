"""Loopback-only API for the local visual RAG laboratory.

This module intentionally calls the same small engine modules as the CLI. It
does not parse CLI output or invent browser-specific RAG mechanics.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tiny_rag_lab.bm25 import BM25Retriever
from tiny_rag_lab.browser_eval import (
    BrowserEvaluationCancelled,
    BrowserEvaluationError,
    EVALUATION_PRESETS,
    RetrievalConfiguration,
    run_browser_comparison,
    validate_evaluation_bundle,
)
from tiny_rag_lab.chunking import chunk_documents_with_strategy
from tiny_rag_lab.context import FakeTokenCounter, pack_context
from tiny_rag_lab.documents import load_documents
from tiny_rag_lab.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    SentenceTransformerEmbedder,
)
from tiny_rag_lab.generation import OpenAIGenerator
from tiny_rag_lab.hybrid import reciprocal_rank_fusion_with_explanation
from tiny_rag_lab.index_backend import NumpyIndexBackend, backend_from_manifest
from tiny_rag_lab.index_loader import load_index
from tiny_rag_lab.index_writer import write_index
from tiny_rag_lab.lab_trace import EvidenceSnapshot, build_lab_run, load_lab_run, write_lab_run
from tiny_rag_lab.jobs import (
    JobConflictError,
    JobNotFoundError,
    LocalJobStore,
    atomic_write_json,
)
from tiny_rag_lab.prompting import assemble_prompt, extract_source_citations
from tiny_rag_lab.qdrant_backend import (
    QdrantBackendError,
    QdrantIndexBackend,
    compare_exact_rankings,
    qdrant_is_available,
    source_vector_fingerprint,
)
from tiny_rag_lab.retrieval import explain_dense_results
from tiny_rag_lab.reranker import (
    CrossEncoderReranker,
    apply_reranker,
    chunk_traces_from_rerank,
    explain_rerank,
)
from tiny_rag_lab.seed_assets import SeedAssetError, seed_bundled_assets
from tiny_rag_lab.trace import AskTrace, ChunkTrace, RetrieveTrace

MAX_UPLOAD_FILES = 100
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
UPLOAD_READ_BYTES = 1024 * 1024
ALLOWED_SUFFIXES = {".md", ".txt"}
_job_lock = threading.Lock()
logger = logging.getLogger(__name__)


class ProviderOverride(BaseModel):
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None


class IndexRequest(BaseModel):
    corpus_id: str
    chunk_size: int = Field(default=800, ge=1)
    chunk_overlap: int = Field(default=120, ge=0)
    chunking_strategy: Literal["fixed_character", "structural", "semantic"] = "fixed_character"
    semantic_similarity_threshold: float = 0.5
    index_backend: Literal["numpy", "qdrant"] = "numpy"


class RunRequest(BaseModel):
    index_id: str
    query: str | None = Field(default=None, min_length=1)
    catalog_question_id: str | None = None
    retrieval_material_id: str | None = None
    retriever: Literal["dense", "bm25", "hybrid"] = "dense"
    top_k: int = Field(default=5, ge=1, le=50)
    reranker: Literal["none", "cross-encoder"] = "none"
    rerank_top_n: int = Field(default=20, ge=1, le=50)
    context_budget: int = Field(default=0, ge=0)
    explain: bool = False
    provider: ProviderOverride | None = None


class QdrantCompareRequest(BaseModel):
    retrieval_material_id: str
    top_k: int = Field(default=5, ge=1, le=20)
    source_group: Literal[
        "durable-objects", "queues", "kv", "r2", "workflows",
    ] | None = None


class EvaluationConfigurationRequest(BaseModel):
    retriever: Literal["bm25", "dense", "hybrid"]
    top_k: int = Field(default=5, ge=1, le=20)
    reranker: Literal["none", "cross-encoder"] = "none"
    rerank_top_n: int = Field(default=20, ge=1, le=50)

    def engine_config(self) -> RetrievalConfiguration:
        return RetrievalConfiguration(**self.model_dump())


class EvaluationRequest(BaseModel):
    left: EvaluationConfigurationRequest
    right: EvaluationConfigurationRequest


def _safe_id(value: str, kind: str) -> str:
    if not value or value != Path(value).name or value in {".", ".."}:
        raise HTTPException(422, f"Invalid {kind} identifier")
    return value


def _write_json(path: Path, value: dict) -> None:
    atomic_write_json(path, value)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _chunk_trace(result) -> ChunkTrace:
    return ChunkTrace(
        rank=result.rank,
        chunk_id=result.chunk.chunk_id,
        doc_id=result.chunk.doc_id,
        title=result.chunk.metadata.get("title", ""),
        path=result.chunk.metadata.get("path", result.chunk.doc_id),
        score=result.score,
        text_preview=result.chunk.text[:120].replace("\n", " ").strip(),
    )


def _evidence(
    result, semantics: str, *, score_components: dict[str, float] | None = None,
    selected_for_context: bool | None = None,
) -> EvidenceSnapshot:
    return EvidenceSnapshot(
        chunk_id=result.chunk.chunk_id,
        doc_id=result.chunk.doc_id,
        title=result.chunk.metadata.get("title", ""),
        path=result.chunk.metadata.get("path", result.chunk.doc_id),
        text=result.chunk.text,
        rank=result.rank,
        score=result.score,
        score_semantics=semantics,
        score_components=score_components or {},
        selected_for_context=selected_for_context,
    )


def create_app(data_root: Path | None = None, static_dir: Path | None = None) -> FastAPI:
    """Create a local API app. `data_root` makes tests and local installs isolated."""
    root = Path(data_root or os.environ.get("TINY_RAG_LAB_DATA_DIR", ".tiny-rag-lab"))
    corpora_dir = root / "corpora"
    indexes_dir = root / "indexes"
    runs_dir = root / "runs"
    jobs_dir = root / "jobs"
    for directory in (corpora_dir, indexes_dir, runs_dir, jobs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    job_store = LocalJobStore(jobs_dir)
    seed_root = Path(os.environ.get("TINY_RAG_LAB_SEED_DIR", "/opt/tiny-rag-lab/seeds/v2"))
    try:
        seed_results = seed_bundled_assets(root, seed_root)
    except SeedAssetError as exc:
        # The lab still starts so custom local work remains accessible, while
        # health makes an image-seed problem explicit instead of hiding it.
        logger.exception("Bundled seed assets are unavailable")
        seed_results = [{"status": "error", "detail": str(exc)}]
    # Recover the short publication boundary before failing other process-bound
    # work. A final artifact proves publication won; otherwise staging and any
    # unpublished Qdrant ownership are cleaned before the job becomes terminal.
    for job in job_store.all():
        artifact = job.get("artifact") or {}
        artifact_kind = artifact.get("kind")
        artifact_id = artifact.get("id")
        final_path = (
            corpora_dir / artifact_id if artifact_kind == "corpus" and artifact_id
            else indexes_dir / artifact_id if artifact_kind == "index" and artifact_id
            else None
        )
        if job.get("status") == "publishing" and final_path is not None and final_path.exists():
            fields = {"corpus_id": artifact_id} if artifact_kind == "corpus" else {"index_id": artifact_id}
            job_store.complete(job["id"], **fields)
            continue
        collection = artifact.get("qdrant_collection")
        if collection and (final_path is None or not final_path.exists()):
            if qdrant_is_available(os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")):
                try:
                    QdrantIndexBackend(
                        os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
                    ).delete(collection)
                    job_store.update(job["id"], cleanup_pending=False)
                except Exception:
                    logger.exception("Failed to clean recovered Qdrant collection %s", collection)
                    job_store.update(job["id"], cleanup_pending=True)
            else:
                job_store.update(job["id"], cleanup_pending=True)
    for parent in (corpora_dir, indexes_dir):
        for staging in parent.glob(".*.staging"):
            if staging.is_dir():
                shutil.rmtree(staging, ignore_errors=True)

    # Other process-bound tasks cannot resume after restart. Persist a
    # terminal, actionable state so the browser never polls forever.
    job_store.recover_interrupted()

    app = FastAPI(title="tiny-rag-lab local API", docs_url=None, redoc_url=None)
    # The packaged browser client uses the same origin. This only helps native
    # development and deliberately does not open the server beyond loopback.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["*"], allow_headers=["*"], allow_credentials=False,
    )

    def resolve_provider(override: ProviderOverride | None = None) -> dict[str, str | None]:
        override = override or ProviderOverride()
        api_key = override.api_key or os.environ.get("OPENAI_API_KEY")
        base_url = override.base_url or os.environ.get("OPENAI_BASE_URL")
        # OpenAI's client accepts an API root and appends /chat/completions.
        # Learners commonly paste the complete provider URL, so accept both
        # forms while keeping the client-facing base URL unambiguous.
        if base_url:
            base_url = base_url.rstrip("/")
            if base_url.endswith("/chat/completions"):
                base_url = base_url.removesuffix("/chat/completions")
        model = override.model or os.environ.get("OPENAI_MODEL") or getattr(OpenAIGenerator, "DEFAULT_MODEL", "gpt-4o-mini")
        return {"api_key": api_key, "base_url": base_url, "model": model}

    def provider_is_usable(config: dict[str, str | None]) -> bool:
        return bool(config["model"] and (config["api_key"] or config["base_url"]))

    def provider_status() -> dict:
        config = resolve_provider()
        return {
            "configured": provider_is_usable(config),
            "base_url_configured": bool(os.environ.get("OPENAI_BASE_URL")),
            "model_configured": bool(os.environ.get("OPENAI_MODEL")),
            "api_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
            "default_model": getattr(OpenAIGenerator, "DEFAULT_MODEL", "gpt-4o-mini"),
            "base_url": os.environ.get("OPENAI_BASE_URL"),
            "model": os.environ.get("OPENAI_MODEL"),
        }

    def run_config(request: RunRequest) -> dict:
        config = {
            "retriever": request.retriever,
            "top_k": request.top_k,
            "context_budget": request.context_budget,
        }
        if request.reranker != "none":
            config.update({
                "reranker": request.reranker,
                "rerank_top_n": request.rerank_top_n,
            })
        return config

    def ask_chunk_traces(results, retrieve_trace: RetrieveTrace) -> list[ChunkTrace]:
        """Keep retrieval/rerank audit fields when Ask selects its context."""
        by_chunk_id = {chunk.chunk_id: chunk for chunk in retrieve_trace.chunks}
        return [
            by_chunk_id.get(result.chunk.chunk_id, _chunk_trace(result))
            for result in results
        ]

    def model_status() -> dict:
        try:
            SentenceTransformerEmbedder(local_files_only=True)
        except Exception:
            ready = False
        else:
            ready = True
        return {
            "ready": ready,
            "variant": os.environ.get("LAB_IMAGE_VARIANT", "native"),
            "model": DEFAULT_EMBEDDING_MODEL,
            "revision": DEFAULT_EMBEDDING_REVISION,
            "dimension": 384,
        }

    def save_run(run) -> dict:
        path = runs_dir / f"{run.run_id}.json"
        write_lab_run(run, path)
        return load_lab_run(path)

    def admit_job(kind: str, **fields) -> str:
        """Persist one visible queued job or reject a concurrent request."""
        try:
            job = job_store.admit(kind, **fields)
        except JobConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        return job["id"]

    def resolve_index(index_id: str):
        index_id = _safe_id(index_id, "index")
        index_dir = indexes_dir / index_id
        if not index_dir.exists():
            raise HTTPException(404, "Index not found")
        index = NumpyIndexBackend().open(index_dir)
        try:
            backend = backend_from_manifest(
                index.manifest,
                qdrant_url=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"),
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return index_id, index, backend

    def resolve_teaching_index():
        index_id = "cloudflare-state-structural-v1"
        path = indexes_dir / index_id
        if not path.exists():
            raise HTTPException(409, "The bundled structural index is not available")
        return index_id, NumpyIndexBackend().open(path)

    def evaluation_paths() -> tuple[Path, Path, Path]:
        corpus = corpora_dir / "cloudflare-state-v1"
        return (
            indexes_dir / "cloudflare-state-structural-v1",
            corpus / "retrieval-questions.jsonl",
            corpus / "evaluation-manifest.json",
        )

    def evaluation_bundle():
        try:
            return validate_evaluation_bundle(*evaluation_paths())
        except BrowserEvaluationError as exc:
            raise HTTPException(409, str(exc)) from exc

    def qdrant_url() -> str:
        return os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")

    def load_catalog_question(corpus_id: str, question_id: str) -> dict:
        questions_path = corpora_dir / corpus_id / "questions.jsonl"
        if not questions_path.exists():
            raise HTTPException(409, "This corpus does not provide a question catalog")
        for line in questions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("question_id") == question_id:
                return item
        raise HTTPException(404, "Catalog question not found")

    def load_retrieval_material(question_id: str) -> dict:
        questions_path = corpora_dir / "cloudflare-state-v1" / "retrieval-questions.jsonl"
        if not questions_path.exists():
            raise HTTPException(409, "The bundled retrieval course is not available")
        for line in questions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("question_id") == question_id:
                return item
        raise HTTPException(404, "Retrieval course question not found")

    def resolve_run_question(request: RunRequest, index) -> tuple[str, dict | None]:
        """Resolve catalog IDs on the server, never from browser-provided gold."""
        if request.retrieval_material_id:
            if request.catalog_question_id:
                raise HTTPException(422, "Choose one question source")
            if index.manifest.get("source_corpus_id") != "cloudflare-state-v1":
                raise HTTPException(409, "Retrieval course questions require the bundled Cloudflare index")
            material = load_retrieval_material(request.retrieval_material_id)
            return material["question"], material
        if not request.catalog_question_id:
            if request.query is None:
                raise HTTPException(422, "Provide a query or catalog_question_id")
            return request.query, None

        corpus_id = index.manifest.get("source_corpus_id")
        if not corpus_id:
            raise HTTPException(409, "This legacy index cannot be used with catalog questions")
        question = load_catalog_question(corpus_id, request.catalog_question_id)
        # The canonical catalog text wins even when an older browser sends its
        # own copy.  This keeps the question/gold pairing durable across restarts.
        return question["question"], question

    def catalog_check(question: dict | None, results) -> dict | None:
        if question is None:
            return None
        expected = list(question.get("gold_doc_ids", []))
        retrieved = [result.chunk.doc_id for result in results]
        return {
            "question_id": question["question_id"],
            "expected_document_ids": expected,
            "retrieved_document_ids": retrieved,
            "hit": bool(set(expected) & set(retrieved)),
        }

    def run_retrieval(request: RunRequest):
        index_id, index, vector_backend = resolve_index(request.index_id)
        query, catalog_question = resolve_run_question(request, index)
        if request.retriever != "bm25" and not model_status()["ready"]:
            raise HTTPException(409, "Download the default embedding model before dense or hybrid retrieval")
        t0 = time.perf_counter()
        query_vector: list[float] | None = None
        latency: dict[str, float] = {}
        semantics = "cosine_similarity[-1,1]"
        explanations: dict | None = None
        if request.reranker != "none" and request.rerank_top_n < request.top_k:
            raise HTTPException(422, "rerank_top_n must be greater than or equal to top_k")
        retrieval_k = request.rerank_top_n if request.reranker != "none" else request.top_k

        score_components: dict[str, dict[str, float]] = {}
        if request.retriever == "bm25":
            bm25 = BM25Retriever(index.chunks)
            if request.explain:
                results, bm25_explanation = bm25.retrieve_with_explanation(query, retrieval_k)
                explanations = {"kind": "bm25", "bm25": asdict(bm25_explanation)}
            else:
                results = bm25.retrieve(query, retrieval_k)
            semantics = "bm25_score"
            score_components = {
                result.chunk.chunk_id: {"bm25_score": result.score, "bm25_rank": float(result.rank)}
                for result in results
            }
        else:
            model_name = index.manifest.get("embedding_model")
            embedder = SentenceTransformerEmbedder(model_name, local_files_only=True)
            query_vec = embedder.embed([query])[0]
            query_vector = [float(value) for value in query_vec]
            latency["embed"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            try:
                dense_hits = vector_backend.search(query_vec, index, retrieval_k)
            except Exception as exc:
                from tiny_rag_lab.qdrant_backend import QdrantBackendError
                if isinstance(exc, QdrantBackendError):
                    raise HTTPException(503, str(exc)) from exc
                raise
            dense_results = [hit.result for hit in dense_hits]
            if request.explain and request.retriever == "dense":
                explanations = {
                    "kind": "dense",
                    "dense": {
                        "dimension": len(query_vec),
                        "candidates": [
                            asdict(item)
                            for item in explain_dense_results(query_vec, index, dense_results)
                        ],
                    },
                }
            semantics = dense_hits[0].score_semantics if dense_hits else vector_backend.score_semantics
            if request.retriever == "hybrid":
                bm25_results = BM25Retriever(index.chunks).retrieve(query, retrieval_k)
                results, hybrid_explanation = reciprocal_rank_fusion_with_explanation(
                    [dense_results, bm25_results], retrieval_k,
                    source_names=["dense", "bm25"],
                )
                if request.explain:
                    explanations = {
                        "kind": "hybrid",
                        "hybrid": {
                            "rrf_k": 60,
                            "dense": [
                                {"chunk_id": result.chunk.chunk_id, "rank": result.rank, "score": result.score}
                                for result in dense_results
                            ],
                            "bm25": [
                                {"chunk_id": result.chunk.chunk_id, "rank": result.rank, "score": result.score}
                                for result in bm25_results
                            ],
                            "candidates": [asdict(item) for item in hybrid_explanation],
                        },
                    }
                semantics = "reciprocal_rank_fusion"
                dense_by_id = {result.chunk.chunk_id: result for result in dense_results}
                bm25_by_id = {result.chunk.chunk_id: result for result in bm25_results}
                score_components = {
                    result.chunk.chunk_id: {
                        "rrf_score": result.score,
                        **({"dense_score": dense_by_id[result.chunk.chunk_id].score, "dense_rank": float(dense_by_id[result.chunk.chunk_id].rank)} if result.chunk.chunk_id in dense_by_id else {}),
                        **({"bm25_score": bm25_by_id[result.chunk.chunk_id].score, "bm25_rank": float(bm25_by_id[result.chunk.chunk_id].rank)} if result.chunk.chunk_id in bm25_by_id else {}),
                    }
                    for result in results
                }
            else:
                results = dense_results
                score_components = {
                    result.chunk.chunk_id: {"dense_score": result.score, "dense_rank": float(result.rank)}
                    for result in results
                }
            latency["retrieve"] = time.perf_counter() - t0

        if request.retriever == "bm25":
            latency["retrieve"] = time.perf_counter() - t0
        candidate_results = list(results)
        candidate_semantics = semantics
        rerank_audit = None
        if request.reranker == "cross-encoder":
            t0 = time.perf_counter()
            try:
                results, rerank_audit = apply_reranker(
                    query,
                    candidate_results,
                    CrossEncoderReranker(local_files_only=True),
                    request.top_k,
                )
            except OSError as exc:
                logger.info("Pinned local reranker snapshot is unavailable: %s", exc)
                raise HTTPException(
                    409,
                    "Download the default reranker model before using cross-encoder reranking",
                ) from exc
            except Exception as exc:
                logger.exception("Cross-encoder reranking failed")
                raise HTTPException(
                    500,
                    "Cross-encoder reranking failed. Check the local server logs and try again.",
                ) from exc
            latency["rerank"] = time.perf_counter() - t0
            semantics = "cross_encoder_relevance"
            audit_by_id = {item.chunk_id: item for item in rerank_audit}
            score_components = {
                result.chunk.chunk_id: {
                    **score_components.get(result.chunk.chunk_id, {}),
                    "reranker_score": result.score,
                    "pre_rerank_score": audit_by_id[result.chunk.chunk_id].pre_score,
                    "pre_rerank_rank": float(audit_by_id[result.chunk.chunk_id].pre_rank),
                }
                for result in results
            }
            if request.explain:
                explanations = {
                    **(explanations or {}),
                    "kind": "reranking",
                    "reranking": {
                        "candidate_count": len(candidate_results),
                        "final_top_k": request.top_k,
                        "candidates": [
                            asdict(item) for item in explain_rerank(rerank_audit, request.top_k)
                        ],
                    },
                }
        trace = RetrieveTrace(
            query=query, retriever=request.retriever, top_k=request.top_k,
            chunks=(
                chunk_traces_from_rerank(results, rerank_audit)
                if rerank_audit is not None else [_chunk_trace(result) for result in results]
            ),
            latency_by_stage=latency,
            reranker=request.reranker,
            rerank_top_n=(
                request.rerank_top_n if request.reranker != "none" else None
            ),
        )
        return (
            index_id, index, results, trace, query_vector, semantics,
            score_components, catalog_question, explanations, candidate_results,
            candidate_semantics,
        )

    @app.get("/api/health")
    def health():
        return {"status": "ok", "data_dir": str(root), "seed_assets": [
            asdict(result) if hasattr(result, "asset_id") else result for result in seed_results
        ]}

    @app.get("/api/provider-status")
    def get_provider_status():
        return provider_status()

    @app.get("/api/backends")
    def get_backend_status():
        """Expose safe local backend readiness for the Build & Inspect UI."""
        return {"items": [
            {"id": "numpy", "available": True},
            {
                "id": "qdrant",
                "available": qdrant_is_available(
                    os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
                ),
            },
        ]}

    @app.post("/api/provider/test")
    def test_provider(override: ProviderOverride | None = None):
        config = resolve_provider(override)
        if not provider_is_usable(config):
            return JSONResponse(status_code=409, content={
                "ok": False,
                "error": "Configure a model and either a provider base URL or API key before testing.",
            })
        api_key = config["api_key"] or "local-provider-no-key"
        try:
            OpenAIGenerator(
                model=config["model"], api_key=api_key, base_url=config["base_url"],
                # The SDK retries failed calls by default.  Disable those
                # retries so this is a single request with a true 10-second
                # server-side deadline rather than several 10-second attempts.
                timeout=10.0, max_retries=0,
            ).generate("Reply only with OK.", max_tokens=4)
        except Exception as exc:
            logger.exception("Provider connection test failed")
            status = getattr(exc, "status_code", None)
            category = type(exc).__name__
            detail = f"Provider connection failed ({category})"
            if isinstance(status, int):
                detail += f"; provider returned HTTP {status}"
            detail += ". Check the base URL, model, and credentials."
            return JSONResponse(status_code=502, content={
                "ok": False,
                "error": detail,
            })
        return {"ok": True, "message": "Provider connection verified"}

    @app.get("/api/models/default/status")
    def get_model_status():
        return model_status()

    @app.get("/api/models/reranker/status")
    def get_reranker_model_status():
        return {
            "ready": CrossEncoderReranker.default_model_available(),
            "model": CrossEncoderReranker.DEFAULT_MODEL,
            "revision": CrossEncoderReranker.DEFAULT_REVISION,
        }

    @app.post("/api/models/default/download", status_code=202)
    def download_default_model(background_tasks: BackgroundTasks):
        if model_status()["ready"]:
            return {"id": "embedding-model-ready", "status": "complete"}
        job_id = admit_job("embedding-model")

        def download_job():
            with _job_lock:
                if not job_store.start(job_id, total=1, message="Downloading embedding model"):
                    return
                try:
                    SentenceTransformerEmbedder()
                    if not job_store.progress(job_id, 1, total=1, message="Embedding model downloaded"):
                        return
                    job_store.complete(job_id)
                except Exception:
                    logger.exception("Embedding-model job %s failed", job_id)
                    job_store.fail(job_id, "Model download failed. Check your network and try again.")

        background_tasks.add_task(download_job)
        return {"id": job_id, "status": "queued"}

    @app.post("/api/models/reranker/download", status_code=202)
    def download_reranker_model(background_tasks: BackgroundTasks):
        if CrossEncoderReranker.default_model_available():
            return {"id": "reranker-model-ready", "status": "complete"}
        job_id = admit_job("reranker-model")

        def download_job():
            with _job_lock:
                if not job_store.start(job_id, total=1, message="Downloading reranker model"):
                    return
                try:
                    CrossEncoderReranker.ensure_default_model(local_files_only=False)
                    if not CrossEncoderReranker.default_model_available():
                        raise RuntimeError("Downloaded reranker snapshot could not be verified")
                    if not job_store.progress(job_id, 1, total=1, message="Reranker model downloaded"):
                        return
                    job_store.complete(job_id)
                except Exception:
                    logger.exception("Reranker-model job %s failed", job_id)
                    job_store.fail(job_id, "Reranker download failed. Check your network and try again.")

        background_tasks.add_task(download_job)
        return {"id": job_id, "status": "queued"}

    @app.get("/api/starter-run")
    def starter_run():
        """A deterministic offline replay, intentionally not a fake live ask."""
        trace = RetrieveTrace(
            query="What does a RAG retriever return?",
            retriever="dense",
            top_k=1,
            chunks=[ChunkTrace(
                rank=1, chunk_id="starter-retrieval", doc_id="retrieval.md",
                title="Retrieval", path="starter/retrieval.md", score=0.91,
                text_preview="Retrieval ranks chunks that may contain evidence for a query.",
            )],
            latency_by_stage={"embed": 0.012, "retrieve": 0.003},
        )
        run = build_lab_run(
            trace, index_id="starter-replay", mode="replay",
            manifest={"index_backend": "numpy", "distance_metric": "cosine", "chunk_count": 1},
            document_count=1, query_vector=[0.12, -0.08, 0.44],
            evidence=[EvidenceSnapshot(
                chunk_id="starter-retrieval", doc_id="retrieval.md", title="Retrieval",
                path="starter/retrieval.md", rank=1, score=0.91,
                score_semantics="cosine_similarity[-1,1]",
                text="Retrieval ranks chunks that may contain evidence for a query. The selected chunks become grounded context for generation.",
            )],
        )
        return save_run(run)

    @app.get("/api/lessons")
    def list_lessons():
        items = []
        for package_dir in sorted((root / "lessons").glob("*")):
            manifest_path = package_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            package = _read_json(manifest_path)
            for lesson in package.get("lessons", []):
                items.append({
                    "id": lesson["id"], "package_id": package["id"],
                    "order": lesson["order"], "title": lesson["title"],
                    "question": lesson["question"], "focus": lesson["focus"],
                })
        return {"items": sorted(items, key=lambda item: (item["package_id"], item["order"]))}

    @app.get("/api/lessons/{lesson_id}")
    def get_lesson(lesson_id: str):
        lesson_id = _safe_id(lesson_id, "lesson")
        for package_dir in sorted((root / "lessons").glob("*")):
            path = package_dir / f"{lesson_id}.json"
            if path.exists():
                return _read_json(path)
        raise HTTPException(404, "Lesson not found")

    @app.get("/api/retrieval/materials")
    def retrieval_materials():
        """Return the reviewed question set without inventing browser copies."""
        path = corpora_dir / "cloudflare-state-v1" / "retrieval-questions.jsonl"
        items = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    items.append(json.loads(line))
        return {
            "corpus_id": "cloudflare-state-v1",
            "index_id": "cloudflare-state-structural-v1",
            "items": items,
        }

    @app.get("/api/retrieval/qdrant/status")
    def retrieval_qdrant_status():
        index_id, index = resolve_teaching_index()
        available = qdrant_is_available(qdrant_url())
        report = None
        if available:
            report = QdrantIndexBackend(qdrant_url()).teaching_status(
                "tiny_rag_cloudflare_state_structural_qdrant_local", index,
            )
        return {
            "available": available,
            "prepared": report is not None,
            "launch_command": "docker compose --profile qdrant up -d",
            "index_id": index_id,
            "source_fingerprint": source_vector_fingerprint(index),
            "collection": asdict(report) if report else None,
            "filters": ["durable-objects", "queues", "kv", "r2", "workflows"],
        }

    @app.post("/api/retrieval/qdrant/prepare", status_code=201)
    def prepare_retrieval_qdrant():
        if not qdrant_is_available(qdrant_url()):
            raise HTTPException(
                409,
                "Qdrant is not ready. Run docker compose --profile qdrant up -d, then try again.",
            )
        _index_id, index = resolve_teaching_index()
        try:
            report = QdrantIndexBackend(qdrant_url()).prepare_teaching_collection(
                "tiny_rag_cloudflare_state_structural_qdrant_local", index,
            )
        except Exception as exc:
            logger.exception("Qdrant teaching collection preparation failed")
            from tiny_rag_lab.qdrant_backend import QdrantBackendError
            if isinstance(exc, QdrantBackendError):
                raise HTTPException(503, str(exc)) from exc
            raise HTTPException(
                500, "Qdrant preparation failed. Check the local server logs and try again."
            ) from exc
        return asdict(report)

    @app.post("/api/retrieval/qdrant/compare")
    def compare_retrieval_qdrant(request: QdrantCompareRequest):
        if not qdrant_is_available(qdrant_url()):
            raise HTTPException(409, "Qdrant is not ready")
        _index_id, index = resolve_teaching_index()
        try:
            backend = QdrantIndexBackend(qdrant_url())
            status = backend.teaching_status(
                "tiny_rag_cloudflare_state_structural_qdrant_local", index,
                raise_on_error=True,
            )
        except (QdrantBackendError, RuntimeError, OSError) as exc:
            logger.exception("Qdrant comparison setup failed")
            raise HTTPException(
                503, "Qdrant became unavailable. Check the local service and try again."
            ) from exc
        if status is None:
            raise HTTPException(409, "Prepare the Qdrant teaching collection before comparing it")
        if not model_status()["ready"]:
            raise HTTPException(409, "Download the default embedding model before comparing vector search")
        material = load_retrieval_material(request.retrieval_material_id)
        embedder = SentenceTransformerEmbedder(
            index.manifest.get("embedding_model"), local_files_only=True,
        )
        query_vector = embedder.embed([material["question"]])[0]
        numpy_hits = NumpyIndexBackend().search(query_vector, index, request.top_k)
        try:
            qdrant_hits = backend.search_exact(
                status.alias, query_vector, index, request.top_k,
            )
        except QdrantBackendError as exc:
            raise HTTPException(503, str(exc)) from exc
        parity = compare_exact_rankings(
            [hit.result for hit in numpy_hits],
            [hit.result for hit in qdrant_hits],
        )
        filtered_hits = []
        if request.source_group:
            try:
                filtered_hits = backend.search_exact(
                    status.alias, query_vector, index, request.top_k,
                    source_group=request.source_group,
                )
            except QdrantBackendError as exc:
                raise HTTPException(503, str(exc)) from exc

        def serialized(result, payload=None):
            return {
                "chunk_id": result.chunk.chunk_id,
                "doc_id": result.chunk.doc_id,
                "title": result.chunk.metadata.get("title", ""),
                "path": result.chunk.doc_id,
                "text": result.chunk.text,
                "rank": result.rank,
                "score": result.score,
                "payload": payload,
            }

        return {
            "question_id": material["question_id"],
            "question": material["question"],
            "collection": asdict(status),
            "numpy": [serialized(hit.result) for hit in numpy_hits],
            "qdrant": [serialized(hit.result, hit.payload) for hit in qdrant_hits],
            "parity": asdict(parity),
            "source_group": request.source_group,
            "filtered_qdrant": [
                serialized(hit.result, hit.payload) for hit in filtered_hits
            ],
        }

    @app.get("/api/evaluations/status")
    def evaluation_status():
        try:
            _index, questions, manifest = validate_evaluation_bundle(*evaluation_paths())
        except BrowserEvaluationError as exc:
            return {
                "ready": False,
                "reason": str(exc),
                "question_count": 0,
                "presets": EVALUATION_PRESETS,
            }
        return {
            "ready": True,
            "reason": None,
            "question_count": len(questions),
            "source_vector_fingerprint": manifest["source_vector_fingerprint"],
            "presets": EVALUATION_PRESETS,
        }

    @app.post("/api/evaluations", status_code=202)
    def create_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
        index, questions, manifest = evaluation_bundle()
        left = request.left.engine_config()
        right = request.right.engine_config()
        try:
            left.validate()
            right.validate()
        except BrowserEvaluationError as exc:
            raise HTTPException(422, str(exc)) from exc
        if left.effective_identity() == right.effective_identity():
            raise HTTPException(422, "Choose two different retrieval configurations")
        needs_embedding = left.retriever != "bm25" or right.retriever != "bm25"
        needs_reranker = left.reranker != "none" or right.reranker != "none"
        if needs_embedding and not model_status()["ready"]:
            raise HTTPException(409, "Download the default embedding model before running this comparison")
        if needs_reranker and not CrossEncoderReranker.default_model_available():
            raise HTTPException(409, "Download the default reranker model before running this comparison")

        job_id = admit_job(
            "evaluation", left=asdict(left), right=asdict(right),
            question_count=len(questions),
            source_vector_fingerprint=manifest["source_vector_fingerprint"],
        )

        def evaluation_job():
            with _job_lock:
                if not job_store.start(
                    job_id, total=len(questions), message="Preparing retrieval comparison",
                ):
                    return
                try:
                    embedder = (
                        SentenceTransformerEmbedder(
                            index.manifest.get("embedding_model"),
                            revision=manifest["embedding_revision"],
                            local_files_only=True,
                        )
                        if needs_embedding else None
                    )

                    def checkpoint(current: int, total: int, side: str) -> bool:
                        return job_store.progress(
                            job_id, current, total=total,
                            message=(
                                f"Question {current + 1} of {total}: first configuration complete"
                                if side == "left" else f"Compared {current} of {total} questions"
                            ),
                        )

                    result = run_browser_comparison(
                        questions, index, left, right,
                        embedder=embedder,
                        reranker_factory=lambda: CrossEncoderReranker(local_files_only=True),
                        checkpoint=checkpoint,
                    )
                    if not job_store.begin_publish(job_id, message="Publishing comparison result"):
                        return
                    job_store.complete(job_id, result={
                        **result,
                        "bundle": {
                            "index_id": "cloudflare-state-structural-v1",
                            "question_count": len(questions),
                            "source_vector_fingerprint": manifest["source_vector_fingerprint"],
                        },
                    })
                except BrowserEvaluationCancelled:
                    return
                except Exception:
                    logger.exception("Evaluation job %s failed", job_id)
                    job_store.fail(
                        job_id,
                        "Evaluation failed. Check local model readiness and the server logs, then try again.",
                    )

        background_tasks.add_task(evaluation_job)
        return {"id": job_id, "status": "queued"}

    @app.get("/api/failure-lessons")
    def failure_lessons():
        from tiny_rag_lab.failure_lessons import FAILURE_LESSONS
        return {"items": FAILURE_LESSONS}

    @app.get("/api/corpora")
    def list_corpora():
        items = []
        for path in sorted(corpora_dir.iterdir()):
            manifest = path / "corpus.json"
            if path.is_dir() and manifest.exists():
                items.append(_read_json(manifest))
        return {"items": items, "limits": {"files": MAX_UPLOAD_FILES, "bytes": MAX_UPLOAD_BYTES}}

    @app.get("/api/corpora/{corpus_id}/questions")
    def list_catalog_questions(corpus_id: str):
        corpus_id = _safe_id(corpus_id, "corpus")
        questions_path = corpora_dir / corpus_id / "questions.jsonl"
        if not questions_path.exists():
            raise HTTPException(404, "Question catalog not found")
        items = []
        for position, line in enumerate(questions_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            question = json.loads(line)
            # Do not expose answer or gold IDs before a validated run.
            items.append({
                "id": question["question_id"], "question": question["question"],
                "featured": position < 8,
            })
        return {"items": items}

    @app.get("/api/indexes")
    def list_indexes():
        items = []
        for path in sorted(indexes_dir.iterdir()):
            manifest = path / "manifest.json"
            if path.is_dir() and manifest.exists():
                data = _read_json(manifest)
                items.append({"id": path.name, "manifest": data})
        return {"items": items}

    @app.get("/api/indexes/{index_id}")
    def get_index(index_id: str):
        index_id, index, backend = resolve_index(index_id)
        capabilities = None
        if isinstance(backend, QdrantIndexBackend):
            collection = index.manifest.get("backend_identity")
            capabilities = {
                "payload_filters": bool(collection) and backend.payload_filters_available(
                    collection, index,
                ),
            }
        return {
            "id": index_id,
            "manifest": index.manifest,
            "document_count": index.manifest.get("document_count", 0),
            "chunk_count": index.manifest.get("chunk_count", len(index.chunks)),
            "capabilities": capabilities,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id,
                    "text": chunk.text, "char_start": chunk.char_start,
                    "char_end": chunk.char_end, "metadata": chunk.metadata,
                    "vector": [float(value) for value in index.embeddings[position]],
                }
                for position, chunk in enumerate(index.chunks)
            ],
        }

    @app.post("/api/corpora/upload", status_code=201)
    async def upload_corpus(
        files: list[UploadFile] = File(...), name: str = Form(default="My corpus")
    ):
        if not files or len(files) > MAX_UPLOAD_FILES:
            raise HTTPException(422, f"Upload between 1 and {MAX_UPLOAD_FILES} files")
        corpus_id = f"custom-{uuid4().hex[:12]}"
        target = corpora_dir / corpus_id / "files"
        target.mkdir(parents=True)
        total = 0
        filenames: set[str] = set()
        try:
            for upload in files:
                filename = Path(upload.filename or "").name
                if not filename or filename in filenames:
                    raise HTTPException(422, "Each uploaded file must have a unique filename")
                if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
                    raise HTTPException(422, "Only Markdown and plain-text files are supported")
                filenames.add(filename)
                with (target / filename).open("wb") as destination:
                    while payload := await upload.read(UPLOAD_READ_BYTES):
                        if total + len(payload) > MAX_UPLOAD_BYTES:
                            raise HTTPException(422, "Upload exceeds the 100 MiB corpus limit")
                        destination.write(payload)
                        total += len(payload)
        except Exception:
            shutil.rmtree(corpora_dir / corpus_id, ignore_errors=True)
            raise
        corpus = {"id": corpus_id, "name": name.strip() or "My corpus", "kind": "custom", "file_count": len(files)}
        _write_json(corpora_dir / corpus_id / "corpus.json", corpus)
        return corpus

    @app.post("/api/corpora/watsonxdocsqa/import", status_code=202)
    def import_watsonxdocsqa(background_tasks: BackgroundTasks):
        """Run the existing reproducible corpus preparer only when requested."""
        corpus_id = "watsonxdocsqa"
        if (corpora_dir / corpus_id / "corpus.json").exists():
            return {"id": "watsonxdocsqa-ready", "status": "complete", "corpus_id": corpus_id}
        job_id = admit_job("watsonxDocsQA")

        def import_job():
            with _job_lock:
                if not job_store.start(job_id, total=2, message="Preparing corpus download"):
                    return
                staging_root = corpora_dir / f".{corpus_id}.staging"
                published = False
                try:
                    shutil.rmtree(staging_root, ignore_errors=True)
                    destination = staging_root / "files"
                    destination.mkdir(parents=True, exist_ok=True)
                    script = Path(__file__).resolve().parent.parent / "scripts" / "prepare_watsonx_docsqa.py"
                    subprocess.run(
                        [sys.executable, str(script), "--output-dir", str(destination)],
                        check=True, capture_output=True, text=True,
                    )
                    if not job_store.progress(job_id, 1, total=2, message="Corpus downloaded"):
                        shutil.rmtree(staging_root, ignore_errors=True)
                        return
                    file_count = len(list(destination.rglob("*.md")))
                    corpus = {"id": corpus_id, "name": "watsonxDocsQA", "kind": "catalog", "file_count": file_count}
                    _write_json(staging_root / "corpus.json", corpus)
                    if not job_store.progress(job_id, 2, total=2, message="Publishing corpus"):
                        shutil.rmtree(staging_root, ignore_errors=True)
                        return
                    job_store.update(job_id, artifact={
                        "kind": "corpus", "id": corpus_id,
                        "staging_name": staging_root.name,
                    })
                    if not job_store.begin_publish(job_id, message="Publishing corpus"):
                        shutil.rmtree(staging_root, ignore_errors=True)
                        return
                    os.replace(staging_root, corpora_dir / corpus_id)
                    published = True
                    job_store.complete(job_id, corpus_id=corpus_id)
                except Exception:
                    logger.exception("watsonxDocsQA import job %s failed", job_id)
                    shutil.rmtree(staging_root, ignore_errors=True)
                    if published:
                        job_store.complete(job_id, corpus_id=corpus_id)
                    else:
                        job_store.fail(job_id, "Corpus import failed. Check the local server logs and try again.")

        background_tasks.add_task(import_job)
        return {"id": job_id, "status": "queued"}

    @app.post("/api/indexes", status_code=202)
    def create_index(request: IndexRequest, background_tasks: BackgroundTasks):
        corpus_id = _safe_id(request.corpus_id, "corpus")
        corpus_path = corpora_dir / corpus_id / "files"
        if not corpus_path.exists():
            raise HTTPException(404, "Corpus not found")
        if not model_status()["ready"]:
            raise HTTPException(409, "Download the default embedding model before indexing this corpus")
        if request.index_backend == "qdrant" and not qdrant_is_available(
            os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
        ):
            raise HTTPException(
                409,
                "Qdrant is not ready. Start the optional Qdrant service, then try building again.",
            )
        job_id = admit_job("index", corpus_id=corpus_id)

        def index_job():
            with _job_lock:
                if not job_store.start(job_id, total=5, message="Loading documents"):
                    return
                staging_dir = None
                vector_backend = None
                collection = None
                qdrant_built = False
                published = False
                try:
                    docs = load_documents(corpus_path)
                    if not job_store.progress(job_id, 1, total=5, message="Chunking documents"):
                        return
                    embedder = SentenceTransformerEmbedder(local_files_only=True)
                    chunks = chunk_documents_with_strategy(
                        docs, strategy=request.chunking_strategy, chunk_size=request.chunk_size,
                        chunk_overlap=request.chunk_overlap,
                        embedder=embedder if request.chunking_strategy == "semantic" else None,
                        similarity_threshold=request.semantic_similarity_threshold,
                    )
                    if not job_store.progress(job_id, 2, total=5, message="Embedding chunks"):
                        return
                    embeddings = embedder.embed([chunk.text for chunk in chunks])
                    if not job_store.progress(job_id, 3, total=5, message="Writing local index"):
                        return
                    index_id = f"index-{uuid4().hex[:12]}"
                    collection = f"tiny_rag_{index_id.replace('-', '_')}"
                    staging_dir = indexes_dir / f".{index_id}.staging"
                    job_store.update(job_id, artifact={
                        "kind": "index", "id": index_id,
                        "staging_name": staging_dir.name,
                        "qdrant_collection": collection if request.index_backend == "qdrant" else None,
                    })
                    write_index(
                        staging_dir, docs, chunks, embeddings, corpus_root=corpus_path,
                        embedding_backend=type(embedder).__name__, embedding_model=embedder.model_name,
                        embedding_revision=getattr(embedder, "revision", None),
                        embedding_dim=embedder.dim, chunk_size=request.chunk_size,
                        chunk_overlap=request.chunk_overlap, chunking_strategy=request.chunking_strategy,
                        chunking_params={"similarity_threshold": request.semantic_similarity_threshold}
                        if request.chunking_strategy == "semantic" else {},
                        index_backend=request.index_backend,
                        backend_identity=collection if request.index_backend == "qdrant" else "numpy",
                        source_corpus_id=corpus_id,
                    )
                    if not job_store.progress(job_id, 4, total=5, message="Publishing vector backend"):
                        shutil.rmtree(staging_dir, ignore_errors=True)
                        return
                    staged_index = load_index(staging_dir)
                    vector_backend = backend_from_manifest(
                        staged_index.manifest,
                        qdrant_url=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"),
                    )
                    if request.index_backend == "qdrant":
                        # Ownership begins before build: Qdrant may create a
                        # partial collection and then raise.
                        qdrant_built = True
                        vector_backend.build(collection, staged_index)
                    if not job_store.progress(job_id, 5, total=5, message="Publishing index"):
                        if request.index_backend == "qdrant":
                            vector_backend.delete(collection)
                        shutil.rmtree(staging_dir, ignore_errors=True)
                        return
                    if not job_store.begin_publish(job_id, message="Publishing index"):
                        if qdrant_built:
                            vector_backend.delete(collection)
                        shutil.rmtree(staging_dir, ignore_errors=True)
                        return
                    staging_dir.replace(indexes_dir / index_id)
                    published = True
                    job_store.complete(job_id, index_id=index_id, corpus_id=corpus_id)
                except Exception:
                    logger.exception("Index job %s failed", job_id)
                    if staging_dir is not None:
                        shutil.rmtree(staging_dir, ignore_errors=True)
                    if qdrant_built and not published:
                        try:
                            vector_backend.delete(collection)
                        except Exception:
                            logger.exception("Failed to clean unpublished Qdrant collection %s", collection)
                            job_store.update(job_id, cleanup_pending=True)
                    if published:
                        job_store.complete(job_id, index_id=index_id, corpus_id=corpus_id)
                    else:
                        error = (
                            "Qdrant could not build this index. Confirm the local Qdrant service is still running, then try again."
                            if request.index_backend == "qdrant"
                            else "Indexing failed. Check the local server logs and try again."
                        )
                        job_store.fail(job_id, error)

        background_tasks.add_task(index_job)
        return {"id": job_id, "status": "queued"}

    @app.get("/api/jobs/active")
    def active_jobs(kind: str | None = None):
        return {"items": job_store.active(kind=kind)}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return job_store.read(_safe_id(job_id, "job"))
        except JobNotFoundError as exc:
            raise HTTPException(404, "Job not found") from exc

    @app.post("/api/jobs/{job_id}/cancel", status_code=202)
    def cancel_job(job_id: str):
        try:
            return job_store.request_cancel(_safe_id(job_id, "job"))
        except JobNotFoundError as exc:
            raise HTTPException(404, "Job not found") from exc

    @app.get("/api/jobs/{job_id}/result")
    def get_job_result(job_id: str):
        try:
            return job_store.result(_safe_id(job_id, "job"))
        except JobNotFoundError as exc:
            raise HTTPException(404, "Complete job result not found") from exc

    @app.post("/api/runs/retrieve", status_code=201)
    def retrieve(request: RunRequest):
        index_id, index, results, trace, query_vector, semantics, components, question, explanations, candidates, candidate_semantics = run_retrieval(request)
        return save_run(build_lab_run(
            trace, index_id=index_id, manifest=index.manifest,
            document_count=index.manifest.get("document_count", 0),
            evidence=[_evidence(result, semantics, score_components=components.get(result.chunk.chunk_id)) for result in results],
            query_vector=query_vector,
            candidates=(
                [_evidence(result, candidate_semantics) for result in candidates]
                if request.reranker != "none" else None
            ),
            explanations=explanations,
            config=run_config(request),
            catalog_check=catalog_check(question, results),
        ))

    @app.post("/api/runs/ask", status_code=201)
    def ask(request: RunRequest):
        provider = resolve_provider(request.provider)
        api_key = provider["api_key"]
        base_url = provider["base_url"]
        model = provider["model"]
        # A local OpenAI-compatible provider may intentionally have no key,
        # but a completely empty browser override is not a configured provider.
        if not provider_is_usable(provider):
            raise HTTPException(409, "Configure an OpenAI-compatible provider before live Ask")
        # The OpenAI SDK requires a non-empty key even for local compatible
        # servers (such as Ollama) that do not authenticate requests. This
        # placeholder exists only for the in-memory client construction and is
        # never stored in a run, job, or response.
        if base_url and not api_key:
            api_key = "local-provider-no-key"
        index_id, index, results, retrieve_trace, query_vector, semantics, components, question, explanations, rerank_candidates, candidate_semantics = run_retrieval(request)
        query = retrieve_trace.query
        candidate_results = list(results)
        if request.context_budget:
            packed = pack_context(results, request.context_budget, FakeTokenCounter(), question=query)
            selected = set(packed.selected)
            results = [result for result in results if result.chunk.chunk_id in selected]
        else:
            packed = None
            selected = {result.chunk.chunk_id for result in results}
        prompt = assemble_prompt(query, results)
        generator = OpenAIGenerator(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        t0 = time.perf_counter()
        try:
            answer = generator.generate(prompt)
        except Exception:
            logger.exception("Live generation failed for index %s", index_id)
            failed_trace = AskTrace(
                query=query, retriever=request.retriever, top_k=request.top_k,
                chunks=ask_chunk_traces(results, retrieve_trace), prompt=prompt,
                latency_by_stage={**retrieve_trace.latency_by_stage, "generate": time.perf_counter() - t0},
                context_pack=packed,
                reranker=retrieve_trace.reranker,
                rerank_top_n=retrieve_trace.rerank_top_n,
            )
            return save_run(build_lab_run(
                failed_trace, index_id=index_id, manifest=index.manifest,
                document_count=index.manifest.get("document_count", 0),
                evidence=[
                    _evidence(
                        result, semantics,
                        score_components=components.get(result.chunk.chunk_id),
                        selected_for_context=result.chunk.chunk_id in selected,
                    )
                    for result in candidate_results
                ],
                query_vector=query_vector,
                candidates=(
                    [_evidence(result, candidate_semantics) for result in rerank_candidates]
                    if request.reranker != "none" else None
                ),
                explanations=explanations,
                config=run_config(request),
                catalog_check=catalog_check(question, candidate_results),
                error="Live generation failed. Check your provider settings and try again.",
            ))
        # A model may repeat a marker or invent one not present in its prompt.
        # Keep only unique citations that resolve to the context actually sent
        # to it; the raw answer remains inspectable in the saved trace.
        available_citations = {result.chunk.chunk_id for result in results}
        citations = [citation for citation in extract_source_citations(answer) if citation in available_citations]
        trace = AskTrace(
            query=query, retriever=request.retriever, top_k=request.top_k,
            chunks=ask_chunk_traces(results, retrieve_trace), prompt=prompt,
            answer=answer, citations=citations,
            latency_by_stage={**retrieve_trace.latency_by_stage, "generate": time.perf_counter() - t0},
            context_pack=packed,
            reranker=retrieve_trace.reranker,
            rerank_top_n=retrieve_trace.rerank_top_n,
        )
        return save_run(build_lab_run(
            trace, index_id=index_id, manifest=index.manifest,
            document_count=index.manifest.get("document_count", 0),
            evidence=[
                _evidence(
                    result, semantics,
                    score_components=components.get(result.chunk.chunk_id),
                    selected_for_context=result.chunk.chunk_id in selected,
                )
                for result in candidate_results
            ],
            query_vector=query_vector,
            candidates=(
                [_evidence(result, candidate_semantics) for result in rerank_candidates]
                if request.reranker != "none" else None
            ),
            explanations=explanations,
            config=run_config(request),
            catalog_check=catalog_check(question, candidate_results),
        ))

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        path = runs_dir / f"{_safe_id(run_id, 'run')}.json"
        if not path.exists():
            raise HTTPException(404, "Run not found")
        return load_lab_run(path)

    if static_dir and Path(static_dir).exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="web")

    return app


app = create_app()


def create_packaged_app() -> FastAPI:
    """Factory used by Docker after the React assets have been built."""
    static_dir = Path(os.environ.get("TINY_RAG_LAB_STATIC_DIR", "/app/web-dist"))
    return create_app(static_dir=static_dir)
