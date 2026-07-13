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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tiny_rag_lab.bm25 import BM25Retriever
from tiny_rag_lab.chunking import chunk_documents_with_strategy
from tiny_rag_lab.context import FakeTokenCounter, pack_context
from tiny_rag_lab.documents import load_documents
from tiny_rag_lab.embeddings import SentenceTransformerEmbedder
from tiny_rag_lab.generation import OpenAIGenerator
from tiny_rag_lab.hybrid import reciprocal_rank_fusion
from tiny_rag_lab.index_backend import NumpyIndexBackend, backend_from_manifest
from tiny_rag_lab.index_loader import load_index
from tiny_rag_lab.index_writer import write_index
from tiny_rag_lab.lab_trace import EvidenceSnapshot, build_lab_run, load_lab_run, write_lab_run
from tiny_rag_lab.prompting import assemble_prompt, extract_source_citations
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
    query: str = Field(min_length=1)
    retriever: Literal["dense", "bm25", "hybrid"] = "dense"
    top_k: int = Field(default=5, ge=1, le=50)
    context_budget: int = Field(default=0, ge=0)
    provider: ProviderOverride | None = None


def _safe_id(value: str, kind: str) -> str:
    if not value or value != Path(value).name or value in {".", ".."}:
        raise HTTPException(422, f"Invalid {kind} identifier")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


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
    # FastAPI BackgroundTasks are process-bound. A persisted queued/running
    # record after restart cannot resume safely, so expose a clear retryable
    # failure instead of leaving the local UI polling forever.
    for job_path in jobs_dir.glob("*.json"):
        try:
            job = _read_json(job_path)
        except (OSError, json.JSONDecodeError):
            continue
        if job.get("status") in {"queued", "running"}:
            job["status"] = "failed"
            job["error"] = "The local server restarted before this job completed. Please start it again."
            _write_json(job_path, job)

    app = FastAPI(title="tiny-rag-lab local API", docs_url=None, redoc_url=None)
    job_admission_lock = threading.Lock()
    # The packaged browser client uses the same origin. This only helps native
    # development and deliberately does not open the server beyond loopback.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["*"], allow_headers=["*"], allow_credentials=False,
    )

    def provider_status() -> dict:
        return {
            "configured": bool(os.environ.get("OPENAI_API_KEY")),
            "base_url_configured": bool(os.environ.get("OPENAI_BASE_URL")),
            "model_configured": bool(os.environ.get("OPENAI_MODEL")),
        }

    def model_status() -> dict:
        try:
            SentenceTransformerEmbedder(local_files_only=True)
        except Exception:
            return {"ready": False, "variant": os.environ.get("LAB_IMAGE_VARIANT", "native")}
        return {"ready": True, "variant": os.environ.get("LAB_IMAGE_VARIANT", "native")}

    def save_run(run) -> dict:
        path = runs_dir / f"{run.run_id}.json"
        write_lab_run(run, path)
        return load_lab_run(path)

    def admit_job(kind: str, **fields: str) -> tuple[str, Path]:
        """Persist one visible queued job or reject a concurrent request."""
        with job_admission_lock:
            for existing_path in jobs_dir.glob("*.json"):
                try:
                    existing = _read_json(existing_path)
                except (OSError, json.JSONDecodeError):
                    continue
                if existing.get("status") in {"queued", "running"}:
                    raise HTTPException(
                        409,
                        f"Local job {existing.get('id', existing_path.stem)} is {existing['status']}. Wait for it before starting another job.",
                    )
            job_id = f"{kind}-{uuid4().hex[:12]}"
            job_path = jobs_dir / f"{job_id}.json"
            _write_json(job_path, {"id": job_id, "status": "queued", "kind": kind, **fields})
            return job_id, job_path

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

    def run_retrieval(request: RunRequest):
        index_id, index, vector_backend = resolve_index(request.index_id)
        t0 = time.perf_counter()
        query_vector: list[float] | None = None
        latency: dict[str, float] = {}
        semantics = "cosine_similarity[-1,1]"

        score_components: dict[str, dict[str, float]] = {}
        if request.retriever == "bm25":
            results = BM25Retriever(index.chunks).retrieve(request.query, request.top_k)
            semantics = "bm25_score"
            score_components = {
                result.chunk.chunk_id: {"bm25_score": result.score, "bm25_rank": float(result.rank)}
                for result in results
            }
        else:
            model_name = index.manifest.get("embedding_model")
            embedder = SentenceTransformerEmbedder(model_name)
            query_vec = embedder.embed([request.query])[0]
            query_vector = [float(value) for value in query_vec]
            latency["embed"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            try:
                dense_hits = vector_backend.search(query_vec, index, request.top_k)
            except Exception as exc:
                from tiny_rag_lab.qdrant_backend import QdrantBackendError
                if isinstance(exc, QdrantBackendError):
                    raise HTTPException(503, str(exc)) from exc
                raise
            dense_results = [hit.result for hit in dense_hits]
            semantics = dense_hits[0].score_semantics if dense_hits else vector_backend.score_semantics
            if request.retriever == "hybrid":
                bm25_results = BM25Retriever(index.chunks).retrieve(request.query, request.top_k)
                results = reciprocal_rank_fusion([dense_results, bm25_results], request.top_k)
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
        trace = RetrieveTrace(
            query=request.query, retriever=request.retriever, top_k=request.top_k,
            chunks=[_chunk_trace(result) for result in results], latency_by_stage=latency,
        )
        return index_id, index, results, trace, query_vector, semantics, score_components

    @app.get("/api/health")
    def health():
        return {"status": "ok", "data_dir": str(root)}

    @app.get("/api/provider-status")
    def get_provider_status():
        return provider_status()

    @app.get("/api/models/default/status")
    def get_model_status():
        return model_status()

    @app.post("/api/models/default/download", status_code=202)
    def download_default_model(background_tasks: BackgroundTasks):
        if model_status()["ready"]:
            return {"id": "embedding-model-ready", "status": "complete"}
        job_id, job_path = admit_job("embedding-model")

        def download_job():
            with _job_lock:
                _write_json(job_path, {"id": job_id, "status": "running", "kind": "embedding-model"})
                try:
                    SentenceTransformerEmbedder()
                    _write_json(job_path, {"id": job_id, "status": "complete", "kind": "embedding-model"})
                except Exception:
                    logger.exception("Embedding-model job %s failed", job_id)
                    _write_json(job_path, {"id": job_id, "status": "failed", "kind": "embedding-model", "error": "Model download failed. Check your network and try again."})

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
        index_id, index, _ = resolve_index(index_id)
        return {
            "id": index_id,
            "manifest": index.manifest,
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
        job_id, job_path = admit_job("watsonxDocsQA")

        def import_job():
            with _job_lock:
                _write_json(job_path, {"id": job_id, "status": "running", "kind": "watsonxDocsQA"})
                try:
                    destination = corpora_dir / corpus_id / "files"
                    destination.mkdir(parents=True, exist_ok=True)
                    script = Path(__file__).resolve().parent.parent / "scripts" / "prepare_watsonx_docsqa.py"
                    subprocess.run(
                        [sys.executable, str(script), "--output-dir", str(destination)],
                        check=True, capture_output=True, text=True,
                    )
                    file_count = len(list(destination.rglob("*.md")))
                    corpus = {"id": corpus_id, "name": "watsonxDocsQA", "kind": "catalog", "file_count": file_count}
                    _write_json(corpora_dir / corpus_id / "corpus.json", corpus)
                    _write_json(job_path, {"id": job_id, "status": "complete", "kind": "watsonxDocsQA", "corpus_id": corpus_id})
                except Exception:
                    logger.exception("watsonxDocsQA import job %s failed", job_id)
                    _write_json(job_path, {"id": job_id, "status": "failed", "kind": "watsonxDocsQA", "error": "Corpus import failed. Check the local server logs and try again."})

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
        job_id, job_path = admit_job("index", corpus_id=corpus_id)

        def index_job():
            with _job_lock:
                _write_json(job_path, {"id": job_id, "status": "running", "kind": "index", "corpus_id": corpus_id})
                try:
                    docs = load_documents(corpus_path)
                    embedder = SentenceTransformerEmbedder(local_files_only=True)
                    chunks = chunk_documents_with_strategy(
                        docs, strategy=request.chunking_strategy, chunk_size=request.chunk_size,
                        chunk_overlap=request.chunk_overlap,
                        embedder=embedder if request.chunking_strategy == "semantic" else None,
                        similarity_threshold=request.semantic_similarity_threshold,
                    )
                    embeddings = embedder.embed([chunk.text for chunk in chunks])
                    index_id = f"index-{uuid4().hex[:12]}"
                    collection = f"tiny_rag_{index_id.replace('-', '_')}"
                    staging_dir = indexes_dir / f".{index_id}.staging"
                    write_index(
                        staging_dir, docs, chunks, embeddings, corpus_root=corpus_path,
                        embedding_backend=type(embedder).__name__, embedding_model=embedder.model_name,
                        embedding_dim=embedder.dim, chunk_size=request.chunk_size,
                        chunk_overlap=request.chunk_overlap, chunking_strategy=request.chunking_strategy,
                        chunking_params={"similarity_threshold": request.semantic_similarity_threshold}
                        if request.chunking_strategy == "semantic" else {},
                        index_backend=request.index_backend,
                        backend_identity=collection if request.index_backend == "qdrant" else "numpy",
                    )
                    staged_index = load_index(staging_dir)
                    vector_backend = backend_from_manifest(
                        staged_index.manifest,
                        qdrant_url=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"),
                    )
                    if request.index_backend == "qdrant":
                        vector_backend.build(collection, staged_index)
                    staging_dir.replace(indexes_dir / index_id)
                    _write_json(job_path, {"id": job_id, "status": "complete", "kind": "index", "index_id": index_id})
                except Exception:
                    logger.exception("Index job %s failed", job_id)
                    shutil.rmtree(locals().get("staging_dir", indexes_dir / ".missing"), ignore_errors=True)
                    _write_json(job_path, {"id": job_id, "status": "failed", "kind": "index", "error": "Indexing failed. Check the local server logs and try again."})

        background_tasks.add_task(index_job)
        return {"id": job_id, "status": "queued"}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        path = jobs_dir / f"{_safe_id(job_id, 'job')}.json"
        if not path.exists():
            raise HTTPException(404, "Job not found")
        return _read_json(path)

    @app.post("/api/runs/retrieve", status_code=201)
    def retrieve(request: RunRequest):
        index_id, index, results, trace, query_vector, semantics, components = run_retrieval(request)
        return save_run(build_lab_run(
            trace, index_id=index_id, manifest=index.manifest,
            document_count=index.manifest.get("document_count", 0),
            evidence=[_evidence(result, semantics, score_components=components.get(result.chunk.chunk_id)) for result in results],
            query_vector=query_vector,
            config={"retriever": request.retriever, "top_k": request.top_k, "context_budget": request.context_budget},
        ))

    @app.post("/api/runs/ask", status_code=201)
    def ask(request: RunRequest):
        override = request.provider or ProviderOverride()
        api_key = override.api_key or os.environ.get("OPENAI_API_KEY")
        base_url = override.base_url or os.environ.get("OPENAI_BASE_URL")
        model = override.model or os.environ.get("OPENAI_MODEL")
        # A local OpenAI-compatible provider may intentionally have no key,
        # but a completely empty browser override is not a configured provider.
        if not api_key and not base_url:
            raise HTTPException(409, "Configure an OpenAI-compatible provider before live Ask")
        # The OpenAI SDK requires a non-empty key even for local compatible
        # servers (such as Ollama) that do not authenticate requests. This
        # placeholder exists only for the in-memory client construction and is
        # never stored in a run, job, or response.
        if base_url and not api_key:
            api_key = "local-provider-no-key"
        index_id, index, results, retrieve_trace, query_vector, semantics, components = run_retrieval(request)
        candidate_results = list(results)
        if request.context_budget:
            packed = pack_context(results, request.context_budget, FakeTokenCounter(), question=request.query)
            selected = set(packed.selected)
            results = [result for result in results if result.chunk.chunk_id in selected]
        else:
            packed = None
            selected = {result.chunk.chunk_id for result in results}
        prompt = assemble_prompt(request.query, results)
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
                query=request.query, retriever=request.retriever, top_k=request.top_k,
                chunks=[_chunk_trace(result) for result in results], prompt=prompt,
                latency_by_stage={**retrieve_trace.latency_by_stage, "generate": time.perf_counter() - t0},
                context_pack=packed,
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
                config={"retriever": request.retriever, "top_k": request.top_k, "context_budget": request.context_budget},
                error="Live generation failed. Check your provider settings and try again.",
            ))
        citations = extract_source_citations(answer)
        trace = AskTrace(
            query=request.query, retriever=request.retriever, top_k=request.top_k,
            chunks=[_chunk_trace(result) for result in results], prompt=prompt,
            answer=answer, citations=citations,
            latency_by_stage={**retrieve_trace.latency_by_stage, "generate": time.perf_counter() - t0},
            context_pack=packed,
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
            config={"retriever": request.retriever, "top_k": request.top_k, "context_budget": request.context_budget},
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
