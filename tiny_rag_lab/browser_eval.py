"""Fingerprint-gated retrieval comparison for the local browser course."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

from tiny_rag_lab.bm25 import BM25Retriever
from tiny_rag_lab.eval import (
    context_precision_at_k,
    context_recall_at_k,
    hit_at_k,
    reciprocal_rank,
)
from tiny_rag_lab.hybrid import reciprocal_rank_fusion
from tiny_rag_lab.index_loader import LoadedIndex, load_index
from tiny_rag_lab.qdrant_backend import source_vector_fingerprint
from tiny_rag_lab.reranker import CrossEncoderReranker, apply_reranker
from tiny_rag_lab.retrieval import retrieve_by_vector


class BrowserEvaluationError(RuntimeError):
    """A non-secret evaluation asset or configuration error."""


class BrowserEvaluationCancelled(RuntimeError):
    """Cooperative cancellation was accepted at a question boundary."""


# This identity is owned by the application release, not by the mutable bundle
# manifest beside the assets. Local metadata therefore cannot redefine which
# reviewed questions, chunks, vectors, or model revisions evaluation accepts.
BUNDLED_EVALUATION_IDENTITY = {
    "questions_sha256": "53dcbd1bd6eb14cc87510ad482032e357b53b6e61e9141d11af1722531981a36",
    "chunks_sha256": "11960c4f72360fdb4dd7fea1f43fbec4dd36a9214e3256bdcffc36ad3aee1f41",
    "embeddings_sha256": "c944c09db6e42fbdac3ec3e25dc74c6e6ea8b23802d612705ec6be512fa29604",
    "source_vector_fingerprint": "b4e7cd9f2a4dff45661dfea67ab4dfe265cce119fe731b39ab01ff8065965ff5",
    "document_count": 40,
    "chunk_count": 537,
    "distance_metric": "cosine",
    "embedding_dimension": 384,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "reranker_revision": "c5ee24cb16019beea0893ab7796b1df96625c6b8",
    "question_count": 16,
}


@dataclass(frozen=True)
class RetrievalConfiguration:
    retriever: Literal["bm25", "dense", "hybrid"]
    top_k: int = 5
    reranker: Literal["none", "cross-encoder"] = "none"
    rerank_top_n: int = 20

    def validate(self) -> None:
        if not 1 <= self.top_k <= 20:
            raise BrowserEvaluationError("top_k must be between 1 and 20")
        if not 1 <= self.rerank_top_n <= 50:
            raise BrowserEvaluationError("rerank_top_n must be between 1 and 50")
        if self.reranker != "none" and self.rerank_top_n < self.top_k:
            raise BrowserEvaluationError("rerank_top_n must be greater than or equal to top_k")

    def effective_identity(self) -> tuple[str, int, str, int | None]:
        """Ignore candidate depth when no reranker can consume it."""
        return (
            self.retriever, self.top_k, self.reranker,
            self.rerank_top_n if self.reranker != "none" else None,
        )


EVALUATION_PRESETS = [
    {
        "id": "bm25-vs-dense",
        "left": asdict(RetrievalConfiguration("bm25")),
        "right": asdict(RetrievalConfiguration("dense")),
    },
    {
        "id": "dense-vs-hybrid",
        "left": asdict(RetrievalConfiguration("dense")),
        "right": asdict(RetrievalConfiguration("hybrid")),
    },
    {
        "id": "hybrid-vs-reranked",
        "left": asdict(RetrievalConfiguration("hybrid")),
        "right": asdict(RetrievalConfiguration(
            "hybrid", reranker="cross-encoder", rerank_top_n=20,
        )),
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reviewed_questions(path: Path) -> list[dict]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    ids = [item.get("question_id") for item in items]
    categories = [item.get("category") for item in items]
    if len(items) != 16 or len(set(ids)) != 16:
        raise BrowserEvaluationError("The reviewed evaluation set must contain 16 unique questions")
    if any(not item.get("question") or not item.get("gold_doc_ids") for item in items):
        raise BrowserEvaluationError("Every reviewed question needs text and gold document IDs")
    if {category: categories.count(category) for category in set(categories)} != {
        "lexical": 4, "dense": 4, "hybrid": 4, "reranking": 4,
    }:
        raise BrowserEvaluationError("The reviewed evaluation set must contain four questions per category")
    return items


def validate_evaluation_bundle(
    index_dir: Path, questions_path: Path, manifest_path: Path,
    *, trusted_identity: dict | None = None,
) -> tuple[LoadedIndex, list[dict], dict]:
    """Validate every immutable input before browser evaluation starts."""
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        index = load_index(index_dir)
        questions = load_reviewed_questions(questions_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise BrowserEvaluationError("The bundled evaluation assets are unavailable or invalid") from exc

    actual = {
        "questions_sha256": sha256_file(questions_path),
        "chunks_sha256": sha256_file(Path(index_dir) / "chunks.jsonl"),
        "embeddings_sha256": sha256_file(Path(index_dir) / "embeddings.npz"),
        "source_vector_fingerprint": source_vector_fingerprint(index),
        "document_count": index.manifest.get("document_count"),
        "chunk_count": len(index.chunks),
        "distance_metric": index.manifest.get("distance_metric"),
        "embedding_dimension": int(index.embeddings.shape[1]),
        "embedding_model": index.manifest.get("embedding_model"),
        "embedding_revision": index.manifest.get("embedding_revision"),
        "reranker_model": CrossEncoderReranker.DEFAULT_MODEL,
        "reranker_revision": CrossEncoderReranker.DEFAULT_REVISION,
        "question_count": len(questions),
    }
    mismatches = [key for key, value in actual.items() if manifest.get(key) != value]
    trusted = BUNDLED_EVALUATION_IDENTITY if trusted_identity is None else trusted_identity
    mismatches.extend(
        key for key, expected in trusted.items()
        if actual.get(key) != expected or manifest.get(key) != expected
    )
    if index.manifest.get("index_backend") != "numpy":
        mismatches.append("index_backend")
    known_documents = {chunk.doc_id for chunk in index.chunks}
    if any(
        not set(question["gold_doc_ids"]).issubset(known_documents)
        for question in questions
    ):
        mismatches.append("gold_doc_ids")
    if mismatches:
        raise BrowserEvaluationError(
            "The bundled evaluation assets do not match their immutable manifest: "
            + ", ".join(sorted(set(mismatches)))
        )
    return index, questions, manifest


def _serialize_results(results) -> list[dict]:
    return [
        {
            "chunk_id": result.chunk.chunk_id,
            "doc_id": result.chunk.doc_id,
            "title": result.chunk.metadata.get("title", ""),
            "path": result.chunk.metadata.get("path", result.chunk.doc_id),
            "text": result.chunk.text,
            "rank": result.rank,
            "score": result.score,
        }
        for result in results
    ]


def _metrics(question: dict, results) -> dict:
    retrieved = [result.chunk.doc_id for result in results]
    gold = list(question["gold_doc_ids"])
    return {
        "hit": hit_at_k(retrieved, gold),
        "reciprocal_rank": reciprocal_rank(retrieved, gold),
        "context_precision": context_precision_at_k(retrieved, gold),
        "context_recall": context_recall_at_k(retrieved, gold),
    }


def _aggregate(items: list[dict]) -> dict:
    count = len(items)
    if count == 0:
        raise BrowserEvaluationError("The evaluation set cannot be empty")
    return {
        "n_questions": count,
        "hit_rate": sum(float(item["metrics"]["hit"]) for item in items) / count,
        "mrr": sum(item["metrics"]["reciprocal_rank"] for item in items) / count,
        "context_precision": sum(item["metrics"]["context_precision"] for item in items) / count,
        "context_recall": sum(item["metrics"]["context_recall"] for item in items) / count,
    }


def run_browser_comparison(
    questions: list[dict], index: LoadedIndex,
    left: RetrievalConfiguration, right: RetrievalConfiguration,
    *, embedder, reranker_factory: Callable[[], object] | None = None,
    checkpoint: Callable[[int, int, str], bool] | None = None,
) -> dict:
    """Run two configurations and preserve every final evidence list."""
    left.validate()
    right.validate()
    if left.effective_identity() == right.effective_identity():
        raise BrowserEvaluationError("Choose two different retrieval configurations")
    bm25 = BM25Retriever(index.chunks)
    factory = reranker_factory or (
        lambda: CrossEncoderReranker(local_files_only=True)
    )
    rerankers = {
        "left": factory() if left.reranker == "cross-encoder" else None,
        "right": factory() if right.reranker == "cross-encoder" else None,
    }

    def run_one(question: dict, config: RetrievalConfiguration, reranker):
        retrieval_k = config.rerank_top_n if reranker is not None else config.top_k
        if config.retriever == "bm25":
            results = bm25.retrieve(question["question"], retrieval_k)
        else:
            query_vector = embedder.embed([question["question"]])[0]
            dense = retrieve_by_vector(query_vector, index, retrieval_k)
            if config.retriever == "hybrid":
                lexical = bm25.retrieve(question["question"], retrieval_k)
                results = reciprocal_rank_fusion([dense, lexical], retrieval_k)
            else:
                results = dense
        if reranker is not None:
            results, _audit = apply_reranker(
                question["question"], results, reranker, config.top_k,
            )
        return results

    left_items = []
    right_items = []
    total = len(questions)
    for position, question in enumerate(questions, start=1):
        left_results = run_one(question, left, rerankers["left"])
        if checkpoint is not None and not checkpoint(position - 1, total, "left"):
            raise BrowserEvaluationCancelled()
        right_results = run_one(question, right, rerankers["right"])
        base = {
            "question_id": question["question_id"],
            "question": question["question"],
            "gold_doc_ids": list(question["gold_doc_ids"]),
            "category": question["category"],
        }
        left_items.append({**base, "metrics": _metrics(question, left_results), "evidence": _serialize_results(left_results)})
        right_items.append({**base, "metrics": _metrics(question, right_results), "evidence": _serialize_results(right_results)})
        if checkpoint is not None and not checkpoint(position, total, "right"):
            raise BrowserEvaluationCancelled()

    return {
        "question_count": total,
        "left": {"config": asdict(left), "metrics": _aggregate(left_items), "questions": left_items},
        "right": {"config": asdict(right), "metrics": _aggregate(right_items), "questions": right_items},
    }
