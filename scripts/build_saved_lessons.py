"""Capture the four offline Guided Learn artifacts from the pinned index.

The answers are short, reviewed teaching copy.  Retrieval, context packing,
prompt assembly, vectors, evidence, and citations are captured from the real
structural Cloudflare index so a replay never needs a provider or a network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from tiny_rag_lab.context import FakeTokenCounter, pack_context
from tiny_rag_lab.embeddings import SentenceTransformerEmbedder
from tiny_rag_lab.index_backend import NumpyIndexBackend
from tiny_rag_lab.index_loader import load_index
from tiny_rag_lab.lab_trace import EvidenceSnapshot, build_lab_run, lab_run_to_dict
from tiny_rag_lab.prompting import assemble_prompt
from tiny_rag_lab.trace import AskTrace, ChunkTrace


PACKAGE_ID = "cloudflare-state-coordination-v1"
CORPUS_ID = "cloudflare-state-v1"
INDEX_ID = "cloudflare-state-structural-v1"
LESSONS = (
    {
        "id": "cloudflare-do-coordinator-v1",
        "order": 1,
        "title": "Route one entity to one coordinator",
        "question": "How can a Worker use a Durable Object namespace, stable ID, and stub to send requests for the same entity to one stateful coordinator?",
        "focus": "Document-to-chunk provenance, Durable Object identity/stubs, and rank ordering.",
        "required_supporting_document_ids": [
            "durable-objects/best-practices/rules-of-durable-objects.md",
            "durable-objects/best-practices/create-durable-object-stubs-and-send-requests.md",
        ],
        "answer": (
            "Bind a Durable Object namespace to the Worker, derive a stable Durable Object ID for the entity, "
            "obtain that object's stub from the namespace, and send the request to the stub. The stable ID makes "
            "requests for the same entity address the same single Durable Object coordinator."
        ),
    },
    {
        "id": "cloudflare-queues-retries-v1",
        "order": 2,
        "title": "Inspect retry and batching decisions",
        "question": "If a queue consumer fails while processing a message, how should retries and batching be considered?",
        "focus": "Retrieval candidates versus context selection for delivery behavior.",
        "required_supporting_document_ids": ["queues/configuration/batching-retries.md"],
        "answer": (
            "Treat a failed delivery as a retry decision rather than assuming exactly-once processing. Configure the "
            "consumer's batch size and timeout for the work it performs, then use retry and dead-letter settings to "
            "bound repeated failures. Consumer code should be safe when a message is delivered again."
        ),
    },
    {
        "id": "cloudflare-kv-r2-choice-v1",
        "order": 3,
        "title": "Choose KV or R2 from access semantics",
        "question": "What is the tradeoff between Workers KV eventual consistency for global configuration and R2 object storage for mutable files?",
        "focus": "Multi-document evidence and a constrained context pack.",
        "required_supporting_document_ids": [
            "kv/concepts/how-kv-works.md", "r2/reference/consistency.md",
        ],
        "answer": (
            "Use KV for configuration that is read globally and benefits from low-latency replicated reads, while "
            "accounting for its eventual-consistency behavior. Use R2 for mutable file or object content, where object "
            "storage and its consistency/durability model match the application's file lifecycle."
        ),
    },
    {
        "id": "cloudflare-workflows-resume-v1",
        "order": 4,
        "title": "Pause and resume durable work",
        "question": "How can a long-running multi-step process pause, retry, and resume safely?",
        "focus": "Prompt construction, saved grounded answer, and citations.",
        "required_supporting_document_ids": [
            "workflows/get-started/guide.md", "workflows/build/sleeping-and-retrying.md",
        ],
        "answer": (
            "Model the process as a Workflow with durable steps. Put retryable work in steps, use the Workflow sleep "
            "mechanism for waits, and keep state at step boundaries so the runtime can resume from completed progress "
            "instead of replaying the entire process after an interruption."
        ),
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_digest(root: Path) -> str:
    """Stable digest of every immutable index artifact, not just its manifest."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _chunk_trace(result) -> ChunkTrace:
    return ChunkTrace(
        rank=result.rank, chunk_id=result.chunk.chunk_id, doc_id=result.chunk.doc_id,
        title=result.chunk.metadata.get("title", ""),
        path=result.chunk.metadata.get("path", result.chunk.doc_id), score=result.score,
        text_preview=result.chunk.text[:120].replace("\n", " ").strip(),
    )


def _evidence(result, selected: set[str]) -> EvidenceSnapshot:
    return EvidenceSnapshot(
        chunk_id=result.chunk.chunk_id, doc_id=result.chunk.doc_id,
        title=result.chunk.metadata.get("title", ""),
        path=result.chunk.metadata.get("path", result.chunk.doc_id), text=result.chunk.text,
        rank=result.rank, score=result.score, score_semantics="cosine_similarity[-1,1]",
        score_components={"dense_score": result.score, "dense_rank": float(result.rank)},
        selected_for_context=result.chunk.chunk_id in selected,
    )


def _pack_with_supporting_contrast(results, lesson: dict):
    """Find the smallest teaching budget that keeps required support visible.

    Saved lessons should demonstrate both a selected and an omitted candidate.
    This searches the small, fixed five-result candidate set rather than
    assuming a hand-picked token limit will remain correct if chunk text moves.
    """
    required_docs = set(lesson["required_supporting_document_ids"])
    counter = FakeTokenCounter()
    for budget in range(1, 4_001):
        packed = pack_context(results, budget, counter, question=lesson["question"])
        selected = [result for result in results if result.chunk.chunk_id in packed.selected]
        if (
            packed.selected
            and packed.omitted
            and required_docs.issubset({result.chunk.doc_id for result in selected})
        ):
            return packed
    raise ValueError(
        f"Lesson {lesson['id']} cannot retain its required support while omitting a candidate"
    )


def build(seed_root: Path) -> None:
    seed_root = Path(seed_root)
    corpus_dir = seed_root / "corpora" / CORPUS_ID
    index_dir = seed_root / "indexes" / INDEX_ID
    index = load_index(index_dir)
    source_manifest = json.loads((corpus_dir / "source-manifest.json").read_text(encoding="utf-8"))
    source_snapshot = {
        "corpus_id": CORPUS_ID,
        "source_revision": source_manifest["revision"],
        "source_repository": source_manifest["repository"],
        "source_license": source_manifest["license"],
        "corpus_digest": _directory_digest(corpus_dir),
        "index_id": INDEX_ID,
        "index_digest": _directory_digest(index_dir),
    }
    output = seed_root / "lessons" / PACKAGE_ID
    output.mkdir(parents=True, exist_ok=True)
    embedder = SentenceTransformerEmbedder(local_files_only=True)
    backend = NumpyIndexBackend()

    for lesson in LESSONS:
        started = time.perf_counter()
        vector = embedder.embed([lesson["question"]])[0]
        embed_seconds = time.perf_counter() - started
        started = time.perf_counter()
        results = [hit.result for hit in backend.search(vector, index, top_k=5)]
        retrieve_seconds = time.perf_counter() - started
        packed = _pack_with_supporting_contrast(results, lesson)
        selected_ids = set(packed.selected)
        selected = [result for result in results if result.chunk.chunk_id in selected_ids]
        prompt = assemble_prompt(lesson["question"], selected)
        supporting_ids = [
            result.chunk.chunk_id for result in selected
            if result.chunk.doc_id in lesson["required_supporting_document_ids"]
        ]
        if not supporting_ids:
            raise ValueError(f"Lesson {lesson['id']} has no selected supporting chunks")
        # Saved answers are reviewed teaching copy.  Cite the selected chunks
        # that explicitly support it, just as a generated answer must do.
        citations = list(dict.fromkeys(supporting_ids))
        selected_ids = {result.chunk.chunk_id for result in selected}
        if not set(supporting_ids).issubset(selected_ids):
            raise ValueError(f"Lesson {lesson['id']} cites an unselected supporting chunk")
        if not set(lesson["required_supporting_document_ids"]).issubset(
            {result.chunk.doc_id for result in selected if result.chunk.chunk_id in supporting_ids}
        ):
            raise ValueError(f"Lesson {lesson['id']} does not cover every required support document")
        trace = AskTrace(
            query=lesson["question"], retriever="dense", top_k=5,
            chunks=[_chunk_trace(result) for result in selected], prompt=prompt,
            answer=lesson["answer"], citations=citations,
            latency_by_stage={"embed": embed_seconds, "retrieve": retrieve_seconds, "prompt_assembly": 0.0},
            context_pack=packed,
        )
        run = build_lab_run(
            trace, index_id=INDEX_ID, manifest=index.manifest,
            document_count=index.manifest["document_count"],
            evidence=[_evidence(result, selected_ids) for result in results],
            query_vector=[float(value) for value in vector],
            config={"retriever": "dense", "top_k": 5, "context_budget": packed.budget,
                    "answer_provenance": "recorded_lesson_result"},
            source_snapshot=source_snapshot, mode="saved_lesson",
        )
        run.run_id = f"lesson-{lesson['id']}"
        payload = {
            "lesson": {**lesson, "package_id": PACKAGE_ID,
                       "answer_supporting_chunk_ids": supporting_ids,
                       "retrieval_configuration": {"retriever": "dense", "top_k": 5, "context_budget": packed.budget},
                       "answer_provenance": "recorded_lesson_result", "source_snapshot": source_snapshot},
            "run": lab_run_to_dict(run),
        }
        (output / f"{lesson['id']}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    manifest = {"id": PACKAGE_ID, "schema_version": 1,
                "lessons": [{key: lesson[key] for key in ("id", "order", "title", "question", "focus")} for lesson in LESSONS]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Built {len(LESSONS)} saved lessons in {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", type=Path, default=Path("assets/seed/v1"))
    args = parser.parse_args()
    build(args.seed_root)


if __name__ == "__main__":
    main()
