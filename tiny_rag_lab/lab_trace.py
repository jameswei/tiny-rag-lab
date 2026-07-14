"""Immutable, JSON-safe artifacts for visual RAG-lab replay.

CLI traces intentionally stay compact.  These records add the full evidence
and configuration a browser needs to replay a run without recomputing it or
depending on a later-mutated index.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from tiny_rag_lab.trace import AskTrace, RetrieveTrace, trace_to_dict

LAB_TRACE_SCHEMA_VERSION = "1.0"


@dataclass
class EvidenceSnapshot:
    chunk_id: str
    doc_id: str
    title: str
    path: str
    text: str
    rank: int
    score: float
    score_semantics: str
    score_components: dict[str, float] = field(default_factory=dict)
    selected_for_context: bool | None = None
    pre_rerank_rank: int | None = None
    pre_rerank_score: float | None = None


@dataclass
class IndexSnapshot:
    index_id: str
    manifest: dict[str, Any]
    document_count: int
    chunk_count: int


@dataclass
class LabRun:
    """A replayable retrieve or ask run with no provider secret fields."""

    operation: Literal["retrieve", "ask"]
    index: IndexSnapshot
    trace: dict[str, Any]
    evidence: list[EvidenceSnapshot]
    query_vector: list[float] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    # Present only for a catalog-backed run.  Keeping this server-produced
    # makes it impossible for the browser to accidentally compare a question
    # against gold data from a different corpus.
    catalog_check: dict[str, Any] | None = None
    # Saved lessons record the pinned corpus/source identity alongside the
    # immutable index snapshot.  Ordinary interactive runs need not duplicate
    # that data because it already lives in their index manifest.
    source_snapshot: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    mode: Literal["live", "replay", "saved_lesson"] = "live"
    run_id: str = field(default_factory=lambda: str(uuid4()))
    schema_version: str = LAB_TRACE_SCHEMA_VERSION
    error: str | None = None


def build_lab_run(
    trace: RetrieveTrace | AskTrace,
    *,
    index_id: str,
    manifest: dict[str, Any],
    document_count: int,
    evidence: list[EvidenceSnapshot],
    query_vector: list[float] | None = None,
    config: dict[str, Any] | None = None,
    catalog_check: dict[str, Any] | None = None,
    source_snapshot: dict[str, Any] | None = None,
    mode: Literal["live", "replay", "saved_lesson"] = "live",
    error: str | None = None,
) -> LabRun:
    return LabRun(
        operation="ask" if isinstance(trace, AskTrace) else "retrieve",
        index=IndexSnapshot(
            index_id=index_id,
            manifest=manifest,
            document_count=document_count,
            chunk_count=manifest.get("chunk_count", len(evidence)),
        ),
        trace=trace_to_dict(trace),
        evidence=evidence,
        query_vector=query_vector,
        config=config or {},
        catalog_check=catalog_check,
        source_snapshot=source_snapshot,
        mode=mode,
        error=error,
    )


def lab_run_to_dict(run: LabRun) -> dict[str, Any]:
    return dataclasses.asdict(run)


def write_lab_run(run: LabRun, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lab_run_to_dict(run), indent=2), encoding="utf-8")


def load_lab_run(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
