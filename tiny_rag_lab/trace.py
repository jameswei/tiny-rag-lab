"""Trace data contracts for the observability plane.

Phase 1.7 scope: per-query trace records for the retrieve and ask flows.
Phase 2.0: AskTrace gains an optional verdict field (JudgeVerdict | None).
Phase 2.1: AskTrace gains an optional context_pack field (ContextPackResult | None).

Serialization and formatters live in the same module to keep the observability
mechanics visible in one place.

All dataclass fields are JSON-native types (str, int, float, list, dict, None)
so dataclasses.asdict() + json.dumps() serializes any trace without a custom
encoder. JudgeVerdict and ContextPackResult are also dataclasses with JSON-native
fields — they nest cleanly under their respective keys.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from tiny_rag_lab.context import ContextPackResult

if TYPE_CHECKING:
    from tiny_rag_lab.judge import JudgeVerdict


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class ChunkTrace:
    """Compact, JSON-safe summary of one retrieved chunk.

    Covers the roadmap's enumerated fields: rank, score, chunk_id, doc_id,
    title, path. text_preview (first 120 chars of chunk.text) is added for
    the human-readable formatter — it avoids embedding full chunk text in the
    JSON trace while still making the output readable.

    pre_rerank_rank and pre_rerank_score are populated by Phase 1.9 reranking
    integrations. They stay None when no reranker ran, so existing Phase 1
    through 1.8 trace consumers see the same rank / score semantics they
    always have.
    """

    rank: int
    chunk_id: str
    doc_id: str
    title: str
    path: str
    score: float
    text_preview: str
    pre_rerank_rank: int | None = None
    pre_rerank_score: float | None = None


@dataclass
class RetrieveTrace:
    """Full trace record for one rag retrieve call.

    latency_by_stage keys:
      "load"     — index loading from disk
      "embed"    — query embedding (dense and hybrid only; absent for bm25)
      "retrieve" — ranking and top-k selection
      "rerank"   — cross-encoder reranking (Phase 1.9; absent when no reranker)
    """

    query: str
    retriever: str            # "dense" | "bm25" | "hybrid"
    top_k: int
    chunks: list[ChunkTrace] = field(default_factory=list)
    latency_by_stage: dict[str, float] = field(default_factory=dict)
    reranker: str = "none"          # Phase 1.9
    rerank_top_n: int | None = None # Phase 1.9


@dataclass
class AskTrace:
    """Full trace record for one rag ask call. Replaces models.RagTrace.

    latency_by_stage keys:
      "load"            — index loading from disk
      "embed"           — query embedding
      "retrieve"        — ranking and top-k selection
      "prompt_assembly" — prompt construction from retrieved chunks
      "generate"        — LLM generation call
    """

    query: str
    retriever: str            # currently always "dense"
    top_k: int
    chunks: list[ChunkTrace] = field(default_factory=list)
    prompt: str = ""
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    latency_by_stage: dict[str, float] = field(default_factory=dict)
    reranker: str = "none"          # Phase 1.9
    rerank_top_n: int | None = None # Phase 1.9
    verdict: "JudgeVerdict | None" = None  # Phase 2.0
    context_pack: ContextPackResult | None = None  # Phase 2.1


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def trace_to_dict(trace: RetrieveTrace | AskTrace) -> dict:
    """Convert a trace to a plain dict suitable for json.dumps.

    Uses dataclasses.asdict() which recurses into nested dataclasses.
    All field types are JSON-native so no custom encoder is needed.
    """
    return dataclasses.asdict(trace)


def write_trace_json(trace: RetrieveTrace | AskTrace, path: Path) -> None:
    """Serialize trace to JSON and write to path.

    Creates parent directories if they do not exist.
    Writes UTF-8 with indent=2 for human readability.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace_to_dict(trace), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Human-readable formatters
# ---------------------------------------------------------------------------

_SEP = "─" * 44


def format_retrieve_trace(trace: RetrieveTrace) -> str:
    """Return a plain-text debug view of a retrieve trace.

    No ANSI escape codes. Scores use 4 decimal places; latencies use 3.

    Example output:
      Retrieve trace
      ────────────────────────────────────────────
        query     : "how to deploy a model"
        retriever : dense
        top_k     : 5
        latency   : load=0.051s  embed=0.012s  retrieve=0.003s
      ────────────────────────────────────────────
      Rank 1  score=0.8432  chunk_id=abc123def456...
        doc_id  : docs/deploy.md
        title   : Deploying Models
        path    : /corpus/docs/deploy.md
        preview : First 120 chars of chunk text...
    """
    latency_str = "  ".join(
        f"{k}={v:.3f}s" for k, v in trace.latency_by_stage.items()
    )
    # Phase 1.9: show reranker info when active.
    reranker_line = ""
    if trace.reranker != "none":
        reranker_line = f"  reranker  : {trace.reranker}"
        if trace.rerank_top_n is not None:
            reranker_line += f"  (rerank_top_n={trace.rerank_top_n})"
    lines = [
        "Retrieve trace",
        _SEP,
        f'  query     : {trace.query!r}',
    ] + ([reranker_line] if reranker_line else []) + [
        f"  retriever : {trace.retriever}",
        f"  top_k     : {trace.top_k}",
        f"  latency   : {latency_str}",
        _SEP,
    ]
    if not trace.chunks:
        lines.append("  No results found.")
    for c in trace.chunks:
        lines.append(
            f"Rank {c.rank}  score={c.score:.4f}  chunk_id={c.chunk_id}"
        )
        lines.append(f"  doc_id  : {c.doc_id}")
        if c.pre_rerank_rank is not None:
            lines.append(
                f"  pre     : rank={c.pre_rerank_rank}  score={c.pre_rerank_score:.4f}"
            )
        lines.append(f"  title   : {c.title}")
        lines.append(f"  path    : {c.path}")
        lines.append(f"  preview : {c.text_preview}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_ask_trace(trace: AskTrace) -> str:
    """Return a plain-text debug view of an ask trace.

    Sections: header (query/retriever/top_k/latency), retrieved chunks,
    answer, citations. No ANSI escape codes.
    """
    latency_str = "  ".join(
        f"{k}={v:.3f}s" for k, v in trace.latency_by_stage.items()
    )
    # Phase 1.9: show reranker info when active.
    reranker_line = ""
    if trace.reranker != "none":
        reranker_line = f"  reranker  : {trace.reranker}"
        if trace.rerank_top_n is not None:
            reranker_line += f"  (rerank_top_n={trace.rerank_top_n})"
    lines = [
        "Ask trace",
        _SEP,
        f'  query     : {trace.query!r}',
    ] + ([reranker_line] if reranker_line else []) + [
        f"  retriever : {trace.retriever}",
        f"  top_k     : {trace.top_k}",
        f"  latency   : {latency_str}",
        _SEP,
    ]
    for c in trace.chunks:
        lines.append(
            f"Rank {c.rank}  score={c.score:.4f}  chunk_id={c.chunk_id}"
        )
        lines.append(f"  doc_id  : {c.doc_id}")
        lines.append(f"  title   : {c.title}")
        if c.pre_rerank_rank is not None:
            lines.append(
                f"  pre     : rank={c.pre_rerank_rank}  score={c.pre_rerank_score:.4f}"
            )
        lines.append(f"  preview : {c.text_preview}")
        lines.append("")
    if trace.context_pack is not None:
        cp = trace.context_pack
        n_selected = len(cp.selected)
        n_omitted = len(cp.omitted)
        lines.append(
            f"Context packing  (budget={cp.budget}, counter={cp.counter_name})"
        )
        lines.append(
            f"  Selected  : {n_selected} chunk{'s' if n_selected != 1 else ''}"
            f"   (~{cp.estimated_tokens} tokens used)"
        )
        if n_omitted == 0:
            lines.append("  Omitted   : 0 chunks")
        else:
            lines.append(
                f"  Omitted   : {n_omitted} chunk{'s' if n_omitted != 1 else ''}"
            )
            for cid in cp.omitted:
                lines.append(f"    - {cid}")
        lines.append("")
    lines.append(_SEP)
    lines.append("Answer:")
    lines.append(trace.answer)
    if trace.citations:
        lines.append("")
        lines.append("Citations: " + ", ".join(trace.citations))
    if trace.verdict is not None:
        v = trace.verdict
        lines.append("")
        lines.append(f"Judge verdict  (judge={v.judge_name})")
        lines.append(f"  Faithfulness     : {v.faithfulness:.3f}")
        lines.append(f"  Answer Relevance : {v.answer_relevance:.3f}")
        lines.append(f"  Citation Support : {v.citation_support:.3f}")
        if v.answer_correctness is not None:
            lines.append(f"  Answer Correct.  : {v.answer_correctness:.3f}")
        if v.notes:
            lines.append(f"  Notes            : {v.notes}")
    return "\n".join(lines)