"""Second-pass reranking for the retrieval plane.

Phase 1.9 scope (T01): data contracts and helpers that downstream Phase 1.9
tasks (CLI, eval, ask, diagnose) compose with the existing retrievers.

The reranker is opt-in. Phase 1 through 1.8 behavior is preserved when no
Reranker is passed: callers simply skip apply_reranker entirely.

Bi-encoder retrievers (dense, BM25, hybrid) score each candidate by
comparing two independently-computed representations. A cross-encoder
reranker encodes (query, chunk) jointly, so it can attend across the pair
and produce a sharper relevance score at the cost of running the model
once per candidate. This module hides whichever scoring mechanism is used
behind the Reranker protocol; only the fake implementation lives here in
T01.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from tiny_rag_lab.models import RetrievalResult
from tiny_rag_lab.trace import ChunkTrace


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class RerankResult:
    """One chunk's reranking audit record.

    pre_rank and post_rank are both 1-indexed, matching RetrievalResult.rank.
    pre_score is the score from the base retriever (cosine, BM25, or RRF).
    post_score is the score produced by the reranker.

    All fields are JSON-native so dataclasses.asdict() + json.dumps()
    serialize without a custom encoder.
    """

    chunk_id: str
    pre_rank: int
    post_rank: int
    pre_score: float
    post_score: float


class Reranker(Protocol):
    """Second-pass reranker over a pre-retrieved candidate set.

    Implementations must be deterministic for the same (query, candidates)
    pair so trace and eval outputs are reproducible.

    name is the short string written into traces. Implementations should
    use the same value across calls.
    """

    name: str

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
    ) -> list[RerankResult]:
        """Return RerankResults ordered by post_rank ascending.

        len(return) == len(candidates). Every input candidate appears once.
        The returned audit covers the full input set even when a downstream
        caller will only slice the top-k.
        """
        ...


# ---------------------------------------------------------------------------
# Fake reranker for deterministic tests
# ---------------------------------------------------------------------------

@dataclass
class FakeReranker:
    """Deterministic reranker for tests.

    score_map maps chunk_id -> post_score. Chunks not in the map get
    post_score = 0.0. When score_map is None the reranker is a no-op:
    post_score == pre_score and post_rank == pre_rank.

    Ties on post_score break by pre_rank (smaller pre_rank wins) so the
    output is fully deterministic.
    """

    name: str = "fake"
    score_map: dict[str, float] | None = field(default=None)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
    ) -> list[RerankResult]:
        if not candidates:
            return []

        if self.score_map is None:
            # No-op: post == pre, preserve order.
            return [
                RerankResult(
                    chunk_id=c.chunk.chunk_id,
                    pre_rank=c.rank,
                    post_rank=c.rank,
                    pre_score=c.score,
                    post_score=c.score,
                )
                for c in candidates
            ]

        # Score-map mode: assign post_score from the map (0.0 for misses),
        # sort by post_score desc with pre_rank asc as the tie-break.
        # Python's sort is stable, so sorting twice — first by the
        # tie-breaker, then by the primary key — produces the right order
        # without a compound key tuple.
        scored: list[tuple[RetrievalResult, float]] = [
            (c, self.score_map.get(c.chunk.chunk_id, 0.0))
            for c in candidates
        ]
        scored.sort(key=lambda pair: pair[0].rank)             # tie-break asc
        scored.sort(key=lambda pair: pair[1], reverse=True)    # primary desc

        return [
            RerankResult(
                chunk_id=c.chunk.chunk_id,
                pre_rank=c.rank,
                post_rank=new_rank,
                pre_score=c.score,
                post_score=score,
            )
            for new_rank, (c, score) in enumerate(scored, start=1)
        ]


# ---------------------------------------------------------------------------
# Cross-encoder reranker (T02)
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """Local cross-encoder reranker backed by sentence-transformers.

    The model is lazily loaded on the first rerank() call. Construction is
    free of side effects (no network, no disk read) — importing this class
    does not import sentence_transformers.

    DEFAULT_MODEL is the well-known MS MARCO MiniLM cross-encoder (~80 MB),
    chosen as a small English-focused default that a learner can download
    on a laptop without hesitating. Pass model_name to swap in a different
    cross-encoder (e.g. BAAI/bge-reranker-base for multilingual corpora).
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    name: str

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = None  # lazy — no I/O in __init__
        self.name = "cross-encoder"

    def rerank(
        self,
        query: str,
        candidates: list["RetrievalResult"],
    ) -> list["RerankResult"]:
        if not candidates:
            return []

        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)

        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)

        # Build scored pairs, then stable-sort twice (tie-break first,
        # primary key second) so the sort policy is visible. Same pattern
        # as FakeReranker.
        scored: list[tuple["RetrievalResult", float]] = [
            (c, float(score)) for c, score in zip(candidates, scores)
        ]
        scored.sort(key=lambda pair: pair[0].rank)             # tie-break asc
        scored.sort(key=lambda pair: pair[1], reverse=True)    # primary desc

        return [
            RerankResult(
                chunk_id=c.chunk.chunk_id,
                pre_rank=c.rank,
                post_rank=new_rank,
                pre_score=c.score,
                post_score=score,
            )
            for new_rank, (c, score) in enumerate(scored, start=1)
        ]


# ---------------------------------------------------------------------------
# Helpers used by CLI, eval, and ask integrations (later T03-T05)
# ---------------------------------------------------------------------------

def apply_reranker(
    query: str,
    results: list[RetrievalResult],
    reranker: Reranker,
    top_k: int,
) -> tuple[list[RetrievalResult], list[RerankResult]]:
    """Run reranker over results and return (reordered, rerank_audit).

    reordered is the top min(top_k, len(results)) post-rerank slice as
    list[RetrievalResult] with rank reassigned to the post-rerank position
    and score replaced with the post-rerank score. Clipping matches
    retrieve_by_vector's behavior on small indexes so callers do not have
    to special-case sparse pools.

    rerank_audit is the full list[RerankResult] (length == len(results)),
    suitable for assembling trace fields even when only a slice is
    returned to the caller.

    Returns ([], []) if results is empty.
    Raises ValueError if top_k < 0.

    The invariant rerank_top_n >= top_k is enforced one layer up at the
    CLI / runner config check, before any base retrieval runs. This
    function does not re-check that invariant; it only deals with the
    candidate set it is actually handed.
    """
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")

    if not results:
        return [], []

    audit = reranker.rerank(query, results)

    actual_k = min(top_k, len(results))
    by_chunk_id = {r.chunk.chunk_id: r for r in results}

    reordered: list[RetrievalResult] = []
    for new_rank, rr in enumerate(audit[:actual_k], start=1):
        src = by_chunk_id[rr.chunk_id]
        reordered.append(
            RetrievalResult(
                chunk=src.chunk,
                score=rr.post_score,
                rank=new_rank,
            )
        )

    return reordered, audit


def chunk_traces_from_rerank(
    results: list[RetrievalResult],
    rerank_audit: list[RerankResult] | None,
) -> list[ChunkTrace]:
    """Build ChunkTrace records, populating pre_rerank_* when rerank ran.

    results is the post-rerank slice (or the original results when no
    rerank ran).
    rerank_audit is the full audit returned by apply_reranker, or None
    when reranking was skipped.

    When rerank_audit is None the returned ChunkTraces have
    pre_rerank_rank == None and pre_rerank_score == None — the existing
    Phase 1.7 / 1.8 trace shape.

    When rerank_audit is given, the corresponding pre_rank / pre_score
    are looked up by chunk_id for every result. Chunks present in
    results but absent from rerank_audit (should not happen in normal
    flows) get None for the pre_rerank fields.
    """
    audit_by_id: dict[str, RerankResult] = {}
    if rerank_audit is not None:
        audit_by_id = {rr.chunk_id: rr for rr in rerank_audit}

    traces: list[ChunkTrace] = []
    for r in results:
        rr = audit_by_id.get(r.chunk.chunk_id)
        traces.append(
            ChunkTrace(
                rank=r.rank,
                chunk_id=r.chunk.chunk_id,
                doc_id=r.chunk.doc_id,
                title=r.chunk.metadata.get("title", ""),
                path=r.chunk.metadata.get("path", r.chunk.doc_id),
                score=r.score,
                text_preview=r.chunk.text[:120].replace("\n", " ").strip(),
                pre_rerank_rank=rr.pre_rank if rr is not None else None,
                pre_rerank_score=rr.pre_score if rr is not None else None,
            )
        )
    return traces
