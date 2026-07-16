from dataclasses import dataclass

from tiny_rag_lab.bm25 import BM25Retriever
from tiny_rag_lab.embeddings import Embedder
from tiny_rag_lab.index_loader import LoadedIndex
from tiny_rag_lab.models import RetrievalResult
from tiny_rag_lab.retrieval import retrieve


@dataclass(frozen=True)
class RRFSourceExplanation:
    source: str
    rank: int
    score: float
    contribution: float


@dataclass(frozen=True)
class RRFCandidateExplanation:
    chunk_id: str
    rank: int
    score: float
    sources: list[RRFSourceExplanation]


def reciprocal_rank_fusion(
    results_lists: list[list[RetrievalResult]],
    top_k: int,
    k: int = 60,
) -> list[RetrievalResult]:
    """Fuse multiple ranked lists via Reciprocal Rank Fusion.

    rrf_score(chunk) = sum(1 / (k + rank_i) for each list where chunk appears)

    rank_i is 1-indexed, matching RetrievalResult.rank.
    Returned results have fused RRF score and re-assigned 1-indexed ranks.
    Tie-breaking: Python stable sort preserves original order; dense list wins
    because it is always passed first.
    """
    fused, _ = reciprocal_rank_fusion_with_explanation(
        results_lists, top_k=top_k, k=k,
    )
    return fused


def reciprocal_rank_fusion_with_explanation(
    results_lists: list[list[RetrievalResult]],
    top_k: int,
    k: int = 60,
    source_names: list[str] | None = None,
) -> tuple[list[RetrievalResult], list[RRFCandidateExplanation]]:
    """Fuse ranked lists and expose every contribution used in each score."""
    if top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")
    if source_names is None:
        source_names = [f"source_{position + 1}" for position in range(len(results_lists))]
    if len(source_names) != len(results_lists):
        raise ValueError("source_names must match results_lists")

    scores: dict[str, float] = {}
    first_seen: dict[str, RetrievalResult] = {}
    contributions: dict[str, list[RRFSourceExplanation]] = {}

    for source, results in zip(source_names, results_lists):
        for result in results:
            cid = result.chunk.chunk_id
            contribution = 1.0 / (k + result.rank)
            scores[cid] = scores.get(cid, 0.0) + contribution
            if cid not in first_seen:
                first_seen[cid] = result
            contributions.setdefault(cid, []).append(RRFSourceExplanation(
                source=source,
                rank=result.rank,
                score=result.score,
                contribution=contribution,
            ))

    ranked = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    fused = []
    for rank, cid in enumerate(ranked[:top_k], start=1):
        fused.append(RetrievalResult(
            chunk=first_seen[cid].chunk,
            score=scores[cid],
            rank=rank,
        ))
    explanation = [
        RRFCandidateExplanation(
            chunk_id=result.chunk.chunk_id,
            rank=result.rank,
            score=result.score,
            sources=contributions[result.chunk.chunk_id],
        )
        for result in fused
    ]
    return fused, explanation


def retrieve_hybrid(
    query: str,
    index: LoadedIndex,
    embedder: Embedder,
    top_k: int = 5,
    bm25_retriever: BM25Retriever | None = None,
) -> list[RetrievalResult]:
    """Return top_k chunks fused from dense and BM25 retrieval via RRF.

    If bm25_retriever is None, a BM25Retriever is built over index.chunks
    internally. Callers running many queries should build one BM25Retriever
    and inject it to avoid rebuilding per query.
    """
    if bm25_retriever is None:
        bm25_retriever = BM25Retriever(index.chunks)
    dense_results = retrieve(query, index, embedder, top_k=top_k)
    bm25_results = bm25_retriever.retrieve(query, top_k=top_k)
    return reciprocal_rank_fusion([dense_results, bm25_results], top_k=top_k)
