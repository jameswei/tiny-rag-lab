from tiny_rag_lab.bm25 import BM25Retriever
from tiny_rag_lab.embeddings import Embedder
from tiny_rag_lab.index_loader import LoadedIndex
from tiny_rag_lab.models import RetrievalResult
from tiny_rag_lab.retrieval import retrieve


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
    scores: dict[str, float] = {}
    first_seen: dict[str, RetrievalResult] = {}

    for results in results_lists:
        for result in results:
            cid = result.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + result.rank)
            if cid not in first_seen:
                first_seen[cid] = result

    ranked = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    fused = []
    for rank, cid in enumerate(ranked[:top_k], start=1):
        fused.append(RetrievalResult(
            chunk=first_seen[cid].chunk,
            score=scores[cid],
            rank=rank,
        ))
    return fused


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
