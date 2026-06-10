from rank_bm25 import BM25Okapi

from tiny_rag_lab.models import Chunk, RetrievalResult


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        # Guard: if all chunks tokenize to empty lists, BM25Okapi raises ZeroDivisionError.
        # Treat that the same as an empty corpus — _bm25 stays None.
        if chunks and any(tokens for tokens in tokenized):
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {top_k}")
        if self._bm25 is None or not query.strip():
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for rank, (idx, score) in enumerate(ranked[:top_k], start=1):
            results.append(RetrievalResult(chunk=self._chunks[idx], score=float(score), rank=rank))
        return results
