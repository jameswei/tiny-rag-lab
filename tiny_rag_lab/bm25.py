from collections import Counter
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from tiny_rag_lab.models import Chunk, RetrievalResult


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


@dataclass(frozen=True)
class BM25TermExplanation:
    """One query term's contribution to one candidate's BM25 score."""

    term: str
    query_frequency: int
    term_frequency: int
    document_frequency: int
    inverse_document_frequency: float
    contribution: float


@dataclass(frozen=True)
class BM25CandidateExplanation:
    chunk_id: str
    rank: int
    score: float
    document_length: int
    average_document_length: float
    terms: list[BM25TermExplanation]


@dataclass(frozen=True)
class BM25Explanation:
    query_tokens: list[str]
    corpus_size: int
    k1: float
    b: float
    candidates: list[BM25CandidateExplanation]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._tokenized = [_tokenize(c.text) for c in chunks]
        # Guard: if all chunks tokenize to empty lists, BM25Okapi raises ZeroDivisionError.
        # Treat that the same as an empty corpus — _bm25 stays None.
        if chunks and any(tokens for tokens in self._tokenized):
            self._bm25 = BM25Okapi(self._tokenized)
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

    def retrieve_with_explanation(
        self, query: str, top_k: int = 5,
    ) -> tuple[list[RetrievalResult], BM25Explanation]:
        """Rank once and expose the exact BM25 components used for that rank."""
        results = self.retrieve(query, top_k=top_k)
        tokens = _tokenize(query)
        if self._bm25 is None or not tokens:
            return results, BM25Explanation(
                query_tokens=tokens,
                corpus_size=len(self._chunks),
                k1=float(getattr(self._bm25, "k1", 1.5)),
                b=float(getattr(self._bm25, "b", 0.75)),
                candidates=[],
            )

        query_counts = Counter(tokens)
        document_frequency = {
            term: sum(1 for frequencies in self._bm25.doc_freqs if term in frequencies)
            for term in query_counts
        }
        position_by_id = {chunk.chunk_id: position for position, chunk in enumerate(self._chunks)}
        candidates: list[BM25CandidateExplanation] = []
        for result in results:
            position = position_by_id[result.chunk.chunk_id]
            frequencies = self._bm25.doc_freqs[position]
            document_length = self._bm25.doc_len[position]
            normalizer = self._bm25.k1 * (
                1.0 - self._bm25.b
                + self._bm25.b * document_length / self._bm25.avgdl
            )
            terms: list[BM25TermExplanation] = []
            for term, query_frequency in query_counts.items():
                term_frequency = frequencies.get(term, 0)
                inverse_document_frequency = float(self._bm25.idf.get(term, 0.0))
                per_occurrence = 0.0
                if term_frequency:
                    per_occurrence = inverse_document_frequency * (
                        term_frequency * (self._bm25.k1 + 1.0)
                        / (term_frequency + normalizer)
                    )
                terms.append(BM25TermExplanation(
                    term=term,
                    query_frequency=query_frequency,
                    term_frequency=int(term_frequency),
                    document_frequency=document_frequency[term],
                    inverse_document_frequency=inverse_document_frequency,
                    contribution=float(per_occurrence * query_frequency),
                ))
            candidates.append(BM25CandidateExplanation(
                chunk_id=result.chunk.chunk_id,
                rank=result.rank,
                score=result.score,
                document_length=int(document_length),
                average_document_length=float(self._bm25.avgdl),
                terms=terms,
            ))
        return results, BM25Explanation(
            query_tokens=tokens,
            corpus_size=len(self._chunks),
            k1=float(self._bm25.k1),
            b=float(self._bm25.b),
            candidates=candidates,
        )
