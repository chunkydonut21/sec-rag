from __future__ import annotations

import voyageai

from ..config import get_settings


class Reranker:
    """Cross-encoder reranking via Voyage, applied to vector-search candidates."""

    def __init__(self) -> None:
        s = get_settings()
        self._client = voyageai.AsyncClient(api_key=s.voyage_api_key)
        self._model = s.rerank_model

    async def rerank(
        self, query: str, documents: list[str], top_k: int
    ) -> list[tuple[int, float]]:
        """Return [(original_index, relevance_score), ...] for the top_k documents."""
        if not documents:
            return []
        resp = await self._client.rerank(
            query, documents, model=self._model, top_k=min(top_k, len(documents))
        )
        return [(r.index, r.relevance_score) for r in resp.results]


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
