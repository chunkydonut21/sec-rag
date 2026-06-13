from __future__ import annotations

import asyncio

import voyageai
from voyageai.error import RateLimitError

from ..config import get_settings


class Embedder:
    """Async wrapper around Voyage embeddings.

    Behind a small interface so the backend is swappable (e.g. to a local
    sentence-transformers model) without touching ingestion/retrieval code.
    Voyage distinguishes 'document' vs 'query' inputs for better retrieval.
    """

    def __init__(self) -> None:
        s = get_settings()
        self._client = voyageai.AsyncClient(api_key=s.voyage_api_key)
        self._model = s.embed_model
        self._batch = s.embed_batch_size

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, input_type="document")

    async def embed_query(self, text: str) -> list[float]:
        out = await self._embed([text], input_type="query")
        return out[0]

    async def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            batch = texts[i : i + self._batch]
            vectors.extend(await self._embed_batch(batch, input_type))
        return vectors

    async def _embed_batch(
        self, batch: list[str], input_type: str, max_retries: int = 5
    ) -> list[list[float]]:
        """Embed one batch, retrying on Voyage rate limits with exponential backoff."""
        delay = 5.0
        for attempt in range(max_retries):
            try:
                resp = await self._client.embed(
                    batch, model=self._model, input_type=input_type
                )
                return resp.embeddings
            except RateLimitError:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)
        return []  # unreachable


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
