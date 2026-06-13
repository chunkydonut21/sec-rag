from __future__ import annotations

from dataclasses import dataclass

from ..config import get_settings
from . import store
from .embeddings import get_embedder
from .rerank import get_reranker


@dataclass
class Citation:
    content: str
    section: str
    ticker: str
    form_type: str
    filing_date: str
    accession: str
    source_url: str
    score: float


async def retrieve(
    question: str,
    ticker: str | None = None,
    form_type: str | None = None,
    top_k: int | None = None,
) -> list[Citation]:
    """Embed the query, vector-search with metadata filters, rerank, return citations."""
    s = get_settings()
    top_k = top_k or s.retrieve_top_k

    qvec = await get_embedder().embed_query(question)
    candidates = await store.search(
        qvec, top_k=s.retrieve_candidates, ticker=ticker, form_type=form_type
    )
    if not candidates:
        return []

    ranked = await get_reranker().rerank(
        question, [c["content"] for c in candidates], top_k=top_k
    )

    citations: list[Citation] = []
    for orig_idx, score in ranked:
        c = candidates[orig_idx]
        citations.append(
            Citation(
                content=c["content"],
                section=c["section"],
                ticker=c["ticker"],
                form_type=c["form_type"],
                filing_date=c["filing_date"].isoformat(),
                accession=c["accession"],
                source_url=c["source_url"],
                score=float(score),
            )
        )
    return citations
