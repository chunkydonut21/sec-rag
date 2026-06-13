from __future__ import annotations

from dataclasses import dataclass, field

from .ingestion.chunk import chunk_sections
from .ingestion.edgar import EdgarClient
from .ingestion.parse import html_to_text, looks_like_html, section_filing
from .rag import store
from .rag.embeddings import get_embedder


@dataclass
class IngestResult:
    ticker: str
    filings_ingested: int = 0
    filings_skipped: int = 0
    chunks_created: int = 0
    chunks_existing: int = 0  # chunks already in the DB for skipped (already-ingested) filings
    details: list[dict] = field(default_factory=list)


async def ingest_ticker(ticker: str, forms: list[str], limit: int) -> IngestResult:
    """Full ingestion path for one ticker: EDGAR -> parse -> chunk -> embed -> store."""
    ticker = ticker.upper()
    edgar = EdgarClient()
    embedder = get_embedder()
    result = IngestResult(ticker=ticker)
    try:
        for ref in await edgar.list_filings(ticker, forms, limit):
            if await store.filing_exists(ref.accession):
                result.filings_skipped += 1
                result.chunks_existing += await store.chunk_count_for_accession(
                    ref.accession
                )
                continue

            raw = await edgar.fetch_document(ref)
            text = html_to_text(raw) if looks_like_html(raw) else raw
            chunks = chunk_sections(section_filing(text))
            if not chunks:
                result.filings_skipped += 1
                continue

            embeddings = await embedder.embed_documents([c.content for c in chunks])

            filing_id = await store.upsert_filing(
                ticker=ticker,
                cik=str(ref.cik),
                form_type=ref.form_type,
                filing_date=ref.filing_date,
                accession=ref.accession,
                title=ref.title,
                source_url=ref.source_url,
            )
            if filing_id is None:  # inserted concurrently between the check and now
                result.filings_skipped += 1
                continue

            rows = [
                {
                    "section": c.section,
                    "chunk_index": c.chunk_index,
                    "content": c.content,
                    "embedding": emb,
                }
                for c, emb in zip(chunks, embeddings)
            ]
            n = await store.insert_chunks(
                filing_id, ticker, ref.form_type, ref.filing_date, rows
            )

            result.filings_ingested += 1
            result.chunks_created += n
            result.details.append(
                {
                    "accession": ref.accession,
                    "form_type": ref.form_type,
                    "filing_date": ref.filing_date.isoformat(),
                    "chunks": n,
                    "sections": sorted({c.section for c in chunks}),
                }
            )
    finally:
        await edgar.aclose()
    return result
