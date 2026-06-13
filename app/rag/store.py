from __future__ import annotations

from datetime import date
from typing import Any

from ..db import pool


async def filing_exists(accession: str) -> bool:
    async with pool().acquire() as conn:
        row = await conn.fetchrow("SELECT 1 FROM filings WHERE accession = $1", accession)
        return row is not None


async def chunk_count_for_accession(accession: str) -> int:
    async with pool().acquire() as conn:
        row = await conn.fetchrow(
            "SELECT count(*) AS n FROM chunks c "
            "JOIN filings f ON f.id = c.filing_id WHERE f.accession = $1",
            accession,
        )
        return int(row["n"]) if row else 0


async def upsert_filing(
    ticker: str,
    cik: str,
    form_type: str,
    filing_date: date,
    accession: str,
    title: str,
    source_url: str,
) -> int | None:
    """Insert a filing row; return its id, or None if the accession already exists."""
    async with pool().acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO filings (ticker, cik, form_type, filing_date, accession, title, source_url)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (accession) DO NOTHING
            RETURNING id
            """,
            ticker, cik, form_type, filing_date, accession, title, source_url,
        )
        return row["id"] if row else None


async def insert_chunks(
    filing_id: int,
    ticker: str,
    form_type: str,
    filing_date: date,
    rows: list[dict[str, Any]],
) -> int:
    """Bulk-insert chunk rows. Each row: {section, chunk_index, content, embedding}."""
    if not rows:
        return 0
    records = [
        (
            filing_id, ticker, form_type, filing_date,
            r["section"], r["chunk_index"], r["content"], r["embedding"],
        )
        for r in rows
    ]
    async with pool().acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO chunks
                (filing_id, ticker, form_type, filing_date, section, chunk_index, content, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            records,
        )
    return len(records)


async def search(
    query_embedding: list[float],
    top_k: int,
    ticker: str | None = None,
    form_type: str | None = None,
) -> list[dict[str, Any]]:
    """Cosine-similarity search with optional metadata filtering.

    Score is 1 - cosine_distance, so higher is more similar.
    """
    conds: list[str] = []
    params: list[Any] = [query_embedding]
    if ticker:
        params.append(ticker.upper())
        conds.append(f"c.ticker = ${len(params)}")
    if form_type:
        params.append(form_type.upper())
        conds.append(f"c.form_type = ${len(params)}")
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    params.append(top_k)
    limit_pos = len(params)

    sql = f"""
        SELECT c.id, c.content, c.section, c.ticker, c.form_type, c.filing_date,
               c.chunk_index, f.accession, f.source_url, f.title,
               1 - (c.embedding <=> $1) AS score
        FROM chunks c
        JOIN filings f ON f.id = c.filing_id
        {where}
        ORDER BY c.embedding <=> $1
        LIMIT ${limit_pos}
    """
    async with pool().acquire() as conn:
        rows = await conn.fetch(sql, *params)
    return [dict(r) for r in rows]
