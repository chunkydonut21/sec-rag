from __future__ import annotations

import asyncpg
from pgvector.asyncpg import register_vector

from .config import get_settings

_pool: asyncpg.Pool | None = None


def _dsn() -> str:
    # asyncpg wants a bare postgresql:// DSN (no SQLAlchemy-style +driver suffix).
    return get_settings().database_url.replace("postgresql+asyncpg://", "postgresql://")


async def _init_connection(conn: asyncpg.Connection) -> None:
    # Teach asyncpg how to encode/decode the pgvector `vector` type on this conn.
    await register_vector(conn)


async def connect() -> None:
    """Create the connection pool and initialise the schema (idempotent)."""
    global _pool
    if _pool is not None:
        return
    dsn = _dsn()
    # The extension must exist before register_vector() can resolve the type OID,
    # so create it on a one-off connection before the pool's init hook runs.
    sys_conn = await asyncpg.connect(dsn)
    try:
        await sys_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await sys_conn.close()
    _pool = await asyncpg.create_pool(dsn, init=_init_connection, min_size=1, max_size=10)
    await _init_schema()


async def disconnect() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool not initialised; call connect() first.")
    return _pool


async def _init_schema() -> None:
    dim = get_settings().embed_dim
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS filings (
                id          BIGSERIAL PRIMARY KEY,
                ticker      TEXT NOT NULL,
                cik         TEXT NOT NULL,
                form_type   TEXT NOT NULL,
                filing_date DATE NOT NULL,
                accession   TEXT NOT NULL UNIQUE,
                title       TEXT,
                source_url  TEXT NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id          BIGSERIAL PRIMARY KEY,
                filing_id   BIGINT NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
                ticker      TEXT NOT NULL,
                form_type   TEXT NOT NULL,
                filing_date DATE NOT NULL,
                section     TEXT NOT NULL,
                chunk_index INT NOT NULL,
                content     TEXT NOT NULL,
                embedding   vector({dim}) NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        # Metadata-filter indexes (the WHERE side of retrieval).
        await conn.execute("CREATE INDEX IF NOT EXISTS chunks_ticker_idx ON chunks (ticker)")
        await conn.execute("CREATE INDEX IF NOT EXISTS chunks_form_idx ON chunks (form_type)")
        # Approximate-nearest-neighbour index for cosine similarity search.
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
            "ON chunks USING hnsw (embedding vector_cosine_ops)"
        )
