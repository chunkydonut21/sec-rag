from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, sourced from environment variables / .env.

    Every embedding/LLM backend is behind a config var so the stack can be
    swapped (e.g. Voyage -> local model, Sonnet -> Opus) without code changes.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Postgres / pgvector -------------------------------------------------
    database_url: str = "postgresql://postgres:postgres@localhost:5432/sec"

    # --- SEC EDGAR -----------------------------------------------------------
    # The SEC REQUIRES a descriptive User-Agent with a contact email on every
    # request; anonymous traffic is throttled/blocked. See sec.gov/os/webmaster-faq.
    sec_user_agent: str = "sec-rag-demo contact@example.com"
    sec_request_delay: float = 0.2  # seconds between requests (SEC caps ~10 req/s)

    # --- Voyage AI (embeddings + reranker) -----------------------------------
    voyage_api_key: str = ""
    # voyage-finance-2 is domain-tuned for financial text (1024-dim, 50M free
    # tokens). For a general/cheaper option use voyage-4-lite (1024-dim, 200M free).
    embed_model: str = "voyage-finance-2"
    embed_dim: int = 1024
    embed_batch_size: int = 100
    rerank_model: str = "rerank-2.5"

    # --- Anthropic (agent layer — Phase 2) -----------------------------------
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"  # bump to claude-opus-4-8 for max quality
    judge_model: str = "claude-haiku-4-5"  # cheaper model for LLM-as-judge eval

    # --- Chunking ------------------------------------------------------------
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # --- Retrieval -----------------------------------------------------------
    retrieve_candidates: int = 40  # vector-search breadth before reranking
    retrieve_top_k: int = 6        # passages returned after reranking


@lru_cache
def get_settings() -> Settings:
    return Settings()
