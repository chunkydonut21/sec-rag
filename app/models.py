from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    forms: list[str] = Field(default_factory=lambda: ["10-K"], examples=[["10-K", "10-Q"]])
    limit: int = Field(default=1, ge=1, le=10, description="Max filings per request")


class IngestResponse(BaseModel):
    ticker: str
    filings_ingested: int
    filings_skipped: int
    chunks_created: int
    chunks_existing: int
    details: list[dict]


class QueryRequest(BaseModel):
    question: str = Field(..., examples=["What are the main risk factors?"])
    ticker: str | None = Field(default=None, examples=["AAPL"])
    form_type: str | None = Field(default=None, examples=["10-K"])
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    content: str
    section: str
    ticker: str
    form_type: str
    filing_date: str
    accession: str
    source_url: str
    score: float


class QueryResponse(BaseModel):
    question: str
    citations: list[CitationOut]


class AnswerResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut]
