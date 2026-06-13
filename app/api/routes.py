from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from ..agent.graph import run_agent, run_agent_events
from ..config import get_settings
from ..eval.runner import latest_report, run_eval
from ..observability import recorder
from ..models import (
    AnswerResponse,
    CitationOut,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from ..pipeline import ingest_ticker
from ..rag.retrieve import retrieve

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    """Fetch, parse, chunk, embed, and store filings for a ticker."""
    try:
        result = await ingest_ticker(req.ticker, req.forms, req.limit)
    except ValueError as exc:  # unknown ticker
        raise HTTPException(status_code=404, detail=str(exc))
    return IngestResponse(
        ticker=result.ticker,
        filings_ingested=result.filings_ingested,
        filings_skipped=result.filings_skipped,
        chunks_created=result.chunks_created,
        chunks_existing=result.chunks_existing,
        details=result.details,
    )


@router.post("/retrieve", response_model=QueryResponse)
async def retrieve_passages(req: QueryRequest) -> QueryResponse:
    """Raw retrieval: the most relevant filing passages, with citations (no LLM)."""
    citations = await retrieve(req.question, req.ticker, req.form_type, req.top_k)
    return QueryResponse(
        question=req.question,
        citations=[CitationOut(**c.__dict__) for c in citations],
    )


def _require_anthropic_key() -> None:
    if not get_settings().anthropic_api_key:
        raise HTTPException(
            status_code=400,
            detail="ANTHROPIC_API_KEY is not set — required for /query (the agent).",
        )


@router.post("/query", response_model=AnswerResponse)
async def query(req: QueryRequest) -> AnswerResponse:
    """Ask the agent: it retrieves, reasons, and writes a cited answer."""
    _require_anthropic_key()
    result = await run_agent(req.question, req.ticker, req.form_type)
    return AnswerResponse(
        question=req.question,
        answer=result["answer"],
        citations=[CitationOut(**c) for c in result["citations"]],
    )


@router.post("/query/stream")
async def query_stream(req: QueryRequest) -> EventSourceResponse:
    """Same as /query, but streams the agent's steps + answer tokens over SSE.

    Each SSE message has an `event` (token | tool_start | tool_end | final) and a
    JSON `data` payload.
    """
    _require_anthropic_key()

    async def event_generator():
        async for event in run_agent_events(req.question, req.ticker, req.form_type):
            yield {"event": event["type"], "data": json.dumps(event)}

    return EventSourceResponse(event_generator())


@router.post("/eval/run")
async def eval_run() -> dict:
    """Run the eval set (ingest known filings, query, judge) and return the report.

    Slow — runs every case through the agent plus LLM-as-judge scoring.
    """
    _require_anthropic_key()
    return await run_eval()


@router.get("/eval/results")
async def eval_results() -> dict:
    """Return the most recent eval report."""
    report = latest_report()
    if report is None:
        raise HTTPException(status_code=404, detail="No eval run yet — POST /eval/run.")
    return report


@router.get("/observability")
async def observability() -> dict:
    """Aggregate token/cost/latency metrics plus the most recent LLM/tool calls."""
    return {"summary": recorder.summary(), "recent": recorder.recent(20)}
