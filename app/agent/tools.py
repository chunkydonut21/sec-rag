from __future__ import annotations

import json
from contextvars import ContextVar

from langchain_core.tools import tool

from ..rag.retrieve import Citation, retrieve as _retrieve
from .calculator import safe_calc
from .market import get_market_data as _get_market_data

# Per-request citation collector. The agent loop sets a fresh list before each
# run; the retrieve tool appends to it so the API can return structured citations
# alongside the natural-language answer.
citation_collector: ContextVar[list[Citation]] = ContextVar("citation_collector")


@tool
async def retrieve_filings(
    query: str, ticker: str | None = None, form_type: str | None = None
) -> str:
    """Search ingested SEC filings for passages relevant to `query`.

    Use this to ground every factual claim about a company in its actual filings.
    Optionally filter by `ticker` (e.g. "AAPL") and `form_type` ("10-K", "10-Q",
    "8-K"). Returns passages, each prefixed with a [n] citation marker you must
    cite inline in your answer.
    """
    results = await _retrieve(query, ticker=ticker, form_type=form_type)
    try:
        collector = citation_collector.get()
    except LookupError:
        collector = []
    if not results:
        return "No relevant passages found. The relevant filing may not be ingested yet."

    start = len(collector)
    collector.extend(results)
    blocks = []
    for offset, c in enumerate(results):
        n = start + offset + 1
        blocks.append(
            f"[{n}] ({c.ticker} {c.form_type} {c.filing_date} — {c.section})\n{c.content}"
        )
    return "\n\n".join(blocks)


@tool
async def get_market_data(ticker: str) -> str:
    """Get the current/last stock price and basic market figures for a ticker.

    Use this for live price, day range, 52-week range, and market cap. This is
    market data, not from the filings.
    """
    return json.dumps(await _get_market_data(ticker))


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression (growth %, ratios, sums, etc.).

    Supports + - * / // % ** and parentheses. Always use this for arithmetic
    rather than computing in your head.
    """
    try:
        return str(safe_calc(expression))
    except Exception as exc:  # surface the error back to the model
        return f"Error: {exc}"


TOOLS = [retrieve_filings, get_market_data, calculator]
TOOL_MAP = {t.name: t for t in TOOLS}
