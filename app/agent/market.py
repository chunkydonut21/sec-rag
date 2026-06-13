from __future__ import annotations

import asyncio
from typing import Any

import yfinance as yf

# yfinance is keyless (it reads Yahoo Finance), which keeps the demo free and
# zero-setup. It's also synchronous, so we run it in a thread to stay async.

_FIELDS = (
    "last_price",
    "previous_close",
    "open",
    "day_high",
    "day_low",
    "year_high",
    "year_low",
    "market_cap",
    "currency",
)


def _fetch(ticker: str) -> dict[str, Any]:
    fast = yf.Ticker(ticker).fast_info
    out: dict[str, Any] = {"ticker": ticker.upper()}
    for field in _FIELDS:
        try:
            out[field] = fast[field]
        except Exception:
            out[field] = None
    if all(out.get(f) is None for f in _FIELDS):
        out["error"] = "No market data returned (unknown or delisted ticker?)."
    return out


async def get_market_data(ticker: str) -> dict[str, Any]:
    return await asyncio.to_thread(_fetch, ticker)
