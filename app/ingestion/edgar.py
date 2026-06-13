from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime

import httpx

from ..config import get_settings

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"


@dataclass
class FilingRef:
    accession: str
    form_type: str
    filing_date: date
    primary_document: str
    cik: int
    source_url: str
    title: str


class EdgarClient:
    """Minimal async SEC EDGAR client: ticker -> CIK -> filing list -> document.

    Uses the public data.sec.gov JSON APIs. Requests are spaced out and carry the
    SEC-mandated User-Agent header.
    """

    def __init__(self) -> None:
        s = get_settings()
        self._delay = s.sec_request_delay
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": s.sec_user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=30.0,
        )
        self._ticker_map: dict[str, int] | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, url: str) -> httpx.Response:
        await asyncio.sleep(self._delay)  # stay under the SEC ~10 req/s cap
        resp = await self._client.get(url)
        resp.raise_for_status()
        return resp

    async def _load_ticker_map(self) -> dict[str, int]:
        if self._ticker_map is None:
            data = (await self._get(TICKERS_URL)).json()
            self._ticker_map = {
                row["ticker"].upper(): int(row["cik_str"]) for row in data.values()
            }
        return self._ticker_map

    async def cik_for_ticker(self, ticker: str) -> int:
        cik = (await self._load_ticker_map()).get(ticker.upper())
        if cik is None:
            raise ValueError(f"Unknown ticker: {ticker!r}")
        return cik

    async def list_filings(
        self, ticker: str, forms: list[str], limit: int
    ) -> list[FilingRef]:
        cik = await self.cik_for_ticker(ticker)
        data = (await self._get(SUBMISSIONS_URL.format(cik=cik))).json()
        recent = data["filings"]["recent"]
        company = data.get("name", ticker.upper())
        wanted = {f.upper() for f in forms}

        out: list[FilingRef] = []
        for i, form in enumerate(recent["form"]):
            if form.upper() not in wanted:
                continue
            acc = recent["accessionNumber"][i]
            doc = recent["primaryDocument"][i]
            if not doc:  # some entries (e.g. pure XBRL) lack a primary document
                continue
            fdate = datetime.strptime(recent["filingDate"][i], "%Y-%m-%d").date()
            url = ARCHIVE_URL.format(cik=cik, acc_nodash=acc.replace("-", ""), doc=doc)
            out.append(
                FilingRef(
                    accession=acc,
                    form_type=form.upper(),
                    filing_date=fdate,
                    primary_document=doc,
                    cik=cik,
                    source_url=url,
                    title=f"{company} {form.upper()} ({fdate.isoformat()})",
                )
            )
            if len(out) >= limit:
                break
        return out

    async def fetch_document(self, ref: FilingRef) -> str:
        return (await self._get(ref.source_url)).text
