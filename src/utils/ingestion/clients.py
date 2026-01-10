"""HTTP clients for external data providers."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any, Dict, List, Optional

import httpx

from config.config import settings
from src.utils.ingestion.common import parse_iso_date

ISO_DATE = "%Y-%m-%d"


class PolygonClient:
    """Client for Polygon.io endpoints used by the pipeline."""

    def __init__(self) -> None:
        self.api_key = settings.polygon_api_key
        if not self.api_key:
            raise RuntimeError("Polygon API key missing in config/secrets.yml")
        self.base_url = settings.polygon_base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

    async def close(self) -> None:
        await self._client.aclose()

    async def get_ticker_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/v3/reference/tickers/{symbol.upper()}"
        params = {"apiKey": self.api_key}
        resp = await self._client.get(url, params=params)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return data.get("results")

    async def get_daily_ohlc(self, symbol: str, start: date, end: date) -> List[Dict[str, Any]]:
        url = (
            f"{self.base_url}/v2/aggs/ticker/{symbol.upper()}/range/1/day/"
            f"{start.strftime(ISO_DATE)}/{end.strftime(ISO_DATE)}"
        )
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self.api_key}
        resp = await self._client.get(url, params=params)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("results", []) or []

    async def get_financials(
        self,
        symbol: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
        timeframe: str = "quarterly",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/vX/reference/financials"
        limit = max(1, min(limit, 100))
        params: Dict[str, Any] = {
            "ticker": symbol.upper(),
            "timeframe": timeframe,
            "limit": limit,
            "apiKey": self.api_key,
        }
        if start:
            params["filing_date.gte"] = start.strftime(ISO_DATE)
        if end:
            params["filing_date.lte"] = end.strftime(ISO_DATE)
        results: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while True:
            if cursor:
                params["cursor"] = cursor
            resp = await self._client.get(url, params=params)
            if resp.status_code == 404:
                break
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("results", []) or []
            results.extend(batch)
            cursor = data.get("next_url")
            if not cursor:
                break
            url = cursor
            params = {"apiKey": self.api_key}
            await asyncio.sleep(0.2)
        return results

    async def get_earnings(self, symbol: str, start: date, end: date) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/vX/reference/earnings"
        params = {
            "ticker": symbol.upper(),
            "report_date.gte": start.strftime(ISO_DATE),
            "report_date.lte": end.strftime(ISO_DATE),
            "limit": 1000,
            "apiKey": self.api_key,
        }
        results: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while True:
            if cursor:
                params["cursor"] = cursor
            resp = await self._client.get(url, params=params)
            if resp.status_code == 404:
                break
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("results", []) or []
            results.extend(batch)
            cursor = data.get("next_url")
            if not cursor:
                break
            await asyncio.sleep(0.2)
        return results

    async def get_dividends(self, symbol: str, start: date, end: date) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/v3/reference/dividends"
        params = {
            "ticker": symbol.upper(),
            "ex_dividend_date.gte": start.strftime(ISO_DATE),
            "ex_dividend_date.lte": end.strftime(ISO_DATE),
            "limit": 1000,
            "apiKey": self.api_key,
        }
        results: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while True:
            if cursor:
                params["cursor"] = cursor
            resp = await self._client.get(url, params=params)
            if resp.status_code == 404:
                break
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("results", []) or []
            results.extend(batch)
            cursor = data.get("next_url")
            if not cursor:
                break
            await asyncio.sleep(0.2)
        return results

    async def get_splits(self, symbol: str, start: date, end: date) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/v3/reference/splits"
        params = {
            "ticker": symbol.upper(),
            "execution_date.gte": start.strftime(ISO_DATE),
            "execution_date.lte": end.strftime(ISO_DATE),
            "limit": 1000,
            "apiKey": self.api_key,
        }
        results: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while True:
            if cursor:
                params["cursor"] = cursor
            resp = await self._client.get(url, params=params)
            if resp.status_code == 404:
                break
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("results", []) or []
            results.extend(batch)
            cursor = data.get("next_url")
            if not cursor:
                break
            await asyncio.sleep(0.2)
        return results


class FinnhubClient:
    """Client for Finnhub endpoints (earnings, news)."""

    def __init__(self) -> None:
        self.api_key = settings.finnhub_api_key
        if not self.api_key:
            raise RuntimeError("Finnhub API key missing in config/secrets.yml")
        self.base_url = settings.finnhub_base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

    async def close(self) -> None:
        await self._client.aclose()

    async def get_earnings_calendar(self, start: date, end: date) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/calendar/earnings"
        params = {
            "from": start.strftime(ISO_DATE),
            "to": end.strftime(ISO_DATE),
            "token": self.api_key,
        }
        resp = await self._client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("earningsCalendar", []) or []

    async def get_company_news(self, symbol: str, start: date, end: date) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/company-news"
        params = {
            "symbol": symbol.upper(),
            "from": start.strftime(ISO_DATE),
            "to": end.strftime(ISO_DATE),
            "token": self.api_key,
        }
        resp = await self._client.get(url, params=params)
        if resp.status_code == 429:
            await asyncio.sleep(0.5)
            resp = await self._client.get(url, params=params)
            if resp.status_code == 429:
                return []
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    async def get_general_news(self, category: str = "general", limit: int = 100) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/news"
        params = {"category": category, "token": self.api_key}
        resp = await self._client.get(url, params=params)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
        news = data if isinstance(data, list) else []
        return news[:limit]

    async def get_ipo_calendar(self, start: date, end: date) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/calendar/ipo"
        params = {
            "from": start.strftime(ISO_DATE),
            "to": end.strftime(ISO_DATE),
            "token": self.api_key,
        }
        resp = await self._client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("ipoCalendar", []) or []

    async def get_economic_calendar(self, start: date, end: date) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/calendar/economic"
        params = {
            "from": start.strftime(ISO_DATE),
            "to": end.strftime(ISO_DATE),
            "token": self.api_key,
        }
        resp = await self._client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("economicCalendar", []) or []


class AlphaVantageClient:
    """Client for Alpha Vantage earnings endpoints."""

    def __init__(self) -> None:
        self.api_key = settings.alphavantage_api_key
        if not self.api_key:
            raise RuntimeError("AlphaVantage API key missing in config/secrets.yml")
        self.base_url = "https://www.alphavantage.co/query"
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

    async def close(self) -> None:
        await self._client.aclose()

    async def get_earnings(self, symbol: str) -> List[Dict[str, Any]]:
        params = {
            "function": "EARNINGS",
            "symbol": symbol.upper(),
            "apikey": self.api_key,
        }
        resp = await self._client.get(self.base_url, params=params)
        resp.raise_for_status()
        data = resp.json()
        quarterly = data.get("quarterlyEarnings", []) or []
        return quarterly


class FREDClient:
    """Client for FRED economic series."""

    def __init__(self) -> None:
        self.api_key = settings.fred_api_key
        if not self.api_key:
            raise RuntimeError("FRED API key missing in config/secrets.yml")
        self.base_url = settings.fred_base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

    async def close(self) -> None:
        await self._client.aclose()

    async def get_observations(
        self,
        series_id: str,
        start: date,
        end: date,
    ) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/series/observations"
        params = {
            "series_id": series_id,
            "observation_start": start.strftime(ISO_DATE),
            "observation_end": end.strftime(ISO_DATE),
            "api_key": self.api_key,
            "file_type": "json",
        }
        resp = await self._client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("observations", []) or []

    async def get_series_info(self, series_id: str) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/series"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        resp = await self._client.get(url, params=params)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        series = data.get("seriess", []) or []
        return series[0] if series else None


class NewsAPIClient:
    """Client for NewsAPI.org articles."""

    def __init__(self) -> None:
        self.api_key = settings.news_api_key
        if not self.api_key:
            raise RuntimeError("NewsAPI key missing in config/secrets.yml")
        self.base_url = settings.news_api_base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

    async def close(self) -> None:
        await self._client.aclose()

    async def get_articles(
        self,
        query: str,
        start: date,
        end: date,
        page_size: int = 100,
        max_pages: int = 5,
        language: str = "en",
    ) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/everything"
        results: List[Dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            params = {
                "q": query,
                "from": start.strftime(ISO_DATE),
                "to": end.strftime(ISO_DATE),
                "language": language,
                "pageSize": page_size,
                "page": page,
                "sortBy": "publishedAt",
                "apiKey": self.api_key,
            }
            resp = await self._client.get(url, params=params)
            if resp.status_code == 426:
                break
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", []) or []
            if not articles:
                break
            results.extend(articles)
            await asyncio.sleep(0.2)
        return results


class SECClient:
    """Client for SEC EDGAR submissions and filings."""

    def __init__(self) -> None:
        self.user_agent = settings.sec_user_agent
        if not self.user_agent:
            raise RuntimeError("SEC user agent missing in config/secrets.yml")
        self.base_url = "https://data.sec.gov"
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
        )
        self._ticker_map: Optional[Dict[str, str]] = None

    async def close(self) -> None:
        await self._client.aclose()

    async def _get_ticker_map(self) -> Dict[str, str]:
        if self._ticker_map is not None:
            return self._ticker_map
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = await self._client.get(url)
        resp.raise_for_status()
        data = resp.json()
        mapping: Dict[str, str] = {}
        for entry in data.values():
            ticker = (entry.get("ticker") or "").upper()
            cik = entry.get("cik_str")
            if ticker and cik is not None:
                mapping[ticker] = str(cik).zfill(10)
        self._ticker_map = mapping
        return mapping

    def _extract_filings(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        if "filings" in payload and isinstance(payload.get("filings"), dict):
            filings = payload.get("filings") or {}
            recent = filings.get("recent")
            if isinstance(recent, dict):
                payload = recent
        forms = payload.get("form", []) or []
        dates = payload.get("filingDate", []) or []
        acceptance = payload.get("acceptanceDateTime", []) or []
        accessions = payload.get("accessionNumber", []) or []
        primary_docs = payload.get("primaryDocument", []) or []
        report_dates = payload.get("reportDate", []) or []
        file_numbers = payload.get("fileNumber", []) or []
        items = payload.get("items", []) or []

        results: List[Dict[str, Any]] = []
        total = len(forms)
        for idx in range(total):
            filing_date = dates[idx] if idx < len(dates) else None
            form = forms[idx] if idx < len(forms) else None
            if not filing_date:
                continue
            results.append(
                {
                    "form": form,
                    "filing_date": filing_date,
                    "acceptance_datetime": acceptance[idx] if idx < len(acceptance) else None,
                    "accession_number": accessions[idx] if idx < len(accessions) else None,
                    "primary_document": primary_docs[idx] if idx < len(primary_docs) else None,
                    "report_date": report_dates[idx] if idx < len(report_dates) else None,
                    "file_number": file_numbers[idx] if idx < len(file_numbers) else None,
                    "items": items[idx] if idx < len(items) else None,
                }
            )
        return results

    async def get_filings(self, symbol: str, start: date, end: date) -> List[Dict[str, Any]]:
        ticker_map = await self._get_ticker_map()
        cik = ticker_map.get(symbol.upper())
        if not cik:
            return []

        resp = await self._client.get(f"{self.base_url}/submissions/CIK{cik}.json")
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()

        results: List[Dict[str, Any]] = []
        results.extend(self._extract_filings(data))

        files = (data.get("filings") or {}).get("files", []) or []
        for entry in files:
            filing_from = parse_iso_date(entry.get("filingFrom"))
            filing_to = parse_iso_date(entry.get("filingTo"))
            if filing_from and filing_from > end:
                continue
            if filing_to and filing_to < start:
                continue
            name = entry.get("name")
            if not name:
                continue
            await asyncio.sleep(0.2)
            file_resp = await self._client.get(f"{self.base_url}/submissions/{name}")
            if file_resp.status_code == 404:
                continue
            file_resp.raise_for_status()
            file_data = file_resp.json()
            results.extend(self._extract_filings(file_data))

        filtered: List[Dict[str, Any]] = []
        for filing in results:
            filing_date = parse_iso_date(filing.get("filing_date"))
            if not filing_date:
                continue
            if filing_date < start or filing_date > end:
                continue
            filing["filing_date"] = filing_date
            filtered.append(filing)
        return filtered
