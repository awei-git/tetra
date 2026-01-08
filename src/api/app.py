"""FastAPI app for data status and ingestion triggers."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text

from src.db.session import engine
from src.pipelines.data.runner import run_all_pipelines

UTC = timezone.utc

app = FastAPI(title="Tetra Data Console")

frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

pipeline_state: Dict[str, Any] = {
    "status": "idle",
    "last_run": None,
    "last_error": None,
    "last_result": None,
}


class IngestRequest(BaseModel):
    start_date: Optional[date] = None
    end_date: Optional[date] = None


def _isoformat(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


async def _fetch_scalar(query: str) -> Optional[Any]:
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text(query))
            return result.scalar_one_or_none()
    except Exception:
        return None


async def _gather_status() -> Dict[str, Any]:
    counts = {
        "assets": await _fetch_scalar("SELECT COUNT(*) FROM market.assets"),
        "ohlcv": await _fetch_scalar("SELECT COUNT(*) FROM market.ohlcv"),
        "events": await _fetch_scalar("SELECT COUNT(*) FROM event.events"),
        "economic_series": await _fetch_scalar("SELECT COUNT(*) FROM economic.series"),
        "economic_values": await _fetch_scalar("SELECT COUNT(*) FROM economic.values"),
        "news": await _fetch_scalar("SELECT COUNT(*) FROM news.articles"),
    }
    latest = {
        "ohlcv": await _fetch_scalar("SELECT MAX(timestamp) FROM market.ohlcv"),
        "events": await _fetch_scalar("SELECT MAX(event_time) FROM event.events"),
        "economic": await _fetch_scalar("SELECT MAX(timestamp) FROM economic.values"),
        "news": await _fetch_scalar("SELECT MAX(published_at) FROM news.articles"),
    }
    return {
        "counts": counts,
        "latest": latest,
        "pipeline": pipeline_state,
    }


async def _run_ingestion(start: date, end: date) -> None:
    pipeline_state["status"] = "running"
    pipeline_state["last_run"] = datetime.now(tz=UTC).isoformat()
    pipeline_state["last_error"] = None
    pipeline_state["last_result"] = None
    try:
        results = await run_all_pipelines(start=start, end=end)
        pipeline_state["status"] = "success"
        pipeline_state["last_result"] = [asdict(result) for result in results]
    except Exception as exc:
        pipeline_state["status"] = "failed"
        pipeline_state["last_error"] = str(exc)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(frontend_dir / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/strats")
async def strats() -> FileResponse:
    return FileResponse(frontend_dir / "strats.html", headers={"Cache-Control": "no-store"})


@app.get("/api/status")
async def get_status() -> Dict[str, Any]:
    return await _gather_status()


@app.post("/api/ingest")
async def trigger_ingest(request: IngestRequest) -> Dict[str, Any]:
    if pipeline_state["status"] == "running":
        raise HTTPException(status_code=409, detail="Ingestion already running")

    today = date.today()
    start = request.start_date or (today - timedelta(days=1))
    end = request.end_date or today

    asyncio.create_task(_run_ingestion(start, end))
    return {"status": "started", "start": start.isoformat(), "end": end.isoformat()}


@app.get("/api/market/coverage")
async def get_market_coverage(limit: int = 500) -> Dict[str, Any]:
    limit = max(1, min(limit, 2000))
    query = text(
        """
        SELECT
          o.symbol,
          a.name,
          COUNT(*) AS rows,
          MIN(o.timestamp) AS start_ts,
          MAX(o.timestamp) AS end_ts,
          ARRAY_AGG(DISTINCT o.source) AS sources
        FROM market.ohlcv o
        LEFT JOIN market.assets a ON a.symbol = o.symbol
        GROUP BY o.symbol, a.name
        ORDER BY o.symbol
        LIMIT :limit
        """
    )
    total_query = text("SELECT COUNT(DISTINCT symbol) FROM market.ohlcv")
    async with engine.begin() as conn:
        total_result = await conn.execute(total_query)
        total_symbols = total_result.scalar_one_or_none() or 0
        result = await conn.execute(query, {"limit": limit})
        rows = result.fetchall()

    coverage: List[Dict[str, Any]] = []
    for row in rows:
        start_ts = row.start_ts
        end_ts = row.end_ts
        span_days = None
        if start_ts and end_ts:
            span_days = max(0, (end_ts - start_ts).days)
        coverage.append(
            {
                "symbol": row.symbol,
                "name": row.name,
                "rows": row.rows,
                "start": _isoformat(start_ts),
                "end": _isoformat(end_ts),
                "days": span_days,
                "sources": [s for s in (row.sources or []) if s],
            }
        )

    return {"total_symbols": total_symbols, "coverage": coverage}


@app.get("/api/events/summary")
async def get_events_summary(limit: int = 50) -> Dict[str, Any]:
    limit = max(1, min(limit, 200))
    query = text(
        """
        SELECT
          event_type,
          COUNT(*) AS events,
          COUNT(DISTINCT symbol) AS symbols,
          COUNT(DISTINCT source) AS sources,
          MIN(event_time) AS start_ts,
          MAX(event_time) AS end_ts
        FROM event.events
        GROUP BY event_type
        ORDER BY events DESC, event_type
        LIMIT :limit
        """
    )
    total_events_query = text("SELECT COUNT(*) FROM event.events")
    total_types_query = text("SELECT COUNT(DISTINCT event_type) FROM event.events")
    async with engine.begin() as conn:
        total_events_result = await conn.execute(total_events_query)
        total_types_result = await conn.execute(total_types_query)
        total_events = total_events_result.scalar_one_or_none() or 0
        total_types = total_types_result.scalar_one_or_none() or 0
        result = await conn.execute(query, {"limit": limit})
        rows = result.fetchall()

    summary: List[Dict[str, Any]] = []
    for row in rows:
        summary.append(
            {
                "event_type": row.event_type,
                "events": row.events,
                "symbols": row.symbols,
                "sources": row.sources,
                "start": _isoformat(row.start_ts),
                "end": _isoformat(row.end_ts),
            }
        )

    return {"total_events": total_events, "total_types": total_types, "summary": summary}


@app.get("/api/economic/summary")
async def get_economic_summary(limit: int = 50) -> Dict[str, Any]:
    limit = max(1, min(limit, 200))
    query = text(
        """
        SELECT
          s.series_id,
          s.name,
          s.frequency,
          COUNT(v.timestamp) AS values,
          MIN(v.timestamp) AS start_ts,
          MAX(v.timestamp) AS end_ts
        FROM economic.series s
        LEFT JOIN economic.values v ON v.series_id = s.series_id
        GROUP BY s.series_id, s.name, s.frequency
        ORDER BY values DESC NULLS LAST, s.series_id
        LIMIT :limit
        """
    )
    total_series_query = text("SELECT COUNT(*) FROM economic.series")
    total_values_query = text("SELECT COUNT(*) FROM economic.values")
    async with engine.begin() as conn:
        total_series_result = await conn.execute(total_series_query)
        total_values_result = await conn.execute(total_values_query)
        total_series = total_series_result.scalar_one_or_none() or 0
        total_values = total_values_result.scalar_one_or_none() or 0
        result = await conn.execute(query, {"limit": limit})
        rows = result.fetchall()

    summary: List[Dict[str, Any]] = []
    for row in rows:
        summary.append(
            {
                "series_id": row.series_id,
                "name": row.name,
                "frequency": row.frequency,
                "values": row.values,
                "start": _isoformat(row.start_ts),
                "end": _isoformat(row.end_ts),
            }
        )

    return {"total_series": total_series, "total_values": total_values, "summary": summary}


@app.get("/api/market/ohlcv")
async def get_market_ohlcv(symbol: str, limit: int = 3650) -> Dict[str, Any]:
    symbol = symbol.upper()
    limit = max(1, min(limit, 5000))
    query = text(
        """
        SELECT timestamp, open, high, low, close, volume, source
        FROM market.ohlcv
        WHERE symbol = :symbol
        ORDER BY timestamp DESC
        LIMIT :limit
        """
    )
    async with engine.begin() as conn:
        result = await conn.execute(query, {"symbol": symbol, "limit": limit})
        rows = result.fetchall()

    series = [
        {
            "timestamp": _isoformat(row.timestamp),
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
            "source": row.source,
        }
        for row in reversed(rows)
    ]
    return {"symbol": symbol, "series": series}


@app.get("/api/news/sentiment")
async def get_news_sentiment(limit: int = 50) -> Dict[str, Any]:
    limit = max(1, min(limit, 500))
    symbol_query = text(
        """
        SELECT
          symbol,
          COUNT(*) AS articles,
          AVG(sentiment) AS avg_sentiment,
          MIN(published_at) AS start_ts,
          MAX(published_at) AS end_ts
        FROM news.articles
        CROSS JOIN LATERAL unnest(coalesce(tickers, ARRAY[]::varchar[])) AS symbol
        WHERE sentiment IS NOT NULL
        GROUP BY symbol
        ORDER BY articles DESC
        LIMIT :limit
        """
    )
    macro_query = text(
        """
        SELECT
          topic,
          COUNT(*) AS articles,
          AVG(sentiment) AS avg_sentiment,
          MIN(published_at) AS start_ts,
          MAX(published_at) AS end_ts
        FROM news.articles
        CROSS JOIN LATERAL jsonb_array_elements_text(
          COALESCE(payload->'analysis'->'topics', '[]'::jsonb)
        ) AS topic
        WHERE sentiment IS NOT NULL
        GROUP BY topic
        ORDER BY articles DESC
        LIMIT :limit
        """
    )
    async with engine.begin() as conn:
        symbol_rows = (await conn.execute(symbol_query, {"limit": limit})).fetchall()
        macro_rows = (await conn.execute(macro_query, {"limit": limit})).fetchall()

    symbols = [
        {
            "symbol": row.symbol,
            "articles": row.articles,
            "avg_sentiment": float(row.avg_sentiment) if row.avg_sentiment is not None else None,
            "start": _isoformat(row.start_ts),
            "end": _isoformat(row.end_ts),
        }
        for row in symbol_rows
    ]
    macro = [
        {
            "topic": row.topic,
            "articles": row.articles,
            "avg_sentiment": float(row.avg_sentiment) if row.avg_sentiment is not None else None,
            "start": _isoformat(row.start_ts),
            "end": _isoformat(row.end_ts),
        }
        for row in macro_rows
    ]
    return {"symbols": symbols, "macro": macro}
