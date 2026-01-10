"""Market data ingestion helpers."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.schema import market_assets, market_ohlcv
from src.db.session import engine
from src.definitions.market_universe import MarketUniverse
from src.utils.ingestion.clients import PolygonClient
from src.utils.ingestion.common import chunk, parse_iso_date
from src.utils.ingestion.types import IngestionSummary
from src.utils.ingestion.yfinance import fetch_yfinance_asset_info, fetch_yfinance_ohlcv, yfinance_available

UTC = timezone.utc
NY_TZ = ZoneInfo("America/New_York")
CLOSE_HOUR = 16
CLOSE_MINUTE = 15


def last_market_close_date(now: Optional[datetime] = None) -> date:
    current = now.astimezone(NY_TZ) if now else datetime.now(tz=NY_TZ)
    local_date = current.date()

    def prev_business_day(value: date) -> date:
        while value.weekday() >= 5:
            value -= timedelta(days=1)
        return value

    if local_date.weekday() >= 5:
        return prev_business_day(local_date)
    if (current.hour, current.minute) < (CLOSE_HOUR, CLOSE_MINUTE):
        return prev_business_day(local_date - timedelta(days=1))
    return local_date


async def ingest_market_data(
    start: date,
    end: date,
    symbols: Optional[List[str]] = None,
) -> IngestionSummary:
    polygon_client: Optional[PolygonClient] = None
    try:
        polygon_client = PolygonClient()
    except RuntimeError as exc:
        print(f"Polygon market ingestion disabled: {exc}")
    yf_enabled = yfinance_available()
    if not polygon_client and not yf_enabled:
        return IngestionSummary(records=0, details={"assets": 0, "ohlcv": 0, "symbols": 0})
    symbols = symbols or MarketUniverse.get_all_symbols()

    asset_count = 0
    ohlcv_count = 0
    ohlcv_polygon = 0
    ohlcv_yfinance = 0
    processed_symbols = 0

    market_end = min(end, last_market_close_date())
    async with engine.begin() as conn:
        for symbol in symbols:
            symbol = symbol.upper()
            symbol_start = max(start, MarketUniverse.get_symbol_start_date(symbol))
            if symbol_start > market_end:
                continue

            try:
                if polygon_client and not MarketUniverse.is_crypto(symbol):
                    details = await polygon_client.get_ticker_details(symbol)
                else:
                    details = None
            except Exception:
                details = None
            yf_info = None
            if not details and yf_enabled:
                try:
                    yf_info = fetch_yfinance_asset_info(symbol)
                except Exception as exc:
                    print(f"yfinance asset info failed for {symbol}: {exc}")

            metadata: Dict[str, Optional[Dict]] = {}
            if details:
                metadata["polygon"] = details
            if yf_info:
                metadata["yfinance"] = yf_info

            yf_listed = None
            if yf_info and yf_info.get("firstTradeDateEpochUtc"):
                try:
                    yf_listed = datetime.fromtimestamp(
                        int(yf_info["firstTradeDateEpochUtc"]),
                        tz=UTC,
                    ).date()
                except Exception:
                    yf_listed = None

            asset_payload = {
                "symbol": symbol,
                "asset_type": (details or {}).get("type") or (yf_info or {}).get("quoteType") or "unknown",
                "name": (details or {}).get("name") or (yf_info or {}).get("shortName") or (yf_info or {}).get("longName"),
                "exchange": (details or {}).get("primary_exchange") or (yf_info or {}).get("exchange"),
                "currency": (details or {}).get("currency_name") or (yf_info or {}).get("currency"),
                "sector": (details or {}).get("sic_description") or (yf_info or {}).get("sector"),
                "industry": (yf_info or {}).get("industry"),
                "is_active": (details or {}).get("active", True),
                "listed_at": parse_iso_date((details or {}).get("list_date")) or yf_listed,
                "delisted_at": parse_iso_date((details or {}).get("delisted_utc")),
                "metadata": metadata or None,
            }
            asset_stmt = pg_insert(market_assets).values(asset_payload)
            await conn.execute(
                asset_stmt.on_conflict_do_update(
                    index_elements=[market_assets.c.symbol],
                    set_={
                        "asset_type": asset_stmt.excluded.asset_type,
                        "name": asset_stmt.excluded.name,
                        "exchange": asset_stmt.excluded.exchange,
                        "currency": asset_stmt.excluded.currency,
                        "sector": asset_stmt.excluded.sector,
                        "industry": asset_stmt.excluded.industry,
                        "is_active": asset_stmt.excluded.is_active,
                        "listed_at": asset_stmt.excluded.listed_at,
                        "delisted_at": asset_stmt.excluded.delisted_at,
                        "metadata": asset_stmt.excluded.metadata,
                        "updated_at": func.now(),
                    },
                )
            )
            asset_count += 1
            processed_symbols += 1

            ohlcv: List[Dict] = []
            polygon_start_dt: Optional[datetime] = None
            if polygon_client and not MarketUniverse.is_crypto(symbol):
                try:
                    ohlcv = await polygon_client.get_daily_ohlc(symbol, symbol_start, market_end)
                except Exception:
                    ohlcv = []
                if ohlcv:
                    polygon_start_dt = datetime.fromtimestamp(ohlcv[0]["t"] / 1000, tz=UTC)
                    ohlcv_polygon += len(ohlcv)
            if yf_enabled:
                missing_start = symbol_start
                missing_end = None
                if polygon_start_dt and polygon_start_dt.date() > symbol_start:
                    missing_end = polygon_start_dt.date() - timedelta(days=1)
                if not ohlcv:
                    missing_end = market_end
                yfinance_rows: List[Dict] = []
                if missing_end and missing_end >= missing_start:
                    try:
                        yfinance_rows = fetch_yfinance_ohlcv(symbol, missing_start, missing_end)
                    except Exception as exc:
                        print(f"yfinance OHLCV failed for {symbol}: {exc}")
                        yfinance_rows = []
                    ohlcv_yfinance += len(yfinance_rows)
                if yfinance_rows:
                    ohlcv = yfinance_rows + ohlcv
            if not ohlcv:
                await asyncio.sleep(0.05)
                continue

            rows: List[Dict] = []
            for item in ohlcv:
                if "t" in item:
                    ts = datetime.fromtimestamp(item["t"] / 1000, tz=UTC)
                else:
                    ts = item["timestamp"]
                rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": ts,
                        "open": item.get("o") or item.get("open"),
                        "high": item.get("h") or item.get("high"),
                        "low": item.get("l") or item.get("low"),
                        "close": item.get("c") or item.get("close"),
                        "volume": item.get("v") or item.get("volume"),
                        "vwap": item.get("vw") or item.get("vwap"),
                        "turnover": item.get("turnover"),
                        "source": item.get("source") or ("polygon" if "t" in item else "yfinance"),
                        "ingested_at": datetime.now(tz=UTC),
                    }
                )
            ohlcv_count += len(rows)
            for batch in chunk(rows):
                stmt = pg_insert(market_ohlcv).values(batch)
                await conn.execute(
                    stmt.on_conflict_do_update(
                        index_elements=[market_ohlcv.c.symbol, market_ohlcv.c.timestamp],
                        set_={
                            "open": stmt.excluded.open,
                            "high": stmt.excluded.high,
                            "low": stmt.excluded.low,
                            "close": stmt.excluded.close,
                            "volume": stmt.excluded.volume,
                            "vwap": stmt.excluded.vwap,
                            "turnover": stmt.excluded.turnover,
                            "source": stmt.excluded.source,
                            "ingested_at": stmt.excluded.ingested_at,
                        },
                    )
                )
            await asyncio.sleep(0.25)

    if polygon_client:
        await polygon_client.close()
    return IngestionSummary(
        records=asset_count + ohlcv_count,
        details={
            "assets": asset_count,
            "ohlcv": ohlcv_count,
            "ohlcv_polygon": ohlcv_polygon,
            "ohlcv_yfinance": ohlcv_yfinance,
            "symbols": processed_symbols,
            "market_end": market_end.isoformat(),
        },
    )
