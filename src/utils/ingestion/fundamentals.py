"""Fundamental data ingestion helpers."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.schema import fundamentals_financials, fundamentals_shares
from src.db.session import engine
from src.definitions.market_universe import MarketUniverse
from src.utils.ingestion.clients import PolygonClient
from src.utils.ingestion.common import chunk, parse_iso_date
from src.utils.ingestion.types import IngestionSummary

UTC = timezone.utc


def _extract_financial_dates(payload: Dict[str, Any]) -> Dict[str, Optional[date]]:
    period_end = (
        parse_iso_date(payload.get("period_of_report"))
        or parse_iso_date(payload.get("period_end_date"))
        or parse_iso_date(payload.get("fiscal_period_end"))
        or parse_iso_date(payload.get("report_date"))
    )
    filing_date = (
        parse_iso_date(payload.get("filing_date"))
        or parse_iso_date(payload.get("filing_date_end"))
        or parse_iso_date(payload.get("filingDate"))
    )
    return {"period_end": period_end, "filing_date": filing_date}


async def ingest_fundamentals(
    start: date,
    end: date,
    symbols: Optional[List[str]] = None,
) -> IngestionSummary:
    try:
        polygon_client = PolygonClient()
    except RuntimeError as exc:
        print(f"Polygon fundamentals ingestion disabled: {exc}")
        return IngestionSummary(records=0, details={"financials": 0, "shares": 0, "symbols": 0})

    symbols = symbols or MarketUniverse.get_all_symbols()
    financial_count = 0
    shares_count = 0
    processed = 0

    async with engine.begin() as conn:
        existing_symbols = {
            row.symbol
            for row in (await conn.execute(text("SELECT DISTINCT symbol FROM fundamentals.financials"))).fetchall()
        }
        for symbol in symbols:
            symbol = symbol.upper()
            if MarketUniverse.is_crypto(symbol):
                continue

            details = None
            try:
                details = await polygon_client.get_ticker_details(symbol)
            except Exception:
                details = None

            if details:
                as_of = (
                    parse_iso_date(details.get("last_updated_utc"))
                    or parse_iso_date(details.get("last_updated"))
                    or parse_iso_date(details.get("updated"))
                    or end
                )
                shares_payload = {
                    "symbol": symbol,
                    "as_of": as_of,
                    "share_class_shares_outstanding": details.get("share_class_shares_outstanding"),
                    "weighted_shares_outstanding": details.get("weighted_shares_outstanding"),
                    "market_cap": details.get("market_cap"),
                    "source": "polygon",
                    "payload": details,
                }
                stmt = pg_insert(fundamentals_shares).values(shares_payload)
                await conn.execute(
                    stmt.on_conflict_do_update(
                        index_elements=[
                            fundamentals_shares.c.symbol,
                            fundamentals_shares.c.as_of,
                            fundamentals_shares.c.source,
                        ],
                        set_={
                            "share_class_shares_outstanding": stmt.excluded.share_class_shares_outstanding,
                            "weighted_shares_outstanding": stmt.excluded.weighted_shares_outstanding,
                            "market_cap": stmt.excluded.market_cap,
                            "payload": stmt.excluded.payload,
                        },
                    )
                )
                shares_count += 1

            financial_rows: List[Dict[str, Any]] = []
            use_history = symbol not in existing_symbols
            for timeframe in ("quarterly", "annual"):
                try:
                    results = await polygon_client.get_financials(
                        symbol,
                        None if use_history else start,
                        None if use_history else end,
                        timeframe=timeframe,
                    )
                except Exception:
                    results = []
                for payload in results:
                    fiscal_year = payload.get("fiscal_year") or payload.get("fiscalYear")
                    fiscal_period = payload.get("fiscal_period") or payload.get("fiscalPeriod")
                    dates = _extract_financial_dates(payload)
                    row = {
                        "symbol": symbol,
                        "timeframe": payload.get("timeframe") or timeframe,
                        "fiscal_year": int(fiscal_year) if fiscal_year else None,
                        "fiscal_period": str(fiscal_period) if fiscal_period else None,
                        "period_end": dates["period_end"],
                        "filing_date": dates["filing_date"],
                        "source": "polygon",
                        "payload": payload,
                        "created_at": datetime.now(tz=UTC),
                    }
                    financial_rows.append(row)
                await asyncio.sleep(0.2)

            if financial_rows:
                deduped: Dict[tuple, Dict[str, Any]] = {}
                for row in financial_rows:
                    key = (
                        row["symbol"],
                        row["timeframe"],
                        row["fiscal_year"],
                        row["fiscal_period"],
                        row["source"],
                    )
                    deduped[key] = row
                rows_to_insert = list(deduped.values())
                for batch in chunk(rows_to_insert, size=200):
                    stmt = pg_insert(fundamentals_financials).values(batch)
                    await conn.execute(
                        stmt.on_conflict_do_update(
                            index_elements=[
                                fundamentals_financials.c.symbol,
                                fundamentals_financials.c.timeframe,
                                fundamentals_financials.c.fiscal_year,
                                fundamentals_financials.c.fiscal_period,
                                fundamentals_financials.c.source,
                            ],
                            set_={
                                "period_end": stmt.excluded.period_end,
                                "filing_date": stmt.excluded.filing_date,
                                "payload": stmt.excluded.payload,
                            },
                        )
                    )
                financial_count += len(rows_to_insert)

            processed += 1
            await asyncio.sleep(0.05)

    await polygon_client.close()
    return IngestionSummary(
        records=financial_count + shares_count,
        details={
            "financials": financial_count,
            "shares": shares_count,
            "symbols": processed,
        },
    )
