"""Event data ingestion helpers."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

import httpx
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.schema import event_events
from src.db.session import engine
from src.definitions.market_universe import MarketUniverse
from src.utils.ingestion.clients import AlphaVantageClient, FinnhubClient, PolygonClient, SECClient
from src.utils.ingestion.common import chunk, parse_iso_date
from src.utils.ingestion.types import IngestionSummary

UTC = timezone.utc


async def _get_alphavantage_earnings(
    client: AlphaVantageClient,
    symbol: str,
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> List[Dict]:
    delay = base_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return await client.get_earnings(symbol)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {429, 500, 502, 503, 504} and attempt < max_attempts:
                print(
                    f"Retrying AlphaVantage earnings for {symbol} after status {status} "
                    f"(attempt {attempt}/{max_attempts})"
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 8.0)
                continue
            print(f"Skipping AlphaVantage earnings for {symbol}: {status}")
            return []
        except httpx.HTTPError as exc:
            if attempt < max_attempts:
                print(
                    f"AlphaVantage earnings error for {symbol} "
                    f"(attempt {attempt}/{max_attempts}): {exc}"
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 8.0)
                continue
            print(f"Skipping AlphaVantage earnings for {symbol}: {exc}")
            return []
        except Exception as exc:
            print(f"Skipping AlphaVantage earnings for {symbol}: {exc}")
            return []
    return []


async def ingest_event_data(
    start: date,
    end: date,
    symbols: Optional[List[str]] = None,
) -> IngestionSummary:
    symbols = [s.upper() for s in (symbols or MarketUniverse.get_all_symbols())]
    symbol_set = set(symbols)

    polygon_earnings = 0
    polygon_dividends = 0
    polygon_splits = 0
    finnhub_earnings = 0
    finnhub_ipo = 0
    finnhub_macro = 0
    alphavantage_events = 0
    sec_filings = 0

    polygon_client: Optional[PolygonClient] = None
    finnhub_client: Optional[FinnhubClient] = None
    alphavantage_client: Optional[AlphaVantageClient] = None
    sec_client: Optional[SECClient] = None

    try:
        polygon_client = PolygonClient()
    except RuntimeError as exc:
        print(f"Polygon events disabled: {exc}")
    try:
        finnhub_client = FinnhubClient()
    except RuntimeError as exc:
        print(f"Finnhub events disabled: {exc}")
    try:
        alphavantage_client = AlphaVantageClient()
    except RuntimeError as exc:
        print(f"AlphaVantage events disabled: {exc}")
    try:
        sec_client = SECClient()
    except RuntimeError as exc:
        print(f"SEC filings disabled: {exc}")

    if polygon_client:
        async with engine.begin() as conn:
            for symbol in symbols:
                try:
                    earnings = await polygon_client.get_earnings(symbol, start, end)
                except httpx.HTTPError as exc:
                    print(f"Polygon earnings failed for {symbol}: {exc}")
                    earnings = []
                dividend_rows: List[Dict] = []
                split_rows: List[Dict] = []
                earnings_rows: List[Dict] = []

                if earnings:
                    for event in earnings:
                        report_date = event.get("report_date") or event.get("fiscal_period_end")
                        if not report_date:
                            continue
                        event_time = datetime.fromisoformat(report_date).replace(tzinfo=UTC)
                        external_id = event.get("id") or f"earnings-{symbol}-{report_date}"
                        earnings_rows.append(
                            {
                                "event_type": "earnings",
                                "event_time": event_time,
                                "external_id": str(external_id)[:512],
                                "source": "polygon",
                                "importance": str(event.get("importance")) if event.get("importance") else None,
                                "symbol": symbol,
                                "payload": event,
                                "created_at": datetime.now(tz=UTC),
                            }
                        )

                try:
                    dividends = await polygon_client.get_dividends(symbol, start, end)
                except httpx.HTTPError as exc:
                    print(f"Polygon dividends failed for {symbol}: {exc}")
                    dividends = []
                if dividends:
                    for dividend in dividends:
                        ex_date = dividend.get("ex_dividend_date") or dividend.get("ex_date")
                        pay_date = dividend.get("pay_date") or dividend.get("payment_date")
                        event_date = parse_iso_date(ex_date) or parse_iso_date(pay_date)
                        if not event_date:
                            continue
                        event_time = datetime.combine(event_date, datetime.min.time(), tzinfo=UTC)
                        cash_amount = dividend.get("cash_amount")
                        external_id = dividend.get("id") or f"dividend-{symbol}-{event_date}-{cash_amount}"
                        dividend_rows.append(
                            {
                                "event_type": "dividend",
                                "event_time": event_time,
                                "external_id": str(external_id)[:512],
                                "source": "polygon",
                                "importance": str(cash_amount) if cash_amount is not None else None,
                                "symbol": symbol,
                                "payload": dividend,
                                "created_at": datetime.now(tz=UTC),
                            }
                        )

                try:
                    splits = await polygon_client.get_splits(symbol, start, end)
                except httpx.HTTPError as exc:
                    print(f"Polygon splits failed for {symbol}: {exc}")
                    splits = []
                if splits:
                    for split in splits:
                        exec_date = split.get("execution_date") or split.get("ex_date") or split.get("split_date")
                        event_date = parse_iso_date(exec_date)
                        if not event_date:
                            continue
                        event_time = datetime.combine(event_date, datetime.min.time(), tzinfo=UTC)
                        split_from = split.get("split_from") or split.get("split_from_factor")
                        split_to = split.get("split_to") or split.get("split_to_factor")
                        ratio = f"{split_to}/{split_from}" if split_from and split_to else None
                        external_id = split.get("id") or f"split-{symbol}-{event_date}-{ratio}"
                        split_rows.append(
                            {
                                "event_type": "split",
                                "event_time": event_time,
                                "external_id": str(external_id)[:512],
                                "source": "polygon",
                                "importance": ratio,
                                "symbol": symbol,
                                "payload": split,
                                "created_at": datetime.now(tz=UTC),
                            }
                        )

                if earnings_rows:
                    polygon_earnings += len(earnings_rows)
                if dividend_rows:
                    polygon_dividends += len(dividend_rows)
                if split_rows:
                    polygon_splits += len(split_rows)

                rows = earnings_rows + dividend_rows + split_rows
                for batch in chunk(rows):
                    stmt = pg_insert(event_events).values(batch)
                    await conn.execute(
                        stmt.on_conflict_do_update(
                            index_elements=[event_events.c.source, event_events.c.external_id, event_events.c.event_time],
                            set_={
                                "importance": stmt.excluded.importance,
                                "symbol": stmt.excluded.symbol,
                                "payload": stmt.excluded.payload,
                                "created_at": stmt.excluded.created_at,
                            },
                        )
                    )
                await asyncio.sleep(0.2)

    if finnhub_client:
        rows: List[Dict] = []
        try:
            calendar = await finnhub_client.get_earnings_calendar(start, end)
        except httpx.HTTPStatusError as exc:
            print(f"Skipping Finnhub earnings calendar: {exc.response.status_code}")
            calendar = []
        if calendar:
            for event in calendar:
                symbol = (event.get("symbol") or "").upper()
                if symbol not in symbol_set:
                    continue
                report_date = event.get("date") or event.get("reportDate")
                if not report_date:
                    continue
                try:
                    event_time = datetime.fromisoformat(report_date).replace(tzinfo=UTC)
                except ValueError:
                    event_time = datetime.strptime(report_date, "%Y-%m-%d").replace(tzinfo=UTC)
                rows.append(
                    {
                        "event_type": "earnings",
                        "event_time": event_time,
                        "external_id": f"FH-earnings-{symbol}-{report_date}",
                        "source": "finnhub",
                        "importance": event.get("hour"),
                        "symbol": symbol,
                        "payload": event,
                        "created_at": datetime.now(tz=UTC),
                    }
                )
            finnhub_earnings += len(rows)

        try:
            ipo_calendar = await finnhub_client.get_ipo_calendar(start, end)
        except httpx.HTTPStatusError as exc:
            print(f"Skipping Finnhub IPO calendar: {exc.response.status_code}")
            ipo_calendar = []
        if ipo_calendar:
            for event in ipo_calendar:
                symbol = (event.get("symbol") or "").upper()
                report_date = event.get("date")
                if not report_date:
                    continue
                try:
                    event_time = datetime.fromisoformat(report_date).replace(tzinfo=UTC)
                except ValueError:
                    event_time = datetime.strptime(report_date, "%Y-%m-%d").replace(tzinfo=UTC)
                external_id = f"FH-ipo-{symbol or 'unknown'}-{report_date}"
                rows.append(
                    {
                        "event_type": "ipo",
                        "event_time": event_time,
                        "external_id": external_id,
                        "source": "finnhub",
                        "importance": event.get("exchange"),
                        "symbol": symbol or None,
                        "payload": event,
                        "created_at": datetime.now(tz=UTC),
                    }
                )
            finnhub_ipo += len(ipo_calendar)

        try:
            econ_calendar = await finnhub_client.get_economic_calendar(start, end)
        except httpx.HTTPStatusError as exc:
            print(f"Skipping Finnhub economic calendar: {exc.response.status_code}")
            econ_calendar = []
        if econ_calendar:
            for event in econ_calendar:
                date_part = event.get("date")
                time_part = event.get("time")
                if not date_part:
                    continue
                report_date = date_part
                if time_part and isinstance(time_part, str):
                    report_date = f"{date_part}T{time_part}"
                try:
                    event_time = datetime.fromisoformat(report_date.replace("Z", "+00:00"))
                except ValueError:
                    try:
                        event_time = datetime.strptime(date_part, "%Y-%m-%d").replace(tzinfo=UTC)
                    except ValueError:
                        continue
                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=UTC)
                country = event.get("country") or "unknown"
                name = event.get("event") or event.get("indicator") or "macro"
                external_id = f"FH-macro-{country}-{name}-{event_time.date()}"
                rows.append(
                    {
                        "event_type": "macro",
                        "event_time": event_time,
                        "external_id": external_id[:512],
                        "source": "finnhub",
                        "importance": event.get("impact"),
                        "symbol": None,
                        "payload": event,
                        "created_at": datetime.now(tz=UTC),
                    }
                )
            finnhub_macro += len(econ_calendar)

        if rows:
            async with engine.begin() as conn:
                for batch in chunk(rows):
                    stmt = pg_insert(event_events).values(batch)
                    await conn.execute(
                        stmt.on_conflict_do_update(
                            index_elements=[event_events.c.source, event_events.c.external_id, event_events.c.event_time],
                            set_={
                                "importance": stmt.excluded.importance,
                                "symbol": stmt.excluded.symbol,
                                "payload": stmt.excluded.payload,
                                "created_at": stmt.excluded.created_at,
                            },
                        )
                    )

    if alphavantage_client:
        alpha_symbols = symbols[:20]
        rows = []
        for symbol in alpha_symbols:
            earnings = await _get_alphavantage_earnings(alphavantage_client, symbol)
            if not earnings:
                await asyncio.sleep(1.0)
                continue
            for entry in earnings:
                report_date = entry.get("fiscalDateEnding")
                if not report_date:
                    continue
                try:
                    event_time = datetime.fromisoformat(report_date).replace(tzinfo=UTC)
                except ValueError:
                    event_time = datetime.strptime(report_date, "%Y-%m-%d").replace(tzinfo=UTC)
                rows.append(
                    {
                        "event_type": "earnings",
                        "event_time": event_time,
                        "external_id": f"AV-{symbol}-{report_date}",
                        "source": "alphavantage",
                        "importance": entry.get("reportedEPS"),
                        "symbol": symbol,
                        "payload": entry,
                        "created_at": datetime.now(tz=UTC),
                    }
                )
            await asyncio.sleep(1.0)
        alphavantage_events = len(rows)
        if rows:
            async with engine.begin() as conn:
                for batch in chunk(rows):
                    stmt = pg_insert(event_events).values(batch)
                    await conn.execute(
                        stmt.on_conflict_do_update(
                            index_elements=[event_events.c.source, event_events.c.external_id, event_events.c.event_time],
                            set_={
                                "importance": stmt.excluded.importance,
                                "symbol": stmt.excluded.symbol,
                                "payload": stmt.excluded.payload,
                                "created_at": stmt.excluded.created_at,
                            },
                        )
                    )

    if sec_client:
        async with engine.begin() as conn:
            for symbol in symbols:
                if MarketUniverse.is_crypto(symbol):
                    continue
                filings = await sec_client.get_filings(symbol, start, end)
                if not filings:
                    await asyncio.sleep(0.05)
                    continue
                rows: List[Dict] = []
                for filing in filings:
                    filing_date = filing.get("filing_date")
                    if not filing_date:
                        continue
                    event_time = datetime.combine(filing_date, datetime.min.time(), tzinfo=UTC)
                    form = filing.get("form")
                    accession = filing.get("accession_number")
                    external_id = accession or f"SEC-{symbol}-{filing_date}-{form}"
                    payload = {
                        "form": form,
                        "filing_date": filing_date.isoformat(),
                        "acceptance_datetime": filing.get("acceptance_datetime"),
                        "accession_number": accession,
                        "primary_document": filing.get("primary_document"),
                        "report_date": filing.get("report_date"),
                        "file_number": filing.get("file_number"),
                        "items": filing.get("items"),
                    }
                    rows.append(
                        {
                            "event_type": "filing",
                            "event_time": event_time,
                            "external_id": str(external_id)[:512],
                            "source": "sec",
                            "importance": form,
                            "symbol": symbol,
                            "payload": payload,
                            "created_at": datetime.now(tz=UTC),
                        }
                    )
                sec_filings += len(rows)
                for batch in chunk(rows):
                    stmt = pg_insert(event_events).values(batch)
                    await conn.execute(
                        stmt.on_conflict_do_update(
                            index_elements=[event_events.c.source, event_events.c.external_id, event_events.c.event_time],
                            set_={
                                "importance": stmt.excluded.importance,
                                "symbol": stmt.excluded.symbol,
                                "payload": stmt.excluded.payload,
                                "created_at": stmt.excluded.created_at,
                            },
                        )
                    )
                await asyncio.sleep(0.1)

    if polygon_client:
        await polygon_client.close()
    if finnhub_client:
        await finnhub_client.close()
    if alphavantage_client:
        await alphavantage_client.close()
    if sec_client:
        await sec_client.close()

    total = (
        polygon_earnings
        + polygon_dividends
        + polygon_splits
        + finnhub_earnings
        + finnhub_ipo
        + finnhub_macro
        + alphavantage_events
        + sec_filings
    )
    return IngestionSummary(
        records=total,
        details={
            "polygon_earnings": polygon_earnings,
            "polygon_dividends": polygon_dividends,
            "polygon_splits": polygon_splits,
            "finnhub_earnings": finnhub_earnings,
            "finnhub_ipo": finnhub_ipo,
            "finnhub_macro": finnhub_macro,
            "alphavantage": alphavantage_events,
            "sec_filings": sec_filings,
            "symbols": len(symbols),
        },
    )
