"""Ingest insider trades + analyst recommendations from Finnhub."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.session import engine
from src.definitions.market_universe import MarketUniverse
from src.utils.ingestion.clients import FinnhubClient
from src.utils.ingestion.common import chunk
from src.utils.ingestion.types import IngestionSummary

UTC = timezone.utc

# Finnhub free tier: 60 req/min. Use 1.1s delay for safety.
_RATE_DELAY = 1.1


def _get_stock_symbols(symbols: Optional[List[str]] = None) -> List[str]:
    """Get individual stock symbols only (skip ETFs, crypto)."""
    all_syms = [s.upper() for s in (symbols or MarketUniverse.get_all_symbols())]
    return [s for s in all_syms
            if not MarketUniverse.is_crypto(s)
            and not MarketUniverse.is_etf(s)]


async def ingest_insider_trades(
    symbols: Optional[List[str]] = None,
) -> IngestionSummary:
    """Fetch and store insider transactions for all tracked symbols."""
    symbols = _get_stock_symbols(symbols)

    try:
        client = FinnhubClient()
    except RuntimeError as exc:
        return IngestionSummary(records=0, details={"error": str(exc)})

    total = 0
    async with engine.begin() as conn:
        for symbol in symbols:
            try:
                txns = await client.get_insider_transactions(symbol)
            except Exception as exc:
                print(f"Insider trades failed for {symbol}: {exc}")
                txns = []

            if not txns:
                await asyncio.sleep(_RATE_DELAY)
                continue

            rows = []
            for t in txns:
                filing_date = t.get("filingDate")
                if not filing_date:
                    continue
                try:
                    fd = date.fromisoformat(filing_date[:10])
                except ValueError:
                    continue

                td = None
                if t.get("transactionDate"):
                    try:
                        td = date.fromisoformat(t["transactionDate"][:10])
                    except ValueError:
                        pass

                rows.append({
                    "symbol": symbol,
                    "filing_date": fd,
                    "transaction_date": td,
                    "insider_name": t.get("name"),
                    "insider_title": t.get("transactionCode"),
                    "transaction_type": t.get("transactionType"),
                    "shares": float(t["share"]) if t.get("share") else None,
                    "price": float(t["price"]) if t.get("price") else None,
                    "value": float(t["value"]) if t.get("value") else None,
                    "shares_after": float(t["shareAfter"]) if t.get("shareAfter") else None,
                    "source": "finnhub",
                    "payload": t,
                    "created_at": datetime.now(tz=UTC),
                })

            if rows:
                for batch in chunk(rows):
                    await conn.execute(text("""
                        INSERT INTO event.insider_trades
                          (symbol, filing_date, transaction_date, insider_name,
                           insider_title, transaction_type, shares, price, value,
                           shares_after, source, payload, created_at)
                        VALUES
                          (:symbol, :filing_date, :transaction_date, :insider_name,
                           :insider_title, :transaction_type, :shares, :price, :value,
                           :shares_after, :source, CAST(:payload AS JSONB), :created_at)
                        ON CONFLICT (symbol, filing_date, insider_name, transaction_type, shares)
                        DO UPDATE SET
                          price = EXCLUDED.price,
                          value = EXCLUDED.value,
                          payload = EXCLUDED.payload
                    """), [{**r, "payload": __import__("json").dumps(r["payload"])} for r in batch])
                total += len(rows)

            await asyncio.sleep(_RATE_DELAY)

    await client.close()
    return IngestionSummary(records=total, details={"symbols": len(symbols)})


async def ingest_analyst_recommendations(
    symbols: Optional[List[str]] = None,
) -> IngestionSummary:
    """Fetch and store analyst recommendation trends."""
    symbols = _get_stock_symbols(symbols)

    try:
        client = FinnhubClient()
    except RuntimeError as exc:
        return IngestionSummary(records=0, details={"error": str(exc)})

    total = 0
    async with engine.begin() as conn:
        for symbol in symbols:
            try:
                recs = await client.get_recommendation(symbol)
            except Exception as exc:
                print(f"Analyst recs failed for {symbol}: {exc}")
                recs = []

            if not recs:
                await asyncio.sleep(_RATE_DELAY)
                continue

            rows = []
            for r in recs:
                period = r.get("period")
                if not period:
                    continue
                try:
                    pd = date.fromisoformat(period[:10])
                except ValueError:
                    continue

                rows.append({
                    "symbol": symbol,
                    "period": pd,
                    "strong_buy": r.get("strongBuy", 0),
                    "buy": r.get("buy", 0),
                    "hold": r.get("hold", 0),
                    "sell": r.get("sell", 0),
                    "strong_sell": r.get("strongSell", 0),
                    "source": "finnhub",
                    "payload": __import__("json").dumps(r),
                    "created_at": datetime.now(tz=UTC),
                })

            if rows:
                for batch in chunk(rows):
                    await conn.execute(text("""
                        INSERT INTO event.analyst_recommendations
                          (symbol, period, strong_buy, buy, hold, sell, strong_sell,
                           source, payload, created_at)
                        VALUES
                          (:symbol, :period, :strong_buy, :buy, :hold, :sell, :strong_sell,
                           :source, CAST(:payload AS JSONB), :created_at)
                        ON CONFLICT (symbol, period, source)
                        DO UPDATE SET
                          strong_buy = EXCLUDED.strong_buy,
                          buy = EXCLUDED.buy,
                          hold = EXCLUDED.hold,
                          sell = EXCLUDED.sell,
                          strong_sell = EXCLUDED.strong_sell,
                          payload = EXCLUDED.payload
                    """), batch)
                total += len(rows)

            await asyncio.sleep(_RATE_DELAY)

    await client.close()
    return IngestionSummary(records=total, details={"symbols": len(symbols)})


async def ingest_peer_network(
    symbols: Optional[List[str]] = None,
) -> IngestionSummary:
    """Fetch and store peer relationships for co-coverage network."""
    symbols = _get_stock_symbols(symbols)

    try:
        client = FinnhubClient()
    except RuntimeError as exc:
        return IngestionSummary(records=0, details={"error": str(exc)})

    total = 0
    async with engine.begin() as conn:
        for symbol in symbols:
            try:
                peers = await client.get_peers(symbol)
            except Exception as exc:
                print(f"Peers failed for {symbol}: {exc}")
                peers = []

            if not peers:
                await asyncio.sleep(_RATE_DELAY)
                continue

            rows = []
            for peer in peers:
                if peer == symbol:
                    continue
                rows.append({
                    "symbol": symbol,
                    "peer_symbol": peer,
                    "source": "finnhub",
                    "updated_at": datetime.now(tz=UTC),
                })

            if rows:
                for batch in chunk(rows):
                    await conn.execute(text("""
                        INSERT INTO network.analyst_coverage
                          (symbol, peer_symbol, source, updated_at)
                        VALUES (:symbol, :peer_symbol, :source, :updated_at)
                        ON CONFLICT (symbol, peer_symbol, source)
                        DO UPDATE SET updated_at = EXCLUDED.updated_at
                    """), batch)
                total += len(rows)

            await asyncio.sleep(_RATE_DELAY)

    await client.close()
    return IngestionSummary(records=total, details={"symbols": len(symbols)})


async def ingest_supply_chain(
    symbols: Optional[List[str]] = None,
) -> IngestionSummary:
    """Fetch and store supply chain relationships."""
    symbols = _get_stock_symbols(symbols)

    try:
        client = FinnhubClient()
    except RuntimeError as exc:
        return IngestionSummary(records=0, details={"error": str(exc)})

    total = 0
    async with engine.begin() as conn:
        for symbol in symbols:
            try:
                data = await client.get_supply_chain(symbol)
            except Exception as exc:
                print(f"Supply chain failed for {symbol}: {exc}")
                data = {}

            relationships = data.get("data", [])
            if not relationships:
                await asyncio.sleep(_RATE_DELAY)
                continue

            rows = []
            for rel in relationships:
                related = rel.get("symbol2") or rel.get("relatedSymbol")
                if not related:
                    continue
                rows.append({
                    "symbol": symbol,
                    "related_symbol": related.upper(),
                    "relation_type": (rel.get("relationship") or "related").lower(),
                    "revenue_pct": float(rel["twoWayRevenue"]) if rel.get("twoWayRevenue") else None,
                    "source": "finnhub",
                    "payload": __import__("json").dumps(rel),
                    "updated_at": datetime.now(tz=UTC),
                })

            if rows:
                for batch in chunk(rows):
                    await conn.execute(text("""
                        INSERT INTO network.supply_chain
                          (symbol, related_symbol, relation_type, revenue_pct, source, payload, updated_at)
                        VALUES (:symbol, :related_symbol, :relation_type, :revenue_pct, :source,
                                CAST(:payload AS JSONB), :updated_at)
                        ON CONFLICT (symbol, related_symbol, relation_type, source)
                        DO UPDATE SET
                          revenue_pct = EXCLUDED.revenue_pct,
                          payload = EXCLUDED.payload,
                          updated_at = EXCLUDED.updated_at
                    """), batch)
                total += len(rows)

            await asyncio.sleep(_RATE_DELAY)

    await client.close()
    return IngestionSummary(records=total, details={"symbols": len(symbols)})
