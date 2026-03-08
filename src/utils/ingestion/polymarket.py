"""Polymarket data ingestion helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.schema import polymarket_markets, polymarket_snapshots
from src.db.session import engine
from src.utils.ingestion.clients import PolymarketGammaClient
from src.utils.ingestion.common import chunk
from src.utils.ingestion.types import IngestionSummary

UTC = timezone.utc


def _parse_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


async def ingest_polymarket_data(
    active_only: bool = True,
    limit: int = 200,
    max_pages: int = 50,
) -> IngestionSummary:
    client = PolymarketGammaClient()
    markets: List[Dict[str, Any]] = []
    try:
        offset = 0
        for _ in range(max_pages):
            batch = await client.get_markets(active=active_only, closed=not active_only, limit=limit, offset=offset)
            if not batch:
                break
            markets.extend(batch)
            if len(batch) < limit:
                break
            offset += limit
    finally:
        await client.close()

    snapshot_time = datetime.now(tz=UTC)
    market_rows: List[Dict[str, Any]] = []
    snapshot_rows: List[Dict[str, Any]] = []

    for market in markets:
        market_id = str(market.get("id") or "").strip()
        if not market_id:
            continue
        market_rows.append(
            {
                "market_id": market_id,
                "slug": market.get("slug"),
                "question": market.get("question"),
                "category": market.get("category"),
                "description": market.get("description"),
                "active": market.get("active"),
                "closed": market.get("closed"),
                "archived": market.get("archived"),
                "end_time": _parse_datetime(market.get("endDate") or market.get("endDateIso")),
                "created_time": _parse_datetime(market.get("createdAt")),
                "volume": _parse_decimal(market.get("volume")),
                "liquidity": _parse_decimal(market.get("liquidity")),
                "best_bid": _parse_decimal(market.get("bestBid")),
                "best_ask": _parse_decimal(market.get("bestAsk")),
                "condition_id": market.get("conditionId"),
                "clob_token_ids": market.get("clobTokenIds"),
                "payload": market,
                "updated_at": snapshot_time,
            }
        )
        snapshot_rows.append(
            {
                "market_id": market_id,
                "snapshot_time": snapshot_time,
                "active": market.get("active"),
                "closed": market.get("closed"),
                "volume": _parse_decimal(market.get("volume")),
                "liquidity": _parse_decimal(market.get("liquidity")),
                "best_bid": _parse_decimal(market.get("bestBid")),
                "best_ask": _parse_decimal(market.get("bestAsk")),
                "payload": market,
            }
        )

    if market_rows or snapshot_rows:
        async with engine.begin() as conn:
            if market_rows:
                for batch in chunk(market_rows):
                    stmt = pg_insert(polymarket_markets).values(batch)
                    await conn.execute(
                        stmt.on_conflict_do_update(
                            index_elements=[polymarket_markets.c.market_id],
                            set_={
                                "slug": stmt.excluded.slug,
                                "question": stmt.excluded.question,
                                "category": stmt.excluded.category,
                                "description": stmt.excluded.description,
                                "active": stmt.excluded.active,
                                "closed": stmt.excluded.closed,
                                "archived": stmt.excluded.archived,
                                "end_time": stmt.excluded.end_time,
                                "created_time": stmt.excluded.created_time,
                                "volume": stmt.excluded.volume,
                                "liquidity": stmt.excluded.liquidity,
                                "best_bid": stmt.excluded.best_bid,
                                "best_ask": stmt.excluded.best_ask,
                                "condition_id": stmt.excluded.condition_id,
                                "clob_token_ids": stmt.excluded.clob_token_ids,
                                "payload": stmt.excluded.payload,
                                "updated_at": stmt.excluded.updated_at,
                            },
                        )
                    )
            if snapshot_rows:
                for batch in chunk(snapshot_rows):
                    stmt = pg_insert(polymarket_snapshots).values(batch)
                    await conn.execute(
                        stmt.on_conflict_do_update(
                            index_elements=[polymarket_snapshots.c.market_id, polymarket_snapshots.c.snapshot_time],
                            set_={
                                "active": stmt.excluded.active,
                                "closed": stmt.excluded.closed,
                                "volume": stmt.excluded.volume,
                                "liquidity": stmt.excluded.liquidity,
                                "best_bid": stmt.excluded.best_bid,
                                "best_ask": stmt.excluded.best_ask,
                                "payload": stmt.excluded.payload,
                                "ingested_at": stmt.excluded.ingested_at,
                            },
                        )
                    )

    return IngestionSummary(
        records=len(snapshot_rows),
        details={
            "markets": len(market_rows),
            "snapshots": len(snapshot_rows),
            "active_only": active_only,
        },
    )
