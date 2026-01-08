"""Backfill market data for a long history window."""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.definitions.market_universe import MarketUniverse
from src.utils.ingestion.market import ingest_market_data

UTC = timezone.utc


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill market OHLCV history")
    parser.add_argument("--years", type=int, default=10, help="Years of history to backfill")
    parser.add_argument("--end-date", type=str, help="ISO end date (default: today)")
    parser.add_argument("--symbols", type=str, help="Comma separated list of tickers")
    parser.add_argument(
        "--high-priority",
        action="store_true",
        help="Use the high priority symbol subset",
    )
    return parser.parse_args()


def parse_date(value: Optional[str], default: date) -> date:
    if not value:
        return default
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_symbols(args: argparse.Namespace) -> List[str]:
    custom = parse_list(args.symbols)
    if custom:
        return custom
    if args.high_priority:
        return MarketUniverse.get_high_priority_symbols()
    return MarketUniverse.get_all_symbols()


def main(args: argparse.Namespace) -> None:
    setup_logging()
    today = date.today()
    end = parse_date(args.end_date, today)
    years = max(1, args.years)
    start = end - timedelta(days=years * 365)
    symbols = resolve_symbols(args)

    logging.info("Market backfill: %s symbols", len(symbols))
    logging.info("Window: %s -> %s", start.isoformat(), end.isoformat())

    started_at = datetime.now(tz=UTC)
    try:
        summary = asyncio.run(ingest_market_data(start=start, end=end, symbols=symbols))
    except Exception:
        logging.exception("Market backfill failed")
        raise
    duration = datetime.now(tz=UTC) - started_at

    logging.info("Market backfill completed in %s", duration)
    logging.info("Records: %s", summary.records)
    logging.info("Details: %s", summary.details)


if __name__ == "__main__":
    args = parse_args()
    main(args)
