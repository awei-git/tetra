"""Run all data pipelines for a date range."""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipelines.data.runner import run_all_pipelines

UTC = timezone.utc


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data pipelines")
    parser.add_argument("--start-date", type=str, help="ISO start date (default: yesterday)")
    parser.add_argument("--end-date", type=str, help="ISO end date (default: today)")
    parser.add_argument("--symbols", type=str, help="Comma separated list of tickers")
    parser.add_argument("--query", type=str, help="News query override")
    parser.add_argument("--series-ids", type=str, help="Comma separated FRED series IDs")
    parser.add_argument("--skip-market", action="store_true", help="Skip market ingestion")
    parser.add_argument("--skip-events", action="store_true", help="Skip event ingestion")
    parser.add_argument("--skip-economic", action="store_true", help="Skip economic ingestion")
    parser.add_argument("--skip-news", action="store_true", help="Skip news ingestion")
    return parser.parse_args()


def parse_date(value: Optional[str], default: date) -> date:
    if not value:
        return default
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    setup_logging()
    args = parse_args()
    today = date.today()
    start_default = today - timedelta(days=1)
    start = parse_date(args.start_date, start_default)
    end = parse_date(args.end_date, today)
    logging.info("Starting ingestion run")
    logging.info("Window: %s -> %s", start.isoformat(), end.isoformat())
    logging.info("Symbols: %s", args.symbols or "default")
    logging.info("Series: %s", args.series_ids or "default")
    logging.info("News query: %s", args.query or "default")
    logging.info(
        "Pipelines: market=%s events=%s economic=%s news=%s",
        "off" if args.skip_market else "on",
        "off" if args.skip_events else "on",
        "off" if args.skip_economic else "on",
        "off" if args.skip_news else "on",
    )

    started_at = datetime.now(tz=UTC)
    try:
        results = asyncio.run(
            run_all_pipelines(
                start=start,
                end=end,
                symbols=parse_list(args.symbols),
            query=args.query,
            series_ids=parse_list(args.series_ids),
            include_market=not args.skip_market,
            include_events=not args.skip_events,
            include_economic=not args.skip_economic,
            include_news=not args.skip_news,
        )
        )
    except Exception:
        logging.exception("Ingestion run failed")
        raise
    finished_at = datetime.now(tz=UTC)
    duration = finished_at - started_at
    logging.info("Ingestion completed in %s", duration)
    for result in results:
        logging.info(
            "Pipeline=%s records=%s details=%s",
            result.pipeline,
            result.records,
            result.details,
        )


if __name__ == "__main__":
    main()
