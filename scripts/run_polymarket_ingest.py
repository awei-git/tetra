"""Run Polymarket ingestion."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.ingestion.polymarket import ingest_polymarket_data

UTC = timezone.utc


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Polymarket ingestion")
    parser.add_argument("--include-inactive", action="store_true", help="Include inactive/closed markets")
    parser.add_argument("--limit", type=int, default=200, help="Page size (default: 200)")
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages to fetch (default: 50)")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    active_only = not args.include_inactive
    started_at = datetime.now(tz=UTC)
    logging.info("Starting Polymarket ingest (active_only=%s)", active_only)
    logging.info("Pagination: limit=%s max_pages=%s", args.limit, args.max_pages)
    try:
        result = asyncio.run(
            ingest_polymarket_data(
                active_only=active_only,
                limit=args.limit,
                max_pages=args.max_pages,
            )
        )
    except Exception:
        logging.exception("Polymarket ingest failed")
        raise
    finished_at = datetime.now(tz=UTC)
    logging.info("Polymarket ingest completed in %s", finished_at - started_at)
    logging.info("Summary: %s", result)


if __name__ == "__main__":
    main()
