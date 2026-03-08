"""Run daily factor computation."""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, datetime, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipelines.factors.daily import run_daily_factors

UTC = timezone.utc


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily factor computation")
    parser.add_argument("--as-of", type=str, help="ISO date (YYYY-MM-DD)")
    return parser.parse_args()


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    setup_logging()
    args = parse_args()
    as_of = parse_date(args.as_of)
    started_at = datetime.now(tz=UTC)
    logging.info("Starting factor run for %s", as_of.isoformat() if as_of else "latest")
    try:
        result = asyncio.run(run_daily_factors(as_of=as_of))
    except Exception:
        logging.exception("Factor run failed")
        raise
    finished_at = datetime.now(tz=UTC)
    logging.info("Factor run completed in %s", finished_at - started_at)
    logging.info("Summary: %s", result)


if __name__ == "__main__":
    main()
