"""Run inference pipeline (signals, events, polymarket)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.inference import run_all_inference

UTC = timezone.utc


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference pipeline")
    parser.add_argument("--as-of", type=str, help="Use a specific as-of date (YYYY-MM-DD)")
    return parser.parse_args()


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    setup_logging()
    args = parse_args()
    as_of = _parse_date(args.as_of)
    started_at = datetime.now(tz=UTC)
    logging.info("Starting inference run (as_of=%s)", as_of or "latest")
    try:
        result = asyncio.run(run_all_inference(as_of=as_of))
    except Exception:
        logging.exception("Inference run failed")
        raise
    finished_at = datetime.now(tz=UTC)
    logging.info("Inference completed in %s", finished_at - started_at)
    logging.info("Summary: %s", result)


if __name__ == "__main__":
    main()
