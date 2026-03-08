"""Run GPT factor review for the latest factor picks."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.gpt.factor_review import run_gpt_factor_reviews

UTC = timezone.utc


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Run GPT factor review")
    parser.add_argument("--session", choices=["pre", "post"], help="Force pre/post session")
    parser.add_argument("--as-of", type=str, help="Factor as-of date (YYYY-MM-DD)")
    args = parser.parse_args()
    started_at = datetime.now(tz=UTC)
    logging.info("Starting GPT factor review (%s)", args.session or "auto")
    try:
        result = asyncio.run(run_gpt_factor_reviews(session=args.session, as_of=_parse_date(args.as_of)))
    except Exception:
        logging.exception("GPT factor review failed")
        raise
    finished_at = datetime.now(tz=UTC)
    logging.info("GPT factor review completed in %s", finished_at - started_at)
    logging.info("Summary: %s", result)


if __name__ == "__main__":
    main()
