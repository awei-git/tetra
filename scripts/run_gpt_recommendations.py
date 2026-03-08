"""Run GPT recommendations and store them in the database."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.gpt.recommendations import run_gpt_recommendations

UTC = timezone.utc


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Run GPT recommendations")
    parser.add_argument("--session", choices=["pre", "post"], help="Force pre/post session")
    args = parser.parse_args()
    started_at = datetime.now(tz=UTC)
    logging.info("Starting GPT recommendations (%s)", args.session or "auto")
    try:
        result = asyncio.run(run_gpt_recommendations(session=args.session))
    except Exception:
        logging.exception("GPT recommendations failed")
        raise
    finished_at = datetime.now(tz=UTC)
    logging.info("GPT recommendations completed in %s", finished_at - started_at)
    logging.info("Summary: %s", result)


if __name__ == "__main__":
    main()
