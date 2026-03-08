"""Run GPT challenge pass to critique prior recommendations."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.gpt.challenge import run_gpt_challenge

UTC = timezone.utc


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Run GPT challenge pass")
    parser.add_argument("--session", choices=["pre", "post"], help="Force pre/post session")
    args = parser.parse_args()
    started_at = datetime.now(tz=UTC)
    logging.info("Starting GPT challenge (%s)", args.session or "auto")
    try:
        result = asyncio.run(run_gpt_challenge(session=args.session))
    except Exception:
        logging.exception("GPT challenge failed")
        raise
    finished_at = datetime.now(tz=UTC)
    logging.info("GPT challenge completed in %s", finished_at - started_at)
    logging.info("Summary: %s", result)


if __name__ == "__main__":
    main()
