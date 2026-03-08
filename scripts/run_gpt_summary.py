"""Generate GPT summary for consolidated recommendations."""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.api.app import get_gpt_summary

UTC = timezone.utc


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def main() -> None:
    setup_logging()
    started_at = datetime.now(tz=UTC)
    logging.info("Starting GPT summary")
    try:
        result = asyncio.run(get_gpt_summary(min_factors=3, signal_threshold=0.1))
    except Exception:
        logging.exception("GPT summary failed")
        raise
    finished_at = datetime.now(tz=UTC)
    logging.info("GPT summary completed in %s", finished_at - started_at)
    logging.info("Summary: %s", result)


if __name__ == "__main__":
    main()
