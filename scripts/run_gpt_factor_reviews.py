"""Run GPT factor review for the latest factor picks."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.gpt.factor_review import run_gpt_factor_reviews


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT factor review")
    parser.add_argument("--session", choices=["pre", "post"], help="Force pre/post session")
    parser.add_argument("--as-of", type=str, help="Factor as-of date (YYYY-MM-DD)")
    args = parser.parse_args()
    asyncio.run(run_gpt_factor_reviews(session=args.session, as_of=_parse_date(args.as_of)))


if __name__ == "__main__":
    main()
