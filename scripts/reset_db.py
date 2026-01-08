"""Reset database tables for a fresh ingest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.config import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset Tetra database tables")
    parser.add_argument(
        "--market-only",
        action="store_true",
        help="Only truncate market assets and OHLCV tables",
    )
    return parser.parse_args()


def build_tables(market_only: bool) -> List[str]:
    if market_only:
        return ["market.ohlcv", "market.assets"]
    return [
        "news.articles",
        "event.events",
        "economic.values",
        "economic.series",
        "market.ohlcv",
        "market.assets",
    ]


def main() -> None:
    args = parse_args()
    tables = build_tables(args.market_only)
    engine = create_engine(settings.sync_database_url, future=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE "
                + ", ".join(tables)
                + " RESTART IDENTITY"
            )
        )


if __name__ == "__main__":
    main()
