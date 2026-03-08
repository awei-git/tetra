"""Ingest Phase 2 data: insider trades, analyst recommendations, peer networks.

Usage:
  python scripts/run_phase2_ingest.py
  python scripts/run_phase2_ingest.py --skip-supply-chain  # supply chain API may be premium
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

UTC = timezone.utc


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Phase 2 data")
    parser.add_argument("--skip-insider", action="store_true")
    parser.add_argument("--skip-analyst", action="store_true")
    parser.add_argument("--skip-peers", action="store_true")
    parser.add_argument("--skip-supply-chain", action="store_true")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    from src.utils.ingestion.insider_analyst import (
        ingest_insider_trades,
        ingest_analyst_recommendations,
        ingest_peer_network,
        ingest_supply_chain,
    )

    if not args.skip_insider:
        logging.info("=== Ingesting insider trades ===")
        result = await ingest_insider_trades()
        logging.info(f"Insider trades: {result.records} records, {result.details}")

    if not args.skip_analyst:
        logging.info("=== Ingesting analyst recommendations ===")
        result = await ingest_analyst_recommendations()
        logging.info(f"Analyst recs: {result.records} records, {result.details}")

    if not args.skip_peers:
        logging.info("=== Ingesting peer network ===")
        result = await ingest_peer_network()
        logging.info(f"Peer network: {result.records} records, {result.details}")

    if not args.skip_supply_chain:
        logging.info("=== Ingesting supply chain ===")
        result = await ingest_supply_chain()
        logging.info(f"Supply chain: {result.records} records, {result.details}")


def main() -> None:
    setup_logging()
    args = parse_args()
    started = datetime.now(tz=UTC)
    logging.info("Starting Phase 2 data ingestion")

    try:
        asyncio.run(run(args))
    except Exception:
        logging.exception("Phase 2 ingestion failed")
        raise

    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    logging.info(f"Phase 2 ingestion complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
