"""Lightweight daily scheduler for data ingestion."""

from __future__ import annotations

import argparse
import asyncio
from datetime import date, datetime, timedelta, timezone

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipelines.data.runner import run_all_pipelines

UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schedule daily data ingestion")
    parser.add_argument("--hour", type=int, default=2, help="Hour to run (0-23, default: 2)")
    parser.add_argument("--minute", type=int, default=0, help="Minute to run (0-59, default: 0)")
    parser.add_argument("--once", action="store_true", help="Run once immediately")
    return parser.parse_args()


def next_run(now: datetime, hour: int, minute: int) -> datetime:
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


async def run_once() -> None:
    today = datetime.now(tz=UTC).date()
    start = today - timedelta(days=1)
    end = today
    await run_all_pipelines(start=start, end=end)


async def run_schedule(hour: int, minute: int) -> None:
    while True:
        now = datetime.now(tz=UTC)
        target = next_run(now, hour, minute)
        wait_seconds = max(0, (target - now).total_seconds())
        print(f"Next ingest scheduled at {target.isoformat()}")
        await asyncio.sleep(wait_seconds)
        try:
            await run_once()
        except Exception as exc:
            print(f"Ingestion run failed: {exc}")


def main() -> None:
    args = parse_args()
    if args.once:
        asyncio.run(run_once())
        return
    asyncio.run(run_schedule(args.hour, args.minute))


if __name__ == "__main__":
    main()
