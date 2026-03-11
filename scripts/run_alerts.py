"""Run intraday alert check.

Usage:
  python scripts/run_alerts.py                       # check portfolio + debate recs
  python scripts/run_alerts.py --symbols AAPL META    # check specific symbols
  python scripts/run_alerts.py --threshold 1.5        # lower sigma threshold
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.append(str(Path(__file__).resolve().parents[1]))


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def is_market_hours() -> bool:
    """Check if US equity market is currently open (weekday 9:30-16:00 ET)."""
    et = ZoneInfo("America/New_York")
    now = datetime.now(et)

    # Weekend check (Monday=0, Sunday=6)
    if now.weekday() >= 5:
        return False

    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now <= market_close


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tetra intraday alert monitor")
    parser.add_argument(
        "--symbols", nargs="+", type=str, default=None,
        help="Symbols to check (default: portfolio + debate recs)",
    )
    parser.add_argument(
        "--threshold", type=float, default=2.0,
        help="Sigma threshold for alerts (default: 2.0)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Run even outside market hours",
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    from src.alerts.monitor import check_alerts, send_alerts

    alerts = await check_alerts(
        symbols=args.symbols,
        sigma_threshold=args.threshold,
    )

    if alerts:
        print(f"\n{'='*60}")
        print(f"TETRA ALERTS — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*60}")
        for a in alerts:
            level_tag = "CRITICAL" if a["alert_level"] == "critical" else "WARNING "
            sign = "+" if a["change_pct"] >= 0 else ""
            print(
                f"  [{level_tag}] {a['symbol']:8s} "
                f"{sign}{a['change_pct']:.2%}  "
                f"({a['z_score']:.1f}σ)  "
                f"${a['current_price']:>10,.2f}  "
                f"prev ${a['prev_close']:>10,.2f}"
            )
        print(f"{'='*60}\n")

        msg_path = send_alerts(alerts)
        if msg_path:
            print(f"Push notification sent: {msg_path}")
    else:
        print("No alerts triggered.")


def main() -> None:
    setup_logging()
    args = parse_args()

    if not args.force and not is_market_hours():
        logging.info("Outside market hours — skipping alert check (use --force to override)")
        return

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
