"""Intraday alert monitor — detect significant moves in portfolio positions.

Checks portfolio positions and key symbols for intraday returns exceeding
2 standard deviations (20-day historical volatility) or 3% absolute move.
Sends push notifications via Mira bridge.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from src.db.session import engine

logger = logging.getLogger(__name__)
UTC = timezone.utc

MIRA_DIR = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/MtJoy/Mira"
BRIDGE_OUTBOX = MIRA_DIR / "Mira-bridge" / "outbox"


async def _get_watchlist_symbols() -> List[str]:
    """Get symbols from portfolio positions + top debate recommendations."""
    symbols: set[str] = set()

    async with engine.begin() as conn:
        # Portfolio positions
        result = await conn.execute(
            text("SELECT symbol FROM portfolio.positions WHERE shares > 0")
        )
        for row in result.fetchall():
            symbols.add(row.symbol)

        # Recent open recommendations from debate
        result = await conn.execute(text("""
            SELECT DISTINCT symbol FROM tracker.recommendations
            WHERE status = 'open'
            ORDER BY symbol
            LIMIT 20
        """))
        for row in result.fetchall():
            symbols.add(row.symbol)

    return sorted(symbols)


async def _get_price_data(
    symbols: List[str], lookback_days: int = 25,
) -> Dict[str, Dict[str, Any]]:
    """Fetch recent OHLCV data for sigma calculation and current price.

    Returns dict keyed by symbol with:
      - closes: list of daily close prices (oldest first)
      - latest_close: most recent close (proxy for current price)
      - prev_close: previous trading day close
    """
    cutoff = datetime.now(tz=UTC) - timedelta(days=lookback_days + 10)
    data: Dict[str, Dict[str, Any]] = {}

    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT symbol, timestamp::date AS trade_date, close
            FROM market.ohlcv
            WHERE symbol = ANY(:symbols)
              AND timestamp >= :cutoff
            ORDER BY symbol, timestamp
        """), {"symbols": symbols, "cutoff": cutoff})

        rows = result.fetchall()

    # Group by symbol
    by_symbol: Dict[str, list] = {}
    for row in rows:
        by_symbol.setdefault(row.symbol, []).append(float(row.close))

    for sym, closes in by_symbol.items():
        if len(closes) < 3:
            continue
        data[sym] = {
            "closes": closes,
            "latest_close": closes[-1],
            "prev_close": closes[-2],
        }

    return data


def _compute_sigma(closes: List[float], window: int = 20) -> float:
    """Compute annualized daily return std dev over trailing window."""
    if len(closes) < window + 1:
        # Use whatever we have
        n = len(closes)
    else:
        n = window + 1
        closes = closes[-n:]

    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            returns.append(closes[i] / closes[i - 1] - 1)

    if len(returns) < 2:
        return 0.0

    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return variance ** 0.5


async def check_alerts(
    symbols: Optional[List[str]] = None,
    sigma_threshold: float = 2.0,
    abs_threshold: float = 0.03,
) -> List[Dict[str, Any]]:
    """Check for significant intraday moves.

    Args:
        symbols: Symbols to check. If None, uses portfolio + debate recommendations.
        sigma_threshold: Z-score threshold for alerts (default 2.0).
        abs_threshold: Absolute return threshold (default 3%).

    Returns:
        List of alert dicts with symbol, prices, change_pct, sigma, z_score, alert_level.
    """
    if symbols is None:
        symbols = await _get_watchlist_symbols()

    if not symbols:
        logger.info("No symbols to monitor")
        return []

    logger.info(f"Checking alerts for {len(symbols)} symbols: {symbols}")

    price_data = await _get_price_data(symbols)
    alerts: List[Dict[str, Any]] = []

    for sym in symbols:
        pd = price_data.get(sym)
        if pd is None:
            continue

        current = pd["latest_close"]
        prev = pd["prev_close"]
        if prev == 0:
            continue

        change_pct = current / prev - 1
        sigma = _compute_sigma(pd["closes"])

        if sigma > 0:
            z_score = abs(change_pct) / sigma
        else:
            z_score = 0.0

        # Alert if |return| > sigma_threshold * sigma OR |return| > abs_threshold
        if z_score >= sigma_threshold or abs(change_pct) >= abs_threshold:
            if z_score >= 3.0:
                alert_level = "critical"
            else:
                alert_level = "warning"

            alerts.append({
                "symbol": sym,
                "current_price": round(current, 2),
                "prev_close": round(prev, 2),
                "change_pct": round(change_pct, 4),
                "sigma": round(sigma, 4),
                "z_score": round(z_score, 2),
                "alert_level": alert_level,
            })

    alerts.sort(key=lambda a: -abs(a["z_score"]))
    logger.info(f"Found {len(alerts)} alerts")
    return alerts


def send_alerts(alerts: List[Dict[str, Any]]) -> Optional[str]:
    """Send alert notifications via Mira bridge.

    Uses the same bridge outbox message format as src/mira/push.py.

    Returns:
        Path to written message file, or None if no alerts.
    """
    if not alerts:
        return None

    # Build message content
    lines = []
    for a in alerts:
        prefix = "🔴" if a["alert_level"] == "critical" else "⚠️"
        sign = "+" if a["change_pct"] >= 0 else ""
        lines.append(
            f"{prefix} {a['symbol']} {sign}{a['change_pct']:.1%} "
            f"({a['z_score']:.1f}σ) | ${a['current_price']}"
        )

    header = "TETRA ALERT"
    content = f"{header}\n" + "\n".join(lines)

    # Write bridge outbox message (same format as push.py)
    try:
        BRIDGE_OUTBOX.mkdir(parents=True, exist_ok=True)
        msg_id = uuid.uuid4().hex[:8]
        ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"agent_{ts}_{msg_id}.json"

        message = {
            "id": msg_id,
            "sender": "agent",
            "timestamp": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "type": "text",
            "content": content,
            "thread_id": "",
        }

        msg_path = BRIDGE_OUTBOX / filename
        msg_path.write_text(json.dumps(message, indent=2), encoding="utf-8")
        logger.info(f"Alert bridge message written: {msg_path.name}")
        return str(msg_path)
    except Exception as e:
        logger.error(f"Failed to write alert bridge message: {e}")
        return None
