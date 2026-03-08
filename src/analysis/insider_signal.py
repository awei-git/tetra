"""Insider Trading ML Signal — Pattern Detection from Form 4 Filings.

Based on SSRN Dec 2025: "ML-Enhanced Insider Trading Signals"
Key features that predict alpha:
1. Cluster buying: multiple insiders buying within 2 weeks
2. Pattern breaks: insider who always sells suddenly buys
3. Contrarian signals: insider buys during stock decline
4. Value-weighted: weight by transaction size relative to holdings

Instead of XGBoost (which needs training data we don't have yet),
we use a rule-based scoring system that captures the same signals.
Can upgrade to ML once we have enough historical data.

Pipeline:
1. Fetch recent insider trades from DB
2. Compute features: cluster, pattern break, contrarian, size
3. Score each symbol
4. Store in signals.informed_trading
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from src.db.session import engine

logger = logging.getLogger(__name__)
UTC = timezone.utc


async def _fetch_insider_trades(
    as_of: date, lookback_days: int = 90,
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch insider trades grouped by symbol."""
    query = text("""
        SELECT symbol, filing_date, transaction_date, insider_name,
               transaction_type, shares, price, value, shares_after
        FROM event.insider_trades
        WHERE filing_date >= :start AND filing_date <= :end
        ORDER BY symbol, filing_date DESC
    """)
    start = as_of - timedelta(days=lookback_days)
    async with engine.begin() as conn:
        result = await conn.execute(query, {"start": start, "end": as_of})
        rows = result.fetchall()

    trades: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        trades[r.symbol].append({
            "filing_date": r.filing_date,
            "transaction_date": r.transaction_date,
            "insider_name": r.insider_name,
            "transaction_type": r.transaction_type or "",
            "shares": float(r.shares) if r.shares else 0,
            "price": float(r.price) if r.price else 0,
            "value": float(r.value) if r.value else 0,
            "shares_after": float(r.shares_after) if r.shares_after else 0,
        })
    return dict(trades)


async def _fetch_recent_returns(
    as_of: date, symbols: List[str], lookback_days: int = 30,
) -> Dict[str, float]:
    """Fetch recent returns for contrarian detection."""
    if not symbols:
        return {}
    query = text("""
        WITH ranked AS (
            SELECT symbol,
                   close,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS rn_latest,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp ASC) AS rn_oldest
            FROM market.ohlcv
            WHERE symbol = ANY(:symbols)
              AND timestamp::date >= :start
              AND timestamp::date <= :end
        )
        SELECT
            latest.symbol,
            (latest.close / NULLIF(oldest.close, 0)) - 1 AS period_return
        FROM (SELECT symbol, close FROM ranked WHERE rn_latest = 1) latest
        JOIN (SELECT symbol, close FROM ranked WHERE rn_oldest = 1) oldest
            ON latest.symbol = oldest.symbol
    """)
    start = as_of - timedelta(days=lookback_days)
    async with engine.begin() as conn:
        result = await conn.execute(query, {
            "symbols": symbols, "start": start, "end": as_of,
        })
        rows = result.fetchall()
    return {r.symbol: float(r.period_return) for r in rows if r.period_return is not None}


def _score_insider_signal(
    trades: List[Dict[str, Any]],
    recent_return: Optional[float],
    as_of: date,
) -> Dict[str, Any]:
    """Score insider trading activity for a single symbol.

    Returns signal dict with score components.
    """
    if not trades:
        return {"score": 0, "components": {}}

    recent_window = as_of - timedelta(days=14)
    recent_trades = [t for t in trades if t["filing_date"] >= recent_window]

    # Classify trades
    buys = [t for t in recent_trades if "P" in t["transaction_type"].upper()
            or "purchase" in t["transaction_type"].lower()]
    sells = [t for t in recent_trades if "S" in t["transaction_type"].upper()
             and "P" not in t["transaction_type"].upper()]

    # 1. Cluster buying: multiple unique insiders buying within 2 weeks
    unique_buyers = len(set(t["insider_name"] for t in buys if t["insider_name"]))
    cluster_score = min(unique_buyers / 3.0, 1.0)  # 3+ insiders = max

    # 2. Pattern break: check if recent activity differs from historical pattern
    all_buys = [t for t in trades if "P" in t["transaction_type"].upper()
                or "purchase" in t["transaction_type"].lower()]
    all_sells = [t for t in trades if "S" in t["transaction_type"].upper()
                 and "P" not in t["transaction_type"].upper()]
    historical_buy_ratio = len(all_buys) / max(len(all_buys) + len(all_sells), 1)
    recent_buy_ratio = len(buys) / max(len(buys) + len(sells), 1)
    pattern_break = abs(recent_buy_ratio - historical_buy_ratio)

    # 3. Contrarian: insiders buying during stock decline
    contrarian_score = 0.0
    if recent_return is not None and buys:
        if recent_return < -0.05:  # Stock down >5%
            contrarian_score = min(abs(recent_return) * 5, 1.0)
        elif recent_return > 0.15 and sells:  # Selling after big run-up (bearish)
            contrarian_score = -min(recent_return * 2, 1.0)

    # 4. Size score: total value of purchases
    buy_value = sum(t["value"] for t in buys if t["value"] > 0)
    sell_value = sum(abs(t["value"]) for t in sells if t["value"])
    net_value = buy_value - sell_value
    # Normalize: $1M+ is significant
    size_score = min(net_value / 1_000_000, 1.0) if net_value > 0 else max(net_value / 1_000_000, -1.0)

    # Composite score (-1 to +1)
    weights = {"cluster": 0.3, "pattern_break": 0.2, "contrarian": 0.3, "size": 0.2}
    direction = 1.0 if len(buys) >= len(sells) else -1.0

    raw_score = (
        weights["cluster"] * cluster_score * direction
        + weights["pattern_break"] * pattern_break * direction
        + weights["contrarian"] * contrarian_score
        + weights["size"] * size_score
    )
    score = max(-1.0, min(1.0, raw_score))

    return {
        "score": round(score, 4),
        "components": {
            "cluster_buyers": unique_buyers,
            "cluster_score": round(cluster_score, 3),
            "pattern_break": round(pattern_break, 3),
            "contrarian_score": round(contrarian_score, 3),
            "size_score": round(size_score, 3),
            "buy_value": round(buy_value, 0),
            "sell_value": round(sell_value, 0),
            "recent_buys": len(buys),
            "recent_sells": len(sells),
            "total_trades_90d": len(trades),
        },
    }


async def _store_insider_signals(
    as_of: date, signals: List[Dict[str, Any]],
) -> None:
    """Store insider signals in signals.informed_trading."""
    if not signals:
        return
    query = text("""
        INSERT INTO signals.informed_trading
          (symbol, date, signal_type, strength, context, raw_data)
        VALUES (:symbol, :date, :signal_type, :strength, :context,
                CAST(:raw_data AS JSONB))
        ON CONFLICT (symbol, date, signal_type)
        DO UPDATE SET
          strength = EXCLUDED.strength,
          context = EXCLUDED.context,
          raw_data = EXCLUDED.raw_data
    """)
    async with engine.begin() as conn:
        for sig in signals:
            await conn.execute(query, sig)


async def run_insider_signal(
    as_of: Optional[date] = None,
) -> Dict[str, Any]:
    """Run insider trading signal pipeline.

    1. Fetch trades from DB (already ingested)
    2. Score each symbol
    3. Store significant signals
    """
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()

    logger.info(f"Running insider trading signal for {as_of}")

    trades_by_symbol = await _fetch_insider_trades(as_of, lookback_days=90)
    if not trades_by_symbol:
        logger.warning("No insider trades found")
        return {"as_of": as_of.isoformat(), "status": "no_data"}

    symbols = list(trades_by_symbol.keys())
    returns = await _fetch_recent_returns(as_of, symbols)

    signals = []
    all_scores = []

    for symbol, trades in trades_by_symbol.items():
        result = _score_insider_signal(trades, returns.get(symbol), as_of)
        score = result["score"]
        all_scores.append({"symbol": symbol, "score": score})

        # Only store significant signals (|score| > 0.15)
        if abs(score) > 0.15:
            signal_type = "insider_cluster" if result["components"]["cluster_buyers"] >= 2 else "insider_activity"
            if result["components"]["contrarian_score"] > 0.3:
                signal_type = "insider_contrarian"
            if result["components"]["pattern_break"] > 0.5:
                signal_type = "insider_pattern_break"

            signals.append({
                "symbol": symbol,
                "date": as_of,
                "signal_type": signal_type,
                "strength": score,
                "context": json.dumps({
                    "type": signal_type,
                    **result["components"],
                }),
                "raw_data": json.dumps(result),
            })

    await _store_insider_signals(as_of, signals)

    # Sort by absolute score
    all_scores.sort(key=lambda x: abs(x["score"]), reverse=True)

    logger.info(
        f"Insider signal: {len(trades_by_symbol)} symbols scored, "
        f"{len(signals)} significant signals"
    )

    return {
        "as_of": as_of.isoformat(),
        "status": "success",
        "symbols_scored": len(trades_by_symbol),
        "significant_signals": len(signals),
        "top_signals": [
            {"symbol": s["symbol"], "score": s["score"]}
            for s in all_scores[:10]
        ],
    }
