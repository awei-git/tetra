"""Analyst Network Alpha — Co-Coverage Momentum Propagation.

Based on arxiv:2410.20597: "Graph Attention on Analyst Co-Coverage Networks"
Key insight: Stocks covered by the same analysts tend to co-move. When one
stock in a co-coverage cluster gets upgraded, the others often follow.

We use analyst recommendation changes + peer network to propagate signals:
1. Detect recommendation shifts (month-over-month changes)
2. Propagate through co-coverage network (peer relationships)
3. Weight by network distance and shift magnitude

Sharpe improvement: 0.35 → 4.06 in the paper (with full graph attention).
Our simplified version captures the core signal without GNN complexity.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from src.db.session import engine

logger = logging.getLogger(__name__)
UTC = timezone.utc


async def _fetch_recommendation_shifts(
    as_of: date,
) -> Dict[str, Dict[str, Any]]:
    """Fetch analyst recommendation changes (current vs previous period).

    Returns {symbol: {current_consensus, previous_consensus, shift, magnitude}}
    """
    query = text("""
        WITH ranked AS (
            SELECT symbol, period, strong_buy, buy, hold, sell, strong_sell,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY period DESC) AS rn
            FROM event.analyst_recommendations
            WHERE period <= :as_of
        )
        SELECT * FROM ranked WHERE rn <= 2
        ORDER BY symbol, rn
    """)
    async with engine.begin() as conn:
        result = await conn.execute(query, {"as_of": as_of})
        rows = result.fetchall()

    # Group by symbol
    by_symbol: Dict[str, List] = defaultdict(list)
    for r in rows:
        by_symbol[r.symbol].append(r)

    shifts = {}
    for symbol, recs in by_symbol.items():
        if len(recs) < 1:
            continue

        def consensus_score(r):
            """Weighted consensus: -2 (strong sell) to +2 (strong buy)."""
            total = (r.strong_buy or 0) + (r.buy or 0) + (r.hold or 0) + (r.sell or 0) + (r.strong_sell or 0)
            if total == 0:
                return 0.0
            return (
                2 * (r.strong_buy or 0) + 1 * (r.buy or 0)
                - 1 * (r.sell or 0) - 2 * (r.strong_sell or 0)
            ) / total

        current = consensus_score(recs[0])
        previous = consensus_score(recs[1]) if len(recs) > 1 else current
        shift = current - previous

        shifts[symbol] = {
            "current_consensus": round(current, 3),
            "previous_consensus": round(previous, 3),
            "shift": round(shift, 3),
            "total_analysts": sum([
                recs[0].strong_buy or 0, recs[0].buy or 0,
                recs[0].hold or 0, recs[0].sell or 0,
                recs[0].strong_sell or 0,
            ]),
            "distribution": {
                "strong_buy": recs[0].strong_buy or 0,
                "buy": recs[0].buy or 0,
                "hold": recs[0].hold or 0,
                "sell": recs[0].sell or 0,
                "strong_sell": recs[0].strong_sell or 0,
            },
        }

    return shifts


async def _fetch_peer_network() -> Dict[str, List[str]]:
    """Fetch peer/co-coverage network."""
    peers: Dict[str, List[str]] = defaultdict(list)
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT symbol, peer_symbol FROM network.analyst_coverage
        """))
        for r in result.fetchall():
            peers[r.symbol].append(r.peer_symbol)
    return dict(peers)


def _propagate_analyst_signal(
    shifts: Dict[str, Dict[str, Any]],
    peers: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Propagate recommendation shifts through peer network.

    Key rules:
    - Direct shift > 0.3: strong signal
    - Peer average shift: weaker propagated signal
    - Contrarian: if most peers upgrade but one doesn't → opportunity
    """
    signals = []

    for symbol, data in shifts.items():
        direct_shift = data["shift"]

        # Direct signal (own recommendation change)
        if abs(direct_shift) > 0.1:
            signals.append({
                "symbol": symbol,
                "signal_type": "analyst_shift",
                "strength": round(direct_shift / 2.0, 4),  # Normalize to [-1, 1]
                "context": {
                    "type": "direct_shift",
                    "consensus": data["current_consensus"],
                    "previous": data["previous_consensus"],
                    "shift": direct_shift,
                    "analysts": data["total_analysts"],
                },
            })

        # Peer propagation
        if symbol in peers:
            peer_shifts = []
            for peer in peers[symbol]:
                if peer in shifts:
                    peer_shifts.append(shifts[peer]["shift"])

            if peer_shifts:
                avg_peer_shift = sum(peer_shifts) / len(peer_shifts)

                # Peer momentum: propagate average shift (discounted)
                if abs(avg_peer_shift) > 0.15:
                    signals.append({
                        "symbol": symbol,
                        "signal_type": "analyst_peer_momentum",
                        "strength": round(avg_peer_shift * 0.4 / 2.0, 4),
                        "context": {
                            "type": "peer_momentum",
                            "avg_peer_shift": round(avg_peer_shift, 3),
                            "peer_count": len(peer_shifts),
                            "own_shift": direct_shift,
                        },
                    })

                # Contrarian: peers upgrading but this stock hasn't moved
                if avg_peer_shift > 0.3 and direct_shift < 0.05:
                    signals.append({
                        "symbol": symbol,
                        "signal_type": "analyst_contrarian",
                        "strength": round(avg_peer_shift * 0.3 / 2.0, 4),
                        "context": {
                            "type": "contrarian_laggard",
                            "avg_peer_shift": round(avg_peer_shift, 3),
                            "own_shift": direct_shift,
                            "peers_upgrading": sum(1 for s in peer_shifts if s > 0.1),
                        },
                    })

    return signals


async def _store_analyst_signals(
    as_of: date, signals: List[Dict[str, Any]],
) -> None:
    """Store analyst network signals."""
    if not signals:
        return
    async with engine.begin() as conn:
        for sig in signals:
            await conn.execute(text("""
                INSERT INTO signals.informed_trading
                  (symbol, date, signal_type, strength, context, raw_data)
                VALUES (:symbol, :date, :signal_type, :strength, :context,
                        CAST(:raw_data AS JSONB))
                ON CONFLICT (symbol, date, signal_type)
                DO UPDATE SET
                  strength = EXCLUDED.strength,
                  context = EXCLUDED.context,
                  raw_data = EXCLUDED.raw_data
            """), {
                "symbol": sig["symbol"],
                "date": as_of,
                "signal_type": sig["signal_type"],
                "strength": sig["strength"],
                "context": json.dumps(sig["context"]),
                "raw_data": json.dumps(sig),
            })


async def run_analyst_network(
    as_of: Optional[date] = None,
) -> Dict[str, Any]:
    """Run analyst network alpha pipeline.

    1. Fetch recommendation shifts
    2. Fetch peer network
    3. Propagate signals
    4. Store results
    """
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()

    logger.info(f"Running analyst network analysis for {as_of}")

    shifts = await _fetch_recommendation_shifts(as_of)
    peers = await _fetch_peer_network()

    if not shifts:
        logger.warning("No analyst recommendation data")
        return {"as_of": as_of.isoformat(), "status": "no_data"}

    signals = _propagate_analyst_signal(shifts, peers)

    # Filter significant
    significant = [s for s in signals if abs(s["strength"]) > 0.03]
    significant.sort(key=lambda x: abs(x["strength"]), reverse=True)

    await _store_analyst_signals(as_of, significant)

    logger.info(
        f"Analyst network: {len(shifts)} symbols with recs, "
        f"{len(peers)} peer groups, {len(significant)} signals"
    )

    return {
        "as_of": as_of.isoformat(),
        "status": "success",
        "symbols_with_recs": len(shifts),
        "peer_groups": len(peers),
        "total_signals": len(significant),
        "signal_breakdown": {
            "analyst_shift": sum(1 for s in significant if s["signal_type"] == "analyst_shift"),
            "peer_momentum": sum(1 for s in significant if s["signal_type"] == "analyst_peer_momentum"),
            "contrarian": sum(1 for s in significant if s["signal_type"] == "analyst_contrarian"),
        },
        "top_signals": [
            {
                "symbol": s["symbol"],
                "type": s["signal_type"],
                "strength": s["strength"],
            }
            for s in significant[:10]
        ],
    }
