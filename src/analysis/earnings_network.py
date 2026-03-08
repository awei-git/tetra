"""Earnings Cascade & Supply Chain Network Signal.

Based on FactSet 2025: "Centrality-Weighted Customer Momentum"
Key insight: When a major customer reports strong earnings, its suppliers
tend to outperform. The effect is stronger for:
1. Higher revenue concentration (supplier depends heavily on customer)
2. Higher network centrality (customer is central to the network)
3. Positive earnings surprise (beat vs miss)

Pipeline:
1. Build company relationship graph from DB (peers + supply chain)
2. Identify recent earnings events and their surprise direction
3. Propagate signal through the network (weighted by centrality)
4. Store cascade signals in network.earnings_cascade
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import text

from src.db.session import engine

logger = logging.getLogger(__name__)
UTC = timezone.utc


async def _fetch_company_graph(
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[str]]]:
    """Fetch company relationships: supply chain + peers.

    Returns:
        supply_chain: {symbol: [{related_symbol, relation_type, revenue_pct}]}
        peers: {symbol: [peer_symbols]}
    """
    supply_chain: Dict[str, List[Dict]] = defaultdict(list)
    peers: Dict[str, List[str]] = defaultdict(list)

    async with engine.begin() as conn:
        # Supply chain
        result = await conn.execute(text("""
            SELECT symbol, related_symbol, relation_type, revenue_pct
            FROM network.supply_chain
        """))
        for r in result.fetchall():
            supply_chain[r.symbol].append({
                "related_symbol": r.related_symbol,
                "relation_type": r.relation_type,
                "revenue_pct": float(r.revenue_pct) if r.revenue_pct else 0.1,
            })

        # Peer network
        result = await conn.execute(text("""
            SELECT symbol, peer_symbol FROM network.analyst_coverage
        """))
        for r in result.fetchall():
            peers[r.symbol].append(r.peer_symbol)

        # Also use company_graph if populated
        result = await conn.execute(text("""
            SELECT symbol, related_symbol, relation_type, strength
            FROM network.company_graph
        """))
        for r in result.fetchall():
            rel_type = r.relation_type
            if rel_type in ("supplier", "customer"):
                supply_chain[r.symbol].append({
                    "related_symbol": r.related_symbol,
                    "relation_type": rel_type,
                    "revenue_pct": float(r.strength) if r.strength else 0.1,
                })
            elif rel_type in ("peer", "competitor"):
                peers[r.symbol].append(r.related_symbol)

    return dict(supply_chain), dict(peers)


async def _fetch_recent_earnings(
    as_of: date, lookback_days: int = 30,
) -> List[Dict[str, Any]]:
    """Fetch recent earnings events with surprise direction."""
    query = text("""
        SELECT symbol, event_time, payload
        FROM event.events
        WHERE event_type = 'earnings'
          AND event_time::date >= :start
          AND event_time::date <= :end
          AND payload IS NOT NULL
        ORDER BY event_time DESC
    """)
    start = as_of - timedelta(days=lookback_days)
    async with engine.begin() as conn:
        result = await conn.execute(query, {"start": start, "end": as_of})
        rows = result.fetchall()

    earnings = []
    for r in rows:
        payload = r.payload if isinstance(r.payload, dict) else {}
        # Extract surprise from various payload formats
        eps_actual = payload.get("epsActual") or payload.get("reportedEPS")
        eps_estimate = payload.get("epsEstimate") or payload.get("estimatedEPS")

        surprise = None
        if eps_actual is not None and eps_estimate is not None:
            try:
                actual = float(eps_actual)
                estimate = float(eps_estimate)
                if estimate != 0:
                    surprise = (actual - estimate) / abs(estimate)
                elif actual > 0:
                    surprise = 1.0
                elif actual < 0:
                    surprise = -1.0
            except (ValueError, TypeError):
                pass

        rev_actual = payload.get("revenueActual")
        rev_estimate = payload.get("revenueEstimate")
        rev_surprise = None
        if rev_actual is not None and rev_estimate is not None:
            try:
                ra = float(rev_actual)
                re = float(rev_estimate)
                if re != 0:
                    rev_surprise = (ra - re) / abs(re)
            except (ValueError, TypeError):
                pass

        earnings.append({
            "symbol": r.symbol,
            "date": r.event_time.date() if hasattr(r.event_time, 'date') else r.event_time,
            "eps_surprise": surprise,
            "rev_surprise": rev_surprise,
            "payload": payload,
        })

    return earnings


def _compute_network_centrality(
    supply_chain: Dict[str, List[Dict]],
    peers: Dict[str, List[str]],
) -> Dict[str, float]:
    """Simple degree centrality. Higher = more connected."""
    degree: Dict[str, int] = defaultdict(int)

    for symbol, rels in supply_chain.items():
        degree[symbol] += len(rels)
        for rel in rels:
            degree[rel["related_symbol"]] += 1

    for symbol, peer_list in peers.items():
        degree[symbol] += len(peer_list)
        for p in peer_list:
            degree[p] += 1

    if not degree:
        return {}

    max_degree = max(degree.values())
    if max_degree == 0:
        return {}

    return {s: d / max_degree for s, d in degree.items()}


def _propagate_earnings_signal(
    earnings: List[Dict[str, Any]],
    supply_chain: Dict[str, List[Dict]],
    peers: Dict[str, List[str]],
    centrality: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Propagate earnings signals through the network.

    Key rules:
    - Customer beats → supplier gets positive signal (supply chain effect)
    - Customer misses → supplier gets negative signal
    - Peer beats → positive signal (sector momentum) but weaker
    - Weight by: centrality of source, revenue concentration, surprise magnitude
    """
    cascade_signals: Dict[str, Dict[str, Any]] = {}

    for earning in earnings:
        source = earning["symbol"]
        surprise = earning["eps_surprise"]
        rev_surprise = earning["rev_surprise"]

        if surprise is None and rev_surprise is None:
            continue

        # Combined surprise score
        combined = 0.0
        if surprise is not None:
            combined += surprise * 0.6
        if rev_surprise is not None:
            combined += rev_surprise * 0.4

        if abs(combined) < 0.02:  # Trivial surprise, skip
            continue

        source_centrality = centrality.get(source, 0.1)

        # Propagate through supply chain
        # If source is a customer and they beat, their suppliers benefit
        for sym, rels in supply_chain.items():
            for rel in rels:
                if rel["related_symbol"] == source:
                    # sym is connected to source
                    revenue_weight = min(rel["revenue_pct"] / 100.0, 1.0) if rel["revenue_pct"] > 1 else rel["revenue_pct"]
                    signal_strength = combined * source_centrality * revenue_weight

                    if rel["relation_type"] == "customer":
                        signal_strength *= 0.8  # Customer → supplier effect
                    elif rel["relation_type"] == "supplier":
                        signal_strength *= 0.5  # Supplier → customer (weaker)

                    key = f"{sym}:{source}"
                    if key not in cascade_signals or abs(signal_strength) > abs(cascade_signals[key].get("magnitude", 0)):
                        cascade_signals[key] = {
                            "target": sym,
                            "source": source,
                            "signal_type": f"earnings_{rel['relation_type']}",
                            "magnitude": round(signal_strength, 4),
                            "confidence": round(source_centrality * min(abs(combined), 1.0), 3),
                            "context": {
                                "eps_surprise": earning["eps_surprise"],
                                "rev_surprise": earning["rev_surprise"],
                                "relation": rel["relation_type"],
                                "revenue_pct": rel["revenue_pct"],
                                "centrality": round(source_centrality, 3),
                            },
                        }

        # Check reverse direction too
        if source in supply_chain:
            for rel in supply_chain[source]:
                target = rel["related_symbol"]
                revenue_weight = min(rel["revenue_pct"] / 100.0, 1.0) if rel["revenue_pct"] > 1 else rel["revenue_pct"]
                signal_strength = combined * source_centrality * revenue_weight * 0.6

                key = f"{target}:{source}"
                if key not in cascade_signals or abs(signal_strength) > abs(cascade_signals[key].get("magnitude", 0)):
                    cascade_signals[key] = {
                        "target": target,
                        "source": source,
                        "signal_type": f"earnings_{rel['relation_type']}_reverse",
                        "magnitude": round(signal_strength, 4),
                        "confidence": round(source_centrality * min(abs(combined), 1.0), 3),
                        "context": {
                            "eps_surprise": earning["eps_surprise"],
                            "rev_surprise": earning["rev_surprise"],
                            "relation": f"reverse_{rel['relation_type']}",
                            "revenue_pct": rel["revenue_pct"],
                            "centrality": round(source_centrality, 3),
                        },
                    }

        # Propagate through peer network (weaker)
        if source in peers:
            for peer in peers[source]:
                signal_strength = combined * source_centrality * 0.3  # Peer effect is weaker
                key = f"{peer}:{source}"
                if key not in cascade_signals or abs(signal_strength) > abs(cascade_signals[key].get("magnitude", 0)):
                    cascade_signals[key] = {
                        "target": peer,
                        "source": source,
                        "signal_type": "earnings_peer_momentum",
                        "magnitude": round(signal_strength, 4),
                        "confidence": round(source_centrality * min(abs(combined), 1.0) * 0.5, 3),
                        "context": {
                            "eps_surprise": earning["eps_surprise"],
                            "rev_surprise": earning["rev_surprise"],
                            "relation": "peer",
                            "centrality": round(source_centrality, 3),
                        },
                    }

    return list(cascade_signals.values())


async def _store_cascade_signals(
    as_of: date, signals: List[Dict[str, Any]],
) -> None:
    """Store cascade signals."""
    if not signals:
        return

    async with engine.begin() as conn:
        for sig in signals:
            await conn.execute(text("""
                INSERT INTO network.earnings_cascade
                  (source_symbol, target_symbol, signal_type, magnitude,
                   confidence, date, context)
                VALUES (:source, :target, :signal_type, :magnitude,
                        :confidence, :date, :context)
            """), {
                "source": sig["source"],
                "target": sig["target"],
                "signal_type": sig["signal_type"],
                "magnitude": sig["magnitude"],
                "confidence": sig["confidence"],
                "date": as_of,
                "context": json.dumps(sig["context"]),
            })

        # Also store as informed_trading signals for the meta-signal layer
        for sig in signals:
            if abs(sig["magnitude"]) > 0.05:
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
                    "symbol": sig["target"],
                    "date": as_of,
                    "signal_type": sig["signal_type"],
                    "strength": sig["magnitude"],
                    "context": json.dumps(sig["context"]),
                    "raw_data": json.dumps(sig),
                })


async def run_earnings_network(
    as_of: Optional[date] = None,
) -> Dict[str, Any]:
    """Run the earnings cascade / supply chain network pipeline."""
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()

    logger.info(f"Running earnings network analysis for {as_of}")

    supply_chain, peers = await _fetch_company_graph()
    earnings = await _fetch_recent_earnings(as_of, lookback_days=30)

    if not supply_chain and not peers:
        logger.warning("No company graph data — run ingestion first")
        return {"as_of": as_of.isoformat(), "status": "no_graph_data"}

    if not earnings:
        logger.warning("No recent earnings events")
        return {"as_of": as_of.isoformat(), "status": "no_earnings"}

    centrality = _compute_network_centrality(supply_chain, peers)
    cascade = _propagate_earnings_signal(earnings, supply_chain, peers, centrality)

    # Filter to meaningful signals
    significant = [s for s in cascade if abs(s["magnitude"]) > 0.02]
    significant.sort(key=lambda x: abs(x["magnitude"]), reverse=True)

    await _store_cascade_signals(as_of, significant)

    logger.info(
        f"Earnings network: {len(earnings)} earnings events, "
        f"{len(supply_chain)} supply chain nodes, {len(peers)} peer groups, "
        f"{len(significant)} cascade signals"
    )

    return {
        "as_of": as_of.isoformat(),
        "status": "success",
        "earnings_events": len(earnings),
        "supply_chain_nodes": len(supply_chain),
        "peer_groups": len(peers),
        "cascade_signals": len(significant),
        "top_signals": [
            {
                "target": s["target"],
                "source": s["source"],
                "type": s["signal_type"],
                "magnitude": s["magnitude"],
            }
            for s in significant[:10]
        ],
    }
