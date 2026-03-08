"""LLM Meta-Signal Layer — Regime-Aware Signal Weighting.

Based on Alpha-R1 (arxiv:2512.23515, Dec 2025):
Instead of fixed weights, an LLM reasons about WHICH signals are contextually
relevant in the current market regime.

Pipeline:
1. Gather all available signals (narrative, polymarket, factors, macro)
2. Summarize current market state
3. LLM reasons about which signals to trust and why
4. Output: per-symbol signal scores with regime-adjusted weights
5. Store in signals.unified
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from src.db.session import engine

logger = logging.getLogger(__name__)
UTC = timezone.utc


async def _fetch_narrative_state(as_of: date) -> Optional[Dict[str, Any]]:
    """Fetch today's narrative analysis results."""
    query = text("""
        SELECT dominant_narrative, narrative_shift, shift_magnitude,
               counter_narrative, novelty, raw_analysis
        FROM narrative.daily_state
        WHERE date = :date AND scope = 'market'
        LIMIT 1
    """)
    async with engine.begin() as conn:
        result = await conn.execute(query, {"date": as_of})
        row = result.fetchone()
    if not row:
        return None
    raw = row.raw_analysis
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    return {
        "dominant_narrative": row.dominant_narrative,
        "narrative_shift": float(row.narrative_shift) if row.narrative_shift else 0.0,
        "shift_magnitude": float(row.shift_magnitude) if row.shift_magnitude else 0.0,
        "counter_narrative": row.counter_narrative,
        "novelty": float(row.novelty) if row.novelty else 0.0,
        "fragmentation": raw.get("fragmentation_index", 0.0) if isinstance(raw, dict) else 0.0,
        "regime_signal": raw.get("llm", {}).get("regime_signal", "unknown") if isinstance(raw, dict) else "unknown",
    }


async def _fetch_polymarket_signals(as_of: date) -> List[Dict[str, Any]]:
    """Fetch today's polymarket lead-lag signals."""
    query = text("""
        SELECT source_market, target_market, gap_signal, gap_z_score, context
        FROM signals.information_flow
        WHERE date = :date
        ORDER BY gap_z_score DESC
        LIMIT 20
    """)
    async with engine.begin() as conn:
        result = await conn.execute(query, {"date": as_of})
        rows = result.fetchall()
    signals = []
    for r in rows:
        ctx = r.context
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx)
            except json.JSONDecodeError:
                ctx = {}
        signals.append({
            "source": r.source_market,
            "target": r.target_market,
            "gap_signal": float(r.gap_signal) if r.gap_signal else 0.0,
            "f_stat": float(r.gap_z_score) if r.gap_z_score else 0.0,
            "context": ctx,
        })
    return signals


async def _fetch_macro_state(as_of: date) -> Dict[str, Any]:
    """Fetch current macro factor values."""
    query = text("""
        SELECT factor, value
        FROM factors.daily_factors
        WHERE symbol = '__macro__'
          AND as_of = :date
    """)
    async with engine.begin() as conn:
        result = await conn.execute(query, {"date": as_of})
        rows = result.fetchall()
    return {r.factor: float(r.value) for r in rows}


async def _fetch_top_movers(as_of: date, n: int = 20) -> List[Dict[str, Any]]:
    """Fetch symbols with strongest recent signals from factors."""
    query = text("""
        WITH latest AS (
            SELECT symbol, factor, value
            FROM factors.daily_factors
            WHERE as_of = :date
              AND symbol != '__macro__'
              AND factor IN (
                'mkt.return_1d', 'mkt.return_5d', 'mkt.vol_z_20d',
                'news.sentiment_3d', 'news.volume_momentum_7d',
                'event.momentum_7d'
              )
        ),
        pivoted AS (
            SELECT
                symbol,
                MAX(CASE WHEN factor = 'mkt.return_1d' THEN value END) AS ret_1d,
                MAX(CASE WHEN factor = 'mkt.return_5d' THEN value END) AS ret_5d,
                MAX(CASE WHEN factor = 'mkt.vol_z_20d' THEN value END) AS vol_z,
                MAX(CASE WHEN factor = 'news.sentiment_3d' THEN value END) AS sentiment,
                MAX(CASE WHEN factor = 'news.volume_momentum_7d' THEN value END) AS news_momentum,
                MAX(CASE WHEN factor = 'event.momentum_7d' THEN value END) AS event_momentum
            FROM latest
            GROUP BY symbol
        )
        SELECT * FROM pivoted
        WHERE ret_1d IS NOT NULL
        ORDER BY ABS(ret_1d) DESC
        LIMIT :n
    """)
    async with engine.begin() as conn:
        result = await conn.execute(query, {"date": as_of, "n": n})
        rows = result.fetchall()
    return [
        {
            "symbol": r.symbol,
            "ret_1d": round(float(r.ret_1d), 4) if r.ret_1d else None,
            "ret_5d": round(float(r.ret_5d), 4) if r.ret_5d else None,
            "vol_z": round(float(r.vol_z), 3) if r.vol_z else None,
            "sentiment": round(float(r.sentiment), 3) if r.sentiment else None,
            "news_momentum": round(float(r.news_momentum), 2) if r.news_momentum else None,
            "event_momentum": round(float(r.event_momentum), 2) if r.event_momentum else None,
        }
        for r in rows
    ]


async def _fetch_portfolio(as_of: date) -> List[Dict[str, Any]]:
    """Fetch current portfolio positions."""
    query = text("""
        SELECT symbol, shares, avg_cost, current_price, market_value, weight
        FROM portfolio.positions
    """)
    async with engine.begin() as conn:
        result = await conn.execute(query)
        rows = result.fetchall()
    return [
        {
            "symbol": r.symbol,
            "shares": float(r.shares),
            "avg_cost": float(r.avg_cost),
            "current_price": float(r.current_price) if r.current_price else None,
            "market_value": float(r.market_value) if r.market_value else None,
            "weight": float(r.weight) if r.weight else None,
        }
        for r in rows
    ]


async def _llm_regime_assessment(
    llm_client,
    narrative: Optional[Dict[str, Any]],
    macro: Dict[str, Any],
    polymarket_signals: List[Dict[str, Any]],
    top_movers: List[Dict[str, Any]],
    portfolio: List[Dict[str, Any]],
    as_of: date,
) -> Dict[str, Any]:
    """Core LLM call: assess regime and generate signal weights + recommendations."""

    # Format macro state
    macro_summary = {}
    for k, v in macro.items():
        # Extract series name from factor key like "macro.VIXCLS.z20"
        parts = k.split(".")
        if len(parts) >= 3:
            series = parts[1]
            metric = parts[2]
            macro_summary[f"{series}_{metric}"] = round(v, 3)

    prompt = f"""You are a quantitative portfolio strategist. Analyze the current market state and provide actionable signals.

DATE: {as_of.isoformat()}

=== NARRATIVE STATE ===
{json.dumps(narrative, indent=2) if narrative else "No narrative data available"}

=== MACRO INDICATORS ===
{json.dumps(macro_summary, indent=2)}

Key series: VIXCLS (VIX), DGS10 (10Y yield), DGS2 (2Y yield), T10Y2Y (yield curve),
BAMLH0A0HYM2 (HY credit spread), DCOILWTICO (WTI crude).
z20/z60 = z-score vs 20/60 day mean. chg20/chg60 = % change vs 20/60 day mean.

=== POLYMARKET SIGNALS ===
{json.dumps(polymarket_signals[:10], indent=2) if polymarket_signals else "No polymarket signals"}

=== TOP MOVERS (by |return_1d|) ===
{json.dumps(top_movers[:15], indent=2)}

=== CURRENT PORTFOLIO ===
{json.dumps(portfolio, indent=2) if portfolio else "No portfolio data"}

=== YOUR TASK ===
1. REGIME: Classify current market regime and assess stability
2. SIGNAL WEIGHTS: Given the regime, how should I weight different signal sources?
3. RECOMMENDATIONS: Top 5 actionable ideas for the portfolio

RESPOND WITH JSON ONLY:
{{
  "regime": {{
    "classification": "risk_on" | "risk_off" | "transition" | "range_bound",
    "confidence": 0.0-1.0,
    "key_driver": "one sentence",
    "stability": "stable" | "fragile" | "shifting"
  }},
  "signal_weights": {{
    "narrative": 0.0-1.0,
    "macro": 0.0-1.0,
    "polymarket": 0.0-1.0,
    "momentum": 0.0-1.0,
    "sentiment": 0.0-1.0,
    "reasoning": "why these weights for current regime"
  }},
  "top_signals": [
    {{
      "symbol": "TICKER",
      "direction": "long" | "short" | "reduce" | "hold",
      "confidence": "high" | "medium" | "low",
      "signal_score": -1.0 to 1.0,
      "thesis": "one sentence",
      "risk": "one sentence",
      "time_horizon": "days" | "weeks" | "months"
    }}
  ],
  "portfolio_actions": [
    {{
      "action": "hold" | "add" | "reduce" | "hedge",
      "symbol": "TICKER",
      "reasoning": "one sentence"
    }}
  ],
  "key_risks": ["risk 1", "risk 2", "risk 3"]
}}"""

    system = (
        "You are a quantitative portfolio strategist. You make decisions based on data, "
        "not opinions. Be decisive and specific. Respond only with valid JSON. No markdown."
    )

    try:
        raw = await llm_client.generate(prompt, system=system, temperature=0.3)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"LLM regime assessment failed: {e}")
        return {
            "regime": {
                "classification": "uncertain",
                "confidence": 0.0,
                "key_driver": "LLM analysis failed",
                "stability": "unknown",
            },
            "signal_weights": {
                "narrative": 0.2, "macro": 0.2, "polymarket": 0.2,
                "momentum": 0.2, "sentiment": 0.2,
                "reasoning": "equal weights (LLM failed)",
            },
            "top_signals": [],
            "portfolio_actions": [],
            "key_risks": ["LLM analysis unavailable"],
        }


async def _store_unified_signals(
    as_of: date,
    assessment: Dict[str, Any],
) -> int:
    """Store LLM-generated signals in signals.unified."""
    signals = assessment.get("top_signals", [])
    if not signals:
        return 0

    direction_map = {
        "long": "buy", "short": "sell", "reduce": "sell",
        "hold": "neutral", "add": "buy",
    }

    query = text("""
        INSERT INTO signals.unified
          (symbol, date, signal_score, signal_direction, confidence,
           components, conflicts, key_drivers)
        VALUES
          (:symbol, :date, :score, :direction, :confidence,
           CAST(:components AS JSONB), CAST(:conflicts AS JSONB),
           CAST(:key_drivers AS JSONB))
        ON CONFLICT (symbol, date)
        DO UPDATE SET
          signal_score = EXCLUDED.signal_score,
          signal_direction = EXCLUDED.signal_direction,
          confidence = EXCLUDED.confidence,
          components = EXCLUDED.components,
          key_drivers = EXCLUDED.key_drivers
    """)

    stored = 0
    async with engine.begin() as conn:
        for sig in signals:
            symbol = sig.get("symbol")
            if not symbol:
                continue

            score = sig.get("signal_score", 0.0)
            direction = sig.get("direction", "neutral")
            mapped_dir = direction_map.get(direction, "neutral")

            # Map to strong_buy/buy/neutral/sell/strong_sell
            if mapped_dir == "buy" and abs(score) > 0.6:
                signal_direction = "strong_buy"
            elif mapped_dir == "buy":
                signal_direction = "buy"
            elif mapped_dir == "sell" and abs(score) > 0.6:
                signal_direction = "strong_sell"
            elif mapped_dir == "sell":
                signal_direction = "sell"
            else:
                signal_direction = "neutral"

            conf_map = {"high": 0.8, "medium": 0.5, "low": 0.3}
            confidence = conf_map.get(sig.get("confidence", "medium"), 0.5)

            await conn.execute(query, {
                "symbol": symbol,
                "date": as_of,
                "score": score,
                "direction": signal_direction,
                "confidence": confidence,
                "components": json.dumps({
                    "weights": assessment.get("signal_weights", {}),
                    "regime": assessment.get("regime", {}),
                }),
                "conflicts": json.dumps(None),
                "key_drivers": json.dumps({
                    "thesis": sig.get("thesis", ""),
                    "risk": sig.get("risk", ""),
                    "time_horizon": sig.get("time_horizon", ""),
                }),
            })
            stored += 1

    return stored


async def run_meta_signal(
    as_of: Optional[date] = None,
    llm_client=None,
) -> Dict[str, Any]:
    """Run the LLM meta-signal layer.

    1. Gather all available signals
    2. LLM regime assessment + signal weighting
    3. Store unified signals

    Returns summary dict.
    """
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()

    logger.info(f"Running meta-signal analysis for {as_of}")

    if not llm_client:
        return {"as_of": as_of.isoformat(), "status": "no_llm_client"}

    # Gather inputs (parallel-safe: all read-only)
    narrative = await _fetch_narrative_state(as_of)
    polymarket_signals = await _fetch_polymarket_signals(as_of)
    macro = await _fetch_macro_state(as_of)
    top_movers = await _fetch_top_movers(as_of)
    portfolio = await _fetch_portfolio(as_of)

    # LLM assessment
    assessment = await _llm_regime_assessment(
        llm_client, narrative, macro, polymarket_signals,
        top_movers, portfolio, as_of,
    )

    # Store signals
    stored = await _store_unified_signals(as_of, assessment)

    regime = assessment.get("regime", {})
    weights = assessment.get("signal_weights", {})

    result = {
        "as_of": as_of.isoformat(),
        "status": "success",
        "regime": regime.get("classification", "unknown"),
        "regime_confidence": regime.get("confidence", 0.0),
        "regime_stability": regime.get("stability", "unknown"),
        "signal_weights": {k: v for k, v in weights.items() if k != "reasoning"},
        "signals_stored": stored,
        "top_signals": [
            {
                "symbol": s.get("symbol"),
                "direction": s.get("direction"),
                "score": s.get("signal_score"),
                "thesis": s.get("thesis"),
            }
            for s in assessment.get("top_signals", [])[:5]
        ],
        "portfolio_actions": assessment.get("portfolio_actions", []),
        "key_risks": assessment.get("key_risks", []),
    }

    logger.info(
        f"Meta-signal complete: regime={result['regime']}, "
        f"signals={stored}, confidence={result['regime_confidence']}"
    )
    return result
