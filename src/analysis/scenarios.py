"""Forward Scenario Analysis — LLM-Generated Scenarios with Portfolio Impact.

Per PLAN.md Method 6: Not backward-looking stress tests, but forward-looking
scenario construction using all available data.

Pipeline:
1. Gather current market state (macro, narrative, signals, portfolio)
2. LLM generates 3 most likely near-term scenarios + 1 black swan
3. For each scenario, compute portfolio impact using historical analogues
4. Store results for report and Mira integration
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from src.db.session import engine
from src.utils.simulations.paths import (
    STRESS_WINDOWS,
    apply_log_returns,
    compute_log_returns,
    summarize_paths,
)

logger = logging.getLogger(__name__)
UTC = timezone.utc


async def _fetch_scenario_context(as_of: date) -> Dict[str, Any]:
    """Gather all data needed for scenario generation."""
    ctx: Dict[str, Any] = {}

    async with engine.begin() as conn:
        # Macro state
        result = await conn.execute(text("""
            SELECT factor, value FROM factors.daily_factors
            WHERE symbol = '__macro__' AND as_of = :date
        """), {"date": as_of})
        macro = {}
        for r in result.fetchall():
            parts = r.factor.split(".")
            if len(parts) >= 3:
                macro[f"{parts[1]}_{parts[2]}"] = round(float(r.value), 3)
        ctx["macro"] = macro

        # Narrative state
        result = await conn.execute(text("""
            SELECT dominant_narrative, narrative_shift, counter_narrative
            FROM narrative.daily_state
            WHERE date = :date AND scope = 'market' LIMIT 1
        """), {"date": as_of})
        row = result.fetchone()
        if row:
            ctx["narrative"] = {
                "dominant": row.dominant_narrative,
                "shift": float(row.narrative_shift) if row.narrative_shift else 0,
                "counter": row.counter_narrative,
            }

        # Latest debate synthesis
        result = await conn.execute(text("""
            SELECT payload FROM gpt.recommendations
            WHERE provider = 'debate'
            ORDER BY run_time DESC LIMIT 1
        """))
        row = result.fetchone()
        if row:
            payload = row.payload if isinstance(row.payload, dict) else json.loads(row.payload)
            synthesis = payload.get("synthesis", {})
            ctx["debate"] = {
                "regime": synthesis.get("regime_consensus", ""),
                "risks": synthesis.get("risk_warnings", []),
                "conflicts": [c.get("topic", "") for c in synthesis.get("key_conflicts", [])],
            }

        # Portfolio positions
        result = await conn.execute(text("""
            SELECT symbol, shares, current_price, market_value, weight
            FROM portfolio.positions
            ORDER BY market_value DESC NULLS LAST
        """))
        ctx["portfolio"] = [
            {"symbol": r.symbol, "shares": float(r.shares),
             "price": float(r.current_price) if r.current_price else 0,
             "value": float(r.market_value) if r.market_value else 0,
             "weight": round(float(r.weight) * 100, 1) if r.weight else 0}
            for r in result.fetchall()
        ]

        # Portfolio snapshot
        result = await conn.execute(text("""
            SELECT total_value, cash FROM portfolio.snapshots
            WHERE date <= :date ORDER BY date DESC LIMIT 1
        """), {"date": as_of})
        snap = result.fetchone()
        if snap:
            ctx["total_value"] = float(snap.total_value)
            ctx["cash"] = float(snap.cash)

        # Upcoming events
        result = await conn.execute(text("""
            SELECT symbol, event_type, event_time FROM event.events
            WHERE event_time::date > :start AND event_time::date <= :end
            ORDER BY event_time ASC LIMIT 15
        """), {"start": as_of, "end": as_of + timedelta(days=14)})
        ctx["upcoming_events"] = [
            {"symbol": r.symbol, "type": r.event_type,
             "date": r.event_time.strftime("%Y-%m-%d") if hasattr(r.event_time, "strftime") else str(r.event_time)}
            for r in result.fetchall()
        ]

    return ctx


async def _fetch_position_returns(
    symbols: List[str], lookback_days: int = 252,
) -> Dict[str, List[float]]:
    """Fetch historical daily log returns for portfolio positions."""
    if not symbols:
        return {}

    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT symbol, close, timestamp
            FROM market.ohlcv
            WHERE symbol = ANY(:symbols)
              AND timestamp >= NOW() - INTERVAL ':days days'
            ORDER BY symbol, timestamp ASC
        """.replace(":days", str(lookback_days))), {"symbols": symbols})
        rows = result.fetchall()

    # Group by symbol
    prices_by_sym: Dict[str, List[float]] = {}
    for r in rows:
        sym = r.symbol
        if sym not in prices_by_sym:
            prices_by_sym[sym] = []
        prices_by_sym[sym].append(float(r.close))

    return {sym: compute_log_returns(prices) for sym, prices in prices_by_sym.items()}


def _compute_portfolio_impact(
    scenario: Dict[str, Any],
    portfolio: List[Dict[str, Any]],
    total_value: float,
    cash: float,
) -> Dict[str, Any]:
    """Compute portfolio PnL under a scenario's expected moves."""
    moves = scenario.get("asset_impacts", [])
    move_map = {m["symbol"]: m for m in moves if "symbol" in m}

    position_impacts = []
    total_pnl = 0.0

    for pos in portfolio:
        sym = pos["symbol"]
        impact = move_map.get(sym, {})
        expected_pct = impact.get("expected_move_pct", 0) / 100.0

        pnl = pos["value"] * expected_pct
        total_pnl += pnl

        position_impacts.append({
            "symbol": sym,
            "current_value": pos["value"],
            "expected_move": f"{expected_pct:+.1%}",
            "pnl": round(pnl, 0),
            "reason": impact.get("reason", "No specific impact modeled"),
        })

    return {
        "total_pnl": round(total_pnl, 0),
        "total_pnl_pct": round(total_pnl / total_value * 100, 2) if total_value > 0 else 0,
        "position_impacts": position_impacts,
    }


async def _llm_generate_scenarios(
    llm_client,
    context: Dict[str, Any],
    as_of: date,
) -> Dict[str, Any]:
    """LLM generates forward scenarios."""
    portfolio_symbols = [p["symbol"] for p in context.get("portfolio", [])]

    prompt = f"""You are a macro strategist generating forward scenarios for portfolio risk management.

DATE: {as_of.isoformat()}

=== CURRENT STATE ===
Macro indicators: {json.dumps(context.get("macro", {}), indent=2)}

Narrative: {json.dumps(context.get("narrative"), indent=2, default=str)}

Debate regime: {json.dumps(context.get("debate", {}), indent=2)}

Portfolio: {json.dumps(context.get("portfolio", []), indent=2)}
Total value: ${context.get("total_value", 0):,.0f} | Cash: ${context.get("cash", 0):,.0f}

Upcoming events (next 14 days): {json.dumps(context.get("upcoming_events", []), indent=2)}

=== TASK ===
Generate forward-looking scenarios for the next 1-3 months.

For each scenario, estimate the impact on these specific portfolio holdings: {portfolio_symbols}

RESPOND WITH JSON ONLY:
{{
  "scenarios": [
    {{
      "name": "short scenario name",
      "probability": 0.0-1.0,
      "timeframe": "1-4 weeks" | "1-3 months",
      "description": "2-3 sentences describing the scenario",
      "trigger": "what would cause this scenario",
      "asset_impacts": [
        {{
          "symbol": "TICKER",
          "expected_move_pct": -10 to +20 (percentage),
          "reason": "one sentence"
        }}
      ],
      "hedges": ["hedge suggestion 1", "hedge suggestion 2"],
      "historical_analogue": "most similar past event"
    }}
  ],
  "black_swan": {{
    "name": "low-probability high-impact event",
    "probability": 0.01-0.10,
    "description": "2-3 sentences",
    "portfolio_impact_pct": -30 to +10,
    "hedge": "how to protect"
  }},
  "key_monitoring": ["data point 1 to watch", "data point 2", "data point 3"]
}}

Generate exactly 3 base scenarios (probabilities should sum to ~0.7-0.9) and 1 black swan.
Be specific about percentage impacts on the actual portfolio holdings."""

    system = (
        "You are a macro strategist doing forward scenario analysis for portfolio risk management. "
        "Be specific with numbers. Base scenarios on current data, not generic statements. "
        "Respond only with valid JSON."
    )

    try:
        raw = await llm_client.generate(prompt, system=system, temperature=0.3)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Scenario generation failed: {e}")
        return {"scenarios": [], "black_swan": None, "key_monitoring": []}


async def _store_scenarios(
    as_of: date, result: Dict[str, Any],
) -> None:
    """Store scenario analysis results."""
    async with engine.begin() as conn:
        # Store in report.daily sections (update existing row)
        await conn.execute(text("""
            UPDATE report.daily SET sections = CAST(:sections AS JSONB)
            WHERE date = :date
        """), {
            "date": as_of,
            "sections": json.dumps({"scenarios": result}),
        })


async def run_scenario_analysis(
    as_of: Optional[date] = None,
    llm_client=None,
) -> Dict[str, Any]:
    """Run forward scenario analysis.

    1. Gather current state
    2. LLM generates scenarios
    3. Compute portfolio impact for each
    4. Store results
    """
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()

    logger.info(f"Running scenario analysis for {as_of}")

    if not llm_client:
        return {"as_of": as_of.isoformat(), "status": "no_llm_client"}

    context = await _fetch_scenario_context(as_of)

    if not context.get("portfolio"):
        return {"as_of": as_of.isoformat(), "status": "no_portfolio"}

    # Generate scenarios
    scenarios_raw = await _llm_generate_scenarios(llm_client, context, as_of)

    total_value = context.get("total_value", 0)
    cash = context.get("cash", 0)
    portfolio = context.get("portfolio", [])

    # Compute portfolio impact for each scenario
    scenarios = []
    for s in scenarios_raw.get("scenarios", []):
        impact = _compute_portfolio_impact(s, portfolio, total_value, cash)
        scenarios.append({
            **s,
            "portfolio_impact": impact,
        })

    # Black swan impact
    black_swan = scenarios_raw.get("black_swan")
    if black_swan and black_swan.get("portfolio_impact_pct"):
        black_swan["portfolio_pnl"] = round(
            total_value * black_swan["portfolio_impact_pct"] / 100, 0
        )

    result = {
        "as_of": as_of.isoformat(),
        "status": "success",
        "scenarios": scenarios,
        "black_swan": black_swan,
        "key_monitoring": scenarios_raw.get("key_monitoring", []),
        "portfolio_value": total_value,
    }

    await _store_scenarios(as_of, result)

    logger.info(
        f"Scenario analysis: {len(scenarios)} scenarios generated, "
        f"black_swan={'yes' if black_swan else 'no'}"
    )

    return result
