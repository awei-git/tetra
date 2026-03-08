"""LLM Adversarial Debate with Information Asymmetry.

Each LLM gets a different slice of data, creating genuine information asymmetry:
- Analyst A (macro-first): macro data, yields, credit, FRED — no individual stock news
- Analyst B (micro-first): earnings, insider trades, analyst consensus, fundamentals
- Analyst C (crowd-first): Polymarket, news narratives, sentiment shifts

Debate structure:
1. Round 1: Each analyst gives top 5 trades with data support
2. Round 2: Analysts challenge each other (see each other's R1 output)
3. Round 3: Synthesis — consensus vs contrarian trades
4. Store results in gpt.recommendation_debates
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from src.db.session import engine

logger = logging.getLogger(__name__)
UTC = timezone.utc


# ─── Data fetching (partitioned by analyst role) ────────────────────────


async def _fetch_macro_data(as_of: date) -> Dict[str, Any]:
    """Data for Analyst A: macro indicators, yield curve, credit spreads."""
    factors = {}
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT factor, value FROM factors.daily_factors
            WHERE symbol = '__macro__' AND as_of = :date
        """), {"date": as_of})
        for r in result.fetchall():
            factors[r.factor] = round(float(r.value), 4)

        # Recent economic data points
        result = await conn.execute(text("""
            SELECT s.name, v.value, v.timestamp
            FROM economic.values v
            JOIN economic.series s ON v.series_id = s.series_id
            WHERE v.timestamp >= :start
            ORDER BY v.timestamp DESC
            LIMIT 50
        """), {"start": as_of - timedelta(days=30)})
        econ = [{"name": r.name, "value": float(r.value),
                 "date": r.timestamp.strftime("%Y-%m-%d")} for r in result.fetchall()]

    return {"macro_factors": factors, "recent_economic": econ[:20]}


async def _fetch_micro_data(as_of: date) -> Dict[str, Any]:
    """Data for Analyst B: earnings, insider trades, analyst consensus."""
    async with engine.begin() as conn:
        # Recent earnings with surprise
        result = await conn.execute(text("""
            SELECT symbol, event_time, payload
            FROM event.events
            WHERE event_type = 'earnings'
              AND event_time::date >= :start AND event_time::date <= :end
            ORDER BY event_time DESC LIMIT 30
        """), {"start": as_of - timedelta(days=30), "end": as_of})
        earnings = []
        for r in result.fetchall():
            p = r.payload if isinstance(r.payload, dict) else {}
            earnings.append({
                "symbol": r.symbol,
                "date": r.event_time.strftime("%Y-%m-%d"),
                "eps_actual": p.get("epsActual"),
                "eps_estimate": p.get("epsEstimate"),
                "revenue_actual": p.get("revenueActual"),
                "revenue_estimate": p.get("revenueEstimate"),
            })

        # Insider trading signals
        result = await conn.execute(text("""
            SELECT symbol, signal_type, strength, context
            FROM signals.informed_trading
            WHERE date = :date AND signal_type LIKE 'insider%'
            ORDER BY ABS(strength) DESC LIMIT 20
        """), {"date": as_of})
        insider = [{"symbol": r.symbol, "type": r.signal_type,
                    "strength": float(r.strength)} for r in result.fetchall()]

        # Analyst consensus
        result = await conn.execute(text("""
            WITH latest AS (
                SELECT symbol, period, strong_buy, buy, hold, sell, strong_sell,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY period DESC) AS rn
                FROM event.analyst_recommendations
            )
            SELECT * FROM latest WHERE rn = 1
            ORDER BY (strong_buy + buy)::float / NULLIF(strong_buy+buy+hold+sell+strong_sell, 0) DESC
            LIMIT 30
        """))
        consensus = []
        for r in result.fetchall():
            total = (r.strong_buy or 0) + (r.buy or 0) + (r.hold or 0) + (r.sell or 0) + (r.strong_sell or 0)
            consensus.append({
                "symbol": r.symbol,
                "strong_buy": r.strong_buy, "buy": r.buy,
                "hold": r.hold, "sell": r.sell, "strong_sell": r.strong_sell,
                "bull_pct": round(((r.strong_buy or 0) + (r.buy or 0)) / max(total, 1) * 100, 1),
            })

    return {"recent_earnings": earnings, "insider_signals": insider,
            "analyst_consensus": consensus}


async def _fetch_crowd_data(as_of: date) -> Dict[str, Any]:
    """Data for Analyst C: Polymarket, news narratives, sentiment."""
    async with engine.begin() as conn:
        # Narrative state
        result = await conn.execute(text("""
            SELECT dominant_narrative, narrative_shift, shift_magnitude,
                   counter_narrative, novelty, raw_analysis
            FROM narrative.daily_state
            WHERE date = :date AND scope = 'market' LIMIT 1
        """), {"date": as_of})
        row = result.fetchone()
        narrative = None
        if row:
            raw = row.raw_analysis
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    raw = {}
            narrative = {
                "dominant_narrative": row.dominant_narrative,
                "shift": float(row.narrative_shift) if row.narrative_shift else 0,
                "shift_magnitude": float(row.shift_magnitude) if row.shift_magnitude else 0,
                "counter_narrative": row.counter_narrative,
                "fragmentation": raw.get("fragmentation_index", 0) if isinstance(raw, dict) else 0,
                "top_themes": raw.get("theme_counts", {}) if isinstance(raw, dict) else {},
            }

        # Polymarket signals
        result = await conn.execute(text("""
            SELECT source_market, target_market, gap_signal, gap_z_score, context
            FROM signals.information_flow
            WHERE date = :date ORDER BY gap_z_score DESC LIMIT 15
        """), {"date": as_of})
        polymarket = []
        for r in result.fetchall():
            ctx = r.context
            if isinstance(ctx, str):
                try:
                    ctx = json.loads(ctx)
                except json.JSONDecodeError:
                    ctx = {}
            polymarket.append({
                "source": r.source_market, "target": r.target_market,
                "gap": float(r.gap_signal) if r.gap_signal else 0,
                "f_stat": float(r.gap_z_score) if r.gap_z_score else 0,
                "question": ctx.get("question", "") if isinstance(ctx, dict) else "",
            })

        # News sentiment momentum
        result = await conn.execute(text("""
            SELECT symbol, value FROM factors.daily_factors
            WHERE as_of = :date AND factor = 'news.sentiment_3d'
              AND symbol != '__macro__'
            ORDER BY value DESC LIMIT 20
        """), {"date": as_of})
        sentiment = [{"symbol": r.symbol, "sentiment_3d": round(float(r.value), 3)}
                     for r in result.fetchall()]

        # Earnings network cascade signals
        result = await conn.execute(text("""
            SELECT symbol, signal_type, strength
            FROM signals.informed_trading
            WHERE date = :date AND signal_type LIKE 'earnings%'
            ORDER BY ABS(strength) DESC LIMIT 15
        """), {"date": as_of})
        cascade = [{"symbol": r.symbol, "type": r.signal_type,
                    "strength": float(r.strength)} for r in result.fetchall()]

    return {"narrative": narrative, "polymarket_signals": polymarket,
            "sentiment_leaders": sentiment, "earnings_cascade": cascade}


async def _fetch_portfolio() -> List[Dict[str, Any]]:
    """Fetch portfolio for all analysts."""
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT symbol, shares, avg_cost, current_price, market_value, weight
            FROM portfolio.positions
        """))
        return [{"symbol": r.symbol, "shares": float(r.shares),
                 "avg_cost": float(r.avg_cost),
                 "value": float(r.market_value) if r.market_value else None,
                 "weight": round(float(r.weight) * 100, 1) if r.weight else None}
                for r in result.fetchall()]


# ─── Debate rounds ─────────────────────────────────────────────────────


ANALYST_SYSTEM = {
    "macro": (
        "You are a macro-focused portfolio analyst. You see ONLY macro data — yields, "
        "credit spreads, economic indicators, VIX. You do NOT see individual stock news, "
        "earnings, or sentiment. Your strength: understanding regime shifts and systemic risk. "
        "Be decisive and specific. Respond ONLY with valid JSON."
    ),
    "micro": (
        "You are a bottom-up equity analyst. You see ONLY company-level data — earnings "
        "surprises, insider trading patterns, analyst consensus shifts. You do NOT see "
        "macro indicators or market sentiment. Your strength: finding individual stock "
        "alpha from fundamental signals. Be decisive and specific. Respond ONLY with valid JSON."
    ),
    "crowd": (
        "You are a market sentiment and flow analyst. You see ONLY crowd data — prediction "
        "market signals, news narrative shifts, sentiment momentum, information flow gaps. "
        "You do NOT see fundamentals or macro data. Your strength: detecting mispricing from "
        "narrative shifts and crowd behavior. Be decisive and specific. Respond ONLY with valid JSON."
    ),
}


async def _round1_initial_views(
    clients: Dict[str, Any],
    macro_data: Dict, micro_data: Dict, crowd_data: Dict,
    portfolio: List[Dict], as_of: date,
    track_record_ctx: str = "",
) -> Dict[str, Dict]:
    """Round 1: Each analyst gives independent views."""
    prompt_template = """DATE: {date}

=== YOUR DATA ===
{data}

=== PORTFOLIO (shared context) ===
{portfolio}
{track_record}

=== TASK ===
Give your top 5 trade recommendations based ONLY on the data you can see.
For each trade, provide specific data points that support your thesis.
If track record data is provided, learn from past mistakes — avoid repeating losing patterns.

RESPOND WITH JSON:
{{
  "regime_view": "your assessment of current market environment",
  "conviction_level": "high" | "medium" | "low",
  "trades": [
    {{
      "symbol": "TICKER",
      "direction": "long" | "short",
      "confidence": "high" | "medium" | "low",
      "thesis": "2-3 sentences with specific data references",
      "risk": "main risk to this trade",
      "time_horizon": "days" | "weeks" | "months",
      "target_return": "expected return %"
    }}
  ],
  "portfolio_concern": "biggest concern about current portfolio based on your data",
  "blind_spot": "what you wish you could see but can't"
}}"""

    results = {}
    role_data = {
        "macro": json.dumps(macro_data, indent=2, default=str),
        "micro": json.dumps(micro_data, indent=2, default=str),
        "crowd": json.dumps(crowd_data, indent=2, default=str),
    }

    for role, client in clients.items():
        prompt = prompt_template.format(
            date=as_of.isoformat(),
            data=role_data[role],
            portfolio=json.dumps(portfolio, indent=2),
            track_record=("\n" + track_record_ctx) if track_record_ctx else "",
        )
        try:
            raw = await client.generate(prompt, system=ANALYST_SYSTEM[role], temperature=0.4)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            results[role] = json.loads(raw)
        except Exception as e:
            logger.warning(f"Round 1 failed for {role}: {e}")
            results[role] = {"regime_view": f"analysis failed: {e}", "trades": [],
                             "portfolio_concern": "", "blind_spot": ""}

    return results


async def _round2_challenge(
    clients: Dict[str, Any],
    r1_results: Dict[str, Dict],
    as_of: date,
) -> Dict[str, Dict]:
    """Round 2: Each analyst sees others' R1 output and challenges."""
    results = {}

    for role, client in clients.items():
        other_roles = [r for r in r1_results if r != role]
        others_summary = {r: r1_results[r] for r in other_roles}

        prompt = f"""DATE: {as_of.isoformat()}

=== YOUR ROUND 1 VIEW ===
{json.dumps(r1_results[role], indent=2)}

=== OTHER ANALYSTS' VIEWS ===
{json.dumps(others_summary, indent=2)}

=== TASK ===
Review the other analysts' views. They have data you DON'T have.
1. Which of their trades do you AGREE with (and why)?
2. Which do you DISAGREE with (and why — based on YOUR data)?
3. Did their views change your conviction on any of your own trades?
4. What's the biggest DISAGREEMENT and who is more likely right?

RESPOND WITH JSON:
{{
  "agreements": [
    {{"analyst": "macro|micro|crowd", "trade": "TICKER direction", "why": "your data supports this because..."}}
  ],
  "disagreements": [
    {{"analyst": "macro|micro|crowd", "trade": "TICKER direction", "why": "my data contradicts this because...", "my_data_says": "specific counter-evidence"}}
  ],
  "conviction_changes": [
    {{"trade": "TICKER direction", "old_confidence": "high|medium|low", "new_confidence": "high|medium|low", "reason": "..."}}
  ],
  "biggest_disagreement": {{
    "topic": "what the disagreement is about",
    "who_is_right": "which analyst and why",
    "what_to_watch": "data point that would resolve this"
  }},
  "revised_top3": [
    {{"symbol": "TICKER", "direction": "long|short", "confidence": "high|medium|low", "thesis": "updated thesis incorporating new info"}}
  ]
}}"""

        try:
            raw = await client.generate(prompt, system=ANALYST_SYSTEM[role], temperature=0.3)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            results[role] = json.loads(raw)
        except Exception as e:
            logger.warning(f"Round 2 failed for {role}: {e}")
            results[role] = {"agreements": [], "disagreements": [],
                             "conviction_changes": [], "revised_top3": []}

    return results


async def _round3_synthesis(
    synthesis_client: Any,
    r1_results: Dict[str, Dict],
    r2_results: Dict[str, Dict],
    portfolio: List[Dict],
    as_of: date,
) -> Dict[str, Any]:
    """Round 3: Synthesis — one LLM reads the full debate and produces final output."""
    prompt = f"""DATE: {as_of.isoformat()}

You are the Chief Investment Officer reviewing a debate between three analysts.
Each analyst had DIFFERENT data — their views reflect genuine information asymmetry.

=== ROUND 1: INITIAL VIEWS ===
MACRO ANALYST (sees: yields, credit spreads, economic data):
{json.dumps(r1_results.get('macro', {}), indent=2)}

MICRO ANALYST (sees: earnings, insider trades, analyst consensus):
{json.dumps(r1_results.get('micro', {}), indent=2)}

CROWD ANALYST (sees: Polymarket, narrative shifts, sentiment):
{json.dumps(r1_results.get('crowd', {}), indent=2)}

=== ROUND 2: CHALLENGES ===
MACRO response: {json.dumps(r2_results.get('macro', {}), indent=2)}
MICRO response: {json.dumps(r2_results.get('micro', {}), indent=2)}
CROWD response: {json.dumps(r2_results.get('crowd', {}), indent=2)}

=== CURRENT PORTFOLIO ===
{json.dumps(portfolio, indent=2)}

=== YOUR TASK ===
Synthesize the debate into actionable decisions. Pay special attention to:
1. CONSENSUS trades: all analysts agree → high confidence
2. CONTRARIAN trades: only one analyst sees it, but with strong data → potential alpha
3. CONFLICTS: genuine disagreements that need monitoring

RESPOND WITH JSON:
{{
  "regime_consensus": "what all analysts agree about the current environment",
  "consensus_trades": [
    {{
      "symbol": "TICKER",
      "direction": "long" | "short",
      "confidence": "high" | "medium",
      "supporting_analysts": ["macro", "micro", "crowd"],
      "combined_thesis": "synthesis of all supporting data",
      "risk": "main risk"
    }}
  ],
  "contrarian_trades": [
    {{
      "symbol": "TICKER",
      "direction": "long" | "short",
      "confidence": "medium" | "low",
      "source_analyst": "macro|micro|crowd",
      "thesis": "why this analyst sees something others don't",
      "what_others_miss": "the information gap",
      "risk": "why the majority might be right"
    }}
  ],
  "portfolio_actions": [
    {{
      "symbol": "TICKER",
      "action": "hold" | "add" | "reduce" | "exit" | "hedge",
      "urgency": "immediate" | "this_week" | "monitor",
      "reasoning": "based on debate evidence"
    }}
  ],
  "key_conflicts": [
    {{
      "topic": "what the debate couldn't resolve",
      "bull_case": "...",
      "bear_case": "...",
      "resolution_data": "what data would resolve this"
    }}
  ],
  "risk_warnings": ["risk 1", "risk 2", "risk 3"]
}}"""

    system = (
        "You are a Chief Investment Officer synthesizing analyst debates. "
        "Be decisive — the goal is ACTIONABLE output, not balanced commentary. "
        "Respond only with valid JSON. No markdown."
    )

    try:
        raw = await synthesis_client.generate(prompt, system=system, temperature=0.2)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Round 3 synthesis failed: {e}")
        return {
            "regime_consensus": "synthesis failed",
            "consensus_trades": [], "contrarian_trades": [],
            "portfolio_actions": [], "key_conflicts": [],
            "risk_warnings": [f"Synthesis LLM failed: {e}"],
        }


# ─── Storage ────────────────────────────────────────────────────────────


async def _store_debate(
    as_of: date,
    r1: Dict[str, Dict], r2: Dict[str, Dict],
    synthesis: Dict[str, Any],
    llm_stats: Dict[str, Any],
) -> None:
    """Store debate results."""
    now = datetime.now(tz=UTC)
    payload = {
        "round1": r1,
        "round2": r2,
        "synthesis": synthesis,
        "llm_stats": llm_stats,
    }
    async with engine.begin() as conn:
        await conn.execute(text("""
            INSERT INTO gpt.recommendations
              (provider, session, run_time, payload, created_at)
            VALUES ('debate', :session, :run_time, CAST(:payload AS JSONB), :created_at)
            ON CONFLICT (provider, session, run_time)
            DO UPDATE SET payload = EXCLUDED.payload
        """), {
            "session": as_of.strftime("%Y%m%d"),
            "run_time": now,
            "payload": json.dumps(payload, default=str),
            "created_at": now,
        })


# ─── Main entry point ──────────────────────────────────────────────────


async def run_debate(
    as_of: Optional[date] = None,
    llm_clients: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the adversarial debate pipeline.

    Requires at least 2 LLM clients (ideally 3 for full information asymmetry).
    Client assignment:
    - If 3+ clients: one per role
    - If 2 clients: two roles + shared synthesis
    - If 1 client: same client plays all roles (still valuable for structured analysis)

    Args:
        llm_clients: Dict of {name: BaseLLMClient} — e.g. from create_clients()
    """
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()

    logger.info(f"Running adversarial debate for {as_of}")

    if not llm_clients:
        return {"as_of": as_of.isoformat(), "status": "no_llm_clients"}

    available = list(llm_clients.keys())
    logger.info(f"Debate participants: {available}")

    # Assign roles. Prefer: openai=macro, deepseek=micro, gemini=crowd
    # But work with whatever is available
    role_preference = {
        "macro": ["openai", "claude", "deepseek", "gemini"],
        "micro": ["deepseek", "openai", "gemini", "claude"],
        "crowd": ["gemini", "deepseek", "openai", "claude"],
    }

    assigned: Dict[str, Any] = {}
    used = set()
    for role, prefs in role_preference.items():
        for pref in prefs:
            if pref in llm_clients and pref not in used:
                assigned[role] = llm_clients[pref]
                used.add(pref)
                break
        if role not in assigned:
            # Reuse a client if we don't have enough
            fallback = available[0]
            assigned[role] = llm_clients[fallback]

    logger.info(f"Role assignment: macro={assigned['macro'].name}, "
                f"micro={assigned['micro'].name}, crowd={assigned['crowd'].name}")

    # Fetch partitioned data
    macro_data, micro_data, crowd_data, portfolio = await _fetch_all_data(as_of)

    # Fetch track record for self-critique
    track_record_ctx = ""
    try:
        from src.portfolio.track_record import compute_track_record, format_track_record_for_llm
        record = await compute_track_record()
        if record.get("overall", {}).get("closed", 0) > 0:
            track_record_ctx = format_track_record_for_llm(record)
            logger.info(f"Track record loaded: {record['overall']['closed']} closed recs")
    except Exception as e:
        logger.warning(f"Track record fetch failed (non-critical): {e}")

    # Round 1: Initial views
    logger.info("Debate Round 1: Initial views")
    r1 = await _round1_initial_views(assigned, macro_data, micro_data, crowd_data, portfolio, as_of, track_record_ctx=track_record_ctx)
    r1_trades = sum(len(v.get("trades", [])) for v in r1.values())
    logger.info(f"Round 1 complete: {r1_trades} trades proposed")

    # Round 2: Challenge
    logger.info("Debate Round 2: Challenges")
    r2 = await _round2_challenge(assigned, r1, as_of)

    # Round 3: Synthesis (use the most capable client)
    synthesis_order = ["openai", "claude", "deepseek", "gemini"]
    synthesis_client = None
    for name in synthesis_order:
        if name in llm_clients:
            synthesis_client = llm_clients[name]
            break
    if not synthesis_client:
        synthesis_client = llm_clients[available[0]]

    logger.info(f"Debate Round 3: Synthesis ({synthesis_client.name})")
    synthesis = await _round3_synthesis(synthesis_client, r1, r2, portfolio, as_of)

    # Collect stats
    all_stats = {}
    for name, client in llm_clients.items():
        all_stats[name] = client.get_stats()

    # Store
    await _store_debate(as_of, r1, r2, synthesis, all_stats)

    consensus_count = len(synthesis.get("consensus_trades", []))
    contrarian_count = len(synthesis.get("contrarian_trades", []))

    logger.info(
        f"Debate complete: {consensus_count} consensus + {contrarian_count} contrarian trades"
    )

    return {
        "as_of": as_of.isoformat(),
        "status": "success",
        "participants": [c.name for c in assigned.values()],
        "synthesizer": synthesis_client.name,
        "round1_trades": r1_trades,
        "consensus_trades": consensus_count,
        "contrarian_trades": contrarian_count,
        "portfolio_actions": len(synthesis.get("portfolio_actions", [])),
        "key_conflicts": len(synthesis.get("key_conflicts", [])),
        "regime_consensus": synthesis.get("regime_consensus", ""),
        "top_consensus": [
            {"symbol": t["symbol"], "direction": t["direction"],
             "confidence": t["confidence"]}
            for t in synthesis.get("consensus_trades", [])[:3]
        ],
        "top_contrarian": [
            {"symbol": t["symbol"], "direction": t["direction"],
             "source": t["source_analyst"]}
            for t in synthesis.get("contrarian_trades", [])[:3]
        ],
        "llm_stats": all_stats,
        "synthesis": synthesis,
    }


async def _fetch_all_data(as_of: date) -> Tuple[Dict, Dict, Dict, List]:
    """Fetch all data for the debate (parallel)."""
    macro_data = await _fetch_macro_data(as_of)
    micro_data = await _fetch_micro_data(as_of)
    crowd_data = await _fetch_crowd_data(as_of)
    portfolio = await _fetch_portfolio()
    return macro_data, micro_data, crowd_data, portfolio
