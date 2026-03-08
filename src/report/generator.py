"""Report generator — assembles all analysis into report sections.

Fetches data from DB (analysis results, debate, portfolio, signals, events)
and formats it for the Jinja2 template.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from src.db.session import engine
from src.report.delivery import generate_pdf

logger = logging.getLogger(__name__)
UTC = timezone.utc


async def _fetch_market_snapshot(as_of: date) -> Dict[str, Any]:
    """Fetch latest prices for snapshot panels."""
    snapshot: Dict[str, Any] = {}

    async with engine.begin() as conn:
        # Portfolio positions with prices
        result = await conn.execute(text("""
            SELECT p.symbol, p.shares, p.current_price, p.market_value, p.weight, p.unrealized_pnl
            FROM portfolio.positions p
            ORDER BY p.market_value DESC NULLS LAST
        """))
        positions = result.fetchall()
        if positions:
            snapshot["portfolio_positions"] = [
                {
                    "name": r.symbol,
                    "price": f"${float(r.current_price):,.2f}" if r.current_price else "—",
                    "change": f"${float(r.unrealized_pnl):+,.0f}" if r.unrealized_pnl else "—",
                    "change_pct": float(r.unrealized_pnl) if r.unrealized_pnl else 0,
                }
                for r in positions
            ]

        # Major indices from latest OHLCV
        index_symbols = ["SPY", "QQQ", "IWM", "DIA"]
        result = await conn.execute(text("""
            WITH latest AS (
                SELECT symbol, close,
                       LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp) AS prev_close,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS rn
                FROM market.ohlcv
                WHERE symbol = ANY(:symbols) AND timestamp::date >= :start
            )
            SELECT symbol, close, prev_close FROM latest WHERE rn = 1
        """), {"symbols": index_symbols, "start": as_of - timedelta(days=7)})
        rows = result.fetchall()
        if rows:
            snapshot["indices"] = [
                {
                    "name": r.symbol,
                    "price": f"${float(r.close):,.2f}",
                    "change": f"{(float(r.close)/float(r.prev_close)-1)*100:+.2f}%" if r.prev_close else "—",
                    "change_pct": (float(r.close)/float(r.prev_close)-1) if r.prev_close else 0,
                }
                for r in rows
            ]

        # Rates from FRED — use short display names
        fred_labels = {
            "VIXCLS": "VIX",
            "DGS10": "10Y Treasury",
            "DGS2": "2Y Treasury",
            "T10Y2Y": "10Y-2Y Spread",
            "BAMLH0A0HYM2": "HY Credit Spread",
        }
        fred_ids = list(fred_labels.keys())
        result = await conn.execute(text("""
            SELECT s.series_id, v.value
            FROM economic.values v
            JOIN economic.series s ON v.series_id = s.series_id
            WHERE s.series_id = ANY(:ids)
            ORDER BY v.timestamp DESC
        """), {"ids": fred_ids})
        rates_rows = result.fetchall()
        seen_rates: set = set()
        rates = []
        for r in rates_rows:
            sid = r.series_id
            if sid not in seen_rates:
                seen_rates.add(sid)
                rates.append({
                    "name": fred_labels.get(sid, sid),
                    "price": f"{float(r.value):.2f}",
                })
        if rates:
            snapshot["rates"] = rates

    return snapshot


async def _fetch_narrative_state(as_of: date) -> Optional[Dict[str, Any]]:
    """Fetch narrative analysis results."""
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT dominant_narrative, narrative_shift, shift_magnitude,
                   counter_narrative, novelty, raw_analysis
            FROM narrative.daily_state
            WHERE date = :date AND scope = 'market'
            LIMIT 1
        """), {"date": as_of})
        row = result.fetchone()

    if not row:
        return None

    return {
        "dominant_narrative": row.dominant_narrative,
        "narrative_shift": round(float(row.narrative_shift), 3) if row.narrative_shift else 0,
        "shift_magnitude": round(float(row.shift_magnitude), 3) if row.shift_magnitude else 0,
        "counter_narrative": row.counter_narrative,
    }


async def _fetch_debate_results(as_of: date) -> Optional[Dict[str, Any]]:
    """Fetch the latest debate synthesis."""
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT payload FROM gpt.recommendations
            WHERE provider = 'debate' AND session = :session
            ORDER BY run_time DESC LIMIT 1
        """), {"session": as_of.strftime("%Y%m%d")})
        row = result.fetchone()

    if not row:
        return None

    payload = row.payload
    if isinstance(payload, str):
        payload = json.loads(payload)

    return payload


async def _fetch_portfolio_state(as_of: date) -> Dict[str, Any]:
    """Fetch portfolio positions and snapshot."""
    result_data: Dict[str, Any] = {}

    async with engine.begin() as conn:
        # Positions
        result = await conn.execute(text("""
            SELECT symbol, shares, avg_cost, current_price, market_value, weight, unrealized_pnl
            FROM portfolio.positions
            ORDER BY market_value DESC NULLS LAST
        """))
        positions = result.fetchall()
        result_data["positions"] = [
            {
                "symbol": r.symbol,
                "price": float(r.current_price) if r.current_price else 0,
                "value": float(r.market_value) if r.market_value else 0,
                "weight": round(float(r.weight) * 100, 1) if r.weight else 0,
                "pnl": float(r.unrealized_pnl) if r.unrealized_pnl else 0,
            }
            for r in positions
        ]

        # Latest snapshot
        result = await conn.execute(text("""
            SELECT total_value, cash, daily_return, cumulative_return
            FROM portfolio.snapshots
            WHERE date <= :date ORDER BY date DESC LIMIT 1
        """), {"date": as_of})
        snap = result.fetchone()
        if snap:
            result_data["summary"] = {
                "total_value": float(snap.total_value),
                "cash": float(snap.cash),
                "daily_return": float(snap.daily_return),
                "cumulative_return": float(snap.cumulative_return),
            }

    return result_data


async def _fetch_track_record() -> Dict[str, Any]:
    """Fetch recommendation track record."""
    track: Dict[str, Any] = {}

    async with engine.begin() as conn:
        # Summary counts
        result = await conn.execute(text("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE status != 'open') AS closed,
                COUNT(*) FILTER (WHERE status = 'hit_target') AS hit_target,
                COUNT(*) FILTER (WHERE status = 'hit_stop') AS hit_stop,
                AVG(realized_pnl) FILTER (WHERE status != 'open') AS avg_pnl
            FROM tracker.recommendations
        """))
        row = result.fetchone()
        if row and row.total > 0:
            track["summary"] = {
                "total": row.total,
                "closed": row.closed or 0,
                "hit_target": row.hit_target or 0,
                "hit_stop": row.hit_stop or 0,
                "avg_pnl": float(row.avg_pnl) if row.avg_pnl else 0,
            }

        # Recent closed recs
        result = await conn.execute(text("""
            SELECT symbol, direction, method, entry_price, closed_price,
                   realized_pnl, status, closed_date
            FROM tracker.recommendations
            WHERE status != 'open'
            ORDER BY closed_date DESC
            LIMIT 10
        """))
        closed = result.fetchall()
        if closed:
            track["recent_closed"] = [
                {
                    "symbol": r.symbol,
                    "direction": r.direction,
                    "method": r.method,
                    "entry_price": float(r.entry_price),
                    "closed_price": float(r.closed_price) if r.closed_price else 0,
                    "realized_pnl": float(r.realized_pnl) if r.realized_pnl else 0,
                    "status": r.status,
                }
                for r in closed
            ]

    return track


async def _fetch_signals_summary(as_of: date) -> List[Dict[str, Any]]:
    """Summarize active signals by source."""
    summaries = []

    async with engine.begin() as conn:
        # Informed trading signals by type
        result = await conn.execute(text("""
            SELECT signal_type,
                   COUNT(*) AS cnt,
                   MAX(symbol) FILTER (WHERE ABS(strength) = (
                       SELECT MAX(ABS(s2.strength)) FROM signals.informed_trading s2
                       WHERE s2.date = :date AND s2.signal_type = signals.informed_trading.signal_type
                   )) AS top_symbol,
                   MAX(ABS(strength)) AS top_strength
            FROM signals.informed_trading
            WHERE date = :date
            GROUP BY signal_type
            ORDER BY MAX(ABS(strength)) DESC
        """), {"date": as_of})
        for r in result.fetchall():
            summaries.append({
                "source": r.signal_type,
                "count": r.cnt,
                "top_symbol": r.top_symbol,
                "top_strength": round(float(r.top_strength), 3) if r.top_strength else None,
            })

        # Unified signals
        result = await conn.execute(text("""
            SELECT COUNT(*) AS cnt,
                   MAX(symbol) AS top_symbol,
                   MAX(ABS(signal_score)) AS top_strength
            FROM signals.unified
            WHERE date = :date
        """), {"date": as_of})
        row = result.fetchone()
        if row and row.cnt > 0:
            summaries.append({
                "source": "unified_meta",
                "count": row.cnt,
                "top_symbol": row.top_symbol,
                "top_strength": round(float(row.top_strength), 3) if row.top_strength else None,
            })

    return summaries


async def _fetch_forward_events(as_of: date, days: int = 5) -> List[Dict[str, Any]]:
    """Fetch upcoming events."""
    end = as_of + timedelta(days=days)
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT symbol, event_type, event_time
            FROM event.events
            WHERE event_time::date > :start AND event_time::date <= :end
            ORDER BY event_time ASC
            LIMIT 20
        """), {"start": as_of, "end": end})
        rows = result.fetchall()

    events = []
    for r in rows:
        events.append({
            "date": r.event_time.strftime("%Y-%m-%d") if hasattr(r.event_time, "strftime") else str(r.event_time),
            "event": r.event_type,
            "symbol": r.symbol,
            "impact": "earnings" if r.event_type == "earnings" else "event",
        })
    return events


async def _generate_llm_commentary(
    llm_client,
    section: str,
    context: Dict[str, Any],
    as_of: date,
) -> str:
    """Use LLM to write a section of the report."""
    prompts = {
        "what_happened": f"""Write a concise market summary for {as_of.isoformat()}.

Context data:
{json.dumps(context, indent=2, default=str)}

Write 2-3 paragraphs in markdown covering:
1. What happened in the market today — the essential story, not just numbers
2. Any narrative shifts or anomalies detected
3. What was unusual or notable
{chr(10) + "The user previously asked these questions that weren't covered. Try to address them if relevant:" + chr(10) + chr(10).join("- " + q for q in context.get("user_questions", [])) if context.get("user_questions") else ""}

Be specific with data references. No fluff. No disclaimers.""",

        "what_it_means": f"""Analyze what the current market state means for positioning.

Context data:
{json.dumps(context, indent=2, default=str)}

Write 2-3 paragraphs in markdown covering:
1. Regime assessment — what kind of market are we in?
2. Cross-asset signals — where are the divergences?
3. Which signals should be weighted higher right now and why?

Be specific and decisive. Reference actual data points.""",
    }

    prompt = prompts.get(section, f"Write analysis for: {section}")
    system = (
        "You are a quantitative portfolio strategist writing a daily market report. "
        "Be concise, data-driven, and decisive. Use markdown formatting. "
        "No disclaimers, no 'it remains to be seen' hedging."
    )

    try:
        return await llm_client.generate(prompt, system=system, temperature=0.3)
    except Exception as e:
        logger.warning(f"LLM commentary failed for {section}: {e}")
        return f"*Analysis unavailable: {e}*"


async def generate_report(
    as_of: Optional[date] = None,
    llm_client=None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate the daily market report.

    1. Fetch all analysis results from DB
    2. Optionally use LLM for narrative sections
    3. Assemble template context
    4. Generate PDF

    Returns dict with report path and summary.
    """
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()

    logger.info(f"Generating report for {as_of}")

    # Fetch all data in parallel-safe order
    market_snapshot = await _fetch_market_snapshot(as_of)
    narrative_state = await _fetch_narrative_state(as_of)
    debate_payload = await _fetch_debate_results(as_of)
    portfolio_data = await _fetch_portfolio_state(as_of)
    track_record = await _fetch_track_record()
    signals_summary = await _fetch_signals_summary(as_of)
    forward_events = await _fetch_forward_events(as_of)

    # Extract debate synthesis
    synthesis = debate_payload.get("synthesis", {}) if debate_payload else {}
    debate_r1 = debate_payload.get("round1", {}) if debate_payload else {}

    # Read feedback gaps from Mira (questions the briefing couldn't answer)
    from src.mira.push import read_feedback_gaps
    user_gaps = read_feedback_gaps()
    if user_gaps:
        logger.info(f"Addressing {len(user_gaps)} user feedback gaps")

    # Build LLM commentary context
    commentary_context = {
        "narrative": narrative_state,
        "regime": synthesis.get("regime_consensus", ""),
        "signals_count": len(signals_summary),
        "portfolio_summary": portfolio_data.get("summary"),
        "top_signals": signals_summary[:5],
        "user_questions": [g["question"] for g in user_gaps] if user_gaps else [],
    }

    # Generate narrative sections
    what_happened = ""
    what_it_means = ""

    if llm_client:
        what_happened = await _generate_llm_commentary(
            llm_client, "what_happened", commentary_context, as_of,
        )
        what_it_means = await _generate_llm_commentary(
            llm_client, "what_it_means", {
                **commentary_context,
                "debate_regime": synthesis.get("regime_consensus"),
                "debate_conflicts": synthesis.get("key_conflicts", []),
                "risk_warnings": synthesis.get("risk_warnings", []),
            }, as_of,
        )
    else:
        # Fallback: use debate regime consensus as what_happened
        if synthesis.get("regime_consensus"):
            what_happened = f"**Regime:** {synthesis['regime_consensus']}"
        what_it_means = ""

    # Debate participants — reconstruct role→provider mapping
    debate_participants = None
    llm_stats = debate_payload.get("llm_stats", {}) if debate_payload else {}
    if debate_r1 and llm_stats:
        # Use same preference order as debate.py to reconstruct assignment
        available = list(llm_stats.keys())
        role_preference = {
            "macro": ["openai", "claude", "deepseek", "gemini"],
            "micro": ["deepseek", "openai", "gemini", "claude"],
            "crowd": ["gemini", "deepseek", "openai", "claude"],
        }
        debate_participants = {}
        used = set()
        for role, prefs in role_preference.items():
            for pref in prefs:
                if pref in available and pref not in used:
                    debate_participants[role] = pref
                    used.add(pref)
                    break
            if role not in debate_participants:
                debate_participants[role] = available[0] if available else role

    # Portfolio total for cover
    portfolio_total = None
    if portfolio_data.get("summary"):
        portfolio_total = f"${portfolio_data['summary']['total_value']:,.0f}"

    # Regime for cover
    regime = synthesis.get("regime_consensus", "") or None
    if regime and len(regime) > 30:
        regime = regime[:30] + "..."

    # Assemble template context
    sections = {
        # Cover
        "regime": regime,
        "portfolio_total": portfolio_total,

        # Section 1: What Happened
        "market_snapshot": market_snapshot,
        "what_happened": what_happened,
        "narrative_state": narrative_state,

        # Section 2: What It Means
        "what_it_means": what_it_means,
        "regime_detail": None,  # From meta-signal if available
        "signals_summary": signals_summary,
        "debate_conflicts": synthesis.get("key_conflicts", []),

        # Section 3: What To Do
        "consensus_trades": synthesis.get("consensus_trades", []),
        "contrarian_trades": synthesis.get("contrarian_trades", []),
        "portfolio_actions": synthesis.get("portfolio_actions", []),
        "risk_warnings": synthesis.get("risk_warnings", []),

        # Section 4: Portfolio & Track Record
        "portfolio_positions": portfolio_data.get("positions", []),
        "portfolio_summary": portfolio_data.get("summary"),
        "track_record": track_record,

        # Section 5: Forward Calendar
        "forward_events": forward_events,

        # Appendix
        "debate_participants": debate_participants,
    }

    # Generate PDF
    pdf_path = await generate_pdf(sections, output_path=output_path)

    # Store report metadata
    await _store_report_metadata(as_of, pdf_path, sections)

    logger.info(f"Report generated: {pdf_path}")

    return {
        "as_of": as_of.isoformat(),
        "status": "success",
        "pdf_path": pdf_path,
        "regime": regime,
        "portfolio_total": portfolio_total,
        "consensus_trades": synthesis.get("consensus_trades", []),
        "contrarian_trades": synthesis.get("contrarian_trades", []),
        "portfolio_actions": synthesis.get("portfolio_actions", []),
        "risk_warnings": synthesis.get("risk_warnings", []),
        "portfolio_positions": len(portfolio_data.get("positions", [])),
        "signals": len(signals_summary),
        "forward_events": len(forward_events),
        "user_questions": [g["question"] for g in user_gaps] if user_gaps else [],
    }


async def _store_report_metadata(
    as_of: date, pdf_path: str, sections: Dict[str, Any],
) -> None:
    """Store report metadata in DB."""
    summary = {
        "regime": sections.get("regime", ""),
        "consensus_trades": len(sections.get("consensus_trades", [])),
        "contrarian_trades": len(sections.get("contrarian_trades", [])),
        "portfolio_total": sections.get("portfolio_total", ""),
        "signals": len(sections.get("signals_summary", [])),
    }

    try:
        async with engine.begin() as conn:
            await conn.execute(text("""
                INSERT INTO report.daily
                  (date, pdf_path, summary, regime, new_recommendations, active_recommendations)
                VALUES (:date, :pdf_path, :summary, :regime, :new_recs, :active_recs)
                ON CONFLICT (date) DO UPDATE SET
                  pdf_path = EXCLUDED.pdf_path,
                  summary = EXCLUDED.summary,
                  regime = EXCLUDED.regime,
                  new_recommendations = EXCLUDED.new_recommendations,
                  active_recommendations = EXCLUDED.active_recommendations
            """), {
                "date": as_of,
                "pdf_path": pdf_path,
                "summary": json.dumps(summary),
                "regime": sections.get("regime", ""),
                "new_recs": summary["consensus_trades"] + summary["contrarian_trades"],
                "active_recs": len(sections.get("portfolio_positions", [])),
            })
    except Exception as e:
        # Non-critical — don't fail the report
        logger.warning(f"Failed to store report metadata: {e}")
