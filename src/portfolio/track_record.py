"""Track record analytics — how good are our recommendations?

Computes per-method hit rates, average PnL, and generates a self-critique
that gets fed into the next debate cycle so the LLM can learn from mistakes.
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


async def compute_track_record(
    lookback_days: int = 90,
) -> Dict[str, Any]:
    """Compute recommendation track record analytics.

    Returns per-method stats + recent closed recs for self-critique.
    """
    cutoff = datetime.now(tz=UTC).date() - timedelta(days=lookback_days)

    async with engine.begin() as conn:
        # Per-method stats
        result = await conn.execute(text("""
            SELECT method,
                   COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE status = 'hit_target') AS wins,
                   COUNT(*) FILTER (WHERE status = 'hit_stop') AS losses,
                   COUNT(*) FILTER (WHERE status = 'expired') AS expired,
                   COUNT(*) FILTER (WHERE status = 'open') AS open_count,
                   AVG(realized_pnl) FILTER (WHERE status != 'open') AS avg_pnl,
                   AVG(realized_pnl) FILTER (WHERE status = 'hit_target') AS avg_win,
                   AVG(realized_pnl) FILTER (WHERE status = 'hit_stop') AS avg_loss,
                   AVG(max_favorable) AS avg_max_favorable,
                   AVG(max_adverse) AS avg_max_adverse
            FROM tracker.recommendations
            WHERE created_date >= :cutoff
            GROUP BY method
            ORDER BY method
        """), {"cutoff": cutoff})

        by_method = {}
        for r in result.fetchall():
            closed = r.wins + r.losses + r.expired
            hit_rate = r.wins / closed if closed > 0 else None
            by_method[r.method] = {
                "total": r.total,
                "open": r.open_count,
                "wins": r.wins,
                "losses": r.losses,
                "expired": r.expired,
                "hit_rate": round(hit_rate, 3) if hit_rate is not None else None,
                "avg_pnl": round(float(r.avg_pnl), 4) if r.avg_pnl else None,
                "avg_win": round(float(r.avg_win), 4) if r.avg_win else None,
                "avg_loss": round(float(r.avg_loss), 4) if r.avg_loss else None,
                "avg_max_favorable": round(float(r.avg_max_favorable), 4) if r.avg_max_favorable else None,
                "avg_max_adverse": round(float(r.avg_max_adverse), 4) if r.avg_max_adverse else None,
            }

        # Overall stats
        result = await conn.execute(text("""
            SELECT COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE status = 'hit_target') AS wins,
                   COUNT(*) FILTER (WHERE status != 'open') AS closed,
                   AVG(realized_pnl) FILTER (WHERE status != 'open') AS avg_pnl
            FROM tracker.recommendations
            WHERE created_date >= :cutoff
        """), {"cutoff": cutoff})
        overall = result.fetchone()
        overall_hit_rate = overall.wins / overall.closed if overall.closed > 0 else None

        # Recent closed recs (for self-critique)
        result = await conn.execute(text("""
            SELECT symbol, direction, method, entry_price, closed_price,
                   realized_pnl, status, thesis, risk_factors,
                   created_date, closed_date,
                   max_favorable, max_adverse
            FROM tracker.recommendations
            WHERE status != 'open' AND closed_date >= :cutoff
            ORDER BY closed_date DESC
            LIMIT 20
        """), {"cutoff": cutoff})

        recent_closed = []
        for r in result.fetchall():
            recent_closed.append({
                "symbol": r.symbol,
                "direction": r.direction,
                "method": r.method,
                "entry": float(r.entry_price),
                "exit": float(r.closed_price) if r.closed_price else None,
                "pnl": round(float(r.realized_pnl), 4) if r.realized_pnl else None,
                "status": r.status,
                "thesis": r.thesis,
                "risk": r.risk_factors,
                "opened": r.created_date.isoformat(),
                "closed": r.closed_date.isoformat() if r.closed_date else None,
                "max_favorable": round(float(r.max_favorable), 4) if r.max_favorable else None,
                "max_adverse": round(float(r.max_adverse), 4) if r.max_adverse else None,
            })

        # Direction stats (long vs short)
        result = await conn.execute(text("""
            SELECT direction,
                   COUNT(*) FILTER (WHERE status = 'hit_target') AS wins,
                   COUNT(*) FILTER (WHERE status != 'open') AS closed,
                   AVG(realized_pnl) FILTER (WHERE status != 'open') AS avg_pnl
            FROM tracker.recommendations
            WHERE created_date >= :cutoff
            GROUP BY direction
        """), {"cutoff": cutoff})
        by_direction = {}
        for r in result.fetchall():
            by_direction[r.direction] = {
                "wins": r.wins,
                "closed": r.closed,
                "hit_rate": round(r.wins / r.closed, 3) if r.closed > 0 else None,
                "avg_pnl": round(float(r.avg_pnl), 4) if r.avg_pnl else None,
            }

    return {
        "lookback_days": lookback_days,
        "overall": {
            "total": overall.total,
            "closed": overall.closed,
            "wins": overall.wins,
            "hit_rate": round(overall_hit_rate, 3) if overall_hit_rate is not None else None,
            "avg_pnl": round(float(overall.avg_pnl), 4) if overall.avg_pnl else None,
        },
        "by_method": by_method,
        "by_direction": by_direction,
        "recent_closed": recent_closed,
    }


def format_track_record_for_llm(record: Dict[str, Any]) -> str:
    """Format track record as context for LLM debate/analysis.

    This gets injected into the debate prompt so the LLM can learn from
    past mistakes and adjust its recommendations.
    """
    lines = ["=== TRACK RECORD (self-critique context) ==="]

    overall = record.get("overall", {})
    if overall.get("closed", 0) > 0:
        lines.append(
            f"Overall: {overall['wins']}/{overall['closed']} wins "
            f"({overall['hit_rate']:.0%} hit rate), "
            f"avg PnL: {overall['avg_pnl']:+.2%}"
        )
    else:
        lines.append("No closed recommendations yet.")

    # Per-method breakdown
    by_method = record.get("by_method", {})
    if by_method:
        lines.append("\nBy method:")
        for method, stats in by_method.items():
            if stats.get("wins") is not None and (stats["wins"] + stats["losses"]) > 0:
                lines.append(
                    f"  {method}: {stats['hit_rate']:.0%} hit rate, "
                    f"avg win: {stats['avg_win']:+.2%}, "
                    f"avg loss: {stats['avg_loss']:+.2%}"
                )

    # Recent losses — most important for learning
    recent = record.get("recent_closed", [])
    losses = [r for r in recent if r["status"] in ("hit_stop", "expired") and r.get("pnl", 0) < 0]
    if losses:
        lines.append("\nRecent losses (learn from these):")
        for r in losses[:5]:
            lines.append(
                f"  {r['symbol']} {r['direction']} ({r['method']}): "
                f"{r['pnl']:+.2%} | thesis: {r['thesis'][:80]}"
            )
            if r.get("max_favorable"):
                lines.append(
                    f"    Was up {r['max_favorable']:+.2%} before reversing — "
                    f"consider tighter trailing stop"
                )

    # Recent wins — what worked
    wins = [r for r in recent if r["status"] == "hit_target"]
    if wins:
        lines.append("\nRecent wins:")
        for r in wins[:3]:
            lines.append(
                f"  {r['symbol']} {r['direction']} ({r['method']}): "
                f"{r['pnl']:+.2%} | thesis: {r['thesis'][:80]}"
            )

    return "\n".join(lines)
