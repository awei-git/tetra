"""Push daily market intelligence to Mira — bridge message + briefing artifact.

Per PLAN.md Phase 4:
1. Write Mira-bridge/outbox/ message (200-char summary + PDF path)
2. Write Mira/artifacts/briefings/{date}_market.md (full summary)

Feedback loop:
3. Read tetra/feedback/gaps.jsonl (questions Mira couldn't answer)
4. Clear processed gaps after reading
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
UTC = timezone.utc

MIRA_DIR = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/MtJoy/Mira"
BRIDGE_OUTBOX = MIRA_DIR / "Mira-bridge" / "outbox"
BRIEFINGS_DIR = MIRA_DIR / "artifacts" / "briefings"
TETRA_DIR = Path(__file__).resolve().parents[2]
FEEDBACK_FILE = TETRA_DIR / "feedback" / "gaps.jsonl"


def _generate_summary(report_data: Dict[str, Any]) -> str:
    """Generate a short summary for bridge message (~200 chars)."""
    parts = []

    regime = report_data.get("regime", "")
    if regime:
        parts.append(f"Regime: {regime}")

    portfolio_total = report_data.get("portfolio_total", "")
    daily_ret = report_data.get("daily_return")
    if portfolio_total:
        ret_str = f" ({daily_ret:+.2%})" if daily_ret else ""
        parts.append(f"Portfolio: {portfolio_total}{ret_str}")

    consensus = report_data.get("consensus_trades", [])
    if consensus:
        trades_str = ", ".join(
            f"{t.get('symbol', '?')} {t.get('direction', '?')}"
            for t in consensus[:3]
        )
        parts.append(f"Trades: {trades_str}")

    risks = report_data.get("risk_warnings", [])
    if risks:
        parts.append(f"Risk: {risks[0][:60]}")

    return " | ".join(parts)


def _generate_briefing_md(report_data: Dict[str, Any], as_of: date) -> str:
    """Generate full markdown briefing for Mira artifacts."""
    lines = [f"# Market Report — {as_of.isoformat()}\n"]

    regime = report_data.get("regime", "")
    if regime:
        lines.append(f"**Regime:** {regime}\n")

    portfolio_total = report_data.get("portfolio_total", "")
    if portfolio_total:
        lines.append(f"**Portfolio:** {portfolio_total}")
        daily_ret = report_data.get("daily_return")
        if daily_ret is not None:
            lines.append(f" | Daily: {daily_ret:+.2%}")
        cum_ret = report_data.get("cumulative_return")
        if cum_ret is not None:
            lines.append(f" | Cumulative: {cum_ret:+.2%}")
        lines.append("\n")

    # Consensus trades
    consensus = report_data.get("consensus_trades", [])
    if consensus:
        lines.append("\n## Consensus Trades\n")
        for t in consensus:
            sym = t.get("symbol", "?")
            direction = t.get("direction", "?").upper()
            confidence = t.get("confidence", "?")
            thesis = t.get("combined_thesis", t.get("thesis", ""))
            lines.append(f"- **{sym}** {direction} ({confidence}): {thesis}")

    # Contrarian trades
    contrarian = report_data.get("contrarian_trades", [])
    if contrarian:
        lines.append("\n## Contrarian Trades\n")
        for t in contrarian:
            sym = t.get("symbol", "?")
            direction = t.get("direction", "?").upper()
            source = t.get("source_analyst", "?")
            thesis = t.get("thesis", "")
            lines.append(f"- **{sym}** {direction} (from {source}): {thesis}")

    # Portfolio actions
    actions = report_data.get("portfolio_actions", [])
    if actions:
        lines.append("\n## Portfolio Actions\n")
        for a in actions:
            sym = a.get("symbol", "?")
            action = a.get("action", "?").upper()
            urgency = a.get("urgency", "")
            reason = a.get("reasoning", "")
            lines.append(f"- **{sym}** {action} [{urgency}]: {reason}")

    # Risk warnings
    risks = report_data.get("risk_warnings", [])
    if risks:
        lines.append("\n## Risk Warnings\n")
        for r in risks:
            lines.append(f"- {r}")

    # Scenarios
    scenarios = report_data.get("scenarios", [])
    if scenarios:
        lines.append("\n## Forward Scenarios\n")
        for s in scenarios:
            name = s.get("name", "?")
            prob = s.get("probability", 0)
            desc = s.get("description", "")
            impact = s.get("portfolio_impact", {})
            pnl = impact.get("total_pnl_pct", 0)
            lines.append(f"- **{name}** ({prob:.0%}): {desc} | Portfolio: {pnl:+.1f}%")

    # Previously unanswered questions (feedback loop)
    # Gaps are passed in report_data by generator.py (which reads + clears them)
    user_questions = report_data.get("user_questions", [])
    if user_questions:
        lines.append("\n## Previously Asked (Gaps from Last Briefing)\n")
        for q in user_questions:
            lines.append(f"- {q}")

    # PDF path
    pdf_path = report_data.get("pdf_path", "")
    if pdf_path:
        lines.append(f"\n---\nFull report: `{pdf_path}`")

    return "\n".join(lines)


def push_to_mira(
    report_data: Dict[str, Any],
    as_of: Optional[date] = None,
) -> Dict[str, Any]:
    """Push report summary to Mira bridge + briefings.

    Args:
        report_data: Combined data from report generator + debate + scenarios
        as_of: Report date

    Returns:
        Status dict with paths written.
    """
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()

    results: Dict[str, Any] = {"as_of": as_of.isoformat()}

    # 1. Write bridge outbox message
    try:
        BRIDGE_OUTBOX.mkdir(parents=True, exist_ok=True)
        msg_id = uuid.uuid4().hex[:8]
        ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"agent_{ts}_{msg_id}.json"

        summary = _generate_summary(report_data)
        pdf_path = report_data.get("pdf_path", "")

        message = {
            "id": msg_id,
            "sender": "agent",
            "timestamp": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "type": "text",
            "content": f"Market Report {as_of}\n\n{summary}\n\nPDF: {pdf_path}",
            "thread_id": "",
        }

        msg_path = BRIDGE_OUTBOX / filename
        msg_path.write_text(json.dumps(message, indent=2), encoding="utf-8")
        results["bridge_message"] = str(msg_path)
        logger.info(f"Bridge message written: {msg_path.name}")
    except Exception as e:
        logger.warning(f"Failed to write bridge message: {e}")
        results["bridge_error"] = str(e)

    # 2. Write briefing artifact
    try:
        BRIEFINGS_DIR.mkdir(parents=True, exist_ok=True)
        briefing_md = _generate_briefing_md(report_data, as_of)
        briefing_path = BRIEFINGS_DIR / f"{as_of.isoformat()}_market.md"
        briefing_path.write_text(briefing_md, encoding="utf-8")
        results["briefing"] = str(briefing_path)
        logger.info(f"Briefing written: {briefing_path.name}")
    except Exception as e:
        logger.warning(f"Failed to write briefing: {e}")
        results["briefing_error"] = str(e)

    return results


def read_feedback_gaps() -> List[Dict[str, Any]]:
    """Read unanswered questions from Mira analyst feedback.

    Returns list of gap entries and clears the file.
    Call this at pipeline start to inform LLM analysis.
    """
    if not FEEDBACK_FILE.exists():
        return []

    gaps = []
    try:
        for line in FEEDBACK_FILE.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                gaps.append(json.loads(line))
        # Clear after reading
        FEEDBACK_FILE.write_text("", encoding="utf-8")
        logger.info(f"Read {len(gaps)} feedback gaps from Mira")
    except Exception as e:
        logger.warning(f"Failed to read feedback gaps: {e}")

    return gaps
