"""FastAPI app for data status and ingestion triggers."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text

from src.db.session import engine
from src.pipelines.data.runner import run_all_pipelines
from src.utils.gpt.challenge import normalize_challenges, run_gpt_challenge
from src.utils.gpt.recommendations import normalize_recommendations, run_gpt_recommendations

UTC = timezone.utc
EASTERN = ZoneInfo("America/New_York")

app = FastAPI(title="Tetra Data Console")

frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

pipeline_state: Dict[str, Any] = {
    "status": "idle",
    "last_run": None,
    "last_error": None,
    "last_result": None,
}

gpt_state: Dict[str, Any] = {
    "status": "idle",
    "last_run": None,
    "last_error": None,
    "last_session": None,
}

gpt_challenge_state: Dict[str, Any] = {
    "status": "idle",
    "last_run": None,
    "last_error": None,
    "last_session": None,
}


class IngestRequest(BaseModel):
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class GPTRefreshRequest(BaseModel):
    session: Optional[str] = None


def _isoformat(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _format_est_timestamp(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, datetime):
        dt_value = value
    elif isinstance(value, date):
        dt_value = datetime.combine(value, datetime.min.time(), tzinfo=UTC)
    else:
        text_value = str(value)
        if text_value.endswith("Z"):
            text_value = text_value[:-1] + "+00:00"
        try:
            dt_value = datetime.fromisoformat(text_value)
        except ValueError:
            return str(value)
    if dt_value.tzinfo is None:
        dt_value = dt_value.replace(tzinfo=UTC)
    est_value = dt_value.astimezone(EASTERN)
    tz_name = est_value.tzname() or "EST"
    return f"{est_value:%Y-%m-%d %H:%M} {tz_name}"


def _redact_error(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    redacted = re.sub(r"(key=)[^&\s]+", r"\1[redacted]", value)
    redacted = re.sub(r"(api_key=)[^&\s]+", r"\1[redacted]", redacted)
    redacted = re.sub(r"(key=\[redacted\])[^\s]*", r"\1", redacted)
    redacted = re.sub(r"(api_key=\[redacted\])[^\s]*", r"\1", redacted)
    return redacted


def _format_category(value: Optional[str]) -> str:
    if not value:
        return "—"
    return " ".join(part.capitalize() for part in value.split("_"))


def _format_action(value: Optional[str]) -> str:
    if not value:
        return "—"
    return str(value).upper()


def _action_class(value: Optional[str]) -> str:
    action = str(value or "").lower()
    if action == "buy":
        return "gpt-action buy"
    if action == "sell":
        return "gpt-action sell"
    return "gpt-action neutral"


def _format_price_value(value: Any) -> str:
    if value is None:
        return "—"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(num) >= 1:
        return f"{num:.2f}"
    return f"{num:.4f}"


def _format_percent(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _format_ratio(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _format_confidence(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _parse_confidence_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        conf = float(value)
    except (TypeError, ValueError):
        text_value = str(value)
        match = re.search(r"\d[\d,]*\.?\d*", text_value)
        if not match:
            return None
        conf = float(match.group(0).replace(",", ""))
        if "%" in text_value:
            conf = conf / 100.0
    if conf > 1:
        conf = conf / 100.0
    return max(0.0, min(1.0, conf))


def _parse_range_mid(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text_value = str(value)
    text_value = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " ", text_value)
    matches = re.findall(r"\d[\d,]*\.?\d*", text_value)
    if not matches:
        return None
    numbers: List[float] = []
    for match in matches:
        try:
            numbers.append(float(match.replace(",", "")))
        except ValueError:
            continue
    if not numbers:
        return None
    return (min(numbers) + max(numbers)) / 2


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def _normalize_consensus_prices(
    action: str,
    last_price: Optional[float],
    entry: Optional[float],
    target: Optional[float],
    stop: Optional[float],
    bounds: Tuple[float, float],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if last_price is None or last_price <= 0:
        return entry, target, stop

    def _within(value: Optional[float]) -> bool:
        if value is None:
            return False
        ratio = value / last_price
        return bounds[0] <= ratio <= bounds[1]

    if not _within(entry):
        entry = last_price

    if not _within(target):
        target = last_price * (0.9 if action == "sell" else 1.1)

    if not _within(stop):
        stop = last_price * (1.05 if action == "sell" else 0.95)

    return entry, target, stop


def _change_class(value: Optional[str]) -> str:
    change = str(value or "").lower()
    if change == "keep":
        return "gpt-change keep"
    if change == "replace":
        return "gpt-change replace"
    return "gpt-change adjust"


def _render_gpt_rows(providers: List[Dict[str, Any]]) -> str:
    rows: List[str] = []
    for provider in providers:
        provider_name = html.escape(str(provider.get("provider") or "—"))
        error = provider.get("error")
        if error:
            rows.append(
                "<div class=\"gpt-row\">"
                f"<span>{provider_name}</span>"
                "<span>—</span>"
                "<span>—</span>"
                "<span>—</span>"
                "<span>—</span>"
                "<span>—</span>"
                "<span>—</span>"
                "<span>—</span>"
                "<span>—</span>"
                f"<span>Error: {html.escape(str(error))}</span>"
                "</div>"
            )
            continue

        categories = provider.get("recommendations") or {}
        for category, items in categories.items():
            for item in items or []:
                action = item.get("action")
                rows.append(
                    "<div class=\"gpt-row\">"
                    f"<span>{provider_name}</span>"
                    f"<span>{html.escape(_format_category(category))}</span>"
                    f"<span>{html.escape(str(item.get('symbol') or '—'))}</span>"
                    f"<span>{html.escape(str(item.get('last_price') or '—'))}</span>"
                    f"<span class=\"{_action_class(action)}\">{html.escape(_format_action(action))}</span>"
                    f"<span>{html.escape(str(item.get('entry') or '—'))}</span>"
                    f"<span>{html.escape(str(item.get('target') or '—'))}</span>"
                    f"<span>{html.escape(str(item.get('stop') or '—'))}</span>"
                    f"<span>{html.escape(str(item.get('horizon') or '—'))}</span>"
                    f"<span>{html.escape(str(item.get('thesis') or ''))}</span>"
                    "</div>"
                )

    if not rows:
        return "<div class=\"gpt-row\"><span>No GPT rows yet.</span></div>"
    return "\n".join(rows)


def _build_gpt_consensus(providers: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    provider_list = [p for p in providers if not p.get("error")]
    provider_count = max(1, len(provider_list))
    consensus_min = 2
    min_per_category = 2
    max_per_category = 3
    category_order = ("large_cap", "growth", "etf", "crypto")
    bounds_map = {
        "large_cap": (0.7, 1.3),
        "growth": (0.6, 1.4),
        "etf": (0.8, 1.2),
        "crypto": (0.4, 2.5),
    }
    aggregated: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for provider in provider_list:
        provider_name = provider.get("provider") or "unknown"
        categories = provider.get("recommendations") or {}
        for category, items in categories.items():
            for item in items or []:
                symbol = str(item.get("symbol") or "").upper()
                action = str(item.get("action") or "").lower()
                if not symbol or not action:
                    continue
                key = (category, symbol, action)
                entry_mid = _parse_range_mid(item.get("entry"))
                target_mid = _parse_range_mid(item.get("target"))
                stop_mid = _parse_range_mid(item.get("stop"))
                last_price = _parse_range_mid(item.get("last_price"))
                data = aggregated.setdefault(
                    key,
                    {
                        "category": category,
                        "symbol": symbol,
                        "action": action,
                        "providers": set(),
                        "entry": [],
                        "target": [],
                        "stop": [],
                        "last_price": [],
                        "confidence": [],
                        "horizon": [],
                        "thesis": [],
                    },
                )
                data["providers"].add(provider_name)
                if entry_mid is not None:
                    data["entry"].append(entry_mid)
                if target_mid is not None:
                    data["target"].append(target_mid)
                if stop_mid is not None:
                    data["stop"].append(stop_mid)
                if last_price is not None:
                    data["last_price"].append(last_price)
                item_conf = _parse_confidence_value(item.get("confidence"))
                if item_conf is not None:
                    data["confidence"].append(item_conf)
                if item.get("horizon"):
                    data["horizon"].append(str(item.get("horizon")))
                if item.get("thesis"):
                    thesis_text = str(item.get("thesis"))
                    if thesis_text:
                        data["thesis"].append(f"{provider_name}: {thesis_text}")

    consensus_candidates: Dict[str, List[Dict[str, Any]]] = {key: [] for key in category_order}
    single_candidates: Dict[str, List[Dict[str, Any]]] = {key: [] for key in category_order}
    action_scores: Dict[Tuple[str, str, str], Tuple[int, float]] = {}
    symbol_support: Dict[Tuple[str, str], int] = {}
    best_action_by_symbol: Dict[Tuple[str, str], str] = {}
    best_action_scores: Dict[Tuple[str, str], Tuple[int, float, str]] = {}
    for key, data in aggregated.items():
        support = len(data["providers"])
        avg_conf = 0.0
        if data["confidence"]:
            avg_conf = sum(data["confidence"]) / len(data["confidence"])
        action_scores[key] = (support, avg_conf)
        symbol_key = (data["category"], data["symbol"])
        symbol_support[symbol_key] = symbol_support.get(symbol_key, 0) + support
        action = data["action"]
        score = (support, avg_conf, action)
        best = best_action_scores.get(symbol_key)
        if best is None or score > best:
            best_action_scores[symbol_key] = score
            best_action_by_symbol[symbol_key] = action
    for data in aggregated.values():
        providers = sorted(data["providers"])
        entry_mid = _median(data["entry"])
        target_mid = _median(data["target"])
        stop_mid = _median(data["stop"])
        last_price = _median(data["last_price"])
        if entry_mid is None or target_mid is None or stop_mid is None:
            continue

        action = data["action"]
        symbol_key = (data["category"], data["symbol"])
        if best_action_by_symbol.get(symbol_key) and best_action_by_symbol[symbol_key] != action:
            continue
        bounds = bounds_map.get(data["category"], (0.6, 1.4))
        entry_mid, target_mid, stop_mid = _normalize_consensus_prices(
            action,
            last_price,
            entry_mid,
            target_mid,
            stop_mid,
            bounds,
        )
        if action == "sell":
            expected = (entry_mid - target_mid) / entry_mid
            risk = (stop_mid - entry_mid) / entry_mid
        else:
            expected = (target_mid - entry_mid) / entry_mid
            risk = (entry_mid - stop_mid) / entry_mid

        reward_risk = expected / risk if risk and risk > 0 else None
        support_count = len(providers)
        support_ratio = support_count / provider_count
        symbol_key = (data["category"], data["symbol"])
        total_support = symbol_support.get(symbol_key) or support_count
        agreement_ratio = support_count / total_support if total_support else 1.0
        if data["confidence"]:
            avg_conf = sum(data["confidence"]) / len(data["confidence"])
            confidence = max(0.0, min(1.0, avg_conf * support_ratio * agreement_ratio))
        else:
            confidence = max(0.0, min(1.0, support_ratio * agreement_ratio))
        thesis_entries: List[str] = []
        for entry in data["thesis"]:
            if entry in thesis_entries:
                continue
            thesis_entries.append(entry)
            if len(thesis_entries) >= 3:
                break
        combined_thesis = " | ".join(thesis_entries) if thesis_entries else None
        row = {
            "category": data["category"],
            "symbol": data["symbol"],
            "action": action,
            "entry": entry_mid,
            "target": target_mid,
            "stop": stop_mid,
            "last_price": last_price,
            "expected_return": expected,
            "reward_risk": reward_risk,
            "confidence": confidence,
            "providers": providers,
            "support": support_count,
            "horizon": data["horizon"][0] if data["horizon"] else None,
            "thesis": combined_thesis,
        }
        if len(providers) >= consensus_min:
            consensus_candidates[row["category"]].append(row)
        else:
            single_candidates[row["category"]].append(row)

    def _sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            rows,
            key=lambda item: (
                item.get("confidence") or 0,
                item.get("support") or 0,
                str(item.get("symbol") or ""),
            ),
            reverse=True,
        )

    rows_by_category: Dict[str, List[Dict[str, Any]]] = {}
    for category in category_order:
        primary = _sort_rows(consensus_candidates.get(category, []))
        secondary = _sort_rows(single_candidates.get(category, []))
        selected: List[Dict[str, Any]] = []
        selected_keys = set()
        for row in primary:
            if len(selected) >= max_per_category:
                break
            key = row.get("symbol")
            selected.append(row)
            selected_keys.add(key)
        if len(selected) < min_per_category:
            for row in secondary:
                if len(selected) >= min_per_category:
                    break
                key = row.get("symbol")
                if key in selected_keys:
                    continue
                selected.append(row)
                selected_keys.add(key)
        if len(selected) < max_per_category:
            for row in primary:
                if len(selected) >= max_per_category:
                    break
                key = row.get("symbol")
                if key in selected_keys:
                    continue
                selected.append(row)
                selected_keys.add(key)
            for row in secondary:
                if len(selected) >= max_per_category:
                    break
                key = row.get("symbol")
                if key in selected_keys:
                    continue
                selected.append(row)
                selected_keys.add(key)

        for idx, row in enumerate(selected, start=1):
            row["rank"] = idx
        rows_by_category[category] = selected

    return rows_by_category


def _render_gpt_consensus_rows(rows_by_category: Dict[str, List[Dict[str, Any]]]) -> str:
    rendered: List[str] = []
    for category in ("large_cap", "growth", "etf", "crypto"):
        rows = rows_by_category.get(category, [])
        rendered.append(
            "<div class=\"gpt-consensus-group\">"
            f"<span>{html.escape(_format_category(category))}</span>"
            "</div>"
        )
        for row in rows:
            providers = ", ".join(row.get("providers") or [])
            thesis = html.escape(str(row.get("thesis") or "No reasoning provided."))
            rendered.append(
                f"<div class=\"gpt-consensus-row\" title=\"{thesis}\">"
                f"<span>{html.escape(str(row.get('rank') or '—'))}</span>"
                f"<span>{html.escape(_format_category(row.get('category')))}</span>"
                f"<span>{html.escape(str(row.get('symbol') or '—'))}</span>"
                f"<span>{html.escape(_format_price_value(row.get('last_price')))}</span>"
                f"<span class=\"{_action_class(row.get('action'))}\">{html.escape(_format_action(row.get('action')))}</span>"
                f"<span>{html.escape(_format_price_value(row.get('entry')))}</span>"
                f"<span>{html.escape(_format_price_value(row.get('target')))}</span>"
                f"<span>{html.escape(_format_price_value(row.get('stop')))}</span>"
                f"<span>{html.escape(_format_percent(row.get('expected_return')))}</span>"
                f"<span>{html.escape(_format_ratio(row.get('reward_risk')))}</span>"
                f"<span>{html.escape(_format_confidence(row.get('confidence')))}</span>"
                f"<span>{html.escape(providers)}</span>"
                "</div>"
            )
    if not rendered:
        return "<div class=\"gpt-consensus-row\"><span>No consensus rows yet.</span></div>"
    return "\n".join(rendered)


def _render_gpt_challenge_rows(providers: List[Dict[str, Any]]) -> str:
    rows: List[str] = []
    for provider in providers:
        provider_name = html.escape(str(provider.get("provider") or "—"))
        error = provider.get("error")
        if error:
            rows.append(
                "<div class=\"gpt-challenge-row\">"
                f"<span>{provider_name}</span>"
                "<span>—</span>"
                "<span>—</span>"
                "<span>—</span>"
                "<span>—</span>"
                "<span>—</span>"
                "<span>—</span>"
                "<span>—</span>"
                "<span>—</span>"
                "<span>—</span>"
                f"<span>Error: {html.escape(str(error))}</span>"
                "</div>"
            )
            continue

        categories = provider.get("recommendations") or {}
        for category, items in categories.items():
            for item in items or []:
                action = item.get("action")
                change = item.get("change")
                rows.append(
                    "<div class=\"gpt-challenge-row\">"
                    f"<span>{provider_name}</span>"
                    f"<span>{html.escape(_format_category(category))}</span>"
                    f"<span>{html.escape(str(item.get('symbol') or '—'))}</span>"
                    f"<span>{html.escape(str(item.get('last_price') or '—'))}</span>"
                    f"<span class=\"{_change_class(change)}\">{html.escape(str(change or 'adjust').upper())}</span>"
                    f"<span class=\"{_action_class(action)}\">{html.escape(_format_action(action))}</span>"
                    f"<span>{html.escape(str(item.get('entry') or '—'))}</span>"
                    f"<span>{html.escape(str(item.get('target') or '—'))}</span>"
                    f"<span>{html.escape(str(item.get('stop') or '—'))}</span>"
                    f"<span>{html.escape(str(item.get('horizon') or '—'))}</span>"
                    f"<span>{html.escape(str(item.get('notes') or item.get('thesis') or ''))}</span>"
                    "</div>"
                )

    if not rows:
        return "<div class=\"gpt-challenge-row\"><span>No GPT challenges yet.</span></div>"
    return "\n".join(rows)


async def _fetch_scalar(query: str) -> Optional[Any]:
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text(query))
            return result.scalar_one_or_none()
    except Exception:
        return None


async def _gather_status() -> Dict[str, Any]:
    counts = {
        "assets": await _fetch_scalar("SELECT COUNT(*) FROM market.assets"),
        "ohlcv": await _fetch_scalar("SELECT COUNT(*) FROM market.ohlcv"),
        "events": await _fetch_scalar("SELECT COUNT(*) FROM event.events"),
        "economic_series": await _fetch_scalar("SELECT COUNT(*) FROM economic.series"),
        "economic_values": await _fetch_scalar("SELECT COUNT(*) FROM economic.values"),
        "news": await _fetch_scalar("SELECT COUNT(*) FROM news.articles"),
    }
    latest = {
        "ohlcv": await _fetch_scalar("SELECT MAX(timestamp) FROM market.ohlcv"),
        "events": await _fetch_scalar("SELECT MAX(event_time) FROM event.events"),
        "economic": await _fetch_scalar("SELECT MAX(timestamp) FROM economic.values"),
        "news": await _fetch_scalar("SELECT MAX(published_at) FROM news.articles"),
    }
    return {
        "counts": counts,
        "latest": latest,
        "pipeline": pipeline_state,
    }


async def _run_ingestion(start: date, end: date) -> None:
    pipeline_state["status"] = "running"
    pipeline_state["last_run"] = datetime.now(tz=UTC).isoformat()
    pipeline_state["last_error"] = None
    pipeline_state["last_result"] = None
    try:
        results = await run_all_pipelines(start=start, end=end)
        pipeline_state["status"] = "success"
        pipeline_state["last_result"] = [asdict(result) for result in results]
    except Exception as exc:
        pipeline_state["status"] = "failed"
        pipeline_state["last_error"] = str(exc)


async def _run_gpt_refresh(session: Optional[str]) -> None:
    gpt_state["status"] = "running"
    gpt_state["last_error"] = None
    gpt_state["last_session"] = session
    try:
        result = await run_gpt_recommendations(session=session)
        gpt_state["status"] = "success"
        gpt_state["last_run"] = result.get("run_time")
        gpt_state["last_session"] = result.get("session")
    except Exception as exc:
        gpt_state["status"] = "failed"
        gpt_state["last_error"] = _redact_error(str(exc))


async def _run_gpt_challenge(session: Optional[str]) -> None:
    gpt_challenge_state["status"] = "running"
    gpt_challenge_state["last_error"] = None
    gpt_challenge_state["last_session"] = session
    try:
        result = await run_gpt_challenge(session=session)
        gpt_challenge_state["status"] = "success"
        gpt_challenge_state["last_run"] = result.get("run_time")
        gpt_challenge_state["last_session"] = result.get("session")
    except Exception as exc:
        gpt_challenge_state["status"] = "failed"
        gpt_challenge_state["last_error"] = _redact_error(str(exc))


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(frontend_dir / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/strats")
async def strats() -> FileResponse:
    return FileResponse(frontend_dir / "strats.html", headers={"Cache-Control": "no-store"})


@app.get("/gpt")
async def gpt() -> HTMLResponse:
    html_template = (frontend_dir / "gpt.html").read_text(encoding="utf-8")
    try:
        consensus = await _load_gpt_consensus(session=None)
        rows = _render_gpt_consensus_rows(consensus.get("by_category", {}))
    except Exception:
        rows = "<div class=\"gpt-consensus-row\"><span>Unable to load GPT data.</span></div>"
        consensus = {}
    try:
        challenge_data = await _load_gpt_challenges(session=None)
        challenge_rows = _render_gpt_challenge_rows(challenge_data.get("providers", []))
    except Exception:
        challenge_rows = "<div class=\"gpt-challenge-row\"><span>Unable to load GPT challenges.</span></div>"
    last_run = _format_est_timestamp(consensus.get("run_time"))
    session = consensus.get("session") or consensus.get("last_session") or "—"
    html_content = (
        html_template.replace("{{GPT_CONSENSUS_ROWS}}", rows)
        .replace("{{GPT_CHALLENGE_ROWS}}", challenge_rows)
        .replace("{{GPT_LAST_RUN}}", html.escape(str(last_run)))
        .replace("{{GPT_SESSION}}", html.escape(str(session)))
    )
    return HTMLResponse(html_content, headers={"Cache-Control": "no-store"})


@app.get("/api/status")
async def get_status() -> Dict[str, Any]:
    return await _gather_status()


@app.post("/api/ingest")
async def trigger_ingest(request: IngestRequest) -> Dict[str, Any]:
    if pipeline_state["status"] == "running":
        raise HTTPException(status_code=409, detail="Ingestion already running")

    today = date.today()
    start = request.start_date or (today - timedelta(days=1))
    end = request.end_date or today

    asyncio.create_task(_run_ingestion(start, end))
    return {"status": "started", "start": start.isoformat(), "end": end.isoformat()}


@app.get("/api/market/coverage")
async def get_market_coverage(limit: int = 500) -> Dict[str, Any]:
    limit = max(1, min(limit, 2000))
    query = text(
        """
        SELECT
          o.symbol,
          a.name,
          COUNT(*) AS rows,
          MIN(o.timestamp) AS start_ts,
          MAX(o.timestamp) AS end_ts,
          ARRAY_AGG(DISTINCT o.source) AS sources
        FROM market.ohlcv o
        LEFT JOIN market.assets a ON a.symbol = o.symbol
        GROUP BY o.symbol, a.name
        ORDER BY o.symbol
        LIMIT :limit
        """
    )
    total_query = text("SELECT COUNT(DISTINCT symbol) FROM market.ohlcv")
    async with engine.begin() as conn:
        total_result = await conn.execute(total_query)
        total_symbols = total_result.scalar_one_or_none() or 0
        result = await conn.execute(query, {"limit": limit})
        rows = result.fetchall()

    coverage: List[Dict[str, Any]] = []
    for row in rows:
        start_ts = row.start_ts
        end_ts = row.end_ts
        span_days = None
        if start_ts and end_ts:
            span_days = max(0, (end_ts - start_ts).days)
        coverage.append(
            {
                "symbol": row.symbol,
                "name": row.name,
                "rows": row.rows,
                "start": _isoformat(start_ts),
                "end": _isoformat(end_ts),
                "days": span_days,
                "sources": [s for s in (row.sources or []) if s],
            }
        )

    return {"total_symbols": total_symbols, "coverage": coverage}


async def _load_gpt_recommendations(session: Optional[str]) -> Dict[str, Any]:
    if session:
        query = text(
            """
            SELECT DISTINCT ON (provider)
              provider,
              session,
              run_time,
              payload,
              raw_text,
              error
            FROM gpt.recommendations
            WHERE session = :session
            ORDER BY provider, run_time DESC
            """
        )
        max_query = text(
            """
            SELECT MAX(run_time)
            FROM gpt.recommendations
            WHERE session = :session
            """
        )
        params = {"session": session}
    else:
        query = text(
            """
            SELECT DISTINCT ON (provider)
              provider,
              session,
              run_time,
              payload,
              raw_text,
              error
            FROM gpt.recommendations
            ORDER BY provider, run_time DESC
            """
        )
        max_query = text("SELECT MAX(run_time) FROM gpt.recommendations")
        params = {}
    try:
        async with engine.begin() as conn:
            result = await conn.execute(query, params)
            rows = result.fetchall()
            max_result = await conn.execute(max_query, params)
            latest_run = max_result.scalar_one_or_none()
    except Exception:
        return {"session": session, "run_time": None, "providers": []}

    providers: List[Dict[str, Any]] = []
    for row in rows:
        payload = row.payload
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = None
        normalized = normalize_recommendations(payload, row.session or session or "pre")
        providers.append(
            {
                "provider": row.provider,
                "session": row.session,
                "run_time": _isoformat(row.run_time),
                "recommendations": normalized.get("categories", {}),
                "error": _redact_error(row.error),
            }
        )

    provider_order = {"openai": 0, "deepseek": 1, "gemini": 2}
    providers.sort(key=lambda item: provider_order.get(item.get("provider"), 99))

    return {
        "session": session,
        "run_time": _isoformat(latest_run),
        "providers": providers,
        "status": gpt_state.get("status"),
        "last_error": gpt_state.get("last_error"),
        "last_session": gpt_state.get("last_session"),
    }


async def _load_gpt_consensus(session: Optional[str]) -> Dict[str, Any]:
    data = await _load_gpt_recommendations(session=session)
    by_category = _build_gpt_consensus(data.get("providers", []))
    consensus = [row for rows in by_category.values() for row in rows]
    return {
        "session": data.get("session"),
        "run_time": data.get("run_time"),
        "consensus": consensus,
        "by_category": by_category,
        "status": data.get("status"),
        "last_error": data.get("last_error"),
        "last_session": data.get("last_session"),
    }


async def _load_gpt_challenges(session: Optional[str]) -> Dict[str, Any]:
    if session:
        query = text(
            """
            SELECT DISTINCT ON (provider)
              provider,
              session,
              run_time,
              source_provider,
              source_run_time,
              payload,
              raw_text,
              error
            FROM gpt.recommendation_challenges
            WHERE session = :session
            ORDER BY provider, run_time DESC
            """
        )
        max_query = text(
            """
            SELECT MAX(run_time)
            FROM gpt.recommendation_challenges
            WHERE session = :session
            """
        )
        params = {"session": session}
    else:
        query = text(
            """
            SELECT DISTINCT ON (provider)
              provider,
              session,
              run_time,
              source_provider,
              source_run_time,
              payload,
              raw_text,
              error
            FROM gpt.recommendation_challenges
            ORDER BY provider, run_time DESC
            """
        )
        max_query = text("SELECT MAX(run_time) FROM gpt.recommendation_challenges")
        params = {}
    try:
        async with engine.begin() as conn:
            result = await conn.execute(query, params)
            rows = result.fetchall()
            max_result = await conn.execute(max_query, params)
            latest_run = max_result.scalar_one_or_none()
    except Exception:
        return {"session": session, "run_time": None, "providers": []}

    providers: List[Dict[str, Any]] = []
    for row in rows:
        payload = row.payload
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = None
        normalized = normalize_challenges(payload, row.session or session or "pre")
        providers.append(
            {
                "provider": row.provider,
                "session": row.session,
                "run_time": _isoformat(row.run_time),
                "source_run_time": _isoformat(row.source_run_time),
                "recommendations": normalized.get("categories", {}),
                "error": _redact_error(row.error),
            }
        )

    provider_order = {"openai": 0, "deepseek": 1, "gemini": 2}
    providers.sort(key=lambda item: provider_order.get(item.get("provider"), 99))

    return {
        "session": session,
        "run_time": _isoformat(latest_run),
        "providers": providers,
        "status": gpt_challenge_state.get("status"),
        "last_error": gpt_challenge_state.get("last_error"),
        "last_session": gpt_challenge_state.get("last_session"),
    }


@app.get("/api/gpt/recommendations")
async def get_gpt_recommendations(session: Optional[str] = None) -> Dict[str, Any]:
    return await _load_gpt_recommendations(session=session)


@app.get("/api/gpt/consensus")
async def get_gpt_consensus(session: Optional[str] = None) -> Dict[str, Any]:
    return await _load_gpt_consensus(session=session)


@app.post("/api/gpt/recommendations/refresh")
async def refresh_gpt_recommendations(request: GPTRefreshRequest) -> Dict[str, Any]:
    session = request.session
    if session and session not in {"pre", "post"}:
        raise HTTPException(status_code=400, detail="session must be pre or post")
    if gpt_state.get("status") == "running":
        return {
            "status": "running",
            "message": "gpt refresh already running",
            "last_run": gpt_state.get("last_run"),
            "last_error": gpt_state.get("last_error"),
        }
    asyncio.create_task(_run_gpt_refresh(session))
    return {"status": "running"}


@app.get("/api/gpt/challenges")
async def get_gpt_challenges(session: Optional[str] = None) -> Dict[str, Any]:
    return await _load_gpt_challenges(session=session)


@app.post("/api/gpt/challenges/refresh")
async def refresh_gpt_challenges(request: GPTRefreshRequest) -> Dict[str, Any]:
    session = request.session
    if session and session not in {"pre", "post"}:
        raise HTTPException(status_code=400, detail="session must be pre or post")
    if gpt_challenge_state.get("status") == "running":
        return {
            "status": "running",
            "message": "gpt challenge already running",
            "last_run": gpt_challenge_state.get("last_run"),
            "last_error": gpt_challenge_state.get("last_error"),
        }
    asyncio.create_task(_run_gpt_challenge(session))
    return {"status": "running"}


@app.get("/api/events/summary")
async def get_events_summary(limit: int = 50) -> Dict[str, Any]:
    limit = max(1, min(limit, 200))
    query = text(
        """
        SELECT
          event_type,
          COUNT(*) AS events,
          COUNT(DISTINCT symbol) AS symbols,
          COUNT(DISTINCT source) AS sources,
          MIN(event_time) AS start_ts,
          MAX(event_time) AS end_ts
        FROM event.events
        GROUP BY event_type
        ORDER BY events DESC, event_type
        LIMIT :limit
        """
    )
    total_events_query = text("SELECT COUNT(*) FROM event.events")
    total_types_query = text("SELECT COUNT(DISTINCT event_type) FROM event.events")
    async with engine.begin() as conn:
        total_events_result = await conn.execute(total_events_query)
        total_types_result = await conn.execute(total_types_query)
        total_events = total_events_result.scalar_one_or_none() or 0
        total_types = total_types_result.scalar_one_or_none() or 0
        result = await conn.execute(query, {"limit": limit})
        rows = result.fetchall()

    summary: List[Dict[str, Any]] = []
    for row in rows:
        summary.append(
            {
                "event_type": row.event_type,
                "events": row.events,
                "symbols": row.symbols,
                "sources": row.sources,
                "start": _isoformat(row.start_ts),
                "end": _isoformat(row.end_ts),
            }
        )

    return {"total_events": total_events, "total_types": total_types, "summary": summary}


@app.get("/api/economic/summary")
async def get_economic_summary(limit: int = 50) -> Dict[str, Any]:
    limit = max(1, min(limit, 200))
    query = text(
        """
        SELECT
          s.series_id,
          s.name,
          s.frequency,
          COUNT(v.timestamp) AS values,
          MIN(v.timestamp) AS start_ts,
          MAX(v.timestamp) AS end_ts
        FROM economic.series s
        LEFT JOIN economic.values v ON v.series_id = s.series_id
        GROUP BY s.series_id, s.name, s.frequency
        ORDER BY values DESC NULLS LAST, s.series_id
        LIMIT :limit
        """
    )
    total_series_query = text("SELECT COUNT(*) FROM economic.series")
    total_values_query = text("SELECT COUNT(*) FROM economic.values")
    async with engine.begin() as conn:
        total_series_result = await conn.execute(total_series_query)
        total_values_result = await conn.execute(total_values_query)
        total_series = total_series_result.scalar_one_or_none() or 0
        total_values = total_values_result.scalar_one_or_none() or 0
        result = await conn.execute(query, {"limit": limit})
        rows = result.fetchall()

    summary: List[Dict[str, Any]] = []
    for row in rows:
        summary.append(
            {
                "series_id": row.series_id,
                "name": row.name,
                "frequency": row.frequency,
                "values": row.values,
                "start": _isoformat(row.start_ts),
                "end": _isoformat(row.end_ts),
            }
        )

    return {"total_series": total_series, "total_values": total_values, "summary": summary}


@app.get("/api/market/ohlcv")
async def get_market_ohlcv(symbol: str, limit: int = 3650) -> Dict[str, Any]:
    symbol = symbol.upper()
    limit = max(1, min(limit, 5000))
    query = text(
        """
        SELECT timestamp, open, high, low, close, volume, source
        FROM market.ohlcv
        WHERE symbol = :symbol
        ORDER BY timestamp DESC
        LIMIT :limit
        """
    )
    async with engine.begin() as conn:
        result = await conn.execute(query, {"symbol": symbol, "limit": limit})
        rows = result.fetchall()

    series = [
        {
            "timestamp": _isoformat(row.timestamp),
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
            "source": row.source,
        }
        for row in reversed(rows)
    ]
    return {"symbol": symbol, "series": series}


@app.get("/api/news/sentiment")
async def get_news_sentiment(limit: int = 50) -> Dict[str, Any]:
    limit = max(1, min(limit, 500))
    symbol_query = text(
        """
        SELECT
          symbol,
          COUNT(*) AS articles,
          AVG(sentiment) AS avg_sentiment,
          MIN(published_at) AS start_ts,
          MAX(published_at) AS end_ts
        FROM news.articles
        CROSS JOIN LATERAL unnest(coalesce(tickers, ARRAY[]::varchar[])) AS symbol
        WHERE sentiment IS NOT NULL
        GROUP BY symbol
        ORDER BY articles DESC
        LIMIT :limit
        """
    )
    macro_query = text(
        """
        SELECT
          topic,
          COUNT(*) AS articles,
          AVG(sentiment) AS avg_sentiment,
          MIN(published_at) AS start_ts,
          MAX(published_at) AS end_ts
        FROM news.articles
        CROSS JOIN LATERAL jsonb_array_elements_text(
          COALESCE(payload->'analysis'->'topics', '[]'::jsonb)
        ) AS topic
        WHERE sentiment IS NOT NULL
        GROUP BY topic
        ORDER BY articles DESC
        LIMIT :limit
        """
    )
    async with engine.begin() as conn:
        symbol_rows = (await conn.execute(symbol_query, {"limit": limit})).fetchall()
        macro_rows = (await conn.execute(macro_query, {"limit": limit})).fetchall()

    symbols = [
        {
            "symbol": row.symbol,
            "articles": row.articles,
            "avg_sentiment": float(row.avg_sentiment) if row.avg_sentiment is not None else None,
            "start": _isoformat(row.start_ts),
            "end": _isoformat(row.end_ts),
        }
        for row in symbol_rows
    ]
    macro = [
        {
            "topic": row.topic,
            "articles": row.articles,
            "avg_sentiment": float(row.avg_sentiment) if row.avg_sentiment is not None else None,
            "start": _isoformat(row.start_ts),
            "end": _isoformat(row.end_ts),
        }
        for row in macro_rows
    ]
    return {"symbols": symbols, "macro": macro}
