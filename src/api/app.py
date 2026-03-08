"""FastAPI app for data status and ingestion triggers."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
import html
import json
import re
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text

from src.db.session import engine
from src.pipelines.data.runner import run_all_pipelines
from src.pipelines.factors.daily import run_daily_factors
from src.definitions.market_universe import MarketUniverse
from src.utils.factors.definitions import get_factor_definitions
from src.utils.factors.scoring import (
    action_from_signal,
    build_factor_stats,
    compute_factor_signal,
    score_factor_rows,
    score_symbol_values,
)
from src.utils.gpt.challenge import normalize_challenges, run_gpt_challenge
from src.utils.gpt.factor_review import (
    build_factor_review_consensus,
    normalize_factor_reviews,
    run_gpt_factor_reviews,
)
from src.utils.gpt.recommendations import normalize_recommendations, run_gpt_recommendations
from src.utils.gpt.summary import generate_summary
from src.utils.simulations.paths import (
    STRESS_WINDOWS,
    compute_log_returns,
    generate_historical_paths,
    generate_monte_carlo_paths,
    generate_stress_paths,
    list_stress_windows,
    summarize_paths,
)

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

gpt_factor_state: Dict[str, Any] = {
    "status": "idle",
    "last_run": None,
    "last_error": None,
    "last_session": None,
}

factor_state: Dict[str, Any] = {
    "status": "idle",
    "last_run": None,
    "last_error": None,
    "last_as_of": None,
}


class IngestRequest(BaseModel):
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class GPTRefreshRequest(BaseModel):
    session: Optional[str] = None


class GPTFactorReviewRequest(BaseModel):
    session: Optional[str] = None
    as_of: Optional[date] = None


class FactorRefreshRequest(BaseModel):
    as_of: Optional[date] = None


class FactorSelectionRequest(BaseModel):
    symbols: List[str]
    as_of: Optional[date] = None


def _isoformat(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)



def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_horizon_days(value: Optional[str]) -> int:
    if not value:
        return 20
    text_value = str(value).lower()
    matches = re.findall(r"\d+\.?\d*", text_value)
    if not matches:
        return 20
    numbers = [float(match) for match in matches]
    avg = sum(numbers) / len(numbers)
    if "year" in text_value:
        return max(1, int(round(avg * 252)))
    if "month" in text_value:
        return max(1, int(round(avg * 21)))
    if "week" in text_value:
        return max(1, int(round(avg * 5)))
    if "day" in text_value:
        return max(1, int(round(avg)))
    return max(1, int(round(avg)))


async def _fetch_close_series(
    symbol: str,
    limit: Optional[int] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> List[Tuple[datetime, float]]:
    clauses = ["symbol = :symbol", "close IS NOT NULL"]
    params: Dict[str, Any] = {"symbol": symbol}
    order_desc = limit is not None and start is None and end is None
    if start is not None:
        clauses.append("timestamp >= :start")
        params["start"] = start
    if end is not None:
        clauses.append("timestamp <= :end")
        params["end"] = end
    limit_sql = ""
    if limit is not None:
        limit_sql = "LIMIT :limit"
        params["limit"] = limit
    order_sql = "ORDER BY timestamp DESC" if order_desc else "ORDER BY timestamp ASC"
    query = text(
        f"""
        SELECT timestamp, close
        FROM market.ohlcv
        WHERE {" AND ".join(clauses)}
        {order_sql}
        {limit_sql}
        """
    )
    async with engine.begin() as conn:
        result = await conn.execute(query, params)
        rows = result.fetchall()
    if order_desc:
        rows = list(reversed(rows))
    return [(row.timestamp, float(row.close)) for row in rows]


async def _estimate_base_profit_prob(
    symbol: str,
    action: str,
    horizon_days: int,
) -> Tuple[Optional[float], int]:
    horizon_days = max(1, min(horizon_days, 252))
    limit = min(1200, max(240, horizon_days * 8))
    series = await _fetch_close_series(symbol, limit=limit)
    prices = [price for _, price in series]
    if len(prices) <= horizon_days:
        return None, 0
    wins = 0
    total = 0
    for idx in range(len(prices) - horizon_days):
        entry = prices[idx]
        exit_price = prices[idx + horizon_days]
        if entry <= 0:
            continue
        ret = (exit_price - entry) / entry
        if action == "sell":
            ret = -ret
        if ret > 0:
            wins += 1
        total += 1
    if total == 0:
        return None, 0
    return wins / total, total


async def _build_return_series(symbol: str, lookback_days: int) -> Dict[date, float]:
    lookback_days = max(30, min(lookback_days, 252))
    limit = min(1400, lookback_days + 120)
    series = await _fetch_close_series(symbol, limit=limit)
    close_by_day: Dict[date, float] = {}
    for timestamp, close in series:
        close_by_day[timestamp.date()] = close
    dates = sorted(close_by_day.keys())
    returns: Dict[date, float] = {}
    for idx in range(1, len(dates)):
        prev_close = close_by_day[dates[idx - 1]]
        cur_close = close_by_day[dates[idx]]
        if prev_close <= 0:
            continue
        returns[dates[idx]] = (cur_close - prev_close) / prev_close
    if len(returns) > lookback_days:
        trimmed_dates = sorted(returns.keys())[-lookback_days:]
        returns = {d: returns[d] for d in trimmed_dates}
    return returns


def _compute_correlation(
    left: Dict[date, float],
    right: Dict[date, float],
    min_obs: int,
) -> Optional[float]:
    common = sorted(set(left.keys()) & set(right.keys()))
    if len(common) < min_obs:
        return None
    left_vals = [left[d] for d in common]
    right_vals = [right[d] for d in common]
    mean_left = sum(left_vals) / len(left_vals)
    mean_right = sum(right_vals) / len(right_vals)
    cov = sum((x - mean_left) * (y - mean_right) for x, y in zip(left_vals, right_vals))
    var_left = sum((x - mean_left) ** 2 for x in left_vals)
    var_right = sum((y - mean_right) ** 2 for y in right_vals)
    if var_left <= 0 or var_right <= 0:
        return None
    return cov / (var_left ** 0.5 * var_right ** 0.5)


async def _apply_diversification(
    candidates: List[Dict[str, Any]],
    lookback_days: int,
    corr_threshold: float,
    min_obs: int,
    max_count: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    symbols = [row.get("symbol") for row in candidates if row.get("symbol")]
    unique_symbols = sorted({str(sym) for sym in symbols if sym})
    returns_map: Dict[str, Dict[date, float]] = {}
    for symbol in unique_symbols:
        try:
            returns_map[symbol] = await _build_return_series(symbol, lookback_days)
        except Exception:
            returns_map[symbol] = {}

    selected: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for row in candidates:
        symbol = str(row.get("symbol") or "")
        max_corr = None
        blocked_by = None
        for chosen in selected:
            chosen_symbol = str(chosen.get("symbol") or "")
            corr = _compute_correlation(
                returns_map.get(symbol, {}),
                returns_map.get(chosen_symbol, {}),
                min_obs,
            )
            if corr is None:
                continue
            corr_abs = abs(corr)
            if max_corr is None or corr_abs > max_corr:
                max_corr = corr_abs
            if corr_abs >= corr_threshold:
                blocked_by = chosen_symbol
                max_corr = corr_abs
                break
        if blocked_by:
            skipped.append(
                {
                    "symbol": symbol,
                    "category": row.get("category"),
                    "final_action": row.get("final_action"),
                    "blocked_by": blocked_by,
                    "max_corr": max_corr,
                }
            )
            continue
        selected.append(row)
        if max_count and len(selected) >= max_count:
            break
    meta = {
        "lookback_days": lookback_days,
        "corr_threshold": corr_threshold,
        "min_obs": min_obs,
        "selected": len(selected),
        "skipped": len(skipped),
    }
    return selected, skipped, meta


def _percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    pct = max(0.0, min(1.0, pct))
    idx = pct * (len(sorted_values) - 1)
    lower = int(idx)
    upper = min(len(sorted_values) - 1, lower + 1)
    if lower == upper:
        return sorted_values[lower]
    weight = idx - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


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
    if change in {"replace", "exit"}:
        return "gpt-change replace"
    return "gpt-change adjust"


def _verdict_class(value: Optional[str]) -> str:
    verdict = str(value or "").lower()
    if verdict == "approve":
        return "gpt-verdict approve"
    if verdict == "reject":
        return "gpt-verdict reject"
    return "gpt-verdict watch"


def _majority(values: List[str], fallback: str) -> Tuple[str, int]:
    if not values:
        return fallback, 0
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    top_value = max(counts, key=counts.get)
    top_count = counts[top_value]
    ties = [value for value, count in counts.items() if count == top_count]
    if len(ties) > 1:
        return fallback, top_count
    return top_value, top_count



def _build_gpt_consensus(providers: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    provider_list = [p for p in providers if not p.get("error")]
    provider_count = max(1, len(provider_list))
    consensus_min = 2
    min_per_category = 5
    max_per_category = 5
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





def _summarize_challenges(providers: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for provider in providers:
        provider_name = provider.get("provider") or "unknown"
        if provider.get("error"):
            continue
        categories = provider.get("recommendations") or {}
        for _, items in categories.items():
            for item in items or []:
                symbol = str(item.get("symbol") or "").upper()
                if not symbol:
                    continue
                entry = summary.setdefault(
                    symbol,
                    {"changes": [], "replacements": [], "notes": [], "providers": set()},
                )
                change = str(item.get("change") or "adjust").lower()
                entry["changes"].append(change)
                replacement = item.get("replaces") or item.get("replacement")
                if replacement:
                    entry["replacements"].append(str(replacement).upper())
                notes = item.get("notes") or item.get("thesis")
                if notes:
                    entry["notes"].append(f"{provider_name}: {notes}")
                entry["providers"].add(provider_name)

    summarized: Dict[str, Dict[str, Any]] = {}
    for symbol, entry in summary.items():
        change, _ = _majority(entry["changes"], fallback="adjust")
        replacement, _ = _majority(entry["replacements"], fallback="")
        summarized[symbol] = {
            "change": change,
            "replacement": replacement or None,
            "notes": " | ".join(entry["notes"][:2]) if entry["notes"] else "",
            "providers": sorted(entry["providers"]),
        }
    return summarized


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


async def _run_gpt_factor_review(session: Optional[str], as_of: Optional[date]) -> None:
    gpt_factor_state["status"] = "running"
    gpt_factor_state["last_error"] = None
    gpt_factor_state["last_session"] = session
    try:
        result = await run_gpt_factor_reviews(session=session, as_of=as_of)
        gpt_factor_state["status"] = "success"
        gpt_factor_state["last_run"] = result.get("run_time")
        gpt_factor_state["last_session"] = result.get("session")
    except Exception as exc:
        gpt_factor_state["status"] = "failed"
        gpt_factor_state["last_error"] = _redact_error(str(exc))


async def _run_factor_refresh(as_of: Optional[date]) -> None:
    factor_state["status"] = "running"
    factor_state["last_error"] = None
    factor_state["last_as_of"] = as_of
    try:
        result = await run_daily_factors(as_of=as_of)
        factor_state["status"] = "success"
        factor_state["last_run"] = result.get("run_time")
        factor_state["last_as_of"] = result.get("as_of")
        factor_state["last_as_of"] = result.get("as_of")
    except Exception as exc:
        factor_state["status"] = "failed"
        factor_state["last_error"] = _redact_error(str(exc))


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(frontend_dir / "index.html", headers={"Cache-Control": "no-store"})

@app.get("/analysis")
async def analysis() -> FileResponse:
    return FileResponse(frontend_dir / "analysis.html", headers={"Cache-Control": "no-store"})


@app.get("/strats")
async def strats() -> FileResponse:
    return FileResponse(frontend_dir / "strats.html", headers={"Cache-Control": "no-store"})


@app.get("/alpha")
async def alpha() -> RedirectResponse:
    return RedirectResponse(url="/analysis#factor")


@app.get("/gpt")
async def gpt() -> RedirectResponse:
    return RedirectResponse(url="/analysis#gpt")


@app.get("/inference")
async def inference() -> RedirectResponse:
    return RedirectResponse(url="/analysis#inference")


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


async def _load_gpt_recommendations_history(
    session: Optional[str],
    provider: Optional[str],
    start: Optional[datetime],
    end: Optional[datetime],
    limit: int,
) -> Dict[str, Any]:
    clauses: List[str] = []
    params: Dict[str, Any] = {"limit": limit}
    if session:
        clauses.append("session = :session")
        params["session"] = session
    if provider:
        clauses.append("provider = :provider")
        params["provider"] = provider
    if start:
        clauses.append("run_time >= :start")
        params["start"] = start
    if end:
        clauses.append("run_time <= :end")
        params["end"] = end

    where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = text(
        f"""
        SELECT provider, session, run_time, payload, raw_text, error
        FROM gpt.recommendations
        {where_clause}
        ORDER BY run_time DESC, provider
        LIMIT :limit
        """
    )
    try:
        async with engine.begin() as conn:
            result = await conn.execute(query, params)
            rows = result.fetchall()
    except Exception:
        return {"session": session, "provider": provider, "runs": []}

    runs: List[Dict[str, Any]] = []
    for row in rows:
        payload = row.payload
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = None
        normalized = normalize_recommendations(payload, row.session or session or "pre")
        runs.append(
            {
                "provider": row.provider,
                "session": row.session,
                "run_time": _isoformat(row.run_time),
                "recommendations": normalized.get("categories", {}),
                "payload": payload,
                "raw_text": row.raw_text,
                "error": _redact_error(row.error),
            }
        )
    return {"session": session, "provider": provider, "runs": runs}


async def _load_gpt_consensus(session: Optional[str]) -> Dict[str, Any]:
    data = await _load_gpt_recommendations(session=session)
    by_category = _build_gpt_consensus(data.get("providers", []))
    consensus = [row for rows in by_category.values() for row in rows]
    return {
        "session": data.get("session"),
        "run_time": data.get("run_time"),
        "consensus": consensus,
        "by_category": by_category,
        "providers": data.get("providers", []),
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


async def _load_gpt_factor_reviews(session: Optional[str]) -> Dict[str, Any]:
    if session:
        query = text(
            """
            SELECT DISTINCT ON (provider)
              provider,
              session,
              run_time,
              as_of,
              payload,
              raw_text,
              error
            FROM gpt.factor_reviews
            WHERE session = :session
            ORDER BY provider, run_time DESC
            """
        )
        max_query = text(
            """
            SELECT MAX(run_time)
            FROM gpt.factor_reviews
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
              as_of,
              payload,
              raw_text,
              error
            FROM gpt.factor_reviews
            ORDER BY provider, run_time DESC
            """
        )
        max_query = text("SELECT MAX(run_time) FROM gpt.factor_reviews")
        params = {}
    try:
        async with engine.begin() as conn:
            result = await conn.execute(query, params)
            rows = result.fetchall()
            max_result = await conn.execute(max_query, params)
            latest_run = max_result.scalar_one_or_none()
    except Exception:
        return {"session": session, "run_time": None, "providers": [], "consensus": []}

    providers: List[Dict[str, Any]] = []
    as_of_value: Optional[str] = None
    for row in rows:
        payload = row.payload
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = None
        normalized = normalize_factor_reviews(payload, row.session or session or "pre")
        if as_of_value is None:
            as_of_value = normalized.get("as_of") or _isoformat(row.as_of)
        providers.append(
            {
                "provider": row.provider,
                "session": row.session,
                "run_time": _isoformat(row.run_time),
                "as_of": normalized.get("as_of") or _isoformat(row.as_of),
                "reviews": normalized.get("reviews", []),
                "error": _redact_error(row.error),
            }
        )

    provider_order = {"openai": 0, "deepseek": 1, "gemini": 2}
    providers.sort(key=lambda item: provider_order.get(item.get("provider"), 99))
    consensus = build_factor_review_consensus(providers)

    return {
        "session": session,
        "run_time": _isoformat(latest_run),
        "as_of": as_of_value,
        "providers": providers,
        "consensus": consensus,
        "status": gpt_factor_state.get("status"),
        "last_error": gpt_factor_state.get("last_error"),
        "last_session": gpt_factor_state.get("last_session"),
    }


@app.get("/api/gpt/recommendations")
async def get_gpt_recommendations(session: Optional[str] = None) -> Dict[str, Any]:
    return await _load_gpt_recommendations(session=session)


@app.get("/api/gpt/recommendations/history")
async def get_gpt_recommendation_history(
    session: Optional[str] = None,
    provider: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    limit = max(1, min(limit, 500))
    return await _load_gpt_recommendations_history(session, provider, start, end, limit)


@app.get("/api/gpt/consensus")
async def get_gpt_consensus(session: Optional[str] = None) -> Dict[str, Any]:
    return await _load_gpt_consensus(session=session)


@app.get("/api/analysis/consensus")
async def get_analysis_consensus(session: Optional[str] = None) -> Dict[str, Any]:
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


@app.post("/api/analysis/recommendations/refresh")
async def refresh_analysis_recommendations(request: GPTRefreshRequest) -> Dict[str, Any]:
    return await refresh_gpt_recommendations(request)


@app.get("/api/gpt/challenges")
async def get_gpt_challenges(session: Optional[str] = None) -> Dict[str, Any]:
    return await _load_gpt_challenges(session=session)


@app.get("/api/analysis/challenges")
async def get_analysis_challenges(session: Optional[str] = None) -> Dict[str, Any]:
    return await _load_gpt_challenges(session=session)


@app.get("/api/analysis/recommendations/history")
async def get_analysis_recommendation_history(
    session: Optional[str] = None,
    provider: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    limit = max(1, min(limit, 500))
    return await _load_gpt_recommendations_history(session, provider, start, end, limit)


@app.get("/api/analysis/recommendations/transcript")
async def get_recommendation_transcript(
    provider: str,
    session: Optional[str] = None,
    run_time: Optional[str] = None,
) -> Dict[str, Any]:
    """Get the full transcript (raw_text) for a specific GPT recommendation."""
    if session:
        query = text(
            """
            SELECT provider, session, run_time, payload, raw_text, error
            FROM gpt.recommendations
            WHERE provider = :provider AND session = :session
            ORDER BY run_time DESC
            LIMIT 1
            """
        )
        params = {"provider": provider, "session": session}
    elif run_time:
        query = text(
            """
            SELECT provider, session, run_time, payload, raw_text, error
            FROM gpt.recommendations
            WHERE provider = :provider AND run_time = :run_time
            LIMIT 1
            """
        )
        params = {"provider": provider, "run_time": run_time}
    else:
        query = text(
            """
            SELECT provider, session, run_time, payload, raw_text, error
            FROM gpt.recommendations
            WHERE provider = :provider
            ORDER BY run_time DESC
            LIMIT 1
            """
        )
        params = {"provider": provider}

    try:
        async with engine.begin() as conn:
            result = await conn.execute(query, params)
            row = result.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"No transcript found for provider {provider}")

        payload = row.payload
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = None

        return {
            "provider": row.provider,
            "session": row.session,
            "run_time": _isoformat(row.run_time),
            "raw_text": row.raw_text,
            "payload": payload,
            "error": _redact_error(row.error),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching transcript: {str(e)}")


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


@app.post("/api/analysis/challenges/refresh")
async def refresh_analysis_challenges(request: GPTRefreshRequest) -> Dict[str, Any]:
    return await refresh_gpt_challenges(request)


@app.get("/api/gpt/factor-reviews")
async def get_gpt_factor_reviews(session: Optional[str] = None) -> Dict[str, Any]:
    return await _load_gpt_factor_reviews(session=session)


@app.get("/api/analysis/factor-reviews")
async def get_analysis_factor_reviews(session: Optional[str] = None) -> Dict[str, Any]:
    return await _load_gpt_factor_reviews(session=session)


@app.post("/api/gpt/factor-reviews/refresh")
async def refresh_gpt_factor_reviews(request: GPTFactorReviewRequest) -> Dict[str, Any]:
    session = request.session
    as_of = request.as_of
    if gpt_factor_state.get("status") == "running":
        return {
            "status": "running",
            "message": "gpt factor review already running",
            "last_run": gpt_factor_state.get("last_run"),
            "last_error": gpt_factor_state.get("last_error"),
        }
    asyncio.create_task(_run_gpt_factor_review(session, as_of))
    return {"status": "running"}


@app.post("/api/analysis/factor-reviews/refresh")
async def refresh_analysis_factor_reviews(request: GPTFactorReviewRequest) -> Dict[str, Any]:
    return await refresh_gpt_factor_reviews(request)


@app.get("/api/gpt/summary")
async def get_gpt_summary(
    as_of: Optional[str] = None,
    min_factors: int = 6,
    signal_threshold: float = 0.2,
) -> Dict[str, Any]:
    target_date = _parse_date(as_of)
    async with engine.begin() as conn:
        if target_date is None:
            result = await conn.execute(text("SELECT MAX(as_of) FROM factors.daily_factors"))
            target_date = result.scalar_one_or_none()
        if target_date is None:
            return {"as_of": None, "final": []}

    gpt_data = await _load_gpt_consensus(session=None)
    factor_data = await _load_gpt_factor_reviews(session=None)
    challenge_data = await _load_gpt_challenges(session=None)
    challenge_summary = _summarize_challenges(challenge_data.get("providers", []))

    gpt_rows = gpt_data.get("consensus", [])
    gpt_by_symbol: Dict[str, Dict[str, Any]] = {}
    for row in gpt_rows:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        existing = gpt_by_symbol.get(symbol)
        if not existing or (row.get("confidence") or 0) > (existing.get("confidence") or 0):
            gpt_by_symbol[symbol] = row

    review_by_symbol: Dict[str, Dict[str, Any]] = {
        str(row.get("symbol") or "").upper(): row for row in factor_data.get("consensus", [])
    }
    scored = await _score_all_symbols(target_date, min_factors=min_factors)

    flow_counts = {
        "gpt_total": len(gpt_rows),
        "gpt_actionable": 0,
        "with_signal": 0,
        "signal_aligned": 0,
        "signal_neutral": 0,
        "signal_conflict": 0,
        "review_reject": 0,
        "review_conflict": 0,
        "challenge_exit": 0,
        "final": 0,
    }

    consolidated: List[Dict[str, Any]] = []
    debate: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    confidence_components = {
        "formula": "profit_prob = clamp(base_prob + signal_delta + review_delta + challenge_delta)",
        "signal_weight": 0.1,
        "review_delta": {"approve": 0.05, "watch": 0.0, "reject": -0.1},
        "challenge_delta": {"adjust": -0.03, "replace": -0.05, "exit": -0.1},
        "default_base_prob": 0.5,
        "min_profit_prob": 0.5,
        "corr_threshold": 0.9,
        "corr_lookback_days": 90,
        "corr_min_obs": 30,
    }
    for symbol, gpt_row in gpt_by_symbol.items():
        gpt_action = str(gpt_row.get("action") or "neutral").lower()
        if gpt_action not in {"buy", "sell"}:
            continue
        flow_counts["gpt_actionable"] += 1
        signal = scored.get(symbol)
        score_value = signal.get("score") if signal else None
        coverage = signal.get("coverage") if signal else None
        signal_action = None
        if signal:
            signal_action = str(action_from_signal(score_value, threshold=signal_threshold)).lower()
            flow_counts["with_signal"] += 1
            if signal_action == gpt_action:
                flow_counts["signal_aligned"] += 1
            elif signal_action == "neutral":
                flow_counts["signal_neutral"] += 1
            else:
                flow_counts["signal_conflict"] += 1
        review = review_by_symbol.get(symbol)
        if review is None and signal and score_value is not None:
            # Derive a synthetic review from the factor score when the factor_review
            # pipeline did not cover this symbol (pipelines run independently).
            abs_score = abs(score_value)
            if abs_score >= signal_threshold and signal_action == gpt_action:
                review = {
                    "verdict": "approve",
                    "action": gpt_action,
                    "confidence": min(0.5 + abs_score, 0.95),
                    "notes": f"Factor score {score_value:.3f} aligned with GPT {gpt_action} (synthetic)",
                }
            elif abs_score >= signal_threshold and signal_action not in {"neutral", gpt_action}:
                review = {
                    "verdict": "reject",
                    "action": signal_action,
                    "confidence": min(0.5 + abs_score, 0.95),
                    "notes": f"Factor score {score_value:.3f} conflicts with GPT {gpt_action} (synthetic)",
                }
            else:
                review = {
                    "verdict": "watch",
                    "action": "neutral",
                    "confidence": 0.5,
                    "notes": f"Factor score {score_value:.3f} too weak to confirm direction (synthetic)",
                }
        review_action = str(review.get("action") or "").lower() if review else ""
        challenge = challenge_summary.get(symbol)
        challenge_change = str(challenge.get("change") or "") if challenge else ""

        reason: List[str] = []
        reason_codes: List[str] = []
        warnings: List[str] = []
        if not signal:
            reason.append("No factor signal")
            reason_codes.append("no_factor_signal")
        elif signal_action == "neutral":
            reason.append("Signal neutral")
            warnings.append("signal_neutral")
        elif gpt_action != signal_action:
            reason.append(f"Signal says {signal_action.upper()}")
            reason_codes.append("signal_conflict")
        if review and str(review.get("verdict") or "").lower() == "reject":
            reason.append("Review: reject")
            reason_codes.append("review_reject")
        if review_action in {"buy", "sell"} and review_action != gpt_action:
            reason.append(f"Review action: {review_action.upper()}")
            reason_codes.append("review_action_conflict")
        if challenge_change in {"adjust", "replace", "exit"}:
            reason.append(f"Challenge: {challenge_change.upper()}")
            if challenge_change == "exit":
                reason_codes.append("challenge_exit")
            else:
                warnings.append(f"challenge_{challenge_change}")

        if review:
            verdict = str(review.get("verdict") or "").lower()
            if verdict in {"approve", "watch"}:
                warnings.append(f"review_{verdict}")

        if reason:
            debate.append(
                {
                    "symbol": symbol,
                    "category": gpt_row.get("category"),
                    "gpt_action": gpt_action,
                    "gpt_confidence": gpt_row.get("confidence"),
                    "providers": gpt_row.get("providers"),
                    "signal_action": signal_action if signal_action else "missing",
                    "score": score_value,
                    "coverage": coverage,
                    "review_verdict": review.get("verdict") if review else None,
                    "review_action": review.get("action") if review else None,
                    "review_confidence": review.get("confidence") if review else None,
                    "challenge_change": challenge_change or "keep",
                    "replacement": challenge.get("replacement") if challenge else None,
                    "notes": review.get("notes") if review else None,
                    "reason": ", ".join(reason),
                    "reason_codes": reason_codes,
                    "warnings": warnings,
                }
            )

        if not signal:
            excluded.append(
                {
                    "symbol": symbol,
                    "category": gpt_row.get("category"),
                    "gpt_action": gpt_action,
                    "signal_action": None,
                    "review_verdict": review.get("verdict") if review else None,
                    "review_action": review.get("action") if review else None,
                    "challenge_change": challenge_change or None,
                    "providers": gpt_row.get("providers"),
                    "reason": "No factor signal",
                    "reason_codes": reason_codes or ["no_factor_signal"],
                    "warnings": warnings,
                }
            )
            continue
        if gpt_action != signal_action and signal_action != "neutral":
            excluded.append(
                {
                    "symbol": symbol,
                    "category": gpt_row.get("category"),
                    "gpt_action": gpt_action,
                    "signal_action": signal_action,
                    "review_verdict": review.get("verdict") if review else None,
                    "review_action": review.get("action") if review else None,
                    "challenge_change": challenge_change or None,
                    "providers": gpt_row.get("providers"),
                    "reason": "Signal conflict",
                    "reason_codes": reason_codes or ["signal_conflict"],
                    "warnings": warnings,
                }
            )
            continue
        if review and str(review.get("verdict") or "").lower() == "reject":
            flow_counts["review_reject"] += 1
            excluded.append(
                {
                    "symbol": symbol,
                    "category": gpt_row.get("category"),
                    "gpt_action": gpt_action,
                    "signal_action": signal_action,
                    "review_verdict": review.get("verdict") if review else None,
                    "review_action": review.get("action") if review else None,
                    "challenge_change": challenge_change or None,
                    "providers": gpt_row.get("providers"),
                    "reason": "Review rejected",
                    "reason_codes": reason_codes or ["review_reject"],
                    "warnings": warnings,
                }
            )
            continue
        if review_action in {"buy", "sell"} and review_action != gpt_action:
            flow_counts["review_conflict"] += 1
            excluded.append(
                {
                    "symbol": symbol,
                    "category": gpt_row.get("category"),
                    "gpt_action": gpt_action,
                    "signal_action": signal_action,
                    "review_verdict": review.get("verdict") if review else None,
                    "review_action": review.get("action") if review else None,
                    "challenge_change": challenge_change or None,
                    "providers": gpt_row.get("providers"),
                    "reason": "Review action conflict",
                    "reason_codes": reason_codes or ["review_action_conflict"],
                    "warnings": warnings,
                }
            )
            continue
        if challenge_change == "exit":
            flow_counts["challenge_exit"] += 1
            excluded.append(
                {
                    "symbol": symbol,
                    "category": gpt_row.get("category"),
                    "gpt_action": gpt_action,
                    "signal_action": signal_action,
                    "review_verdict": review.get("verdict") if review else None,
                    "review_action": review.get("action") if review else None,
                    "challenge_change": challenge_change or None,
                    "providers": gpt_row.get("providers"),
                    "reason": "Challenge exit",
                    "reason_codes": reason_codes or ["challenge_exit"],
                    "warnings": warnings,
                }
            )
            continue

        confidence = gpt_row.get("confidence")
        if confidence is None:
            confidence = 0.5
        factor_strength = min(abs(score_value or 0), 1.0)
        confidence = confidence * (0.7 + 0.3 * factor_strength)
        if review:
            verdict = str(review.get("verdict") or "").lower()
            if verdict == "approve":
                confidence *= 1.05
            elif verdict == "watch":
                confidence *= 0.85
        if challenge_change == "replace":
            confidence *= 0.8
        elif challenge_change == "adjust":
            confidence *= 0.9
        confidence = max(0.0, min(1.0, confidence))

        horizon_days = _parse_horizon_days(gpt_row.get("horizon"))
        base_prob, base_samples = await _estimate_base_profit_prob(symbol, gpt_action, horizon_days)
        base_prob = base_prob if base_prob is not None else confidence_components["default_base_prob"]
        signal_delta = 0.0
        if signal_action == gpt_action:
            signal_delta = confidence_components["signal_weight"] * factor_strength
        review_delta_map = confidence_components["review_delta"]
        review_delta = 0.0
        if review:
            verdict = str(review.get("verdict") or "").lower()
            review_delta = review_delta_map.get(verdict, 0.0)
        challenge_delta_map = confidence_components["challenge_delta"]
        challenge_delta = 0.0
        if challenge_change:
            challenge_delta = challenge_delta_map.get(challenge_change, 0.0)
        profit_prob = max(0.0, min(1.0, base_prob + signal_delta + review_delta + challenge_delta))
        if profit_prob < confidence_components["min_profit_prob"]:
            excluded.append(
                {
                    "symbol": symbol,
                    "category": gpt_row.get("category"),
                    "gpt_action": gpt_action,
                    "signal_action": signal_action,
                    "review_verdict": review.get("verdict") if review else None,
                    "review_action": review.get("action") if review else None,
                    "challenge_change": challenge_change or None,
                    "providers": gpt_row.get("providers"),
                    "reason": "Profit probability below threshold",
                    "reason_codes": ["profit_prob_below_threshold"],
                    "warnings": warnings,
                }
            )
            continue

        reason_bits = [
            f"GPT {gpt_action.upper()}",
            f"Signal {(signal_action or 'missing').upper()}",
        ]
        if review and review.get("verdict"):
            reason_bits.append(f"Review {str(review.get('verdict')).upper()}")
        if review_action in {"buy", "sell"}:
            reason_bits.append(f"Review action {review_action.upper()}")
        if challenge_change:
            reason_bits.append(f"Challenge {challenge_change.upper()}")

        consolidated.append(
            {
                "symbol": symbol,
                "category": gpt_row.get("category"),
                "final_action": review_action if review_action in {"buy", "sell"} else gpt_action,
                "confidence": profit_prob,
                "profit_prob": profit_prob,
                "base_prob": base_prob,
                "base_samples": base_samples,
                "signal_delta": signal_delta,
                "review_delta": review_delta,
                "challenge_delta": challenge_delta,
                "score": score_value,
                "coverage": coverage,
                "gpt_action": gpt_action,
                "signal_action": signal_action,
                "review_verdict": review.get("verdict") if review else None,
                "review_action": review.get("action") if review else None,
                "review_confidence": review.get("confidence") if review else None,
                "challenge_change": challenge_change or None,
                "replacement": challenge.get("replacement") if challenge else None,
                "providers": gpt_row.get("providers"),
                "notes": review.get("notes") if review else None,
                "entry": gpt_row.get("entry"),
                "target": gpt_row.get("target"),
                "stop": gpt_row.get("stop"),
                "last_price": gpt_row.get("last_price"),
                "horizon": gpt_row.get("horizon"),
                "horizon_days": horizon_days,
                "thesis": gpt_row.get("thesis"),
                "reason": "; ".join(reason_bits),
                "reason_codes": reason_codes,
                "warnings": warnings,
            }
        )

    consolidated.sort(key=lambda item: item.get("confidence") or 0, reverse=True)
    debate.sort(key=lambda item: item.get("gpt_confidence") or 0, reverse=True)
    excluded.sort(key=lambda item: item.get("symbol") or "")
    diversified, corr_skipped, corr_meta = await _apply_diversification(
        consolidated,
        lookback_days=confidence_components["corr_lookback_days"],
        corr_threshold=confidence_components["corr_threshold"],
        min_obs=confidence_components["corr_min_obs"],
        max_count=None,
    )
    flow_counts["final"] = len(diversified)
    summary_payload = None
    summary_provider = None
    summary_error = None
    if gpt_data.get("run_time"):
        try:
            run_time = datetime.fromisoformat(str(gpt_data.get("run_time")).replace("Z", "+00:00"))
        except ValueError:
            run_time = datetime.now(tz=UTC)
        summary_result = await generate_summary(diversified, gpt_data.get("session"), target_date, run_time)
        summary_payload = summary_result.get("payload")
        summary_provider = summary_result.get("provider")
        summary_error = summary_result.get("error")

    return {
        "as_of": target_date.isoformat(),
        "session": gpt_data.get("session"),
        "run_time": gpt_data.get("run_time"),
        "final": diversified,
        "undiversified": consolidated,
        "excluded": excluded,
        "corr_skipped": corr_skipped,
        "diversification": corr_meta,
        "debate": debate,
        "min_factors": min_factors,
        "signal_threshold": signal_threshold,
        "flow": flow_counts,
        "factor_run_time": factor_data.get("run_time"),
        "factor_session": factor_data.get("session"),
        "challenge_run_time": challenge_data.get("run_time"),
        "confidence_components": confidence_components,
        "summary": summary_payload.get("summary") if isinstance(summary_payload, dict) else None,
        "summary_payload": summary_payload,
        "summary_provider": summary_provider,
        "summary_error": _redact_error(summary_error) if summary_error else None,
    }


@app.get("/api/analysis/summary")
async def get_analysis_summary(
    as_of: Optional[str] = None,
    min_factors: int = 6,
    signal_threshold: float = 0.2,
) -> Dict[str, Any]:
    return await get_gpt_summary(
        as_of=as_of,
        min_factors=min_factors,
        signal_threshold=signal_threshold,
    )


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


@app.get("/api/market/simulations")
async def get_market_simulations(
    symbol: str,
    method: str = "historical",
    horizon: int = 252,
    paths: int = 30,
    lookback: int = 2520,
    stress: Optional[str] = None,
    mode: str = "block",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    symbol = symbol.upper()
    method = (method or "historical").lower()
    if method not in {"historical", "stress", "monte_carlo"}:
        raise HTTPException(status_code=400, detail="method must be historical, stress, or monte_carlo")
    horizon = max(10, min(horizon, 2520))
    paths = max(1, min(paths, 100))
    lookback = max(horizon + 1, min(lookback, 6000))
    mode = (mode or "block").lower()
    if mode not in {"block", "bootstrap"}:
        raise HTTPException(status_code=400, detail="mode must be block or bootstrap")

    history_rows = await _fetch_close_series(symbol, limit=lookback)
    if len(history_rows) < 2:
        raise HTTPException(status_code=404, detail="Not enough market data for symbol")
    timestamps, prices = zip(*history_rows)
    last_price = prices[-1]
    last_ts = timestamps[-1]
    returns = compute_log_returns(prices)
    rng = random.Random(seed)

    stress_info = None
    if method == "stress":
        stress_key = stress or "covid_2020"
        window = STRESS_WINDOWS.get(stress_key)
        if not window:
            raise HTTPException(status_code=400, detail="Unknown stress window")
        stress_rows = await _fetch_close_series(symbol, start=window.start, end=window.end)
        if len(stress_rows) < 2:
            raise HTTPException(status_code=404, detail="Not enough data for stress window")
        _, stress_prices = zip(*stress_rows)
        stress_returns = compute_log_returns(stress_prices)
        simulations = generate_stress_paths(stress_returns, last_price, horizon, paths, rng)
        stress_info = {
            "key": window.key,
            "label": window.label,
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
        }
    elif method == "monte_carlo":
        simulations = generate_monte_carlo_paths(returns, last_price, horizon, paths, rng)
    else:
        simulations = generate_historical_paths(returns, last_price, horizon, paths, mode, rng)

    summary = summarize_paths(simulations, last_price)
    end_returns = [(path[-1] / last_price) - 1 for path in simulations if path]
    if end_returns:
        sorted_returns = sorted(end_returns)
        summary.update(
            {
                "min_return": sorted_returns[0],
                "max_return": sorted_returns[-1],
                "median_return": _percentile(sorted_returns, 0.5),
                "p05_return": _percentile(sorted_returns, 0.05),
                "p95_return": _percentile(sorted_returns, 0.95),
                "mean_return": sum(sorted_returns) / len(sorted_returns),
            }
        )

    return {
        "symbol": symbol,
        "method": method,
        "horizon": horizon,
        "paths": paths,
        "as_of": _isoformat(last_ts),
        "start_price": last_price,
        "summary": summary,
        "stress": stress_info,
        "available_stress": list_stress_windows(),
        "steps": list(range(horizon + 1)),
        "paths_data": [{"id": idx + 1, "prices": path} for idx, path in enumerate(simulations)],
    }


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


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_symbols(symbols: List[str]) -> List[str]:
    seen = set()
    cleaned = []
    for symbol in symbols:
        if not symbol:
            continue
        symbol = symbol.strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        cleaned.append(symbol)
    return cleaned


def _decode_metadata(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return None


async def _load_factor_rows(target_date: date) -> List[Any]:
    query = text(
        """
        SELECT symbol, factor, value, source, window_days, metadata
        FROM factors.daily_factors
        WHERE as_of = :as_of
          AND value IS NOT NULL
        """
    )
    async with engine.begin() as conn:
        result = await conn.execute(query, {"as_of": target_date})
        return result.fetchall()


async def _score_all_symbols(
    target_date: date,
    min_factors: int = 1,
) -> Dict[str, Dict[str, Any]]:
    rows = await _load_factor_rows(target_date)
    definitions = get_factor_definitions()
    stats = build_factor_stats(
        [
            (row.symbol, row.factor, _coerce_float(row.value))
            for row in rows
            if _coerce_float(row.value) is not None
        ],
        definitions,
    )

    symbol_values: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if row.symbol == "__macro__":
            continue
        value = _coerce_float(row.value)
        if value is None:
            continue
        symbol_values.setdefault(row.symbol, {})[row.factor] = value

    results: Dict[str, Dict[str, Any]] = {}
    for symbol, values in symbol_values.items():
        summary = score_symbol_values(values, definitions, stats)
        score = summary.get("score")
        coverage = summary.get("coverage", 0)
        if score is None or coverage < min_factors:
            continue
        results[symbol] = {
            "symbol": symbol,
            "score": score,
            "coverage": coverage,
            "action": action_from_signal(score),
            "category": MarketUniverse.get_symbol_info(symbol).get("category"),
        }
    return results


def _build_factor_entry(
    row: Any,
    definitions: Dict[str, Dict[str, object]],
    stats: Dict[str, Dict[str, float]],
) -> Optional[Dict[str, Any]]:
    definition = definitions.get(row.factor)
    if not definition:
        return None
    value = _coerce_float(row.value)
    if value is None:
        return None
    signal = compute_factor_signal(row.factor, value, definition, stats)
    return {
        "factor": row.factor,
        "value": value,
        "signal": signal,
        "action": action_from_signal(signal),
        "description": definition.get("description"),
        "source": row.source or definition.get("source"),
        "window": row.window_days if row.window_days is not None else definition.get("window"),
        "weight": definition.get("weight"),
        "metadata": _decode_metadata(row.metadata),
    }


@app.get("/api/factors/summary")
async def get_factors_summary(as_of: Optional[str] = None) -> Dict[str, Any]:
    target_date = _parse_date(as_of)
    async with engine.begin() as conn:
        if target_date is None:
            result = await conn.execute(text("SELECT MAX(as_of) FROM factors.daily_factors"))
            target_date = result.scalar_one_or_none()
        if target_date is None:
            return {"as_of": None, "symbols": 0, "factors": 0, "rows": 0}
        result = await conn.execute(
            text(
                """
                SELECT COUNT(DISTINCT symbol) AS symbols,
                       COUNT(DISTINCT factor) AS factors,
                       COUNT(*) AS rows
                FROM factors.daily_factors
                WHERE as_of = :as_of
                """
            ),
            {"as_of": target_date},
        )
        row = result.fetchone()
    return {
        "as_of": target_date.isoformat(),
        "symbols": row.symbols if row else 0,
        "factors": row.factors if row else 0,
        "rows": row.rows if row else 0,
        "status": factor_state.get("status"),
        "last_run": factor_state.get("last_run"),
        "last_error": factor_state.get("last_error"),
    }


@app.get("/api/inference/signal-leaderboard")
async def get_inference_signal_leaderboard(as_of: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
    limit = max(1, min(limit, 500))
    target_date = _parse_date(as_of)
    async with engine.begin() as conn:
        if target_date is None:
            result = await conn.execute(text("SELECT MAX(as_of) FROM inference.signal_leaderboard"))
            target_date = result.scalar_one_or_none()
        if target_date is None:
            return {"as_of": None, "rows": []}
        result = await conn.execute(
            text(
                """
                SELECT factor, horizon_days, avg_ic, median_ic, hit_rate, days, observations
                FROM inference.signal_leaderboard
                WHERE as_of = :as_of
                ORDER BY ABS(avg_ic) DESC NULLS LAST, factor
                LIMIT :limit
                """
            ),
            {"as_of": target_date, "limit": limit},
        )
        rows = result.fetchall()

    payload = [
        {
            "factor": row.factor,
            "horizon_days": row.horizon_days,
            "avg_ic": row.avg_ic,
            "median_ic": row.median_ic,
            "hit_rate": row.hit_rate,
            "days": row.days,
            "observations": row.observations,
        }
        for row in rows
    ]
    return {"as_of": _isoformat(target_date), "rows": payload, "run_time": _isoformat(datetime.now(tz=UTC))}


@app.get("/api/inference/event-study")
async def get_inference_event_study(as_of: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
    limit = max(1, min(limit, 500))
    target_date = _parse_date(as_of)
    async with engine.begin() as conn:
        if target_date is None:
            result = await conn.execute(text("SELECT MAX(as_of) FROM inference.event_study"))
            target_date = result.scalar_one_or_none()
        if target_date is None:
            return {"as_of": None, "rows": []}
        result = await conn.execute(
            text(
                """
                SELECT event_type, window_days, avg_return, median_return, observations
                FROM inference.event_study
                WHERE as_of = :as_of
                ORDER BY ABS(avg_return) DESC NULLS LAST, event_type
                LIMIT :limit
                """
            ),
            {"as_of": target_date, "limit": limit},
        )
        rows = result.fetchall()

    payload = [
        {
            "event_type": row.event_type,
            "window_days": row.window_days,
            "avg_return": row.avg_return,
            "median_return": row.median_return,
            "observations": row.observations,
        }
        for row in rows
    ]
    return {"as_of": _isoformat(target_date), "rows": payload, "run_time": _isoformat(datetime.now(tz=UTC))}


@app.get("/api/inference/polymarket")
async def get_inference_polymarket(as_of: Optional[str] = None) -> Dict[str, Any]:
    target_date = _parse_date(as_of)
    async with engine.begin() as conn:
        if target_date is None:
            result = await conn.execute(text("SELECT MAX(as_of) FROM inference.polymarket_summary"))
            target_date = result.scalar_one_or_none()
        if target_date is None:
            return {"as_of": None, "summary": {}, "bins": []}
        summary_result = await conn.execute(
            text(
                """
                SELECT markets, closed_markets, resolved_proxy, avg_spread, avg_volume, avg_brier
                FROM inference.polymarket_summary
                WHERE as_of = :as_of
                """
            ),
            {"as_of": target_date},
        )
        summary_row = summary_result.first()
        bins_result = await conn.execute(
            text(
                """
                SELECT bin_low, bin_high, count, avg_pred, proxy_accuracy, avg_brier
                FROM inference.polymarket_bins
                WHERE as_of = :as_of
                ORDER BY bin_low
                """
            ),
            {"as_of": target_date},
        )
        bins = bins_result.fetchall()

    summary = (
        {
            "markets": summary_row.markets,
            "closed_markets": summary_row.closed_markets,
            "resolved_proxy": summary_row.resolved_proxy,
            "avg_spread": summary_row.avg_spread,
            "avg_volume": summary_row.avg_volume,
            "avg_brier": summary_row.avg_brier,
        }
        if summary_row
        else {}
    )
    payload = [
        {
            "bin_low": row.bin_low,
            "bin_high": row.bin_high,
            "count": row.count,
            "avg_pred": row.avg_pred,
            "proxy_accuracy": row.proxy_accuracy,
            "avg_brier": row.avg_brier,
        }
        for row in bins
    ]
    return {
        "as_of": _isoformat(target_date),
        "summary": summary,
        "bins": payload,
        "run_time": _isoformat(datetime.now(tz=UTC)),
    }


@app.get("/api/factors/alpha")
async def get_factors_alpha(
    as_of: Optional[str] = None,
    limit: int = 20,
    min_factors: int = 6,
) -> Dict[str, Any]:
    target_date = _parse_date(as_of)
    async with engine.begin() as conn:
        if target_date is None:
            result = await conn.execute(text("SELECT MAX(as_of) FROM factors.daily_factors"))
            target_date = result.scalar_one_or_none()
        if target_date is None:
            return {"as_of": None, "longs": [], "shorts": [], "factors_used": []}
        result = await conn.execute(
            text(
                """
                SELECT symbol, factor, value
                FROM factors.daily_factors
                WHERE as_of = :as_of
                  AND symbol != '__macro__'
                  AND value IS NOT NULL
                """
            ),
            {"as_of": target_date},
        )
        rows = result.fetchall()

    row_values = []
    for row in rows:
        value = _coerce_float(row.value)
        if value is None:
            continue
        row_values.append((row.symbol, row.factor, value))

    scoring = score_factor_rows(row_values)
    scores = scoring["scores"]
    factor_list = scoring["factors_used"]

    filtered = [
        (symbol, data)
        for symbol, data in scores.items()
        if symbol != "__macro__" and data.get("coverage", 0) >= min_factors and data.get("score") is not None
    ]
    if not filtered:
        return {"as_of": target_date.isoformat(), "longs": [], "shorts": [], "factors_used": factor_list}

    filtered.sort(key=lambda item: item[1]["score"], reverse=True)
    limit = max(5, min(limit, 50))
    longs = filtered[:limit]
    shorts = list(reversed(filtered[-limit:]))

    def _format_entry(symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        contributions = [
            {"factor": factor, "score": value}
            for factor, value in data["contributions"][:5]
        ]
        return {
            "symbol": symbol,
            "score": data["score"],
            "coverage": data["coverage"],
            "drivers": contributions,
            "action": action_from_signal(data["score"]),
        }

    return {
        "as_of": target_date.isoformat(),
        "longs": [_format_entry(symbol, data) for symbol, data in longs],
        "shorts": [_format_entry(symbol, data) for symbol, data in shorts],
        "factors_used": factor_list,
        "definitions": scoring.get("definitions", {}),
        "symbols_scored": len(filtered),
        "min_factors": min_factors,
    }


@app.get("/api/factors/symbol")
async def get_factors_symbol(symbol: str, as_of: Optional[str] = None) -> Dict[str, Any]:
    symbol = symbol.upper()
    target_date = _parse_date(as_of)
    async with engine.begin() as conn:
        if target_date is None:
            result = await conn.execute(text("SELECT MAX(as_of) FROM factors.daily_factors"))
            target_date = result.scalar_one_or_none()
        if target_date is None:
            return {"symbol": symbol, "as_of": None, "factors": []}

    rows = await _load_factor_rows(target_date)
    if not rows:
        return {"symbol": symbol, "as_of": target_date.isoformat(), "factors": []}

    definitions = get_factor_definitions()
    stats = build_factor_stats(
        [
            (row.symbol, row.factor, _coerce_float(row.value))
            for row in rows
            if _coerce_float(row.value) is not None
        ],
        definitions,
    )

    symbol_rows = [row for row in rows if row.symbol == symbol]
    macro_rows = [row for row in rows if row.symbol == "__macro__"]
    if not symbol_rows and not macro_rows:
        return {"symbol": symbol, "as_of": target_date.isoformat(), "factors": []}

    factor_entries = []
    symbol_values: Dict[str, float] = {}
    for row in symbol_rows + macro_rows:
        entry = _build_factor_entry(row, definitions, stats)
        if entry is None:
            continue
        factor_entries.append(entry)
        symbol_values[row.factor] = entry["value"]

    factor_entries.sort(key=lambda item: abs(item.get("signal") or 0), reverse=True)
    summary = score_symbol_values(symbol_values, definitions, stats)

    return {
        "symbol": symbol,
        "as_of": target_date.isoformat(),
        "score": summary.get("score"),
        "coverage": summary.get("coverage"),
        "action": action_from_signal(summary.get("score")),
        "category": MarketUniverse.get_symbol_info(symbol).get("category"),
        "factors": factor_entries,
    }


@app.get("/api/factors/list")
async def get_factors_list(
    as_of: Optional[str] = None,
    min_factors: int = 1,
) -> Dict[str, Any]:
    target_date = _parse_date(as_of)
    async with engine.begin() as conn:
        if target_date is None:
            result = await conn.execute(text("SELECT MAX(as_of) FROM factors.daily_factors"))
            target_date = result.scalar_one_or_none()
        if target_date is None:
            return {"as_of": None, "symbols": []}

    scored = await _score_all_symbols(target_date, min_factors=min_factors)
    results = list(scored.values())
    results.sort(key=lambda item: item.get("score") or 0, reverse=True)
    return {
        "as_of": target_date.isoformat(),
        "symbols": results,
        "last_run": factor_state.get("last_run"),
        "last_error": factor_state.get("last_error"),
    }


@app.get("/api/opinions/validated")
async def get_validated_opinions(
    as_of: Optional[str] = None,
    min_factors: int = 6,
    signal_threshold: float = 0.2,
) -> Dict[str, Any]:
    target_date = _parse_date(as_of)
    async with engine.begin() as conn:
        if target_date is None:
            result = await conn.execute(text("SELECT MAX(as_of) FROM factors.daily_factors"))
            target_date = result.scalar_one_or_none()
        if target_date is None:
            return {"as_of": None, "validated": []}

    gpt_data = await _load_gpt_consensus(session=None)
    gpt_rows = gpt_data.get("consensus", [])
    gpt_by_symbol: Dict[str, Dict[str, Any]] = {}
    for row in gpt_rows:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        existing = gpt_by_symbol.get(symbol)
        if not existing or (row.get("confidence") or 0) > (existing.get("confidence") or 0):
            gpt_by_symbol[symbol] = row

    scored = await _score_all_symbols(target_date, min_factors=min_factors)

    validated = []
    disagreements = []
    for symbol, gpt_row in gpt_by_symbol.items():
        signal = scored.get(symbol)
        if not signal:
            disagreements.append({"symbol": symbol, "reason": "no_factor_score"})
            continue
        gpt_action = str(gpt_row.get("action") or "neutral").lower()
        score_value = signal.get("score")
        signal_action = str(action_from_signal(score_value, threshold=signal_threshold)).lower()
        if gpt_action == signal_action and gpt_action != "neutral":
            validated.append(
                {
                    "symbol": symbol,
                    "category": gpt_row.get("category"),
                    "gpt_action": gpt_action,
                    "signal_action": signal_action,
                    "score": score_value,
                    "coverage": signal.get("coverage"),
                    "confidence": gpt_row.get("confidence"),
                    "providers": gpt_row.get("providers"),
                }
            )
        else:
            disagreements.append(
                {
                    "symbol": symbol,
                    "category": gpt_row.get("category"),
                    "gpt_action": gpt_action,
                    "signal_action": signal_action,
                }
            )

    validated.sort(key=lambda item: item.get("confidence") or 0, reverse=True)
    return {
        "as_of": target_date.isoformat(),
        "session": gpt_data.get("session"),
        "run_time": gpt_data.get("run_time"),
        "validated": validated,
        "disagreements": disagreements,
        "min_factors": min_factors,
        "signal_threshold": signal_threshold,
    }


@app.get("/api/opinions/final")
async def get_final_opinions(
    as_of: Optional[str] = None,
    min_factors: int = 6,
    signal_threshold: float = 0.2,
    filter_rejects: bool = True,
) -> Dict[str, Any]:
    target_date = _parse_date(as_of)
    async with engine.begin() as conn:
        if target_date is None:
            result = await conn.execute(text("SELECT MAX(as_of) FROM factors.daily_factors"))
            target_date = result.scalar_one_or_none()
        if target_date is None:
            return {"as_of": None, "final": []}

    gpt_data = await _load_gpt_consensus(session=None)
    gpt_rows = gpt_data.get("consensus", [])
    gpt_by_symbol: Dict[str, Dict[str, Any]] = {}
    for row in gpt_rows:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        existing = gpt_by_symbol.get(symbol)
        if not existing or (row.get("confidence") or 0) > (existing.get("confidence") or 0):
            gpt_by_symbol[symbol] = row

    scored = await _score_all_symbols(target_date, min_factors=min_factors)
    factor_reviews = await _load_gpt_factor_reviews(session=None)
    review_by_symbol: Dict[str, Dict[str, Any]] = {
        str(row.get("symbol") or "").upper(): row for row in factor_reviews.get("consensus", [])
    }

    validated: List[Dict[str, Any]] = []
    disagreements: List[Dict[str, Any]] = []
    for symbol, gpt_row in gpt_by_symbol.items():
        signal = scored.get(symbol)
        if not signal:
            disagreements.append({"symbol": symbol, "reason": "no_factor_score"})
            continue
        gpt_action = str(gpt_row.get("action") or "neutral").lower()
        score_value = signal.get("score")
        signal_action = str(action_from_signal(score_value, threshold=signal_threshold)).lower()
        if gpt_action == signal_action and gpt_action != "neutral":
            validated.append(
                {
                    "symbol": symbol,
                    "category": gpt_row.get("category"),
                    "gpt_action": gpt_action,
                    "signal_action": signal_action,
                    "score": score_value,
                    "coverage": signal.get("coverage"),
                    "confidence": gpt_row.get("confidence"),
                    "providers": gpt_row.get("providers"),
                }
            )
        else:
            disagreements.append(
                {
                    "symbol": symbol,
                    "category": gpt_row.get("category"),
                    "gpt_action": gpt_action,
                    "signal_action": signal_action,
                }
            )

    final: List[Dict[str, Any]] = []
    for row in validated:
        symbol = row.get("symbol")
        review = review_by_symbol.get(str(symbol or "").upper())
        verdict = review.get("verdict") if review else None
        review_action = review.get("action") if review else None
        review_action = review_action if review_action in {"buy", "sell"} else None
        if verdict == "reject" and filter_rejects:
            continue
        final.append(
            {
                **row,
                "final_action": review_action or row.get("gpt_action"),
                "review_verdict": verdict,
                "review_action": review_action,
                "review_confidence": review.get("confidence") if review else None,
                "review_providers": review.get("providers") if review else None,
                "review_notes": review.get("notes") if review else None,
                "replacement": review.get("replacement") if review else None,
            }
        )

    final.sort(key=lambda item: item.get("confidence") or 0, reverse=True)
    return {
        "as_of": target_date.isoformat(),
        "session": gpt_data.get("session"),
        "run_time": gpt_data.get("run_time"),
        "final": final,
        "validated": validated,
        "disagreements": disagreements,
        "min_factors": min_factors,
        "signal_threshold": signal_threshold,
        "filter_rejects": filter_rejects,
    }


@app.post("/api/factors/selected")
async def get_factors_selected(request: FactorSelectionRequest) -> Dict[str, Any]:
    symbols = _normalize_symbols(request.symbols or [])
    target_date = request.as_of
    if target_date is None:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT MAX(as_of) FROM factors.daily_factors"))
            target_date = result.scalar_one_or_none()
    if target_date is None:
        return {"as_of": None, "symbols": [], "missing": symbols}

    rows = await _load_factor_rows(target_date)
    definitions = get_factor_definitions()
    stats = build_factor_stats(
        [
            (row.symbol, row.factor, _coerce_float(row.value))
            for row in rows
            if _coerce_float(row.value) is not None
        ],
        definitions,
    )

    macro_rows = [row for row in rows if row.symbol == "__macro__"]
    symbol_rows: Dict[str, List[Any]] = {symbol: [] for symbol in symbols}
    for row in rows:
        if row.symbol in symbol_rows:
            symbol_rows[row.symbol].append(row)

    results = []
    missing = []
    for symbol in symbols:
        factor_entries = []
        symbol_values: Dict[str, float] = {}
        rows_for_symbol = symbol_rows.get(symbol, [])
        if not rows_for_symbol and not macro_rows:
            missing.append(symbol)
            continue
        for row in rows_for_symbol + macro_rows:
            entry = _build_factor_entry(row, definitions, stats)
            if entry is None:
                continue
            factor_entries.append(entry)
            symbol_values[row.factor] = entry["value"]
        if not factor_entries:
            missing.append(symbol)
            continue
        factor_entries.sort(key=lambda item: abs(item.get("signal") or 0), reverse=True)
        summary = score_symbol_values(symbol_values, definitions, stats)
        results.append(
            {
                "symbol": symbol,
                "score": summary.get("score"),
                "coverage": summary.get("coverage"),
                "action": action_from_signal(summary.get("score")),
                "factors": factor_entries,
            }
        )

    results.sort(key=lambda item: (item.get("score") is None, -(item.get("score") or 0)))
    return {
        "as_of": target_date.isoformat(),
        "symbols": results,
        "missing": missing,
        "total_factors": len(definitions),
        "last_run": factor_state.get("last_run"),
        "last_error": factor_state.get("last_error"),
    }


@app.post("/api/factors/refresh")
async def refresh_factors(request: FactorRefreshRequest) -> Dict[str, Any]:
    if factor_state.get("status") == "running":
        return {
            "status": "running",
            "last_run": factor_state.get("last_run"),
            "last_error": factor_state.get("last_error"),
        }
    asyncio.create_task(_run_factor_refresh(request.as_of))
    return {"status": "running"}
