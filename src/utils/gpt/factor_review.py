"""GPT critique runner for factor-based picks."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from sqlalchemy import text

from config.config import settings
from src.db.session import engine
from src.definitions.market_universe import MarketUniverse
from src.utils.factors.definitions import get_factor_definitions
from src.utils.factors.scoring import (
    action_from_signal,
    build_factor_stats,
    compute_factor_signal,
    score_symbol_values,
)
from src.utils.gpt.recommendations import (
    DEFAULT_GEMINI_MODELS,
    _determine_session,
    _extract_json,
    _fetch_latest_prices,
    _fetch_yfinance_last_prices,
    _parse_confidence,
    _redact_secret,
)

UTC = timezone.utc

CATEGORY_ORDER = ("large_cap", "growth", "etf", "crypto")
MAX_DRIVERS = 4
MIN_FACTOR_COVERAGE = 6
ACTION_THRESHOLD = 0.1
PICKS_PER_SIDE = 3
MAX_PICKS_PER_CATEGORY = 6

SYSTEM_PROMPT = (
    "You are a portfolio committee reviewing factor-driven trade picks. "
    "Use only the provided factor context and prices. Respond with JSON only."
)


def _normalize_category(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    category = str(raw).lower()
    if category in {"crypto"}:
        return "crypto"
    if category.endswith("_etfs") or category in {
        "index_etfs",
        "sector_etfs",
        "international_etfs",
        "bond_etfs",
        "commodity_etfs",
        "thematic_etfs",
        "reit_etfs",
        "volatility_etfs",
    }:
        return "etf"
    if category in {"mega_cap_stocks"}:
        return "large_cap"
    if category in {
        "high_growth_stocks",
        "ai_infrastructure",
        "quantum_computing",
        "consumer",
        "defense",
        "crypto_stocks",
    }:
        return "growth"
    return "growth"


async def _call_openai(client: httpx.AsyncClient, prompt: str) -> str:
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    if settings.openai_org:
        headers["OpenAI-Organization"] = settings.openai_org
    payload = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    resp = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


async def _call_deepseek(client: httpx.AsyncClient, prompt: str) -> str:
    url = f"{settings.deepseek_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.deepseek_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    resp = await client.post(url, json=payload, headers={"Authorization": f"Bearer {settings.deepseek_api_key}"})
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _iter_gemini_models() -> List[str]:
    seen = set()
    candidates: List[str] = []
    if settings.gemini_model:
        candidates.append(settings.gemini_model)
    candidates.extend(settings.gemini_models)
    candidates.extend(DEFAULT_GEMINI_MODELS)
    unique: List[str] = []
    for model in candidates:
        if not model or model in seen:
            continue
        seen.add(model)
        unique.append(model)
    return unique


async def _call_gemini_model(client: httpx.AsyncClient, prompt: str, model: str) -> str:
    url = f"{settings.gemini_base_url.rstrip('/')}/models/{model}:generateContent"
    params = {"key": settings.gemini_api_key}
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": f"{SYSTEM_PROMPT}\n{prompt}"}]},
        ],
        "generationConfig": {"temperature": 0.2},
    }
    resp = await client.post(url, params=params, json=payload)
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts)


async def _call_gemini(client: httpx.AsyncClient, prompt: str) -> str:
    last_error: Optional[Exception] = None
    tried: List[str] = []
    for model in _iter_gemini_models():
        tried.append(model)
        try:
            return await _call_gemini_model(client, prompt, model)
        except Exception as exc:
            last_error = exc
            continue
    if last_error:
        raise RuntimeError(f"gemini model not available (tried: {', '.join(tried)})") from last_error
    raise RuntimeError("gemini model list is empty")


def _normalize_verdict(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"approve", "approved", "agree", "support", "keep"}:
        return "approve"
    if raw in {"reject", "rejected", "disagree", "exit", "drop"}:
        return "reject"
    if raw in {"watch", "hold", "neutral", "mixed"}:
        return "watch"
    return "watch"


def _normalize_action(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"buy", "sell", "neutral"}:
        return raw
    if raw in {"hold"}:
        return "neutral"
    return "neutral"


def _normalize_reviews(payload: Optional[Dict[str, Any]], session: str) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"as_of": None, "session": session, "reviews": []}

    reviews = payload.get("reviews")
    if not isinstance(reviews, list):
        categories = payload.get("categories")
        if isinstance(categories, dict):
            reviews = []
            for items in categories.values():
                if isinstance(items, list):
                    reviews.extend(items)
        else:
            reviews = payload.get("items", [])

    normalized: List[Dict[str, Any]] = []
    if isinstance(reviews, list):
        for item in reviews:
            if not isinstance(item, dict):
                continue
            symbol = item.get("symbol")
            if symbol:
                symbol = str(symbol).upper()
            drivers = item.get("drivers") or []
            if not isinstance(drivers, list):
                drivers = []
            normalized.append(
                {
                    "symbol": symbol,
                    "category": item.get("category"),
                    "factor_action": _normalize_action(item.get("factor_action") or item.get("signal_action")),
                    "factor_score": _coerce_float(item.get("factor_score") or item.get("score")),
                    "coverage": _coerce_float(item.get("coverage")),
                    "last_price": _coerce_float(item.get("last_price")),
                    "drivers": drivers,
                    "verdict": _normalize_verdict(item.get("verdict") or item.get("decision")),
                    "action": _normalize_action(item.get("action")),
                    "confidence": _parse_confidence(item.get("confidence")),
                    "notes": item.get("notes") or item.get("rationale") or item.get("thesis") or "",
                    "replacement": item.get("replacement")
                    or item.get("replacement_symbol")
                    or item.get("replaces"),
                }
            )
    as_of = payload.get("as_of") or payload.get("date")
    return {
        "as_of": as_of,
        "session": payload.get("session") or session,
        "reviews": normalized,
    }


def normalize_factor_reviews(payload: Optional[Dict[str, Any]], session: str) -> Dict[str, Any]:
    return _normalize_reviews(payload, session)


def _align_reviews(
    reviews: List[Dict[str, Any]],
    picks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_symbol = {item.get("symbol"): item for item in reviews if item.get("symbol")}
    merged: List[Dict[str, Any]] = []
    for pick in picks:
        symbol = pick.get("symbol")
        review = by_symbol.get(symbol)
        if not review:
            merged.append(
                {
                    **pick,
                    "verdict": "watch",
                    "action": "neutral",
                    "confidence": None,
                    "notes": "No review output; flagging for manual check.",
                    "replacement": None,
                }
            )
            continue
        merged.append(
            {
                **pick,
                "verdict": review.get("verdict", "watch"),
                "action": review.get("action", "neutral"),
                "confidence": review.get("confidence"),
                "notes": review.get("notes", ""),
                "replacement": review.get("replacement"),
            }
        )
    return merged


def _majority(values: Sequence[str], fallback: str) -> Tuple[str, int]:
    if not values:
        return fallback, 0
    counts = Counter(values)
    top_value, top_count = counts.most_common(1)[0]
    ties = [value for value, count in counts.items() if count == top_count]
    if len(ties) > 1:
        return fallback, top_count
    return top_value, top_count


def build_factor_review_consensus(providers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    usable = [provider for provider in providers if not provider.get("error")]
    provider_count = max(1, len(usable))
    by_symbol: Dict[str, Dict[str, Any]] = {}

    for provider in usable:
        provider_name = provider.get("provider") or "unknown"
        reviews = provider.get("reviews") or []
        for review in reviews:
            symbol = review.get("symbol")
            if not symbol:
                continue
            entry = by_symbol.setdefault(
                symbol,
                {
                    "symbol": symbol,
                    "category": review.get("category"),
                    "factor_action": review.get("factor_action"),
                    "factor_score": review.get("factor_score"),
                    "coverage": review.get("coverage"),
                    "last_price": review.get("last_price"),
                    "drivers": review.get("drivers") or [],
                    "providers": [],
                    "verdicts": [],
                    "actions": [],
                    "confidences": [],
                    "notes": [],
                    "replacements": [],
                },
            )
            entry["providers"].append(provider_name)
            verdict = review.get("verdict")
            if verdict:
                entry["verdicts"].append(str(verdict))
            action = review.get("action")
            if action:
                entry["actions"].append(str(action))
            confidence = review.get("confidence")
            if confidence is not None:
                entry["confidences"].append(float(confidence))
            notes = review.get("notes")
            if notes:
                entry["notes"].append(f"{provider_name}: {notes}")
            replacement = review.get("replacement")
            if replacement:
                entry["replacements"].append(str(replacement).upper())

    consensus: List[Dict[str, Any]] = []
    for entry in by_symbol.values():
        verdict, verdict_support = _majority(entry["verdicts"], fallback="watch")
        action, action_support = _majority(entry["actions"], fallback=entry.get("factor_action") or "neutral")
        support = len(entry["providers"])
        avg_conf = None
        if entry["confidences"]:
            avg_conf = sum(entry["confidences"]) / len(entry["confidences"])
        support_ratio = support / provider_count if provider_count else 1.0
        confidence = None
        if avg_conf is not None:
            confidence = max(0.0, min(1.0, avg_conf * support_ratio))
        elif support_ratio:
            confidence = max(0.0, min(1.0, support_ratio))

        replacement = None
        if entry["replacements"]:
            replacement, _ = _majority(entry["replacements"], fallback=entry["replacements"][0])

        consensus.append(
            {
                "symbol": entry["symbol"],
                "category": entry.get("category"),
                "factor_action": entry.get("factor_action"),
                "factor_score": entry.get("factor_score"),
                "coverage": entry.get("coverage"),
                "last_price": entry.get("last_price"),
                "drivers": entry.get("drivers") or [],
                "verdict": verdict,
                "action": action,
                "confidence": confidence,
                "providers": entry["providers"],
                "support": support,
                "notes": " | ".join(entry["notes"][:3]) if entry["notes"] else "",
                "replacement": replacement,
                "verdict_support": verdict_support,
                "action_support": action_support,
            }
        )

    consensus.sort(key=lambda item: abs(item.get("factor_score") or 0), reverse=True)
    return consensus


async def _resolve_as_of(as_of: Optional[date]) -> date:
    if as_of:
        return as_of
    query = text("SELECT MAX(as_of) FROM factors.daily_factors")
    async with engine.begin() as conn:
        result = await conn.execute(query)
        value = result.scalar_one_or_none()
    if not value:
        raise RuntimeError("No factor data available for GPT factor review")
    return value


async def _load_factor_rows(as_of: date) -> List[Any]:
    query = text(
        """
        SELECT symbol, factor, value
        FROM factors.daily_factors
        WHERE as_of = :as_of
          AND value IS NOT NULL
        """
    )
    async with engine.begin() as conn:
        result = await conn.execute(query, {"as_of": as_of})
        return result.fetchall()


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_driver_details(
    symbol_values: Dict[str, float],
    contributions: List[Tuple[str, float]],
    definitions: Dict[str, Dict[str, object]],
    stats: Dict[str, Dict[str, float]],
    max_drivers: int = MAX_DRIVERS,
) -> List[Dict[str, Any]]:
    drivers: List[Dict[str, Any]] = []
    for factor, _ in contributions[:max_drivers]:
        definition = definitions.get(factor)
        if not definition:
            continue
        value = symbol_values.get(factor)
        if value is None:
            continue
        signal = compute_factor_signal(factor, value, definition, stats)
        drivers.append(
            {
                "factor": factor,
                "value": value,
                "signal": signal,
                "description": definition.get("description"),
            }
        )
    return drivers


def _select_picks(
    scored: Dict[str, Dict[str, Any]],
    per_side: int = PICKS_PER_SIDE,
    max_per_category: int = MAX_PICKS_PER_CATEGORY,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {key: [] for key in CATEGORY_ORDER}
    for entry in scored.values():
        category = entry.get("category")
        if category not in grouped:
            continue
        grouped[category].append(entry)

    picks: List[Dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        entries = grouped.get(category, [])
        buys = [item for item in entries if item.get("factor_action") == "buy"]
        sells = [item for item in entries if item.get("factor_action") == "sell"]
        buys.sort(key=lambda item: item.get("factor_score") or 0, reverse=True)
        sells.sort(key=lambda item: item.get("factor_score") or 0)
        selected = buys[:per_side] + sells[:per_side]
        selected_symbols = {item["symbol"] for item in selected}
        if len(selected) < max_per_category:
            remainder = sorted(entries, key=lambda item: abs(item.get("factor_score") or 0), reverse=True)
            for item in remainder:
                if len(selected) >= max_per_category:
                    break
                if item["symbol"] in selected_symbols:
                    continue
                selected.append(item)
                selected_symbols.add(item["symbol"])
        picks.extend(selected)
    return picks


async def _fetch_last_prices(symbols: Sequence[str]) -> Dict[str, float]:
    if not symbols:
        return {}
    latest_prices = await _fetch_latest_prices(symbols)
    missing = [symbol for symbol in symbols if symbol not in latest_prices]
    if missing:
        yf_prices = _fetch_yfinance_last_prices(missing)
        latest_prices.update(yf_prices)
    return latest_prices


async def _build_factor_picks(as_of: date) -> List[Dict[str, Any]]:
    rows = await _load_factor_rows(as_of)
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

    scored: Dict[str, Dict[str, Any]] = {}
    for symbol, values in symbol_values.items():
        summary = score_symbol_values(values, definitions, stats)
        score = summary.get("score")
        coverage = summary.get("coverage", 0)
        if score is None or coverage < MIN_FACTOR_COVERAGE:
            continue
        action = action_from_signal(score, threshold=ACTION_THRESHOLD)
        if action == "neutral":
            action = "buy" if score >= 0 else "sell"
        raw_category = MarketUniverse.get_symbol_info(symbol).get("category")
        category = _normalize_category(raw_category)
        drivers = _build_driver_details(values, summary.get("contributions", []), definitions, stats)
        scored[symbol] = {
            "symbol": symbol,
            "category": category,
            "factor_score": score,
            "factor_action": action,
            "coverage": coverage,
            "drivers": drivers,
        }

    picks = _select_picks(scored)
    prices = await _fetch_last_prices([pick["symbol"] for pick in picks])
    for pick in picks:
        pick["last_price"] = prices.get(pick["symbol"])
    return picks


def _build_macro_context(
    rows: List[Any],
    definitions: Dict[str, Dict[str, object]],
    stats: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    macro_entries: List[Dict[str, Any]] = []
    for row in rows:
        if row.symbol != "__macro__":
            continue
        definition = definitions.get(row.factor)
        if not definition:
            continue
        value = _coerce_float(row.value)
        if value is None:
            continue
        signal = compute_factor_signal(row.factor, value, definition, stats)
        if signal is None:
            continue
        macro_entries.append(
            {
                "factor": row.factor,
                "value": value,
                "signal": signal,
                "description": definition.get("description"),
            }
        )
    macro_entries.sort(key=lambda item: abs(item.get("signal") or 0), reverse=True)
    return macro_entries[:5]


def _build_prompt(session: str, as_of: date, picks: List[Dict[str, Any]], macro: List[Dict[str, Any]]) -> str:
    pick_payload = json.dumps(picks, separators=(",", ":"), sort_keys=True)
    macro_payload = json.dumps(macro, separators=(",", ":"), sort_keys=True)
    return (
        "Return JSON with keys: as_of (YYYY-MM-DD), session (pre or post), reviews. "
        "Each review must include: symbol, verdict (approve, watch, reject), action (buy, sell, neutral), "
        "confidence (0 to 1), notes, and optional replacement. Use only symbols from the picks list. "
        "Keep exactly one review per pick in the same order; do not add new rows. "
        "Base critiques on factor_score, factor_action, drivers, and last_price. "
        "In notes, cite at least one driver factor by name and explain mismatches between signals and verdicts. "
        "Confidence must be continuous (not buckets). "
        "No markdown or extra text. Session: "
        + session
        + ". Factor as_of: "
        + as_of.isoformat()
        + ". Picks: "
        + pick_payload
        + ". Macro context: "
        + macro_payload
    )


async def _ensure_gpt_factor_review_table(conn) -> None:
    await conn.execute(text("CREATE SCHEMA IF NOT EXISTS gpt"))
    await conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS gpt.factor_reviews (
              id BIGSERIAL PRIMARY KEY,
              provider VARCHAR(32) NOT NULL,
              session VARCHAR(16) NOT NULL,
              run_time TIMESTAMPTZ NOT NULL,
              as_of DATE NOT NULL,
              payload JSONB,
              raw_text TEXT,
              error TEXT,
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              CONSTRAINT gpt_factor_reviews_provider_session_run_time_key
                UNIQUE (provider, session, run_time)
            )
            """
        )
    )


async def _store_factor_review(
    conn,
    provider: str,
    session: str,
    run_time: datetime,
    as_of: date,
    payload: Optional[Dict[str, Any]],
    raw_text: Optional[str],
    error: Optional[str],
) -> None:
    payload_json = json.dumps(payload) if payload is not None else None
    error = _redact_secret(error)
    await conn.execute(
        text(
            """
            INSERT INTO gpt.factor_reviews
              (provider, session, run_time, as_of, payload, raw_text, error)
            VALUES (:provider, :session, :run_time, :as_of, CAST(:payload AS JSONB), :raw_text, :error)
            ON CONFLICT (provider, session, run_time)
            DO UPDATE SET payload = EXCLUDED.payload,
                         raw_text = EXCLUDED.raw_text,
                         error = EXCLUDED.error,
                         as_of = EXCLUDED.as_of
            """
        ),
        {
            "provider": provider,
            "session": session,
            "run_time": run_time,
            "as_of": as_of,
            "payload": payload_json,
            "raw_text": raw_text,
            "error": error,
        },
    )


async def run_gpt_factor_reviews(
    session: Optional[str] = None,
    as_of: Optional[date] = None,
) -> Dict[str, Any]:
    as_of = await _resolve_as_of(as_of)
    now = datetime.now(tz=UTC)
    session = session or _determine_session(now)

    picks = await _build_factor_picks(as_of)
    if not picks:
        raise RuntimeError("No factor picks available for GPT review")

    rows = await _load_factor_rows(as_of)
    definitions = get_factor_definitions()
    stats = build_factor_stats(
        [
            (row.symbol, row.factor, _coerce_float(row.value))
            for row in rows
            if _coerce_float(row.value) is not None
        ],
        definitions,
    )
    macro_context = _build_macro_context(rows, definitions, stats)

    prompt = _build_prompt(session, as_of, picks, macro_context)

    providers: List[Dict[str, Any]] = []
    async with engine.begin() as conn:
        await _ensure_gpt_factor_review_table(conn)
        async with httpx.AsyncClient(timeout=45.0) as client:
            for provider in ("openai", "deepseek", "gemini"):
                api_key = {
                    "openai": settings.openai_api_key,
                    "deepseek": settings.deepseek_api_key,
                    "gemini": settings.gemini_api_key,
                }.get(provider)
                if not api_key:
                    error = "missing api key"
                    await _store_factor_review(conn, provider, session, now, as_of, None, None, error)
                    providers.append({"provider": provider, "error": error})
                    continue

                error = None
                raw_text = None
                payload = None
                try:
                    if provider == "openai":
                        raw_text = await _call_openai(client, prompt)
                    elif provider == "deepseek":
                        raw_text = await _call_deepseek(client, prompt)
                    else:
                        raw_text = await _call_gemini(client, prompt)
                    parsed = _extract_json(raw_text or "")
                    normalized = _normalize_reviews(parsed, session)
                    aligned = _align_reviews(normalized.get("reviews", []), picks)
                    payload = {
                        "as_of": as_of.isoformat(),
                        "session": session,
                        "reviews": aligned,
                    }
                except Exception as exc:
                    error = str(exc)

                await _store_factor_review(conn, provider, session, now, as_of, payload, raw_text, error)
                providers.append(
                    {
                        "provider": provider,
                        "error": error,
                        "reviews": payload.get("reviews") if payload else [],
                    }
                )

    return {"session": session, "run_time": now.isoformat(), "as_of": as_of.isoformat(), "providers": providers}
