"""GPT challenge runner to critique and improve prior recommendations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from sqlalchemy import text

from config.config import settings
from src.db.session import engine
from src.utils.gpt.recommendations import (
    CATEGORIES,
    _apply_price_guards,
    _build_price_snapshot,
    _call_deepseek,
    _call_gemini,
    _call_openai,
    _determine_session,
    _extract_json,
    _fetch_latest_prices,
    _fetch_yfinance_last_prices,
    _normalize_payload,
    _redact_secret,
)

UTC = timezone.utc

SYSTEM_PROMPT = (
    "You are a portfolio risk manager. Critique prior trade ideas and improve them for better "
    "risk-adjusted returns. Respond with JSON only."
)


def _build_challenge_prompt(
    session: str,
    price_snapshot: Dict[str, Dict[str, float]],
    previous_payload: Dict[str, Any],
) -> str:
    previous_json = json.dumps(previous_payload, separators=(",", ":"), sort_keys=True)
    price_json = json.dumps(price_snapshot, separators=(",", ":"), sort_keys=True)
    return (
        "Return JSON with keys: as_of (YYYY-MM-DD), session (pre or post), categories. "
        "Categories must include large_cap, growth, etf, crypto. "
        "Use only symbols from the price snapshot below. Use USD prices; no commas. "
        "Each category must include one review per previous recommendation in that category (same symbols). "
        "Do not add new symbols as separate rows. If you want to replace a symbol, set change=replace and set "
        "replaces to the new symbol; still keep the original symbol row. If you would drop a symbol, use "
        "change=exit. Each review must include: symbol, action (buy or sell), entry, target, stop, horizon, "
        "thesis, change (keep, adjust, replace, exit), notes, and replaces (optional). Notes must explicitly "
        "explain why the prior idea is still valid or not today given the last_close. Entry/target/stop must "
        "be within +/-20% of last_close for stocks/ETFs and +/-40% for crypto. No markdown or extra text. "
        "Session: "
        + session
        + ". Previous recommendations: "
        + previous_json
        + ". Price snapshot: "
        + price_json
    )


def _normalize_change(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"keep", "adjust", "replace", "exit"}:
        return raw
    if raw in {"hold", "retain", "unchanged"}:
        return "keep"
    if raw in {"swap", "new", "replace"}:
        return "replace"
    if raw in {"drop", "remove", "close", "exit"}:
        return "exit"
    return "adjust"


def _normalize_challenge_payload(payload: Dict[str, Any], session: str) -> Dict[str, Any]:
    normalized = _normalize_payload(payload, session)
    categories = normalized.get("categories") or {}
    for items in categories.values():
        for item in items:
            symbol = item.get("symbol")
            if symbol:
                item["symbol"] = str(symbol).upper()
            item["change"] = _normalize_change(item.get("change"))
            replaces = item.get("replaces") or item.get("replacement") or item.get("replacement_symbol")
            if replaces:
                item["replaces"] = str(replaces).upper()
            if "notes" not in item:
                item["notes"] = item.get("rationale") or ""
    normalized["categories"] = categories
    return normalized


def _align_challenge_payload(
    payload: Dict[str, Any],
    previous_payload: Dict[str, Any],
    session: str,
) -> Dict[str, Any]:
    normalized_prev = _normalize_payload(previous_payload, session)
    prev_categories = normalized_prev.get("categories") or {}
    categories = payload.get("categories") or {}
    aligned: Dict[str, List[Dict[str, Any]]] = {}
    for category in CATEGORIES:
        prev_items = prev_categories.get(category, []) or []
        output_items = categories.get(category, []) or []
        by_symbol = {
            str(item.get("symbol")).upper(): item
            for item in output_items
            if item.get("symbol")
        }
        by_replaces = {}
        for item in output_items:
            replaces = item.get("replaces")
            if replaces:
                by_replaces[str(replaces).upper()] = item

        merged: List[Dict[str, Any]] = []
        for prev_item in prev_items:
            symbol = prev_item.get("symbol")
            if not symbol:
                continue
            symbol_key = str(symbol).upper()
            current = by_symbol.get(symbol_key)
            if current:
                if not current.get("action"):
                    current["action"] = prev_item.get("action") or "buy"
                if not current.get("entry"):
                    current["entry"] = prev_item.get("entry")
                if not current.get("target"):
                    current["target"] = prev_item.get("target")
                if not current.get("stop"):
                    current["stop"] = prev_item.get("stop")
                if not current.get("horizon"):
                    current["horizon"] = prev_item.get("horizon")
                if not current.get("change"):
                    current["change"] = "keep"
                merged.append(current)
                continue
            replacement = by_replaces.get(symbol_key)
            if replacement:
                replacement_symbol = replacement.get("symbol")
                notes = replacement.get("notes") or replacement.get("thesis") or ""
                prefix = f"Replace with {replacement_symbol}. " if replacement_symbol else ""
                merged.append(
                    {
                        "symbol": symbol_key,
                        "action": prev_item.get("action") or "buy",
                        "entry": prev_item.get("entry"),
                        "target": prev_item.get("target"),
                        "stop": prev_item.get("stop"),
                        "horizon": prev_item.get("horizon"),
                        "change": "replace",
                        "notes": f"{prefix}{notes}".strip(),
                    }
                )
                continue
            merged.append(
                {
                    "symbol": symbol_key,
                    "action": prev_item.get("action") or "buy",
                    "entry": prev_item.get("entry"),
                    "target": prev_item.get("target"),
                    "stop": prev_item.get("stop"),
                    "horizon": prev_item.get("horizon"),
                    "change": "keep",
                    "notes": "No challenge output; keeping prior idea for review.",
                }
            )
        if merged:
            aligned[category] = merged
        else:
            aligned[category] = output_items
    payload["categories"] = aligned
    return payload


async def _extend_price_snapshot(
    price_snapshot: Dict[str, Dict[str, float]],
    previous_payload: Dict[str, Any],
    session: str,
) -> Dict[str, Dict[str, float]]:
    normalized_prev = _normalize_payload(previous_payload, session)
    prev_categories = normalized_prev.get("categories") or {}
    missing_symbols: List[str] = []
    for category in CATEGORIES:
        category_prices = price_snapshot.setdefault(category, {})
        for item in prev_categories.get(category, []) or []:
            symbol = item.get("symbol")
            if not symbol:
                continue
            symbol_key = str(symbol).upper()
            if symbol_key not in category_prices:
                missing_symbols.append(symbol_key)

    if not missing_symbols:
        return price_snapshot

    latest_prices = await _fetch_latest_prices(missing_symbols)
    still_missing = [symbol for symbol in missing_symbols if symbol not in latest_prices]
    yf_prices = _fetch_yfinance_last_prices(still_missing)

    for category in CATEGORIES:
        category_prices = price_snapshot.setdefault(category, {})
        for item in prev_categories.get(category, []) or []:
            symbol = item.get("symbol")
            if not symbol:
                continue
            symbol_key = str(symbol).upper()
            price = latest_prices.get(symbol_key) or yf_prices.get(symbol_key)
            if price is not None:
                category_prices[symbol_key] = price
    return price_snapshot


async def _ensure_gpt_challenge_table(conn) -> None:
    await conn.execute(text("CREATE SCHEMA IF NOT EXISTS gpt"))
    await conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS gpt.recommendation_challenges (
              id BIGSERIAL PRIMARY KEY,
              provider VARCHAR(32) NOT NULL,
              session VARCHAR(16) NOT NULL,
              run_time TIMESTAMPTZ NOT NULL,
              source_provider VARCHAR(32),
              source_run_time TIMESTAMPTZ,
              source_payload JSONB,
              payload JSONB,
              raw_text TEXT,
              error TEXT,
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              CONSTRAINT gpt_recommendation_challenges_provider_session_run_time_key
                UNIQUE (provider, session, run_time)
            )
            """
        )
    )


async def _fetch_latest_recommendation(
    conn,
    provider: str,
    session: Optional[str],
) -> Optional[Tuple[datetime, Dict[str, Any]]]:
    if session:
        query = text(
            """
            SELECT run_time, payload
            FROM gpt.recommendations
            WHERE provider = :provider
              AND session = :session
              AND payload IS NOT NULL
              AND error IS NULL
            ORDER BY run_time DESC
            LIMIT 1
            """
        )
        params = {"provider": provider, "session": session}
    else:
        query = text(
            """
            SELECT run_time, payload
            FROM gpt.recommendations
            WHERE provider = :provider
              AND payload IS NOT NULL
              AND error IS NULL
            ORDER BY run_time DESC
            LIMIT 1
            """
        )
        params = {"provider": provider}
    result = await conn.execute(query, params)
    row = result.fetchone()
    if row is None and session:
        result = await conn.execute(
            text(
                """
                SELECT run_time, payload
                FROM gpt.recommendations
                WHERE provider = :provider
                  AND payload IS NOT NULL
                  AND error IS NULL
                ORDER BY run_time DESC
                LIMIT 1
                """
            ),
            {"provider": provider},
        )
        row = result.fetchone()
    if row is None:
        return None
    payload = row.payload
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = None
    if not isinstance(payload, dict):
        return None
    return row.run_time, payload


async def _store_challenge(
    conn,
    provider: str,
    session: str,
    run_time: datetime,
    source_provider: Optional[str],
    source_run_time: Optional[datetime],
    source_payload: Optional[Dict[str, Any]],
    payload: Optional[Dict[str, Any]],
    raw_text: Optional[str],
    error: Optional[str],
) -> None:
    payload_json = json.dumps(payload) if payload is not None else None
    source_payload_json = json.dumps(source_payload) if source_payload is not None else None
    error = _redact_secret(error)
    await conn.execute(
        text(
            """
            INSERT INTO gpt.recommendation_challenges
            (provider, session, run_time, source_provider, source_run_time, source_payload, payload, raw_text, error)
            VALUES (:provider, :session, :run_time, :source_provider, :source_run_time, CAST(:source_payload AS JSONB),
                    CAST(:payload AS JSONB), :raw_text, :error)
            ON CONFLICT (provider, session, run_time)
            DO UPDATE SET payload = EXCLUDED.payload,
                          raw_text = EXCLUDED.raw_text,
                          error = EXCLUDED.error,
                          source_provider = EXCLUDED.source_provider,
                          source_run_time = EXCLUDED.source_run_time,
                          source_payload = EXCLUDED.source_payload
            """
        ),
        {
            "provider": provider,
            "session": session,
            "run_time": run_time,
            "source_provider": source_provider,
            "source_run_time": source_run_time,
            "source_payload": source_payload_json,
            "payload": payload_json,
            "raw_text": raw_text,
            "error": error,
        },
    )


async def run_gpt_challenge(session: Optional[str] = None) -> Dict[str, Any]:
    now = datetime.now(tz=UTC)
    session = session or _determine_session(now)
    price_snapshot = await _build_price_snapshot()

    providers: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        async with engine.begin() as conn:
            await _ensure_gpt_challenge_table(conn)
            for provider in ("openai", "deepseek", "gemini"):
                api_key = {
                    "openai": settings.openai_api_key,
                    "deepseek": settings.deepseek_api_key,
                    "gemini": settings.gemini_api_key,
                }.get(provider)

                if not api_key:
                    error = "missing api key"
                    await _store_challenge(conn, provider, session, now, None, None, None, None, None, error)
                    providers.append(
                        {
                            "provider": provider,
                            "session": session,
                            "run_time": now.isoformat(),
                            "recommendations": {},
                            "error": error,
                        }
                    )
                    continue

                source = await _fetch_latest_recommendation(conn, provider, session)
                if not source:
                    error = "no prior recommendations"
                    await _store_challenge(conn, provider, session, now, provider, None, None, None, None, error)
                    providers.append(
                        {
                            "provider": provider,
                            "session": session,
                            "run_time": now.isoformat(),
                            "recommendations": {},
                            "error": error,
                        }
                    )
                    continue

                source_run_time, source_payload = source
                raw_text = None
                payload = None
                error = None
                provider_snapshot = await _extend_price_snapshot(
                    {key: dict(value) for key, value in price_snapshot.items()},
                    source_payload,
                    session,
                )
                prompt = _build_challenge_prompt(session, provider_snapshot, source_payload)
                try:
                    if provider == "openai":
                        raw_text = await _call_openai(client, prompt)
                    elif provider == "deepseek":
                        raw_text = await _call_deepseek(client, prompt)
                    else:
                        raw_text = await _call_gemini(client, prompt)
                    parsed = _extract_json(raw_text or "")
                    if not parsed:
                        raise ValueError("invalid json response")
                    payload = _normalize_challenge_payload(parsed, session)
                    payload = _apply_price_guards(payload, provider_snapshot)
                    payload = _align_challenge_payload(payload, source_payload, session)
                    payload = _apply_price_guards(payload, provider_snapshot)
                except Exception as exc:
                    error = str(exc)

                await _store_challenge(
                    conn,
                    provider,
                    session,
                    now,
                    provider,
                    source_run_time,
                    source_payload,
                    payload,
                    raw_text,
                    error,
                )
                providers.append(
                    {
                        "provider": provider,
                        "session": session,
                        "run_time": now.isoformat(),
                        "source_run_time": source_run_time.isoformat(),
                        "recommendations": (payload or {}).get("categories", {}),
                        "error": _redact_secret(error),
                    }
                )

    return {"session": session, "run_time": now.isoformat(), "providers": providers}


def normalize_challenges(payload: Any, session: str) -> Dict[str, Any]:
    if payload is None:
        return {"as_of": None, "session": session, "categories": {key: [] for key in CATEGORIES}}
    if isinstance(payload, str):
        parsed = _extract_json(payload)
        if parsed:
            return _normalize_challenge_payload(parsed, session)
        return {"as_of": None, "session": session, "categories": {key: [] for key in CATEGORIES}}
    if isinstance(payload, dict):
        return _normalize_challenge_payload(payload, session)
    return {"as_of": None, "session": session, "categories": {key: [] for key in CATEGORIES}}
