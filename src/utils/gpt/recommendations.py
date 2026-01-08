"""LLM-powered trade recommendation helpers."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy import bindparam, text

from config.config import settings
from src.db.session import engine
from src.definitions.market_universe import MarketUniverse
from src.utils.ingestion.yfinance import fetch_yfinance_ohlcv, yfinance_available

UTC = timezone.utc
EASTERN = ZoneInfo("America/New_York")
CATEGORIES = ("large_cap", "growth", "etf", "crypto")
CATEGORY_SYMBOLS: Dict[str, List[str]] = {
    "large_cap": MarketUniverse.MEGA_CAP_STOCKS[:12],
    "growth": MarketUniverse.HIGH_GROWTH_STOCKS[:12],
    "etf": (MarketUniverse.INDEX_ETFS + MarketUniverse.SECTOR_ETFS + MarketUniverse.THEMATIC_ETFS)[:12],
    "crypto": MarketUniverse.CRYPTO_SYMBOLS[:8],
}
PRICE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "large_cap": (0.7, 1.3),
    "growth": (0.6, 1.4),
    "etf": (0.8, 1.2),
    "crypto": (0.4, 2.5),
}

SYSTEM_PROMPT = (
    "You are a markets strategist. Provide concise trade ideas with clear targets. "
    "Respond with JSON only."
)
DEFAULT_GEMINI_MODELS = (
    "gemini-3.0-flash",
    "gemini-3.0-pro",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
)


def _determine_session(now: datetime) -> str:
    local = now.astimezone(EASTERN)
    close_time = local.replace(hour=16, minute=0, second=0, microsecond=0)
    return "post" if local >= close_time else "pre"


def _build_prompt(session: str, price_snapshot: Dict[str, Dict[str, float]]) -> str:
    price_json = json.dumps(price_snapshot, separators=(",", ":"), sort_keys=True)
    return (
        "Return JSON with keys: as_of (YYYY-MM-DD), session (pre or post), categories. "
        "Categories must include large_cap, growth, etf, crypto. "
        "Use only symbols from the price snapshot below. Use USD prices; no commas. "
        "Each category is a list of 3 ideas. Each idea must include: symbol, action (buy or sell), "
        "entry, target, stop, horizon, thesis, confidence (0 to 1). Use price ranges if unsure. "
        "Entry/target/stop must be within +/-20% of last_close for stocks/ETFs and +/-40% for crypto. "
        "No markdown or extra text. Session: "
        + session
        + ". Price snapshot: "
        + price_json
    )


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _normalize_payload(payload: Dict[str, Any], session: str) -> Dict[str, Any]:
    categories = payload.get("categories")
    if not isinstance(categories, dict):
        categories = {key: payload.get(key, []) for key in CATEGORIES}

    normalized: Dict[str, List[Dict[str, Any]]] = {}
    for key in CATEGORIES:
        items = categories.get(key) or []
        if not isinstance(items, list):
            items = []
        for item in items:
            if isinstance(item, dict) and "confidence" in item:
                item["confidence"] = _parse_confidence(item.get("confidence"))
        normalized[key] = items

    as_of = payload.get("as_of") or payload.get("date")
    if not as_of:
        as_of = datetime.now(tz=UTC).date().isoformat()

    return {
        "as_of": as_of,
        "session": payload.get("session") or session,
        "categories": normalized,
    }


def _parse_price_range(value: Any) -> Tuple[Optional[float], Optional[float]]:
    if value is None:
        return None, None
    text_value = str(value)
    text_value = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " ", text_value)
    matches = re.findall(r"\d[\d,]*\.?\d*", text_value)
    if not matches:
        return None, None
    numbers = []
    for match in matches:
        try:
            numbers.append(float(match.replace(",", "")))
        except ValueError:
            continue
    if not numbers:
        return None, None
    return min(numbers), max(numbers)


def _parse_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        conf = float(value)
    else:
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


def _format_price(value: float) -> str:
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _format_range(low: float, high: float) -> str:
    return f"{_format_price(low)}-{_format_price(high)}"


def _entry_within_bounds(entry: Any, last_price: float, bounds: Tuple[float, float]) -> bool:
    if not last_price or last_price <= 0:
        return False
    low, high = _parse_price_range(entry)
    if low is None or high is None:
        return False
    return (low / last_price) >= bounds[0] and (high / last_price) <= bounds[1]


def _normalize_item_prices(item: Dict[str, Any], last_price: float, category: str) -> Dict[str, Any]:
    bounds = PRICE_BOUNDS.get(category, (0.6, 1.4))
    action = str(item.get("action") or "").lower()
    entry_ok = _entry_within_bounds(item.get("entry"), last_price, bounds)
    target_ok = _entry_within_bounds(item.get("target"), last_price, bounds)
    stop_ok = _entry_within_bounds(item.get("stop"), last_price, bounds)

    if not entry_ok:
        item["entry"] = _format_range(last_price * 0.99, last_price * 1.01)
    if not target_ok:
        if action == "buy":
            item["target"] = _format_price(last_price * 1.1)
        elif action == "sell":
            item["target"] = _format_price(last_price * 0.9)
        else:
            item["target"] = _format_price(last_price * 1.03)
    if not stop_ok:
        if action == "buy":
            item["stop"] = _format_price(last_price * 0.95)
        elif action == "sell":
            item["stop"] = _format_price(last_price * 1.05)
        else:
            item["stop"] = _format_price(last_price * 0.97)

    item["last_price"] = float(_format_price(last_price))
    return item


def _apply_price_guards(payload: Dict[str, Any], price_snapshot: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    categories = payload.get("categories") or {}
    updated: Dict[str, List[Dict[str, Any]]] = {}
    for category, items in categories.items():
        allowed = price_snapshot.get(category, {})
        normalized_items: List[Dict[str, Any]] = []
        for item in items or []:
            symbol = item.get("symbol")
            last_price = allowed.get(symbol)
            if not symbol or last_price is None:
                continue
            normalized_items.append(_normalize_item_prices(item, last_price, category))
        updated[category] = normalized_items
    payload["categories"] = updated
    return payload


def _redact_secret(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    redacted = value
    for secret in (settings.openai_api_key, settings.deepseek_api_key, settings.gemini_api_key):
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    redacted = re.sub(r"(key=)[^&\s]+", r"\1[redacted]", redacted)
    redacted = re.sub(r"(api_key=)[^&\s]+", r"\1[redacted]", redacted)
    redacted = re.sub(r"(key=\[redacted\])[^\s]*", r"\1", redacted)
    redacted = re.sub(r"(api_key=\[redacted\])[^\s]*", r"\1", redacted)
    return redacted


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


def _iter_gemini_models() -> Iterable[str]:
    seen = set()
    candidates = []
    if settings.gemini_model:
        candidates.append(settings.gemini_model)
    candidates.extend(settings.gemini_models)
    candidates.extend(DEFAULT_GEMINI_MODELS)
    for model in candidates:
        if not model or model in seen:
            continue
        seen.add(model)
        yield model


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
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {400, 404}:
                last_error = exc
                continue
            raise
        except Exception as exc:
            last_error = exc
            break
    if last_error:
        raise RuntimeError(f"gemini model not available (tried: {', '.join(tried)})") from last_error
    raise RuntimeError("gemini model list is empty")


async def _fetch_latest_prices(symbols: Sequence[str]) -> Dict[str, float]:
    if not symbols:
        return {}
    query = text(
        """
        SELECT DISTINCT ON (symbol)
          symbol,
          close
        FROM market.ohlcv
        WHERE symbol IN :symbols
        ORDER BY symbol, timestamp DESC
        """
    ).bindparams(bindparam("symbols", expanding=True))
    async with engine.begin() as conn:
        result = await conn.execute(query, {"symbols": list(symbols)})
        rows = result.fetchall()
    prices: Dict[str, float] = {}
    for row in rows:
        if row.close is None:
            continue
        try:
            prices[row.symbol] = float(row.close)
        except (TypeError, ValueError):
            continue
    return prices


def _fetch_yfinance_last_prices(symbols: Sequence[str]) -> Dict[str, float]:
    if not symbols or not yfinance_available():
        return {}
    end = date.today()
    start = end - timedelta(days=7)
    prices: Dict[str, float] = {}
    for symbol in symbols:
        try:
            rows = fetch_yfinance_ohlcv(symbol, start, end)
        except Exception:
            continue
        for row in reversed(rows):
            close = row.get("close")
            if close is None:
                continue
            try:
                prices[symbol] = float(close)
                break
            except (TypeError, ValueError):
                continue
    return prices


async def _build_price_snapshot() -> Dict[str, Dict[str, float]]:
    symbols: List[str] = []
    for value in CATEGORY_SYMBOLS.values():
        symbols.extend(value)
    unique_symbols = sorted(set(symbols))
    latest_prices = await _fetch_latest_prices(unique_symbols)
    snapshot: Dict[str, Dict[str, float]] = {}
    for category, symbols in CATEGORY_SYMBOLS.items():
        category_prices: Dict[str, float] = {}
        for symbol in symbols:
            price = latest_prices.get(symbol)
            if price is not None:
                category_prices[symbol] = price
        if category == "crypto":
            yf_prices = _fetch_yfinance_last_prices(symbols)
            if yf_prices:
                category_prices.update(yf_prices)
        snapshot[category] = category_prices
    return snapshot


async def _ensure_gpt_table(conn) -> None:
    await conn.execute(text("CREATE SCHEMA IF NOT EXISTS gpt"))
    await conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS gpt.recommendations (
              id BIGSERIAL PRIMARY KEY,
              provider VARCHAR(32) NOT NULL,
              session VARCHAR(16) NOT NULL,
              run_time TIMESTAMPTZ NOT NULL,
              payload JSONB,
              raw_text TEXT,
              error TEXT,
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              CONSTRAINT gpt_recommendations_provider_session_run_time_key
                UNIQUE (provider, session, run_time)
            )
            """
        )
    )


async def _store_recommendation(
    conn,
    provider: str,
    session: str,
    run_time: datetime,
    payload: Optional[Dict[str, Any]],
    raw_text: Optional[str],
    error: Optional[str],
) -> None:
    payload_json = json.dumps(payload) if payload is not None else None
    error = _redact_secret(error)
    await conn.execute(
        text(
            """
            INSERT INTO gpt.recommendations
            (provider, session, run_time, payload, raw_text, error)
            VALUES (:provider, :session, :run_time, CAST(:payload AS JSONB), :raw_text, :error)
            ON CONFLICT (provider, session, run_time)
            DO UPDATE SET payload = EXCLUDED.payload,
                          raw_text = EXCLUDED.raw_text,
                          error = EXCLUDED.error
            """
        ),
        {
            "provider": provider,
            "session": session,
            "run_time": run_time,
            "payload": payload_json,
            "raw_text": raw_text,
            "error": error,
        },
    )


async def run_gpt_recommendations(session: Optional[str] = None) -> Dict[str, Any]:
    now = datetime.now(tz=UTC)
    session = session or _determine_session(now)
    price_snapshot = await _build_price_snapshot()
    prompt = _build_prompt(session, price_snapshot)

    providers: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(40.0, connect=10.0)) as client:
        async with engine.begin() as conn:
            await _ensure_gpt_table(conn)
            for provider in ("openai", "deepseek", "gemini"):
                api_key = {
                    "openai": settings.openai_api_key,
                    "deepseek": settings.deepseek_api_key,
                    "gemini": settings.gemini_api_key,
                }.get(provider)

                if not api_key:
                    error = "missing api key"
                    await _store_recommendation(conn, provider, session, now, None, None, error)
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

                raw_text = None
                payload = None
                error = None
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
                    payload = _normalize_payload(parsed, session)
                    payload = _apply_price_guards(payload, price_snapshot)
                except Exception as exc:
                    error = str(exc)

                await _store_recommendation(conn, provider, session, now, payload, raw_text, error)
                providers.append(
                    {
                        "provider": provider,
                        "session": session,
                        "run_time": now.isoformat(),
                        "recommendations": (payload or {}).get("categories", {}),
                        "error": _redact_secret(error),
                    }
                )

    return {"session": session, "run_time": now.isoformat(), "providers": providers}


def normalize_recommendations(payload: Any, session: str) -> Dict[str, Any]:
    if payload is None:
        return {"as_of": None, "session": session, "categories": {key: [] for key in CATEGORIES}}
    if isinstance(payload, str):
        parsed = _extract_json(payload)
        if parsed:
            return _normalize_payload(parsed, session)
        return {"as_of": None, "session": session, "categories": {key: [] for key in CATEGORIES}}
    if isinstance(payload, dict):
        return _normalize_payload(payload, session)
    return {"as_of": None, "session": session, "categories": {key: [] for key in CATEGORIES}}
