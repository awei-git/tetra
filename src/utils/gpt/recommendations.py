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
ENTRY_BOUNDS: Dict[str, Tuple[float, float]] = {
    "large_cap": (0.97, 1.03),
    "growth": (0.97, 1.03),
    "etf": (0.97, 1.03),
    "crypto": (0.95, 1.05),
}

SYSTEM_PROMPT = (
    "You are a quantitative portfolio strategist at a systematic macro fund. "
    "You integrate regime detection, risk analytics, stress scenarios, and factor signals "
    "to generate high-conviction trade ideas. Be selective — quality over quantity. "
    "Only recommend trades where your quantitative evidence is strong. "
    "Respond with JSON only."
)
DEFAULT_GEMINI_MODELS = (
    "gemini-3.1-pro-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
)


def _determine_session(now: datetime) -> str:
    local = now.astimezone(EASTERN)
    close_time = local.replace(hour=16, minute=0, second=0, microsecond=0)
    return "post" if local >= close_time else "pre"


async def _fetch_tetra_context() -> str:
    """Fetch regime, risk, scenarios, and portfolio from Tetra's pipeline."""
    parts = []
    try:
        async with engine.begin() as conn:
            # Regime
            r = await conn.execute(text("""
                SELECT current_regime, current_probs
                FROM simulation.regimes ORDER BY as_of DESC LIMIT 1
            """))
            row = r.fetchone()
            if row:
                parts.append(f"REGIME: {row.current_regime} (probs: {row.current_probs})")

            # Risk
            r = await conn.execute(text("""
                SELECT total_vol_ann, var_95_1d, cvar_95_1d, expected_max_drawdown, effective_n
                FROM simulation.risk WHERE method='parametric' ORDER BY as_of DESC LIMIT 1
            """))
            row = r.fetchone()
            if row:
                parts.append(
                    f"PORTFOLIO RISK: vol={float(row.total_vol_ann):.1%}, "
                    f"VaR95=${float(row.var_95_1d):,.0f}, CVaR95=${float(row.cvar_95_1d):,.0f}, "
                    f"E[MDD]={float(row.expected_max_drawdown):.1%}, Eff.N={float(row.effective_n):.1f}"
                )

            # Scenarios (top 5 worst)
            r = await conn.execute(text("""
                SELECT scenario_name, portfolio_pnl_pct, description
                FROM simulation.scenarios
                WHERE as_of = (SELECT MAX(as_of) FROM simulation.scenarios)
                ORDER BY portfolio_pnl_pct ASC LIMIT 5
            """))
            rows = r.fetchall()
            if rows:
                sc = "; ".join(f"{r.scenario_name}={float(r.portfolio_pnl_pct):+.1%}" for r in rows)
                parts.append(f"WORST SCENARIOS: {sc}")

            # Portfolio positions
            r = await conn.execute(text("""
                SELECT symbol, shares, market_value, weight
                FROM portfolio.positions ORDER BY market_value DESC
            """))
            rows = r.fetchall()
            if rows:
                pos = ", ".join(
                    f"{r.symbol}({float(r.weight)*100:.0f}%=${float(r.market_value):,.0f})"
                    for r in rows
                )
                parts.append(f"CURRENT POSITIONS: {pos}")

            cash_r = await conn.execute(text(
                "SELECT amount FROM portfolio.cash ORDER BY id DESC LIMIT 1"
            ))
            cash_row = cash_r.fetchone()
            if cash_row:
                parts.append(f"CASH: ${float(cash_row.amount):,.0f}")

    except Exception:
        pass
    return "\n".join(parts)


def _build_prompt(session: str, as_of: str, price_snapshot: Dict[str, Dict[str, Any]],
                  tetra_context: str = "") -> str:
    price_json = json.dumps(price_snapshot, separators=(",", ":"), sort_keys=True)

    context_block = ""
    if tetra_context:
        context_block = (
            "\n\nQUANTITATIVE CONTEXT FROM TETRA PIPELINE:\n"
            + tetra_context
            + "\n\nUse this context to inform your recommendations. "
            "If the regime is bearish/stressed, be defensive. "
            "If portfolio is concentrated, suggest diversification. "
            "If scenarios show large downside, suggest hedges.\n\n"
        )

    return (
        f"Today is {as_of}. Session: {session}. "
        + context_block
        + "Return JSON with keys: as_of (YYYY-MM-DD today's date), session (pre or post), categories. "
        "Categories must include large_cap, growth, etf, crypto. "
        "Use only symbols from the price snapshot below. Use USD prices; no commas. "
        "Each category is a list of 3 ideas ranked by conviction. "
        "Each idea must include: symbol, action (buy or sell), "
        "entry (number), target (number), stop (number), "
        "horizon (use format: '1w', '2w', '1m', '3m'), "
        "thesis (string), confidence (0.0 to 1.0 — be precise: 0.9 = very high conviction with strong catalyst, "
        "0.7 = solid thesis with some uncertainty, 0.5 = balanced risk/reward, "
        "0.3 = speculative. Explain your conviction level in thesis), "
        "confidence_reason (1 sentence: why this confidence level). "
        "IMPORTANT - use factor_score and factor_action from the snapshot to guide your picks: "
        "prefer symbols where factor_action aligns with your recommended action. "
        "IMPORTANT - set realistic targets using atr14 (average true range): "
        "for a 1-month horizon, target = entry + 2*atr14 for buys, entry - 2*atr14 for sells. "
        "Stop = entry - 1.5*atr14 for buys, entry + 1.5*atr14 for sells. "
        "If atr14 is null, use +/-5% for ETFs, +/-8% for stocks, +/-15% for crypto. "
        "Entry must be within +/-3% of last_close. "
        "No markdown or extra text. Price snapshot: "
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


def _normalize_item_prices(
    item: Dict[str, Any],
    last_price: float,
    category: str,
    atr14: Optional[float] = None,
) -> Dict[str, Any]:
    bounds = PRICE_BOUNDS.get(category, (0.6, 1.4))
    entry_bounds = ENTRY_BOUNDS.get(category, (0.97, 1.03))
    action = str(item.get("action") or "").lower()
    entry_ok = _entry_within_bounds(item.get("entry"), last_price, entry_bounds)
    target_ok = _entry_within_bounds(item.get("target"), last_price, bounds)
    stop_ok = _entry_within_bounds(item.get("stop"), last_price, bounds)

    # ATR-based fallback multipliers (2x ATR target, 1.5x ATR stop)
    if atr14 and atr14 > 0:
        atr_pct = atr14 / last_price
        target_mult = min(1.0 + 2.0 * atr_pct, bounds[1])
        stop_mult = max(1.0 - 1.5 * atr_pct, bounds[0])
        target_mult_sell = max(1.0 - 2.0 * atr_pct, bounds[0])
        stop_mult_sell = min(1.0 + 1.5 * atr_pct, bounds[1])
    else:
        # Conservative defaults without ATR
        fallbacks = {"etf": (1.05, 0.96), "large_cap": (1.08, 0.94), "growth": (1.1, 0.92), "crypto": (1.15, 0.88)}
        t, s = fallbacks.get(category, (1.08, 0.94))
        target_mult, stop_mult = t, s
        target_mult_sell, stop_mult_sell = 2.0 - t, 2.0 - s

    if not entry_ok:
        item["entry"] = _format_range(last_price * 0.99, last_price * 1.01)
    if not target_ok:
        if action == "buy":
            item["target"] = _format_price(last_price * target_mult)
        elif action == "sell":
            item["target"] = _format_price(last_price * target_mult_sell)
        else:
            item["target"] = _format_price(last_price * target_mult)
    if not stop_ok:
        if action == "buy":
            item["stop"] = _format_price(last_price * stop_mult)
        elif action == "sell":
            item["stop"] = _format_price(last_price * stop_mult_sell)
        else:
            item["stop"] = _format_price(last_price * stop_mult)

    item["last_price"] = float(_format_price(last_price))
    return item


def _apply_price_guards(
    payload: Dict[str, Any],
    price_snapshot: Dict[str, Dict[str, Any]],
    atr_by_symbol: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    categories = payload.get("categories") or {}
    updated: Dict[str, List[Dict[str, Any]]] = {}
    for category, items in categories.items():
        allowed = price_snapshot.get(category, {})
        normalized_items: List[Dict[str, Any]] = []
        for item in items or []:
            symbol = item.get("symbol")
            sym_data = allowed.get(symbol)
            if not symbol or sym_data is None:
                continue
            # sym_data is now a dict with last_close, atr14, etc.
            last_price = sym_data.get("last_close") if isinstance(sym_data, dict) else float(sym_data)
            if last_price is None:
                continue
            atr14 = (atr_by_symbol or {}).get(symbol) or (sym_data.get("atr14") if isinstance(sym_data, dict) else None)
            normalized_items.append(_normalize_item_prices(item, last_price, category, atr14=atr14))
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


async def _fetch_atr_and_factors(symbols: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch ATR(14) and latest factor score for each symbol."""
    if not symbols:
        return {}
    # ATR query: average true range over last 14 trading days
    atr_query = text(
        """
        WITH ranked AS (
          SELECT symbol, high, low, close,
            LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp) AS prev_close,
            ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS rn
          FROM market.ohlcv
          WHERE symbol IN :symbols
        ),
        tr AS (
          SELECT symbol,
            GREATEST(
              high - low,
              ABS(high - prev_close),
              ABS(low  - prev_close)
            ) AS true_range
          FROM ranked
          WHERE rn <= 14 AND prev_close IS NOT NULL
        )
        SELECT symbol, AVG(true_range) AS atr14
        FROM tr
        GROUP BY symbol
        """
    ).bindparams(bindparam("symbols", expanding=True))
    # Factor score query: latest composite score
    factor_query = text(
        """
        SELECT DISTINCT ON (symbol)
          symbol, factor_score, action
        FROM factors.daily_factors
        WHERE symbol IN :symbols
        ORDER BY symbol, as_of DESC
        """
    ).bindparams(bindparam("symbols", expanding=True))
    result: Dict[str, Dict[str, Any]] = {}
    async with engine.begin() as conn:
        try:
            atr_rows = (await conn.execute(atr_query, {"symbols": list(symbols)})).fetchall()
            for row in atr_rows:
                result.setdefault(row.symbol, {})["atr14"] = float(row.atr14) if row.atr14 else None
        except Exception:
            pass
        try:
            factor_rows = (await conn.execute(factor_query, {"symbols": list(symbols)})).fetchall()
            for row in factor_rows:
                result.setdefault(row.symbol, {})["factor_score"] = float(row.factor_score) if row.factor_score is not None else None
                result.setdefault(row.symbol, {})["factor_action"] = row.action
        except Exception:
            pass
    return result


async def _build_price_snapshot() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    """Return (enriched snapshot per category, atr_by_symbol)."""
    symbols: List[str] = []
    for value in CATEGORY_SYMBOLS.values():
        symbols.extend(value)
    unique_symbols = sorted(set(symbols))
    latest_prices = await _fetch_latest_prices(unique_symbols)
    extra = await _fetch_atr_and_factors(unique_symbols)
    snapshot: Dict[str, Dict[str, Any]] = {}
    atr_by_symbol: Dict[str, float] = {}
    for category, syms in CATEGORY_SYMBOLS.items():
        category_data: Dict[str, Any] = {}
        for symbol in syms:
            price = latest_prices.get(symbol)
            if price is None and category == "crypto":
                yf = _fetch_yfinance_last_prices([symbol])
                price = yf.get(symbol)
            if price is None:
                continue
            sym_extra = extra.get(symbol, {})
            atr14 = sym_extra.get("atr14")
            if atr14:
                atr_by_symbol[symbol] = atr14
            category_data[symbol] = {
                "last_close": round(price, 4),
                "atr14": round(atr14, 4) if atr14 else None,
                "factor_score": round(sym_extra.get("factor_score") or 0, 3),
                "factor_action": sym_extra.get("factor_action") or "neutral",
            }
        snapshot[category] = category_data
    return snapshot, atr_by_symbol


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
    as_of = now.astimezone(EASTERN).date().isoformat()
    price_snapshot, atr_by_symbol = await _build_price_snapshot()
    tetra_context = await _fetch_tetra_context()
    prompt = _build_prompt(session, as_of, price_snapshot, tetra_context)

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
                    payload = _apply_price_guards(payload, price_snapshot, atr_by_symbol)
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
