"""Generate GPT-written summary of consolidated recommendations."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from sqlalchemy import text

from config.config import settings
from src.db.session import engine
from src.utils.gpt.recommendations import DEFAULT_GEMINI_MODELS, _extract_json, _redact_secret

UTC = timezone.utc

SYSTEM_PROMPT = (
    "You are a trading strategist summarizing consensus trade recommendations. "
    "Be concise, decisive, and avoid disclaimers. Respond with JSON only."
)


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


def _build_prompt(as_of: date, session: Optional[str], payload: Dict[str, Any]) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return (
        "Return JSON with keys: summary, buy, sell. "
        "summary must be 2-4 sentences and state the overall consensus. "
        "buy and sell are arrays of {symbol, confidence} (max 5 each). "
        "Use only the provided data. No markdown. "
        f"Session: {session or 'unknown'}. "
        f"As of: {as_of.isoformat()}. "
        f"Data: {payload_json}"
    )


def _parse_summary(raw_text: str) -> Dict[str, Any]:
    parsed = _extract_json(raw_text)
    if isinstance(parsed, dict):
        return parsed
    return {"summary": raw_text.strip()}


async def _ensure_summary_table(conn) -> None:
    await conn.execute(text("CREATE SCHEMA IF NOT EXISTS gpt"))
    await conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS gpt.recommendation_summaries (
              id BIGSERIAL PRIMARY KEY,
              session VARCHAR(16),
              run_time TIMESTAMPTZ NOT NULL,
              as_of DATE,
              provider VARCHAR(32),
              payload JSONB,
              raw_text TEXT,
              error TEXT,
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              CONSTRAINT gpt_recommendation_summaries_session_run_time_key
                UNIQUE (session, run_time)
            )
            """
        )
    )


async def _load_summary(conn, session: Optional[str], run_time: datetime) -> Optional[Dict[str, Any]]:
    result = await conn.execute(
        text(
            """
            SELECT session, run_time, as_of, provider, payload, error
            FROM gpt.recommendation_summaries
            WHERE session IS NOT DISTINCT FROM :session
              AND run_time = :run_time
            ORDER BY created_at DESC
            LIMIT 1
            """
        ),
        {"session": session, "run_time": run_time},
    )
    row = result.fetchone()
    if not row:
        return None
    payload = row.payload
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = None
    return {
        "session": row.session,
        "run_time": row.run_time,
        "as_of": row.as_of,
        "provider": row.provider,
        "payload": payload,
        "error": row.error,
    }


async def _store_summary(
    conn,
    session: Optional[str],
    run_time: datetime,
    as_of: date,
    provider: Optional[str],
    payload: Optional[Dict[str, Any]],
    raw_text: Optional[str],
    error: Optional[str],
) -> None:
    payload_json = json.dumps(payload) if payload is not None else None
    error = _redact_secret(error)
    await conn.execute(
        text(
            """
            INSERT INTO gpt.recommendation_summaries
              (session, run_time, as_of, provider, payload, raw_text, error)
            VALUES (:session, :run_time, :as_of, :provider, CAST(:payload AS JSONB), :raw_text, :error)
            ON CONFLICT (session, run_time)
            DO UPDATE SET payload = EXCLUDED.payload,
                         raw_text = EXCLUDED.raw_text,
                         error = EXCLUDED.error,
                         provider = EXCLUDED.provider,
                         as_of = EXCLUDED.as_of
            """
        ),
        {
            "session": session,
            "run_time": run_time,
            "as_of": as_of,
            "provider": provider,
            "payload": payload_json,
            "raw_text": raw_text,
            "error": error,
        },
    )


def _summarize_inputs(consolidated: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    buys = [item for item in consolidated if item.get("final_action") == "buy"]
    sells = [item for item in consolidated if item.get("final_action") == "sell"]
    top_buys = sorted(buys, key=lambda item: item.get("confidence") or 0, reverse=True)[:5]
    top_sells = sorted(sells, key=lambda item: item.get("confidence") or 0, reverse=True)[:5]

    def _pack(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        packed: List[Dict[str, Any]] = []
        for item in items:
            packed.append(
                {
                    "symbol": item.get("symbol"),
                    "category": item.get("category"),
                    "confidence": item.get("confidence"),
                }
            )
        return packed

    return {
        "counts": {"buy": len(buys), "sell": len(sells), "total": len(consolidated)},
        "top_buy": _pack(top_buys),
        "top_sell": _pack(top_sells),
    }


async def generate_summary(
    consolidated: Sequence[Dict[str, Any]],
    session: Optional[str],
    as_of: date,
    run_time: datetime,
) -> Dict[str, Any]:
    async with engine.begin() as conn:
        await _ensure_summary_table(conn)
        existing = await _load_summary(conn, session, run_time)
        if existing and existing.get("payload"):
            payload = existing["payload"]
            return {
                "provider": existing.get("provider"),
                "payload": payload,
                "error": existing.get("error"),
            }

        if not consolidated:
            payload = {"summary": "No aligned buy/sell consensus yet."}
            await _store_summary(conn, session, run_time, as_of, None, payload, None, None)
            return {"provider": None, "payload": payload, "error": None}

        summary_input = _summarize_inputs(consolidated)
        prompt = _build_prompt(as_of, session, summary_input)

        provider = None
        raw_text = None
        payload = None
        error = None
        async with httpx.AsyncClient(timeout=45.0) as client:
            for candidate in ("openai", "deepseek", "gemini"):
                api_key = {
                    "openai": settings.openai_api_key,
                    "deepseek": settings.deepseek_api_key,
                    "gemini": settings.gemini_api_key,
                }.get(candidate)
                if not api_key:
                    continue
                try:
                    if candidate == "openai":
                        raw_text = await _call_openai(client, prompt)
                    elif candidate == "deepseek":
                        raw_text = await _call_deepseek(client, prompt)
                    else:
                        raw_text = await _call_gemini(client, prompt)
                    payload = _parse_summary(raw_text or "")
                    provider = candidate
                    break
                except Exception as exc:
                    error = str(exc)
                    continue

        await _store_summary(conn, session, run_time, as_of, provider, payload, raw_text, error)
        return {"provider": provider, "payload": payload, "error": error}
