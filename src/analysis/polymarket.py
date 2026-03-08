"""Polymarket → Equity Lead-Lag Scanner with LLM Semantic Filter.

Based on arxiv:2602.07048 (Feb 2026): "LLM as a Risk Manager"
- Stage 1: Granger causality between prediction market probabilities and equity returns
- Stage 2: LLM semantic filter validates economic transmission mechanism
- Result: win rate 51.4% → 54.5%, losing trade magnitude -46%

Pipeline:
1. Build daily probability series from polymarket.snapshots
2. Compute Granger causality against equity returns
3. LLM filters spurious correlations
4. Store validated signals in signals.information_flow
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import text

from src.db.session import engine

logger = logging.getLogger(__name__)
UTC = timezone.utc


async def _fetch_polymarket_series(
    as_of: date, lookback_days: int = 60, min_snapshots: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Fetch daily midpoint probability series for active markets.

    Returns {market_id: {question, category, prices: [(date, midpoint)]}}
    """
    query = text("""
        WITH daily AS (
            SELECT
                s.market_id,
                s.snapshot_time::date AS day,
                AVG(
                    ((s.payload->>'outcomePrices')::jsonb->>0)::float
                ) AS yes_prob
            FROM polymarket.snapshots s
            WHERE s.snapshot_time::date >= :start
              AND s.snapshot_time::date <= :end
              AND s.payload->>'outcomePrices' IS NOT NULL
            GROUP BY s.market_id, s.snapshot_time::date
            HAVING AVG(((s.payload->>'outcomePrices')::jsonb->>0)::float)
                   BETWEEN 0.05 AND 0.95
        )
        SELECT
            d.market_id,
            m.question,
            m.category,
            ARRAY_AGG(d.day ORDER BY d.day) AS dates,
            ARRAY_AGG(d.yes_prob ORDER BY d.day) AS probs
        FROM daily d
        JOIN polymarket.markets m ON d.market_id = m.market_id
        GROUP BY d.market_id, m.question, m.category
        HAVING COUNT(*) >= :min_snapshots
    """)
    start = as_of - timedelta(days=lookback_days)
    async with engine.begin() as conn:
        result = await conn.execute(query, {
            "start": start, "end": as_of, "min_snapshots": min_snapshots,
        })
        rows = result.fetchall()

    series = {}
    for r in rows:
        series[r.market_id] = {
            "question": r.question,
            "category": r.category,
            "prices": list(zip(
                [d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in r.dates],
                [float(p) for p in r.probs],
            )),
        }
    return series


async def _fetch_equity_returns(
    as_of: date, lookback_days: int = 60,
) -> Dict[str, List[Tuple[str, float]]]:
    """Fetch daily returns for all symbols.

    Returns {symbol: [(date_str, return)]}
    """
    query = text("""
        WITH prices AS (
            SELECT symbol,
                   timestamp::date AS day,
                   close,
                   LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp::date) AS prev_close
            FROM market.ohlcv
            WHERE timestamp::date >= CAST(:start AS DATE) - INTERVAL '1 day'
              AND timestamp::date <= :end
        )
        SELECT symbol, day, (close / NULLIF(prev_close, 0)) - 1 AS ret
        FROM prices
        WHERE prev_close IS NOT NULL AND prev_close > 0
        ORDER BY symbol, day
    """)
    start = as_of - timedelta(days=lookback_days)
    async with engine.begin() as conn:
        result = await conn.execute(query, {"start": start, "end": as_of})
        rows = result.fetchall()

    series: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for r in rows:
        series[r.symbol].append((r.day.isoformat(), float(r.ret)))
    return dict(series)


def _granger_causality_f_stat(
    x: List[float], y: List[float], max_lag: int = 5,
) -> Optional[Tuple[float, int]]:
    """Simplified Granger causality test.

    Tests if x (prediction market prob changes) Granger-causes y (equity returns).
    Returns (best_f_stat, best_lag) or None if insufficient data.

    Uses OLS regression comparison:
      Restricted:  y_t = a + sum(b_i * y_{t-i})
      Unrestricted: y_t = a + sum(b_i * y_{t-i}) + sum(c_i * x_{t-i})
    F = ((RSS_r - RSS_u) / p) / (RSS_u / (n - 2p - 1))
    """
    n = len(y)
    if n < max_lag + 10:
        return None

    # Compute changes in x (prob deltas)
    dx = [x[i] - x[i - 1] for i in range(1, len(x))]
    # Align: need y and dx of same length
    min_len = min(len(y), len(dx))
    y = y[-min_len:]
    dx = dx[-min_len:]

    best_f = 0.0
    best_lag = 1

    for lag in range(1, max_lag + 1):
        if len(y) < lag + 10:
            continue

        n_obs = len(y) - lag

        # Build restricted model: y_t ~ y_{t-1}, ..., y_{t-lag}
        Y = np.array(y[lag:])
        X_r = np.column_stack([
            np.array(y[lag - i - 1: -i - 1 if i + 1 < len(y) else None])[:n_obs]
            for i in range(lag)
        ])
        X_r = np.column_stack([np.ones(n_obs), X_r])

        # Build unrestricted: add dx_{t-1}, ..., dx_{t-lag}
        X_u = np.column_stack([
            X_r,
            *[
                np.array(dx[lag - i - 1: -i - 1 if i + 1 < len(dx) else None])[:n_obs]
                for i in range(lag)
            ],
        ])

        try:
            # OLS: beta = (X'X)^-1 X'Y
            beta_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_u, Y, rcond=None)[0]

            resid_r = Y - X_r @ beta_r
            resid_u = Y - X_u @ beta_u

            rss_r = np.sum(resid_r ** 2)
            rss_u = np.sum(resid_u ** 2)

            p = lag  # number of added parameters
            dof = n_obs - 2 * lag - 1

            if dof <= 0 or rss_u <= 0:
                continue

            f_stat = ((rss_r - rss_u) / p) / (rss_u / dof)

            if f_stat > best_f:
                best_f = f_stat
                best_lag = lag
        except (np.linalg.LinAlgError, ValueError):
            continue

    if best_f <= 0:
        return None
    return (best_f, best_lag)


def _f_critical(p: int, dof: int, alpha: float = 0.05) -> float:
    """Approximate F critical value. For proper implementation, use scipy.

    This uses a rough approximation sufficient for screening.
    Real p-values should be computed with scipy.stats.f.sf().
    """
    # Rough critical values for common cases
    # F(1, 30, 0.05) ≈ 4.17, F(2, 30, 0.05) ≈ 3.32, F(3, 30, 0.05) ≈ 2.92
    # F(1, 50, 0.05) ≈ 4.03, F(2, 50, 0.05) ≈ 3.18
    base = {1: 4.0, 2: 3.3, 3: 2.9, 4: 2.7, 5: 2.5}
    return base.get(p, 2.5)


def _align_series(
    pm_prices: List[Tuple[str, float]],
    eq_returns: List[Tuple[str, float]],
) -> Tuple[List[float], List[float]]:
    """Align polymarket probability series with equity return series by date."""
    pm_dict = {d: v for d, v in pm_prices}
    eq_dict = {d: v for d, v in eq_returns}
    common_dates = sorted(set(pm_dict.keys()) & set(eq_dict.keys()))

    pm_aligned = [pm_dict[d] for d in common_dates]
    eq_aligned = [eq_dict[d] for d in common_dates]
    return pm_aligned, eq_aligned


async def _llm_semantic_filter(
    llm_client,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """LLM validates whether candidate lead-lag pairs have economic logic.

    This is the key innovation from the paper: filtering spurious statistical
    correlations by checking for plausible transmission mechanisms.
    """
    if not candidates or not llm_client:
        return candidates

    # Batch candidates into a single prompt
    pairs_desc = []
    for i, c in enumerate(candidates):
        pairs_desc.append(
            f"{i+1}. Prediction market: \"{c['question']}\" → Equity: {c['symbol']} "
            f"(F-stat={c['f_stat']:.2f}, lag={c['lag']}d)"
        )

    prompt = f"""You are a quantitative analyst evaluating potential lead-lag relationships between prediction markets and equities.

For each candidate pair below, assess whether there is a PLAUSIBLE ECONOMIC TRANSMISSION MECHANISM.
A valid mechanism means: changes in the prediction market probability would logically affect the equity price.

Examples of VALID mechanisms:
- "Will US impose tariffs on China?" → semiconductor stocks (tariffs affect supply chain)
- "Will Fed cut rates in March?" → bank stocks (rates affect net interest margin)
- "Will Trump win election?" → renewable energy stocks (policy implications)

Examples of INVALID (spurious) correlations:
- "Will team X win championship?" → tech stocks (no economic link)
- "Will celebrity Y do Z?" → commodity ETFs (no economic link)

CANDIDATES:
{chr(10).join(pairs_desc)}

RESPOND WITH JSON ONLY — an array of objects:
[
  {{"index": 1, "valid": true/false, "mechanism": "brief explanation", "confidence": 0.0-1.0}},
  ...
]
Only mark as valid if there is a clear, logical economic transmission mechanism."""

    system = "You are a quantitative analyst. Respond only with valid JSON array. No markdown."

    try:
        raw = await llm_client.generate(prompt, system=system, temperature=0.1)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        results = json.loads(raw)

        # Merge LLM verdicts back
        verdict_map = {r["index"]: r for r in results}
        filtered = []
        for i, c in enumerate(candidates):
            v = verdict_map.get(i + 1, {})
            c["llm_valid"] = v.get("valid", False)
            c["llm_mechanism"] = v.get("mechanism", "")
            c["llm_confidence"] = v.get("confidence", 0.0)
            if c["llm_valid"]:
                filtered.append(c)

        logger.info(
            f"LLM semantic filter: {len(candidates)} candidates → "
            f"{len(filtered)} validated"
        )
        return filtered

    except Exception as e:
        logger.warning(f"LLM semantic filter failed: {e}, keeping all candidates")
        for c in candidates:
            c["llm_valid"] = None
            c["llm_mechanism"] = "filter_failed"
            c["llm_confidence"] = 0.0
        return candidates


async def _store_signals(
    as_of: date,
    signals: List[Dict[str, Any]],
) -> None:
    """Store validated lead-lag signals."""
    if not signals:
        return

    query = text("""
        INSERT INTO signals.information_flow
          (date, source_market, target_market, gap_signal, gap_z_score, context)
        VALUES
          (:date, :source_market, :target_market, :gap_signal, :gap_z_score,
           :context)
        ON CONFLICT (date, source_market, target_market)
        DO UPDATE SET
          gap_signal = EXCLUDED.gap_signal,
          gap_z_score = EXCLUDED.gap_z_score,
          context = EXCLUDED.context
    """)

    async with engine.begin() as conn:
        for sig in signals:
            await conn.execute(query, {
                "date": as_of,
                "source_market": f"polymarket:{sig['market_id']}",
                "target_market": f"equity:{sig['symbol']}",
                "gap_signal": sig.get("recent_prob_change", 0.0),
                "gap_z_score": sig.get("f_stat", 0.0),
                "context": json.dumps({
                    "question": sig["question"],
                    "lag_days": sig["lag"],
                    "f_stat": sig["f_stat"],
                    "mechanism": sig.get("llm_mechanism", ""),
                    "confidence": sig.get("llm_confidence", 0.0),
                }),
            })


async def run_polymarket_scanner(
    as_of: Optional[date] = None,
    llm_client=None,
    max_lag: int = 5,
    top_n: int = 20,
) -> Dict[str, Any]:
    """Run the Polymarket → Equity lead-lag scanner.

    1. Fetch polymarket probability series + equity returns
    2. Granger causality screen across all pairs
    3. LLM semantic filter on top candidates
    4. Store validated signals

    Returns summary dict.
    """
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()

    logger.info(f"Running Polymarket lead-lag scanner for {as_of}")

    # Fetch data
    pm_series = await _fetch_polymarket_series(as_of, lookback_days=60)
    eq_returns = await _fetch_equity_returns(as_of, lookback_days=60)

    if not pm_series:
        logger.warning("No active Polymarket series with real probabilities found")
        return {
            "as_of": as_of.isoformat(),
            "status": "no_polymarket_data",
            "note": "Polymarket markets table may need refresh — all markets show as closed",
        }

    if not eq_returns:
        logger.warning("No equity return data found")
        return {"as_of": as_of.isoformat(), "status": "no_equity_data"}

    # Screen all pairs
    candidates = []
    pairs_tested = 0

    for mid, pm_data in pm_series.items():
        for symbol, eq_data in eq_returns.items():
            pm_vals, eq_vals = _align_series(pm_data["prices"], eq_data)
            if len(pm_vals) < max_lag + 10:
                continue

            pairs_tested += 1
            result = _granger_causality_f_stat(pm_vals, eq_vals, max_lag)
            if result is None:
                continue

            f_stat, best_lag = result
            f_crit = _f_critical(best_lag, len(eq_vals) - 2 * best_lag - 1)

            if f_stat > f_crit:
                # Compute recent probability change for signal direction
                recent_change = pm_vals[-1] - pm_vals[-min(5, len(pm_vals))]
                candidates.append({
                    "market_id": mid,
                    "question": pm_data["question"],
                    "category": pm_data["category"],
                    "symbol": symbol,
                    "f_stat": round(f_stat, 3),
                    "lag": best_lag,
                    "recent_prob_change": round(recent_change, 4),
                })

    # Sort by F-stat, take top N
    candidates.sort(key=lambda x: x["f_stat"], reverse=True)
    candidates = candidates[:top_n]

    logger.info(
        f"Granger screen: {pairs_tested} pairs tested, "
        f"{len(candidates)} significant at alpha=0.05"
    )

    # LLM semantic filter
    validated = candidates
    if llm_client and candidates:
        validated = await _llm_semantic_filter(llm_client, candidates)

    # Store
    await _store_signals(as_of, validated)

    return {
        "as_of": as_of.isoformat(),
        "status": "success",
        "polymarket_series": len(pm_series),
        "equity_symbols": len(eq_returns),
        "pairs_tested": pairs_tested,
        "candidates_found": len(candidates),
        "validated_signals": len(validated),
        "signals": [
            {
                "question": s["question"],
                "symbol": s["symbol"],
                "f_stat": s["f_stat"],
                "lag": s["lag"],
                "mechanism": s.get("llm_mechanism", ""),
            }
            for s in validated[:10]
        ],
    }
