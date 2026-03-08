"""Inference pipelines for signal quality, event impact, and Polymarket metrics."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.schema import (
    inference_event_study,
    inference_polymarket_bins,
    inference_polymarket_summary,
    inference_signal_leaderboard,
)
from src.db.session import engine
from src.utils.factors.definitions import get_factor_definitions
from src.utils.factors.scoring import build_factor_stats, compute_factor_signal

UTC = timezone.utc


def _rank(values: List[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    var_x = sum((a - mean_x) ** 2 for a in x)
    var_y = sum((b - mean_y) ** 2 for b in y)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / (var_x * var_y) ** 0.5


def _spearman(x: List[float], y: List[float]) -> Optional[float]:
    return _pearson(_rank(x), _rank(y))


async def _resolve_as_of(table: str, column: str = "as_of") -> Optional[date]:
    query = text(f"SELECT MAX({column}) FROM {table}")
    async with engine.begin() as conn:
        result = await conn.execute(query)
        return result.scalar_one_or_none()


async def _load_factor_rows(start: date, end: date) -> List[Tuple[str, date, str, float]]:
    query = text(
        """
        SELECT symbol, as_of, factor, value
        FROM factors.daily_factors
        WHERE as_of BETWEEN :start AND :end
          AND symbol <> '__macro__'
        """
    )
    async with engine.begin() as conn:
        result = await conn.execute(query, {"start": start, "end": end})
        rows = result.fetchall()
    return [(row.symbol, row.as_of, row.factor, float(row.value)) for row in rows if row.value is not None]


async def _load_close_series(
    symbols: Sequence[str],
    start: date,
    end: date,
) -> Dict[str, List[Tuple[date, float]]]:
    if not symbols:
        return {}
    query = text(
        """
        SELECT symbol, DATE(timestamp) AS day, close
        FROM market.ohlcv
        WHERE symbol = ANY(:symbols)
          AND timestamp >= :start
          AND timestamp <= :end
        ORDER BY symbol, day
        """
    )
    async with engine.begin() as conn:
        result = await conn.execute(
            query,
            {
                "symbols": list(symbols),
                "start": datetime.combine(start, datetime.min.time(), tzinfo=UTC),
                "end": datetime.combine(end, datetime.max.time(), tzinfo=UTC),
            },
        )
        rows = result.fetchall()

    series: Dict[str, List[Tuple[date, float]]] = defaultdict(list)
    for row in rows:
        if row.close is None:
            continue
        series[row.symbol].append((row.day, float(row.close)))
    return series


def _compute_forward_returns(
    series: List[Tuple[date, float]],
    horizons: Sequence[int],
) -> Dict[Tuple[date, int], float]:
    returns: Dict[Tuple[date, int], float] = {}
    for idx, (day, price) in enumerate(series):
        if price == 0:
            continue
        for horizon in horizons:
            future_idx = idx + horizon
            if future_idx >= len(series):
                continue
            future_price = series[future_idx][1]
            returns[(day, horizon)] = (future_price / price) - 1.0
    return returns


async def run_signal_leaderboard(
    as_of: Optional[date] = None,
    lookback_days: int = 180,
    horizons: Sequence[int] = (1, 5, 20),
    min_obs: int = 30,
) -> Dict[str, Any]:
    target_date = as_of or await _resolve_as_of("factors.daily_factors")
    if target_date is None:
        return {"status": "no_data"}
    start_date = target_date - timedelta(days=lookback_days)

    factor_rows = await _load_factor_rows(start_date, target_date)
    if not factor_rows:
        return {"status": "no_data"}

    definitions = get_factor_definitions()
    stats = build_factor_stats(
        [(symbol, factor, value) for symbol, _, factor, value in factor_rows],
        definitions,
    )

    signals_by_factor_date: Dict[str, Dict[date, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    symbols: set[str] = set()
    for symbol, as_of_date, factor, value in factor_rows:
        definition = definitions.get(factor)
        if not definition:
            continue
        signal = compute_factor_signal(factor, value, definition, stats)
        if signal is None:
            continue
        signals_by_factor_date[factor][as_of_date][symbol] = signal
        symbols.add(symbol)

    close_series = await _load_close_series(
        sorted(symbols),
        start_date,
        target_date + timedelta(days=max(horizons) + 5),
    )
    returns_by_symbol: Dict[str, Dict[Tuple[date, int], float]] = {}
    for symbol, series in close_series.items():
        returns_by_symbol[symbol] = _compute_forward_returns(series, horizons)

    rows: List[Dict[str, Any]] = []
    for factor, date_map in signals_by_factor_date.items():
        for horizon in horizons:
            daily_ics: List[float] = []
            observations = 0
            for as_of_date, symbol_map in date_map.items():
                signals: List[float] = []
                returns: List[float] = []
                for symbol, signal in symbol_map.items():
                    ret = returns_by_symbol.get(symbol, {}).get((as_of_date, horizon))
                    if ret is None:
                        continue
                    signals.append(signal)
                    returns.append(ret)
                if len(signals) < min_obs:
                    continue
                ic = _spearman(signals, returns)
                if ic is None:
                    continue
                daily_ics.append(ic)
                observations += len(signals)

            if not daily_ics:
                continue
            avg_ic = sum(daily_ics) / len(daily_ics)
            hit_rate = sum(1 for value in daily_ics if value > 0) / len(daily_ics)
            rows.append(
                {
                    "factor": factor,
                    "horizon_days": horizon,
                    "as_of": target_date,
                    "start_date": start_date,
                    "end_date": target_date,
                    "avg_ic": avg_ic,
                    "median_ic": median(daily_ics),
                    "hit_rate": hit_rate,
                    "days": len(daily_ics),
                    "observations": observations,
                }
            )

    async with engine.begin() as conn:
        for row in rows:
            stmt = pg_insert(inference_signal_leaderboard).values(row)
            await conn.execute(
                stmt.on_conflict_do_update(
                    index_elements=[
                        inference_signal_leaderboard.c.factor,
                        inference_signal_leaderboard.c.horizon_days,
                        inference_signal_leaderboard.c.as_of,
                    ],
                    set_={
                        "start_date": stmt.excluded.start_date,
                        "end_date": stmt.excluded.end_date,
                        "avg_ic": stmt.excluded.avg_ic,
                        "median_ic": stmt.excluded.median_ic,
                        "hit_rate": stmt.excluded.hit_rate,
                        "days": stmt.excluded.days,
                        "observations": stmt.excluded.observations,
                        "created_at": stmt.excluded.created_at,
                    },
                )
            )

    return {"status": "ok", "rows": len(rows), "as_of": target_date.isoformat()}


async def run_event_study(
    as_of: Optional[date] = None,
    lookback_days: int = 365,
    windows: Sequence[int] = (1, 3, 5),
    min_obs: int = 30,
) -> Dict[str, Any]:
    target_date = as_of or await _resolve_as_of("event.events", column="event_time")
    if target_date is None:
        return {"status": "no_data"}
    if isinstance(target_date, datetime):
        target_date = target_date.date()
    start_date = target_date - timedelta(days=lookback_days)

    query = text(
        """
        SELECT symbol, event_type, event_time
        FROM event.events
        WHERE event_time BETWEEN :start AND :end
          AND symbol IS NOT NULL
        """
    )
    async with engine.begin() as conn:
        result = await conn.execute(
            query,
            {
                "start": datetime.combine(start_date, datetime.min.time(), tzinfo=UTC),
                "end": datetime.combine(target_date, datetime.max.time(), tzinfo=UTC),
            },
        )
        event_rows = result.fetchall()

    events: List[Tuple[str, str, date]] = []
    symbols: set[str] = set()
    for row in event_rows:
        symbol = str(row.symbol or "").upper()
        if not symbol:
            continue
        event_day = row.event_time.date()
        events.append((symbol, row.event_type, event_day))
        symbols.add(symbol)

    if not events:
        return {"status": "no_data"}

    close_series = await _load_close_series(
        sorted(symbols),
        start_date - timedelta(days=max(windows)),
        target_date + timedelta(days=max(windows)),
    )
    index_map: Dict[str, Dict[date, int]] = {}
    for symbol, series in close_series.items():
        index_map[symbol] = {day: idx for idx, (day, _) in enumerate(series)}

    returns_by_type_window: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for symbol, event_type, event_day in events:
        series = close_series.get(symbol)
        if not series:
            continue
        idx = index_map.get(symbol, {}).get(event_day)
        if idx is None:
            continue
        for window in windows:
            start_idx = idx - (window // 2)
            end_idx = idx + (window // 2) + (0 if window % 2 == 0 else 0)
            if window == 1:
                start_idx = idx
                end_idx = idx + 1
            if start_idx < 0 or end_idx >= len(series):
                continue
            start_price = series[start_idx][1]
            end_price = series[end_idx][1]
            if start_price == 0:
                continue
            returns_by_type_window[(event_type, window)].append((end_price / start_price) - 1.0)

    rows: List[Dict[str, Any]] = []
    for (event_type, window), values in returns_by_type_window.items():
        if len(values) < min_obs:
            continue
        rows.append(
            {
                "event_type": event_type,
                "window_days": window,
                "as_of": target_date,
                "start_date": start_date,
                "end_date": target_date,
                "avg_return": sum(values) / len(values),
                "median_return": median(values),
                "observations": len(values),
            }
        )

    async with engine.begin() as conn:
        for row in rows:
            stmt = pg_insert(inference_event_study).values(row)
            await conn.execute(
                stmt.on_conflict_do_update(
                    index_elements=[
                        inference_event_study.c.event_type,
                        inference_event_study.c.window_days,
                        inference_event_study.c.as_of,
                    ],
                    set_={
                        "start_date": stmt.excluded.start_date,
                        "end_date": stmt.excluded.end_date,
                        "avg_return": stmt.excluded.avg_return,
                        "median_return": stmt.excluded.median_return,
                        "observations": stmt.excluded.observations,
                        "created_at": stmt.excluded.created_at,
                    },
                )
            )

    return {"status": "ok", "rows": len(rows), "as_of": target_date.isoformat()}


async def run_polymarket_calibration(as_of: Optional[date] = None) -> Dict[str, Any]:
    query_snapshot = text("SELECT MAX(snapshot_time) FROM polymarket.snapshots")
    async with engine.begin() as conn:
        result = await conn.execute(query_snapshot)
        latest_snapshot = result.scalar_one_or_none()
    if latest_snapshot is None:
        return {"status": "no_data"}

    query = text(
        """
        SELECT m.market_id, m.payload, s.best_bid, s.best_ask, s.volume, m.closed, m.active
        FROM polymarket.markets m
        JOIN polymarket.snapshots s
          ON s.market_id = m.market_id
         AND s.snapshot_time = :snapshot_time
        """
    )
    async with engine.begin() as conn:
        result = await conn.execute(query, {"snapshot_time": latest_snapshot})
        rows = result.fetchall()

    bins = [(i / 10, (i + 1) / 10) for i in range(10)]
    bin_stats: Dict[Tuple[float, float], Dict[str, float]] = {
        bin_range: {"count": 0, "sum_pred": 0.0, "sum_outcome": 0.0, "sum_brier": 0.0}
        for bin_range in bins
    }

    markets = 0
    closed_markets = 0
    resolved_proxy = 0
    spreads: List[float] = []
    volumes: List[float] = []
    brier_scores: List[float] = []

    for row in rows:
        markets += 1
        payload = row.payload or {}
        outcomes = payload.get("outcomes") or []
        outcome_prices = payload.get("outcomePrices") or []
        if not outcomes or len(outcomes) < 2 or len(outcome_prices) < 2:
            continue
        try:
            price_yes = float(outcome_prices[0])
            price_no = float(outcome_prices[1])
        except (TypeError, ValueError):
            continue
        best_bid = row.best_bid
        best_ask = row.best_ask
        if best_bid is not None and best_ask is not None:
            spreads.append(float(best_ask) - float(best_bid))
        if row.volume is not None:
            volumes.append(float(row.volume))

        closed = bool(row.closed)
        if closed:
            closed_markets += 1
        max_price = max(price_yes, price_no)
        if closed and max_price >= 0.9:
            outcome = 1.0 if price_yes >= price_no else 0.0
            pred = price_yes
            brier = (pred - outcome) ** 2
            brier_scores.append(brier)
            resolved_proxy += 1
            for bin_low, bin_high in bins:
                if pred >= bin_low and pred <= bin_high:
                    stats = bin_stats[(bin_low, bin_high)]
                    stats["count"] += 1
                    stats["sum_pred"] += pred
                    stats["sum_outcome"] += outcome
                    stats["sum_brier"] += brier
                    break

    as_of_date = as_of or latest_snapshot.date()
    summary_row = {
        "as_of": as_of_date,
        "markets": markets,
        "closed_markets": closed_markets,
        "resolved_proxy": resolved_proxy,
        "avg_spread": sum(spreads) / len(spreads) if spreads else None,
        "avg_volume": sum(volumes) / len(volumes) if volumes else None,
        "avg_brier": sum(brier_scores) / len(brier_scores) if brier_scores else None,
    }

    bin_rows: List[Dict[str, Any]] = []
    for bin_low, bin_high in bins:
        stats = bin_stats[(bin_low, bin_high)]
        if stats["count"] == 0:
            continue
        bin_rows.append(
            {
                "as_of": as_of_date,
                "bin_low": Decimal(str(bin_low)),
                "bin_high": Decimal(str(bin_high)),
                "count": int(stats["count"]),
                "avg_pred": stats["sum_pred"] / stats["count"],
                "proxy_accuracy": stats["sum_outcome"] / stats["count"],
                "avg_brier": stats["sum_brier"] / stats["count"],
            }
        )

    async with engine.begin() as conn:
        stmt = pg_insert(inference_polymarket_summary).values(summary_row)
        await conn.execute(
            stmt.on_conflict_do_update(
                index_elements=[inference_polymarket_summary.c.as_of],
                set_={
                    "markets": stmt.excluded.markets,
                    "closed_markets": stmt.excluded.closed_markets,
                    "resolved_proxy": stmt.excluded.resolved_proxy,
                    "avg_spread": stmt.excluded.avg_spread,
                    "avg_volume": stmt.excluded.avg_volume,
                    "avg_brier": stmt.excluded.avg_brier,
                    "created_at": stmt.excluded.created_at,
                },
            )
        )
        for row in bin_rows:
            stmt = pg_insert(inference_polymarket_bins).values(row)
            await conn.execute(
                stmt.on_conflict_do_update(
                    index_elements=[
                        inference_polymarket_bins.c.as_of,
                        inference_polymarket_bins.c.bin_low,
                        inference_polymarket_bins.c.bin_high,
                    ],
                    set_={
                        "count": stmt.excluded.count,
                        "avg_pred": stmt.excluded.avg_pred,
                        "proxy_accuracy": stmt.excluded.proxy_accuracy,
                        "avg_brier": stmt.excluded.avg_brier,
                        "created_at": stmt.excluded.created_at,
                    },
                )
            )

    return {"status": "ok", "as_of": as_of_date.isoformat(), "bins": len(bin_rows)}


async def run_all_inference(as_of: Optional[date] = None) -> Dict[str, Any]:
    signal = await run_signal_leaderboard(as_of=as_of)
    events = await run_event_study(as_of=as_of)
    polymarket = await run_polymarket_calibration(as_of=as_of)
    return {
        "signal": signal,
        "events": events,
        "polymarket": polymarket,
        "run_time": datetime.now(tz=UTC).isoformat(),
    }
