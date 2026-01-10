"""Compute daily factor values from events, news, econ, and minimal market data."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import text
from zoneinfo import ZoneInfo

from src.db.session import engine

UTC = timezone.utc
EASTERN = ZoneInfo("America/New_York")

MACRO_SERIES = (
    "VIXCLS",
    "DGS10",
    "DGS2",
    "T10Y2Y",
    "BAMLH0A0HYM2",
    "DCOILWTICO",
)


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _add_factor(
    rows: List[Dict[str, Any]],
    symbol: str,
    as_of: date,
    factor: str,
    value: Optional[float],
    source: str,
    window: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    if value is None:
        return False
    rows.append(
        {
            "symbol": symbol,
            "as_of": as_of,
            "factor": factor,
            "value": value,
            "source": source,
            "window_days": window,
            "metadata": json.dumps(metadata) if metadata else None,
        }
    )
    return True


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _stddev(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    avg = _mean(values)
    if avg is None:
        return None
    var = sum((value - avg) ** 2 for value in values) / max(1, len(values) - 1)
    return math.sqrt(var)


def _log_returns(prices: Sequence[float]) -> List[float]:
    returns: List[float] = []
    for idx in range(1, len(prices)):
        prev = prices[idx - 1]
        curr = prices[idx]
        if prev is None or curr is None or prev <= 0 or curr <= 0:
            continue
        returns.append(math.log(curr / prev))
    return returns


def _sma(prices: Sequence[float], window: int) -> Optional[float]:
    if len(prices) < window:
        return None
    return _mean(prices[-window:])


def _rsi(prices: Sequence[float], window: int = 14) -> Optional[float]:
    if len(prices) < window + 1:
        return None
    gains: List[float] = []
    losses: List[float] = []
    for idx in range(-window, 0):
        diff = prices[idx] - prices[idx - 1]
        if diff >= 0:
            gains.append(diff)
        else:
            losses.append(-diff)
    avg_gain = _mean(gains) if gains else 0.0
    avg_loss = _mean(losses) if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


async def _resolve_as_of(as_of: Optional[date]) -> date:
    if as_of:
        return as_of
    query = text("SELECT MAX(timestamp)::date FROM market.ohlcv")
    async with engine.begin() as conn:
        result = await conn.execute(query)
        value = result.scalar_one_or_none()
    if not value:
        raise RuntimeError("No market data found for factor computation")
    return datetime.now(tz=EASTERN).date()


async def _fetch_event_stats(as_of: date) -> Dict[str, Dict[str, Any]]:
    query = text(
        """
        SELECT
          symbol,
          COUNT(*) FILTER (WHERE event_time >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '1 day') AS count_1d,
          COUNT(*) FILTER (WHERE event_time >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '3 days') AS count_3d,
          COUNT(*) FILTER (WHERE event_time >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '7 days') AS count_7d,
          COUNT(*) FILTER (WHERE event_time >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '14 days') AS count_14d,
          COUNT(*) FILTER (WHERE event_time >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '30 days') AS count_30d,
          COUNT(*) FILTER (WHERE event_time >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '60 days') AS count_60d,
          COUNT(*) FILTER (
            WHERE event_time >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '14 days'
              AND event_time < CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '7 days'
          ) AS count_prev_7d,
          COUNT(*) FILTER (
            WHERE event_time >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '60 days'
              AND event_time < CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '30 days'
          ) AS count_prev_30d,
          COUNT(*) FILTER (
            WHERE event_time >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '7 days'
              AND LOWER(COALESCE(importance, '')) IN ('high', 'critical')
          ) AS count_high_7d,
          COUNT(*) FILTER (
            WHERE event_time >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '30 days'
              AND LOWER(COALESCE(importance, '')) IN ('high', 'critical')
          ) AS count_high_30d,
          ARRAY_AGG(DISTINCT event_type) FILTER (
            WHERE event_time >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '30 days'
          ) AS types_30d,
          ARRAY_AGG(DISTINCT event_type) FILTER (
            WHERE event_time >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '90 days'
          ) AS types_90d,
          ARRAY_AGG(DISTINCT event_type) FILTER (
            WHERE event_time >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '210 days'
              AND event_time < CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '30 days'
          ) AS types_prev_180d,
          ARRAY_AGG(DISTINCT event_type) FILTER (
            WHERE event_time >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '455 days'
              AND event_time < CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '90 days'
          ) AS types_prev_365d
        FROM event.events
        WHERE symbol IS NOT NULL
          AND event_time <= CAST(:as_of AS TIMESTAMPTZ) + INTERVAL '1 day'
          AND event_time >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '455 days'
        GROUP BY symbol
        """
    )
    async with engine.begin() as conn:
        result = await conn.execute(query, {"as_of": as_of})
        rows = result.fetchall()
    stats: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        stats[row.symbol] = {
            "count_1d": row.count_1d or 0,
            "count_3d": row.count_3d or 0,
            "count_7d": row.count_7d or 0,
            "count_14d": row.count_14d or 0,
            "count_30d": row.count_30d or 0,
            "count_60d": row.count_60d or 0,
            "count_prev_7d": row.count_prev_7d or 0,
            "count_prev_30d": row.count_prev_30d or 0,
            "count_high_7d": row.count_high_7d or 0,
            "count_high_30d": row.count_high_30d or 0,
            "types_30d": set(row.types_30d or []),
            "types_90d": set(row.types_90d or []),
            "types_prev_180d": set(row.types_prev_180d or []),
            "types_prev_365d": set(row.types_prev_365d or []),
        }
    return stats


async def _fetch_news_stats(as_of: date) -> Dict[str, Dict[str, Any]]:
    query = text(
        """
        WITH base AS (
          SELECT
            id,
            published_at,
            source,
            sentiment,
            unnest(COALESCE(tickers, ARRAY[]::varchar[])) AS symbol
          FROM news.articles
          WHERE published_at <= CAST(:as_of AS TIMESTAMPTZ) + INTERVAL '1 day'
            AND published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '30 days'
        ),
        source_3d AS (
          SELECT source, COUNT(DISTINCT id) AS cnt
          FROM base
          WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '3 days'
          GROUP BY source
        ),
        total_3d AS (
          SELECT COUNT(DISTINCT id) AS total
          FROM base
          WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '3 days'
        ),
        source_7d AS (
          SELECT source, COUNT(DISTINCT id) AS cnt
          FROM base
          WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '7 days'
          GROUP BY source
        ),
        total_7d AS (
          SELECT COUNT(DISTINCT id) AS total
          FROM base
          WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '7 days'
        ),
        source_14d AS (
          SELECT source, COUNT(DISTINCT id) AS cnt
          FROM base
          WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '14 days'
          GROUP BY source
        ),
        total_14d AS (
          SELECT COUNT(DISTINCT id) AS total
          FROM base
          WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '14 days'
        ),
        source_30d AS (
          SELECT source, COUNT(DISTINCT id) AS cnt
          FROM base
          GROUP BY source
        ),
        total_30d AS (
          SELECT COUNT(DISTINCT id) AS total
          FROM base
        ),
        source_prev_7d AS (
          SELECT source, COUNT(DISTINCT id) AS cnt
          FROM base
          WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '14 days'
            AND published_at < CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '7 days'
          GROUP BY source
        ),
        total_prev_7d AS (
          SELECT COUNT(DISTINCT id) AS total
          FROM base
          WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '14 days'
            AND published_at < CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '7 days'
        ),
        source_prev_14d AS (
          SELECT source, COUNT(DISTINCT id) AS cnt
          FROM base
          WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '28 days'
            AND published_at < CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '14 days'
          GROUP BY source
        ),
        total_prev_14d AS (
          SELECT COUNT(DISTINCT id) AS total
          FROM base
          WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '28 days'
            AND published_at < CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '14 days'
        )
        SELECT
          symbol,
          COUNT(DISTINCT id) FILTER (WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '3 days') AS articles_3d,
          COUNT(DISTINCT id) FILTER (WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '7 days') AS articles_7d,
          COUNT(DISTINCT id) FILTER (WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '14 days') AS articles_14d,
          COUNT(DISTINCT id) FILTER (WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '30 days') AS articles_30d,
          COUNT(DISTINCT id) FILTER (
            WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '14 days'
              AND published_at < CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '7 days'
          ) AS articles_prev_7d,
          COUNT(DISTINCT id) FILTER (
            WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '28 days'
              AND published_at < CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '14 days'
          ) AS articles_prev_14d,
          SUM(
            CASE
              WHEN published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '3 days'
              THEN COALESCE(sqrt(total_3d.total::float / NULLIF(source_3d.cnt, 0)), 0)
              ELSE 0
            END
          ) AS articles_3d_weighted,
          SUM(
            CASE
              WHEN published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '7 days'
              THEN COALESCE(sqrt(total_7d.total::float / NULLIF(source_7d.cnt, 0)), 0)
              ELSE 0
            END
          ) AS articles_7d_weighted,
          SUM(
            CASE
              WHEN published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '14 days'
              THEN COALESCE(sqrt(total_14d.total::float / NULLIF(source_14d.cnt, 0)), 0)
              ELSE 0
            END
          ) AS articles_14d_weighted,
          SUM(
            CASE
              WHEN published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '30 days'
              THEN COALESCE(sqrt(total_30d.total::float / NULLIF(source_30d.cnt, 0)), 0)
              ELSE 0
            END
          ) AS articles_30d_weighted,
          SUM(
            CASE
              WHEN published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '14 days'
                AND published_at < CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '7 days'
              THEN COALESCE(sqrt(total_prev_7d.total::float / NULLIF(source_prev_7d.cnt, 0)), 0)
              ELSE 0
            END
          ) AS articles_prev_7d_weighted,
          SUM(
            CASE
              WHEN published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '28 days'
                AND published_at < CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '14 days'
              THEN COALESCE(sqrt(total_prev_14d.total::float / NULLIF(source_prev_14d.cnt, 0)), 0)
              ELSE 0
            END
          ) AS articles_prev_14d_weighted,
          AVG(sentiment) FILTER (WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '1 day') AS sentiment_1d,
          AVG(sentiment) FILTER (WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '3 days') AS sentiment_3d,
          AVG(sentiment) FILTER (WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '7 days') AS sentiment_7d,
          AVG(sentiment) FILTER (WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '14 days') AS sentiment_14d,
          AVG(sentiment) FILTER (WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '30 days') AS sentiment_30d,
          STDDEV_POP(sentiment) FILTER (WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '7 days') AS sentiment_std_7d,
          STDDEV_POP(sentiment) FILTER (WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '30 days') AS sentiment_std_30d
        FROM base
        LEFT JOIN source_3d ON base.source = source_3d.source
        LEFT JOIN source_7d ON base.source = source_7d.source
        LEFT JOIN source_14d ON base.source = source_14d.source
        LEFT JOIN source_30d ON base.source = source_30d.source
        LEFT JOIN source_prev_7d ON base.source = source_prev_7d.source
        LEFT JOIN source_prev_14d ON base.source = source_prev_14d.source
        CROSS JOIN total_3d
        CROSS JOIN total_7d
        CROSS JOIN total_14d
        CROSS JOIN total_30d
        CROSS JOIN total_prev_7d
        CROSS JOIN total_prev_14d
        GROUP BY symbol,
                 total_3d.total,
                 total_7d.total,
                 total_14d.total,
                 total_30d.total,
                 total_prev_7d.total,
                 total_prev_14d.total
        """
    )
    async with engine.begin() as conn:
        result = await conn.execute(query, {"as_of": as_of})
        rows = result.fetchall()
    stats: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        stats[row.symbol] = {
            "articles_3d": row.articles_3d or 0,
            "articles_7d": row.articles_7d or 0,
            "articles_14d": row.articles_14d or 0,
            "articles_30d": row.articles_30d or 0,
            "articles_prev_7d": row.articles_prev_7d or 0,
            "articles_prev_14d": row.articles_prev_14d or 0,
            "articles_3d_weighted": float(row.articles_3d_weighted or 0),
            "articles_7d_weighted": float(row.articles_7d_weighted or 0),
            "articles_14d_weighted": float(row.articles_14d_weighted or 0),
            "articles_30d_weighted": float(row.articles_30d_weighted or 0),
            "articles_prev_7d_weighted": float(row.articles_prev_7d_weighted or 0),
            "articles_prev_14d_weighted": float(row.articles_prev_14d_weighted or 0),
            "sentiment_1d": float(row.sentiment_1d) if row.sentiment_1d is not None else None,
            "sentiment_3d": float(row.sentiment_3d) if row.sentiment_3d is not None else None,
            "sentiment_7d": float(row.sentiment_7d) if row.sentiment_7d is not None else None,
            "sentiment_14d": float(row.sentiment_14d) if row.sentiment_14d is not None else None,
            "sentiment_30d": float(row.sentiment_30d) if row.sentiment_30d is not None else None,
            "sentiment_std_7d": float(row.sentiment_std_7d) if row.sentiment_std_7d is not None else None,
            "sentiment_std_30d": float(row.sentiment_std_30d) if row.sentiment_std_30d is not None else None,
        }
    return stats


async def _fetch_news_topic_stats(as_of: date) -> Dict[str, Dict[str, Any]]:
    query = text(
        """
        SELECT
          symbol,
          COUNT(*) FILTER (WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '3 days') AS topics_3d,
          COUNT(*) FILTER (WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '7 days') AS topics_7d,
          COUNT(*) FILTER (
            WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '6 days'
              AND published_at < CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '3 days'
          ) AS topics_prev_3d,
          COUNT(*) FILTER (
            WHERE published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '14 days'
              AND published_at < CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '7 days'
          ) AS topics_prev_7d
        FROM news.articles
        CROSS JOIN LATERAL unnest(COALESCE(tickers, ARRAY[]::varchar[])) AS symbol
        CROSS JOIN LATERAL jsonb_array_elements_text(
          COALESCE(payload->'analysis'->'topics', '[]'::jsonb)
        ) AS topic
        WHERE published_at <= CAST(:as_of AS TIMESTAMPTZ) + INTERVAL '1 day'
          AND published_at >= CAST(:as_of AS TIMESTAMPTZ) - INTERVAL '14 days'
        GROUP BY symbol
        """
    )
    async with engine.begin() as conn:
        result = await conn.execute(query, {"as_of": as_of})
        rows = result.fetchall()
    stats: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        stats[row.symbol] = {
            "topics_3d": row.topics_3d or 0,
            "topics_7d": row.topics_7d or 0,
            "topics_prev_3d": row.topics_prev_3d or 0,
            "topics_prev_7d": row.topics_prev_7d or 0,
        }
    return stats


async def _fetch_market_series(as_of: date) -> Dict[str, List[Tuple[date, float, Optional[float]]]]:
    query = text(
        """
        SELECT symbol,
               timestamp::date AS day,
               close,
               volume
        FROM market.ohlcv
        WHERE timestamp::date <= :as_of
          AND timestamp::date >= :as_of - INTERVAL '400 days'
        ORDER BY symbol, day
        """
    )
    async with engine.begin() as conn:
        result = await conn.execute(query, {"as_of": as_of})
        rows = result.fetchall()
    series: Dict[str, List[Tuple[date, float, Optional[float]]]] = {}
    for row in rows:
        series.setdefault(row.symbol, []).append((row.day, float(row.close), row.volume))
    return series


async def _fetch_macro_series(as_of: date) -> Dict[str, List[Tuple[date, float]]]:
    query = text(
        """
        SELECT series_id, timestamp::date AS day, value
        FROM economic.values
        WHERE series_id = ANY(:series_ids)
          AND timestamp::date <= :as_of
          AND timestamp::date >= :as_of - INTERVAL '260 days'
        ORDER BY series_id, day
        """
    )
    async with engine.begin() as conn:
        result = await conn.execute(query, {"series_ids": list(MACRO_SERIES), "as_of": as_of})
        rows = result.fetchall()
    series: Dict[str, List[Tuple[date, float]]] = {}
    for row in rows:
        series.setdefault(row.series_id, []).append((row.day, float(row.value)))
    return series


async def run_daily_factors(as_of: Optional[date] = None) -> Dict[str, Any]:
    as_of = await _resolve_as_of(as_of)
    run_time = datetime.now(tz=UTC)
    factor_rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {"event": 0, "news": 0, "market": 0, "macro": 0}

    event_stats = await _fetch_event_stats(as_of)
    for symbol, stats in event_stats.items():
        event_added = 0
        count_1d = stats["count_1d"]
        count_3d = stats["count_3d"]
        count_7d = stats["count_7d"]
        count_14d = stats["count_14d"]
        count_30d = stats["count_30d"]
        count_60d = stats["count_60d"]
        momentum_7d = count_7d - stats["count_prev_7d"]
        momentum_30d = count_30d - stats["count_prev_30d"]
        intensity_ratio_7d = _safe_div(count_7d, count_30d) if count_30d else None
        intensity_ratio_14d = _safe_div(count_14d, count_60d) if count_60d else None
        types_30d = stats["types_30d"]
        types_90d = stats["types_90d"]
        types_prev_180d = stats["types_prev_180d"]
        types_prev_365d = stats["types_prev_365d"]
        novelty_30d = len(types_30d - types_prev_180d)
        novelty_90d = len(types_90d - types_prev_365d)
        breadth_30d = len(types_30d)
        breadth_90d = len(types_90d)
        high_7d = stats["count_high_7d"]
        high_30d = stats["count_high_30d"]
        ratio_high_7d = _safe_div(high_7d, count_7d) if count_7d else None
        ratio_high_30d = _safe_div(high_30d, count_30d) if count_30d else None

        event_added += int(_add_factor(factor_rows, symbol, as_of, "event.count_1d", count_1d, "event", 1))
        event_added += int(_add_factor(factor_rows, symbol, as_of, "event.count_3d", count_3d, "event", 3))
        event_added += int(_add_factor(factor_rows, symbol, as_of, "event.count_7d", count_7d, "event", 7))
        event_added += int(_add_factor(factor_rows, symbol, as_of, "event.count_14d", count_14d, "event", 14))
        event_added += int(_add_factor(factor_rows, symbol, as_of, "event.count_30d", count_30d, "event", 30))
        event_added += int(_add_factor(factor_rows, symbol, as_of, "event.count_60d", count_60d, "event", 60))
        event_added += int(_add_factor(factor_rows, symbol, as_of, "event.momentum_7d", momentum_7d, "event", 7))
        event_added += int(_add_factor(factor_rows, symbol, as_of, "event.momentum_30d", momentum_30d, "event", 30))
        event_added += int(
            _add_factor(factor_rows, symbol, as_of, "event.intensity_ratio_7d", intensity_ratio_7d, "event", 7)
        )
        event_added += int(
            _add_factor(factor_rows, symbol, as_of, "event.intensity_ratio_14d", intensity_ratio_14d, "event", 14)
        )
        event_added += int(_add_factor(factor_rows, symbol, as_of, "event.type_breadth_30d", breadth_30d, "event", 30))
        event_added += int(_add_factor(factor_rows, symbol, as_of, "event.type_breadth_90d", breadth_90d, "event", 90))
        event_added += int(_add_factor(factor_rows, symbol, as_of, "event.type_novelty_30d", novelty_30d, "event", 30))
        event_added += int(_add_factor(factor_rows, symbol, as_of, "event.type_novelty_90d", novelty_90d, "event", 90))
        event_added += int(_add_factor(factor_rows, symbol, as_of, "event.importance_high_7d", high_7d, "event", 7))
        event_added += int(_add_factor(factor_rows, symbol, as_of, "event.importance_high_30d", high_30d, "event", 30))
        event_added += int(_add_factor(factor_rows, symbol, as_of, "event.importance_ratio_7d", ratio_high_7d, "event", 7))
        event_added += int(_add_factor(factor_rows, symbol, as_of, "event.importance_ratio_30d", ratio_high_30d, "event", 30))
        counts["event"] += event_added

    news_stats = await _fetch_news_stats(as_of)
    topic_stats = await _fetch_news_topic_stats(as_of)
    for symbol, stats in news_stats.items():
        news_added = 0
        news_added += int(
            _add_factor(factor_rows, symbol, as_of, "news.volume_3d", stats["articles_3d_weighted"], "news", 3)
        )
        news_added += int(
            _add_factor(factor_rows, symbol, as_of, "news.volume_7d", stats["articles_7d_weighted"], "news", 7)
        )
        news_added += int(
            _add_factor(factor_rows, symbol, as_of, "news.volume_14d", stats["articles_14d_weighted"], "news", 14)
        )
        news_added += int(
            _add_factor(factor_rows, symbol, as_of, "news.volume_30d", stats["articles_30d_weighted"], "news", 30)
        )
        volume_momentum_7d = stats["articles_7d_weighted"] - stats["articles_prev_7d_weighted"]
        volume_momentum_14d = stats["articles_14d_weighted"] - stats["articles_prev_14d_weighted"]
        news_added += int(
            _add_factor(factor_rows, symbol, as_of, "news.volume_momentum_7d", volume_momentum_7d, "news", 7)
        )
        news_added += int(
            _add_factor(factor_rows, symbol, as_of, "news.volume_momentum_14d", volume_momentum_14d, "news", 14)
        )
        news_added += int(_add_factor(factor_rows, symbol, as_of, "news.sentiment_1d", stats["sentiment_1d"], "news", 1))
        news_added += int(_add_factor(factor_rows, symbol, as_of, "news.sentiment_3d", stats["sentiment_3d"], "news", 3))
        news_added += int(_add_factor(factor_rows, symbol, as_of, "news.sentiment_7d", stats["sentiment_7d"], "news", 7))
        news_added += int(_add_factor(factor_rows, symbol, as_of, "news.sentiment_14d", stats["sentiment_14d"], "news", 14))
        news_added += int(_add_factor(factor_rows, symbol, as_of, "news.sentiment_30d", stats["sentiment_30d"], "news", 30))
        news_added += int(
            _add_factor(factor_rows, symbol, as_of, "news.sentiment_dispersion_7d", stats["sentiment_std_7d"], "news", 7)
        )
        news_added += int(
            _add_factor(factor_rows, symbol, as_of, "news.sentiment_dispersion_30d", stats["sentiment_std_30d"], "news", 30)
        )
        sentiment_momentum_3d = None
        if stats["sentiment_3d"] is not None and stats["sentiment_7d"] is not None:
            sentiment_momentum_3d = stats["sentiment_3d"] - stats["sentiment_7d"]
        sentiment_momentum_14d = None
        if stats["sentiment_14d"] is not None and stats["sentiment_30d"] is not None:
            sentiment_momentum_14d = stats["sentiment_14d"] - stats["sentiment_30d"]
        news_added += int(
            _add_factor(factor_rows, symbol, as_of, "news.sentiment_momentum_3d", sentiment_momentum_3d, "news", 3)
        )
        news_added += int(
            _add_factor(factor_rows, symbol, as_of, "news.sentiment_momentum_14d", sentiment_momentum_14d, "news", 14)
        )
        topic = topic_stats.get(symbol)
        if topic:
            velocity_3d = topic["topics_3d"] - topic["topics_prev_3d"]
            velocity_7d = topic["topics_7d"] - topic["topics_prev_7d"]
            news_added += int(
                _add_factor(factor_rows, symbol, as_of, "news.topic_velocity_3d", velocity_3d, "news", 3)
            )
            news_added += int(
                _add_factor(factor_rows, symbol, as_of, "news.topic_velocity_7d", velocity_7d, "news", 7)
            )
            news_added += int(
                _add_factor(factor_rows, symbol, as_of, "news.topic_volume_3d", topic["topics_3d"], "news", 3)
            )
            news_added += int(
                _add_factor(factor_rows, symbol, as_of, "news.topic_volume_7d", topic["topics_7d"], "news", 7)
            )
        counts["news"] += news_added

    market_series = await _fetch_market_series(as_of)
    for symbol, series in market_series.items():
        closes = [row[1] for row in series]
        volumes = [row[2] for row in series if row[2] is not None]
        if len(closes) < 21:
            continue
        last_close = closes[-1]
        returns = _log_returns(closes)
        vol_20 = _stddev(returns[-20:]) if len(returns) >= 20 else None
        vol_60 = _stddev(returns[-60:]) if len(returns) >= 60 else None
        vol_252 = _stddev(returns[-252:]) if len(returns) >= 252 else None
        vol_z_20 = _safe_div(vol_20, vol_252) if vol_20 is not None and vol_252 else None
        vol_z_60 = _safe_div(vol_60, vol_252) if vol_60 is not None and vol_252 else None

        ret_1d = None
        if len(closes) >= 2 and closes[-2] != 0:
            ret_1d = closes[-1] / closes[-2] - 1
        ret_5d = None
        if len(closes) >= 6 and closes[-6] != 0:
            ret_5d = closes[-1] / closes[-6] - 1
        ret_10d = None
        if len(closes) >= 11 and closes[-11] != 0:
            ret_10d = closes[-1] / closes[-11] - 1
        ret_20d = None
        if len(closes) >= 21 and closes[-21] != 0:
            ret_20d = closes[-1] / closes[-21] - 1
        ret_60d = None
        if len(closes) >= 61 and closes[-61] != 0:
            ret_60d = closes[-1] / closes[-61] - 1
        ret_120d = None
        if len(closes) >= 121 and closes[-121] != 0:
            ret_120d = closes[-1] / closes[-121] - 1

        volume_20 = _mean(volumes[-20:]) if len(volumes) >= 20 else None
        volume_60 = _mean(volumes[-60:]) if len(volumes) >= 60 else None
        volume_252 = _mean(volumes[-252:]) if len(volumes) >= 252 else None
        volume_z_20 = _safe_div(volume_20, volume_252) if volume_20 is not None and volume_252 else None
        volume_z_60 = _safe_div(volume_60, volume_252) if volume_60 is not None and volume_252 else None

        sma_20 = _sma(closes, 20)
        sma_50 = _sma(closes, 50)
        sma_200 = _sma(closes, 200)
        sma_20_dist = _safe_div(last_close, sma_20) - 1 if sma_20 else None
        sma_50_dist = _safe_div(last_close, sma_50) - 1 if sma_50 else None
        sma_200_dist = _safe_div(last_close, sma_200) - 1 if sma_200 else None
        ma_cross_50_200 = _safe_div(sma_50 - sma_200, sma_200) if sma_50 and sma_200 else None
        max_120 = max(closes[-120:]) if len(closes) >= 120 else None
        max_252 = max(closes[-252:]) if len(closes) >= 252 else None
        drawdown_120 = _safe_div(last_close, max_120) - 1 if max_120 else None
        breakout_252 = _safe_div(last_close, max_252) - 1 if max_252 else None
        rsi_14 = _rsi(closes, 14)
        rsi_14_dev = rsi_14 - 50.0 if rsi_14 is not None else None

        market_added = 0
        market_added += int(_add_factor(factor_rows, symbol, as_of, "mkt.return_1d", ret_1d, "market", 1))
        market_added += int(_add_factor(factor_rows, symbol, as_of, "mkt.return_5d", ret_5d, "market", 5))
        market_added += int(_add_factor(factor_rows, symbol, as_of, "mkt.return_10d", ret_10d, "market", 10))
        market_added += int(_add_factor(factor_rows, symbol, as_of, "mkt.return_20d", ret_20d, "market", 20))
        market_added += int(_add_factor(factor_rows, symbol, as_of, "mkt.momentum_60d", ret_60d, "market", 60))
        market_added += int(_add_factor(factor_rows, symbol, as_of, "mkt.momentum_120d", ret_120d, "market", 120))
        market_added += int(_add_factor(factor_rows, symbol, as_of, "mkt.vol_20d", vol_20, "market", 20))
        market_added += int(_add_factor(factor_rows, symbol, as_of, "mkt.vol_60d", vol_60, "market", 60))
        market_added += int(_add_factor(factor_rows, symbol, as_of, "mkt.vol_z_20d", vol_z_20, "market", 20))
        market_added += int(_add_factor(factor_rows, symbol, as_of, "mkt.vol_z_60d", vol_z_60, "market", 60))
        market_added += int(_add_factor(factor_rows, symbol, as_of, "mkt.volume_z_20d", volume_z_20, "market", 20))
        market_added += int(_add_factor(factor_rows, symbol, as_of, "mkt.volume_z_60d", volume_z_60, "market", 60))
        market_added += int(_add_factor(factor_rows, symbol, as_of, "mkt.sma_20_dist", sma_20_dist, "market", 20))
        market_added += int(_add_factor(factor_rows, symbol, as_of, "mkt.sma_50_dist", sma_50_dist, "market", 50))
        market_added += int(_add_factor(factor_rows, symbol, as_of, "mkt.sma_200_dist", sma_200_dist, "market", 200))
        market_added += int(_add_factor(factor_rows, symbol, as_of, "mkt.ma_cross_50_200", ma_cross_50_200, "market", 200))
        market_added += int(_add_factor(factor_rows, symbol, as_of, "mkt.drawdown_120d", drawdown_120, "market", 120))
        market_added += int(_add_factor(factor_rows, symbol, as_of, "mkt.breakout_252d", breakout_252, "market", 252))
        market_added += int(_add_factor(factor_rows, symbol, as_of, "mkt.rsi_14", rsi_14_dev, "market", 14))
        counts["market"] += market_added

    macro_series = await _fetch_macro_series(as_of)
    for series_id, values in macro_series.items():
        series_values = [value for _, value in values]
        if len(series_values) < 5:
            continue
        last_value = series_values[-1]
        window_20 = series_values[-20:] if len(series_values) >= 20 else series_values
        window_60 = series_values[-60:] if len(series_values) >= 60 else series_values
        mean_20 = _mean(window_20)
        mean_60 = _mean(window_60)
        std_20 = _stddev(window_20)
        std_60 = _stddev(window_60)
        zscore_20 = None
        zscore_60 = None
        if std_20 and std_20 > 0:
            zscore_20 = (last_value - mean_20) / std_20
        if std_60 and std_60 > 0:
            zscore_60 = (last_value - mean_60) / std_60
        change_20 = _safe_div(last_value - mean_20, mean_20) if mean_20 else None
        change_60 = _safe_div(last_value - mean_60, mean_60) if mean_60 else None
        macro_added = 0
        macro_added += int(
            _add_factor(
                factor_rows,
                "__macro__",
                as_of,
                f"macro.{series_id}.z20",
                zscore_20,
                "economic",
                20,
            )
        )
        macro_added += int(
            _add_factor(
                factor_rows,
                "__macro__",
                as_of,
                f"macro.{series_id}.chg20",
                change_20,
                "economic",
                20,
            )
        )
        macro_added += int(
            _add_factor(
                factor_rows,
                "__macro__",
                as_of,
                f"macro.{series_id}.z60",
                zscore_60,
                "economic",
                60,
            )
        )
        macro_added += int(
            _add_factor(
                factor_rows,
                "__macro__",
                as_of,
                f"macro.{series_id}.chg60",
                change_60,
                "economic",
                60,
            )
        )
        counts["macro"] += macro_added

    if not factor_rows:
        return {"as_of": as_of.isoformat(), "run_time": run_time.isoformat(), "status": "empty"}

    insert_query = text(
        """
        INSERT INTO factors.daily_factors
          (symbol, as_of, factor, value, source, window_days, metadata)
        VALUES
          (:symbol, :as_of, :factor, :value, :source, :window_days, CAST(:metadata AS JSONB))
        ON CONFLICT (symbol, as_of, factor)
        DO UPDATE SET value = EXCLUDED.value,
                      source = EXCLUDED.source,
                      window_days = EXCLUDED.window_days,
                      metadata = EXCLUDED.metadata
        """
    )
    run_query = text(
        """
        INSERT INTO factors.factor_runs
          (as_of, run_time, status, summary, error)
        VALUES
          (:as_of, :run_time, :status, CAST(:summary AS JSONB), :error)
        RETURNING id
        """
    )
    summary = {
        "factors": len(factor_rows),
        "counts": counts,
    }
    async with engine.begin() as conn:
        result = await conn.execute(
            run_query,
            {
                "as_of": as_of,
                "run_time": run_time,
                "status": "running",
                "summary": json.dumps(summary),
                "error": None,
            },
        )
        run_id = result.scalar_one()
        await conn.execute(insert_query, factor_rows)
        await conn.execute(
            text(
                """
                UPDATE factors.factor_runs
                SET status = :status,
                    summary = CAST(:summary AS JSONB)
                WHERE id = :id
                """
            ),
            {"status": "success", "summary": json.dumps(summary), "id": run_id},
        )

    return {
        "as_of": as_of.isoformat(),
        "run_time": run_time.isoformat(),
        "status": "success",
        "counts": counts,
        "factors": len(factor_rows),
    }
