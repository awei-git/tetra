"""Report generator — assembles all analysis into report sections.

Fetches data from DB (OHLCV for 205 symbols, FRED rates, analysis results,
debate, portfolio, signals, events) and formats it for the Jinja2 template.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from src.db.session import engine
from src.report.delivery import generate_pdf

logger = logging.getLogger(__name__)
UTC = timezone.utc

# ---------------------------------------------------------------------------
# Symbol universe grouped by category
# ---------------------------------------------------------------------------
MARKET_GROUPS = [
    ("Mega Cap Tech", ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "TSLA"]),
    ("Large Cap Growth", ["NFLX", "AVGO", "CRM", "ADBE", "NOW", "INTU", "SNOW", "PLTR", "MELI"]),
    ("Sector ETFs", ["XLK", "XLE", "XLF", "XLV", "XLI", "XLC", "XLY", "XLP", "XLU", "XLB", "XLRE"]),
    ("Major Indices", ["SPY", "QQQ", "IWM", "DIA", "VOO", "VTI"]),
    ("International", ["EEM", "EFA", "FXI", "EWJ", "EWY", "EWZ", "INDA", "VEA", "VWO"]),
    ("Commodities", ["GLD", "SLV", "USO", "UNG", "GDX", "COPX", "DBA"]),
    ("Crypto", ["IBIT", "GBTC", "MSTR", "COIN", "MARA", "RIOT", "BITF", "HUT"]),
    ("Bonds & Rates", ["TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG", "EMB", "TIP"]),
    ("Volatility", ["UVXY", "SVXY", "VXX", "VIXY", "VXZ"]),
    ("Thematic ETFs", ["ARKK", "ARKW", "ARKG", "BOTZ", "ROBO", "HACK", "ICLN", "TAN", "LIT"]),
]

ALL_SYMBOLS = []
for _, syms in MARKET_GROUPS:
    ALL_SYMBOLS.extend(syms)
ALL_SYMBOLS = list(dict.fromkeys(ALL_SYMBOLS))  # dedupe, preserve order


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

async def _resolve_latest_trading_day(as_of: date) -> date:
    """Find the most recent trading day on or before as_of with OHLCV data."""
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT MAX(timestamp::date) AS latest
            FROM market.ohlcv
            WHERE timestamp::date <= :as_of
        """), {"as_of": as_of})
        row = result.fetchone()
        if row and row.latest:
            return row.latest
    return as_of


async def _fetch_market_dashboard(trading_day: date) -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]
]:
    """Fetch OHLCV data for all symbols and compute 1d/5d returns.

    Returns:
        (market_groups, top_gainers, top_losers)
    """
    prev_1d = trading_day - timedelta(days=1)
    prev_5d = trading_day - timedelta(days=10)  # generous window for 5 trading days

    async with engine.begin() as conn:
        # Get the latest close on or before trading_day, the previous close (1d),
        # and the close ~5 trading days ago — all in one query.
        result = await conn.execute(text("""
            WITH ranked AS (
                SELECT symbol, timestamp::date AS dt, close,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS rn
                FROM market.ohlcv
                WHERE symbol = ANY(:symbols) AND timestamp::date <= :trading_day
            ),
            latest AS (
                SELECT symbol, close AS close_today
                FROM ranked WHERE rn = 1
            ),
            prev1 AS (
                SELECT symbol, close AS close_prev1
                FROM ranked WHERE rn = 2
            ),
            prev5 AS (
                -- 5th most recent trading day (row 6 = 5 trading days back)
                SELECT symbol, close AS close_prev5
                FROM ranked WHERE rn = 6
            )
            SELECT l.symbol, l.close_today,
                   p1.close_prev1, p5.close_prev5
            FROM latest l
            LEFT JOIN prev1 p1 ON l.symbol = p1.symbol
            LEFT JOIN prev5 p5 ON l.symbol = p5.symbol
        """), {"symbols": ALL_SYMBOLS, "trading_day": trading_day})
        rows = result.fetchall()

    # Build lookup: symbol -> {close, change_1d, change_5d}
    sym_data: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        close = float(r.close_today) if r.close_today else None
        prev1 = float(r.close_prev1) if r.close_prev1 else None
        prev5 = float(r.close_prev5) if r.close_prev5 else None

        change_1d = ((close / prev1) - 1) if (close and prev1) else None
        change_5d = ((close / prev5) - 1) if (close and prev5) else None

        sym_data[r.symbol] = {
            "symbol": r.symbol,
            "close": close,
            "change_1d": round(change_1d * 100, 2) if change_1d is not None else None,
            "change_5d": round(change_5d * 100, 2) if change_5d is not None else None,
        }

    # Assemble grouped output
    market_groups = []
    for group_name, group_symbols in MARKET_GROUPS:
        items = []
        for sym in group_symbols:
            if sym in sym_data:
                items.append(sym_data[sym])
            else:
                items.append({"symbol": sym, "close": None, "change_1d": None, "change_5d": None})
        market_groups.append({"name": group_name, "symbols": items})

    # Top movers — sort by absolute 1d change
    all_with_change = [v for v in sym_data.values() if v["change_1d"] is not None]
    sorted_by_change = sorted(all_with_change, key=lambda x: x["change_1d"], reverse=True)

    top_gainers = sorted_by_change[:5]
    top_losers = sorted_by_change[-5:][::-1]  # worst 5, most negative first

    return market_groups, top_gainers, top_losers


async def _fetch_macro_data() -> List[Dict[str, Any]]:
    """Fetch FRED economic data — VIX, treasuries, credit spreads."""
    fred_labels = {
        "VIXCLS": "VIX",
        "DGS10": "10Y Treasury",
        "DGS2": "2Y Treasury",
        "T10Y2Y": "10Y-2Y Spread",
        "BAMLH0A0HYM2": "HY Credit Spread",
    }
    fred_ids = list(fred_labels.keys())

    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT DISTINCT ON (s.series_id)
                   s.series_id, v.value, v.timestamp
            FROM economic.values v
            JOIN economic.series s ON v.series_id = s.series_id
            WHERE s.series_id = ANY(:ids)
            ORDER BY s.series_id, v.timestamp DESC
        """), {"ids": fred_ids})
        rows = result.fetchall()

    macro = []
    for r in rows:
        sid = r.series_id
        macro.append({
            "name": fred_labels.get(sid, sid),
            "value": round(float(r.value), 2) if r.value is not None else None,
        })
    return macro


async def _fetch_market_snapshot(
    as_of: date,
    market_groups: List[Dict[str, Any]],
    macro_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the market_snapshot dict for backward compat (indices, rates, portfolio)."""
    snapshot: Dict[str, Any] = {}

    # Extract indices from market_groups for backward compat
    for group in market_groups:
        if group["name"] == "Major Indices":
            snapshot["indices"] = [
                {
                    "name": s["symbol"],
                    "price": f"${s['close']:,.2f}" if s["close"] else "—",
                    "change": f"{s['change_1d']:+.2f}%" if s["change_1d"] is not None else "—",
                    "change_pct": s["change_1d"] / 100 if s["change_1d"] is not None else 0,
                }
                for s in group["symbols"]
            ]
            break

    # Rates from macro_data
    if macro_data:
        snapshot["rates"] = [
            {"name": m["name"], "price": f"{m['value']:.2f}" if m["value"] is not None else "—"}
            for m in macro_data
        ]

    # Portfolio positions
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT symbol, shares, current_price, market_value, weight, unrealized_pnl
            FROM portfolio.positions
            ORDER BY market_value DESC NULLS LAST
        """))
        positions = result.fetchall()
        if positions:
            snapshot["portfolio_positions"] = [
                {
                    "name": r.symbol,
                    "price": f"${float(r.current_price):,.2f}" if r.current_price else "—",
                    "change": f"${float(r.unrealized_pnl):+,.0f}" if r.unrealized_pnl else "—",
                    "change_pct": float(r.unrealized_pnl) if r.unrealized_pnl else 0,
                }
                for r in positions
            ]

    return snapshot


async def _fetch_narrative_state(as_of: date) -> Optional[Dict[str, Any]]:
    """Fetch narrative analysis results."""
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT dominant_narrative, narrative_shift, shift_magnitude,
                   counter_narrative, novelty, raw_analysis
            FROM narrative.daily_state
            WHERE date = :date AND scope = 'market'
            LIMIT 1
        """), {"date": as_of})
        row = result.fetchone()

    if not row:
        return None

    return {
        "dominant_narrative": row.dominant_narrative,
        "narrative_shift": round(float(row.narrative_shift), 3) if row.narrative_shift else 0,
        "shift_magnitude": round(float(row.shift_magnitude), 3) if row.shift_magnitude else 0,
        "counter_narrative": row.counter_narrative,
    }


async def _fetch_debate_results(as_of: date) -> Optional[Dict[str, Any]]:
    """Fetch the latest debate synthesis."""
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT payload FROM gpt.recommendations
            WHERE provider = 'debate' AND session = :session
            ORDER BY run_time DESC LIMIT 1
        """), {"session": as_of.strftime("%Y%m%d")})
        row = result.fetchone()

    if not row:
        return None

    payload = row.payload
    if isinstance(payload, str):
        payload = json.loads(payload)

    return payload


async def _fetch_portfolio_state(as_of: date) -> Dict[str, Any]:
    """Fetch portfolio positions and snapshot."""
    result_data: Dict[str, Any] = {}

    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT symbol, shares, avg_cost, current_price, market_value, weight, unrealized_pnl
            FROM portfolio.positions
            ORDER BY market_value DESC NULLS LAST
        """))
        positions = result.fetchall()
        result_data["positions"] = [
            {
                "symbol": r.symbol,
                "price": float(r.current_price) if r.current_price else 0,
                "value": float(r.market_value) if r.market_value else 0,
                "weight": round(float(r.weight) * 100, 1) if r.weight else 0,
                "pnl": float(r.unrealized_pnl) if r.unrealized_pnl else 0,
            }
            for r in positions
        ]

        result = await conn.execute(text("""
            SELECT total_value, cash, daily_return, cumulative_return
            FROM portfolio.snapshots
            WHERE date <= :date ORDER BY date DESC LIMIT 1
        """), {"date": as_of})
        snap = result.fetchone()
        if snap:
            result_data["summary"] = {
                "total_value": float(snap.total_value),
                "cash": float(snap.cash),
                "daily_return": float(snap.daily_return),
                "cumulative_return": float(snap.cumulative_return),
            }

    return result_data


async def _fetch_track_record() -> Dict[str, Any]:
    """Fetch recommendation track record with extended analytics."""
    track: Dict[str, Any] = {}

    async with engine.begin() as conn:
        # --- Basic summary ---
        result = await conn.execute(text("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE status != 'open') AS closed,
                COUNT(*) FILTER (WHERE status = 'hit_target') AS hit_target,
                COUNT(*) FILTER (WHERE status = 'hit_stop') AS hit_stop,
                AVG(realized_pnl) FILTER (WHERE status != 'open') AS avg_pnl
            FROM tracker.recommendations
        """))
        row = result.fetchone()
        if row and row.total > 0:
            track["summary"] = {
                "total": row.total,
                "closed": row.closed or 0,
                "hit_target": row.hit_target or 0,
                "hit_stop": row.hit_stop or 0,
                "avg_pnl": float(row.avg_pnl) if row.avg_pnl else 0,
            }

        # --- Recent closed recs ---
        result = await conn.execute(text("""
            SELECT symbol, direction, method, entry_price, closed_price,
                   realized_pnl, status, closed_date
            FROM tracker.recommendations
            WHERE status != 'open'
            ORDER BY closed_date DESC
            LIMIT 10
        """))
        closed = result.fetchall()
        if closed:
            track["recent_closed"] = [
                {
                    "symbol": r.symbol,
                    "direction": r.direction,
                    "method": r.method,
                    "entry_price": float(r.entry_price),
                    "closed_price": float(r.closed_price) if r.closed_price else 0,
                    "realized_pnl": float(r.realized_pnl) if r.realized_pnl else 0,
                    "status": r.status,
                }
                for r in closed
            ]

        # --- Accuracy by method ---
        result = await conn.execute(text("""
            SELECT method,
                   COUNT(*) FILTER (WHERE status != 'open') AS closed,
                   COUNT(*) FILTER (WHERE status = 'hit_target') AS wins,
                   COUNT(*) FILTER (WHERE status = 'hit_stop') AS losses,
                   COUNT(*) FILTER (WHERE status = 'expired') AS expired,
                   AVG(realized_pnl) FILTER (WHERE status != 'open') AS avg_pnl
            FROM tracker.recommendations
            GROUP BY method
            ORDER BY method
        """))
        by_method = []
        for r in result.fetchall():
            closed_count = (r.closed or 0)
            wins = (r.wins or 0)
            hit_rate = wins / closed_count if closed_count > 0 else None
            by_method.append({
                "method": r.method or "unknown",
                "closed": closed_count,
                "wins": wins,
                "losses": r.losses or 0,
                "expired": r.expired or 0,
                "hit_rate": round(hit_rate, 3) if hit_rate is not None else None,
                "avg_pnl": round(float(r.avg_pnl), 4) if r.avg_pnl else None,
            })
        if by_method:
            track["accuracy_by_method"] = by_method

        # --- Accuracy by confidence ---
        result = await conn.execute(text("""
            SELECT confidence,
                   COUNT(*) FILTER (WHERE status != 'open') AS closed,
                   COUNT(*) FILTER (WHERE status = 'hit_target') AS wins,
                   AVG(realized_pnl) FILTER (WHERE status != 'open') AS avg_pnl
            FROM tracker.recommendations
            GROUP BY confidence
            ORDER BY confidence
        """))
        by_conf = []
        for r in result.fetchall():
            closed_count = (r.closed or 0)
            wins = (r.wins or 0)
            hit_rate = wins / closed_count if closed_count > 0 else None
            by_conf.append({
                "confidence": r.confidence or "unknown",
                "closed": closed_count,
                "wins": wins,
                "hit_rate": round(hit_rate, 3) if hit_rate is not None else None,
                "avg_pnl": round(float(r.avg_pnl), 4) if r.avg_pnl else None,
            })
        if by_conf:
            track["accuracy_by_confidence"] = by_conf

        # --- Average hold days ---
        result = await conn.execute(text("""
            SELECT AVG(closed_date - created_date) AS avg_hold
            FROM tracker.recommendations
            WHERE status != 'open' AND closed_date IS NOT NULL
        """))
        hold_row = result.fetchone()
        if hold_row and hold_row.avg_hold is not None:
            try:
                track["avg_hold_days"] = round(hold_row.avg_hold.total_seconds() / 86400, 1)
            except AttributeError:
                track["avg_hold_days"] = round(float(hold_row.avg_hold), 1)

        # --- Best & worst trades ---
        result = await conn.execute(text("""
            SELECT symbol, direction, method, realized_pnl, status, closed_date
            FROM tracker.recommendations
            WHERE status != 'open' AND realized_pnl IS NOT NULL
            ORDER BY realized_pnl DESC
            LIMIT 1
        """))
        best = result.fetchone()
        if best:
            track["best_trade"] = {
                "symbol": best.symbol,
                "direction": best.direction,
                "method": best.method,
                "pnl": round(float(best.realized_pnl), 4),
            }

        result = await conn.execute(text("""
            SELECT symbol, direction, method, realized_pnl, status, closed_date
            FROM tracker.recommendations
            WHERE status != 'open' AND realized_pnl IS NOT NULL
            ORDER BY realized_pnl ASC
            LIMIT 1
        """))
        worst = result.fetchone()
        if worst:
            track["worst_trade"] = {
                "symbol": worst.symbol,
                "direction": worst.direction,
                "method": worst.method,
                "pnl": round(float(worst.realized_pnl), 4),
            }

        # --- Cumulative PnL ---
        result = await conn.execute(text("""
            SELECT SUM(realized_pnl) AS cumulative
            FROM tracker.recommendations
            WHERE status != 'open' AND realized_pnl IS NOT NULL
        """))
        cum_row = result.fetchone()
        if cum_row and cum_row.cumulative is not None:
            track["cumulative_pnl"] = round(float(cum_row.cumulative), 4)

        # --- Win / lose streaks (current) ---
        result = await conn.execute(text("""
            SELECT status
            FROM tracker.recommendations
            WHERE status != 'open'
            ORDER BY closed_date DESC
        """))
        streak_rows = result.fetchall()
        win_streak = 0
        lose_streak = 0
        if streak_rows:
            first_status = streak_rows[0].status
            if first_status == "hit_target":
                for sr in streak_rows:
                    if sr.status == "hit_target":
                        win_streak += 1
                    else:
                        break
            elif first_status == "hit_stop":
                for sr in streak_rows:
                    if sr.status == "hit_stop":
                        lose_streak += 1
                    else:
                        break
        track["win_streak"] = win_streak
        track["lose_streak"] = lose_streak

    return track


async def _fetch_signals_summary(as_of: date) -> List[Dict[str, Any]]:
    """Summarize active signals by source."""
    summaries = []

    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT signal_type,
                   COUNT(*) AS cnt,
                   MAX(symbol) FILTER (WHERE ABS(strength) = (
                       SELECT MAX(ABS(s2.strength)) FROM signals.informed_trading s2
                       WHERE s2.date = :date AND s2.signal_type = signals.informed_trading.signal_type
                   )) AS top_symbol,
                   MAX(ABS(strength)) AS top_strength
            FROM signals.informed_trading
            WHERE date = :date
            GROUP BY signal_type
            ORDER BY MAX(ABS(strength)) DESC
        """), {"date": as_of})
        for r in result.fetchall():
            summaries.append({
                "source": r.signal_type,
                "count": r.cnt,
                "top_symbol": r.top_symbol,
                "top_strength": round(float(r.top_strength), 3) if r.top_strength else None,
            })

        result = await conn.execute(text("""
            SELECT COUNT(*) AS cnt,
                   MAX(symbol) AS top_symbol,
                   MAX(ABS(signal_score)) AS top_strength
            FROM signals.unified
            WHERE date = :date
        """), {"date": as_of})
        row = result.fetchone()
        if row and row.cnt > 0:
            summaries.append({
                "source": "unified_meta",
                "count": row.cnt,
                "top_symbol": row.top_symbol,
                "top_strength": round(float(row.top_strength), 3) if row.top_strength else None,
            })

    return summaries


CORRELATION_ASSETS = [
    "SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "IBIT", "HYG", "EEM", "VXX", "DXY",
]


async def _compute_correlation_matrix(trading_day: date) -> Dict[str, Any]:
    """Compute 20-day rolling Pearson correlations between key assets.

    Returns dict with:
        labels: list of symbols actually available
        matrix: 2D list of correlation values (symmetric, diagonal = 1.0)
        notable_pairs: list of pairs where |current_rho - 60d_avg_rho| > 0.3
    """
    # We need 25 trading days for 20-day correlation, plus 65 days for 60-day baseline
    lookback_days = 100  # calendar days to cover ~65 trading days

    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT symbol, timestamp::date AS dt, close
            FROM market.ohlcv
            WHERE symbol = ANY(:symbols)
              AND timestamp::date BETWEEN :start AND :end
            ORDER BY symbol, timestamp::date
        """), {
            "symbols": CORRELATION_ASSETS,
            "start": trading_day - timedelta(days=lookback_days),
            "end": trading_day,
        })
        rows = result.fetchall()

    if not rows:
        return {"labels": [], "matrix": [], "notable_pairs": []}

    # Organize: symbol -> sorted list of (date, close)
    from collections import defaultdict
    price_series: Dict[str, List[Tuple[date, float]]] = defaultdict(list)
    for r in rows:
        price_series[r.symbol].append((r.dt, float(r.close)))

    # Only keep symbols with enough data (at least 25 trading days)
    valid_symbols = [s for s in CORRELATION_ASSETS if len(price_series[s]) >= 25]
    if len(valid_symbols) < 2:
        return {"labels": valid_symbols, "matrix": [], "notable_pairs": []}

    # Build aligned date set (intersection of all valid symbols' dates)
    date_sets = [set(dt for dt, _ in price_series[s]) for s in valid_symbols]
    common_dates = sorted(set.intersection(*date_sets))

    if len(common_dates) < 25:
        return {"labels": valid_symbols, "matrix": [], "notable_pairs": []}

    # Build price matrix: each row = symbol, each col = date
    import math
    price_lookup: Dict[str, Dict[date, float]] = {}
    for s in valid_symbols:
        price_lookup[s] = {dt: c for dt, c in price_series[s]}

    # Compute daily returns for common dates
    returns: Dict[str, List[float]] = {}
    for s in valid_symbols:
        ret = []
        for i in range(1, len(common_dates)):
            prev = price_lookup[s].get(common_dates[i - 1])
            curr = price_lookup[s].get(common_dates[i])
            if prev and curr and prev != 0:
                ret.append(curr / prev - 1)
            else:
                ret.append(0.0)
        returns[s] = ret

    n_returns = len(common_dates) - 1

    def pearson(x: List[float], y: List[float]) -> float:
        """Compute Pearson correlation between two return series."""
        n = len(x)
        if n < 5:
            return 0.0
        mx = sum(x) / n
        my = sum(y) / n
        sx = math.sqrt(max(sum((xi - mx) ** 2 for xi in x) / n, 1e-15))
        sy = math.sqrt(max(sum((yi - my) ** 2 for yi in y) / n, 1e-15))
        cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
        return cov / (sx * sy)

    # 20-day correlation (use last 20 returns)
    window_20 = 20
    recent_20 = {s: returns[s][-window_20:] for s in valid_symbols} if n_returns >= window_20 else {s: returns[s] for s in valid_symbols}

    n = len(valid_symbols)
    matrix_20: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix_20[i][i] = 1.0
        for j in range(i + 1, n):
            rho = pearson(recent_20[valid_symbols[i]], recent_20[valid_symbols[j]])
            rho = max(-1.0, min(1.0, rho))
            matrix_20[i][j] = round(rho, 4)
            matrix_20[j][i] = round(rho, 4)

    # 60-day correlation for detecting breaks
    notable_pairs = []
    if n_returns >= 60:
        window_60 = 60
        recent_60 = {s: returns[s][-window_60:] for s in valid_symbols}
        for i in range(n):
            for j in range(i + 1, n):
                rho_60 = pearson(recent_60[valid_symbols[i]], recent_60[valid_symbols[j]])
                rho_20 = matrix_20[i][j]
                delta = abs(rho_20 - rho_60)
                if delta > 0.3:
                    notable_pairs.append({
                        "pair": f"{valid_symbols[i]}/{valid_symbols[j]}",
                        "rho_20d": round(rho_20, 2),
                        "rho_60d": round(rho_60, 2),
                        "delta": round(rho_20 - rho_60, 2),
                    })

    # Round matrix for display
    matrix_display = [[round(v, 2) for v in row] for row in matrix_20]

    return {
        "labels": valid_symbols,
        "matrix": matrix_display,
        "notable_pairs": notable_pairs,
    }


async def _fetch_forward_events(as_of: date, days: int = 5) -> List[Dict[str, Any]]:
    """Fetch upcoming events."""
    end = as_of + timedelta(days=days)
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT symbol, event_type, event_time
            FROM event.events
            WHERE event_time::date > :start AND event_time::date <= :end
            ORDER BY event_time ASC
            LIMIT 20
        """), {"start": as_of, "end": end})
        rows = result.fetchall()

    events = []
    for r in rows:
        events.append({
            "date": r.event_time.strftime("%Y-%m-%d") if hasattr(r.event_time, "strftime") else str(r.event_time),
            "event": r.event_type,
            "symbol": r.symbol,
            "impact": "earnings" if r.event_type == "earnings" else "event",
        })
    return events


async def _fetch_news_headlines(as_of: date, limit: int = 15) -> List[Dict[str, Any]]:
    """Fetch recent news headlines with sentiment."""
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT headline, source, published_at, tickers, sentiment
            FROM news.articles
            WHERE published_at::date >= :start AND published_at::date <= :as_of
            ORDER BY published_at DESC
            LIMIT :limit
        """), {"start": as_of - timedelta(days=1), "as_of": as_of, "limit": limit})
        rows = result.fetchall()

    return [
        {
            "title": r.headline,
            "source": r.source,
            "symbols": list(r.tickers) if r.tickers else [],
            "sentiment": round(float(r.sentiment), 2) if r.sentiment is not None else None,
        }
        for r in rows
    ]


async def _fetch_insider_and_analyst(as_of: date) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch recent insider trades and analyst recommendations."""
    data: Dict[str, List] = {"insider_trades": [], "analyst_recs": []}

    async with engine.begin() as conn:
        # Insider trades — last 3 days
        try:
            result = await conn.execute(text("""
                SELECT symbol, insider_name, transaction_type, shares, price, value, filing_date
                FROM event.insider_trades
                WHERE filing_date >= :start
                ORDER BY ABS(value) DESC NULLS LAST
                LIMIT 10
            """), {"start": as_of - timedelta(days=3)})
            for r in result.fetchall():
                data["insider_trades"].append({
                    "symbol": r.symbol,
                    "insider": r.insider_name,
                    "type": r.transaction_type,
                    "value": float(r.value) if r.value else 0,
                })
        except Exception as e:
            logger.debug(f"Insider trades query failed (table may not exist): {e}")

        # Analyst recommendations — last 3 days
        try:
            result = await conn.execute(text("""
                SELECT symbol, firm, action, rating_to, price_target, date
                FROM event.analyst_recommendations
                WHERE date >= :start
                ORDER BY date DESC
                LIMIT 10
            """), {"start": as_of - timedelta(days=3)})
            for r in result.fetchall():
                data["analyst_recs"].append({
                    "symbol": r.symbol,
                    "firm": r.firm,
                    "action": r.action,
                    "rating": r.rating_to,
                    "target": float(r.price_target) if r.price_target else None,
                })
        except Exception as e:
            logger.debug(f"Analyst recs query failed (table may not exist): {e}")

    return data


# ---------------------------------------------------------------------------
# LLM commentary
# ---------------------------------------------------------------------------

def _build_llm_market_summary(
    market_groups: List[Dict[str, Any]],
    top_gainers: List[Dict[str, Any]],
    top_losers: List[Dict[str, Any]],
    macro_data: List[Dict[str, Any]],
) -> str:
    """Build a compact text summary of market data for the LLM prompt."""
    lines = []

    # Macro
    if macro_data:
        macro_str = ", ".join(f"{m['name']}: {m['value']}" for m in macro_data if m["value"] is not None)
        lines.append(f"MACRO: {macro_str}")

    # Per-group summary (show average 1d change + notable movers)
    for group in market_groups:
        changes = [s["change_1d"] for s in group["symbols"] if s["change_1d"] is not None]
        if not changes:
            continue
        avg = sum(changes) / len(changes)
        best = max(group["symbols"], key=lambda s: s["change_1d"] if s["change_1d"] is not None else -999)
        worst = min(group["symbols"], key=lambda s: s["change_1d"] if s["change_1d"] is not None else 999)

        detail_parts = []
        for s in group["symbols"]:
            if s["close"] is not None and s["change_1d"] is not None:
                detail_parts.append(f"{s['symbol']} ${s['close']:.2f} ({s['change_1d']:+.1f}%)")
        detail = ", ".join(detail_parts)
        lines.append(f"\n{group['name']} (avg {avg:+.1f}%): {detail}")

    # Top movers
    gainer_str = ", ".join(f"{g['symbol']} {g['change_1d']:+.1f}%" for g in top_gainers)
    loser_str = ", ".join(f"{l['symbol']} {l['change_1d']:+.1f}%" for l in top_losers)
    lines.append(f"\nTOP GAINERS: {gainer_str}")
    lines.append(f"TOP LOSERS: {loser_str}")

    return "\n".join(lines)


async def _generate_llm_commentary(
    llm_client,
    section: str,
    context: Dict[str, Any],
    as_of: date,
) -> str:
    """Use LLM to write a section of the report."""
    market_summary = context.get("market_text", "")

    prompts = {
        "what_happened": f"""Market data for {as_of.isoformat()}:

{market_summary}

Narrative: {json.dumps(context.get("narrative"), default=str)}
News: {json.dumps(context.get("news"), default=str)}
Insider/Analyst: {json.dumps(context.get("insider_analyst"), default=str)}

OUTPUT FORMAT — quant desk morning note. Tables, bullets, formulas. NO paragraphs. Be DETAILED and THOROUGH.

## Key Moves (TOP 10 ONLY — do NOT list every symbol)
| Symbol | Close | 1D Chg | Vol Ratio | Driver |
MAX 10 rows. Pick the 10 most significant by |return| × vol_ratio. Quality over quantity.

## Cross-Asset Matrix
| Signal | Reading | 20D Avg | Δ | Z-Score | Interpretation |
Cover ALL of these:
- Credit: HY OAS (bps), IG OAS, HY/IG ratio, CDX spread
- Curves: 2s10s (bps), 5s30s, real rate (10Y - breakeven)
- Ratios: XLY/XLP, XLK/XLF, copper/gold, oil/gold, BTC/NDX correlation (20d rolling)
- Vol surface: VIX level, VIX/VXV (term structure), VVIX, put/call ratio, skew (25Δ)
- FX/Rates: DXY, 10Y yield, TIP breakeven, fed funds futures implied
- Flows: GLD flows, TLT flows, HYG flows (if available from price action)

## Sector Heatmap
| Sector (ETF) | 1D | 5D | 20D | Rel vs SPY 1D | Rel vs SPY 5D | Momentum Signal |
ALL 11 sectors. Momentum signal = {">0" if True else "<0"} based on 5D vs 20D crossover.

## Correlation Breaks
| Pair | 20D ρ | 60D ρ | Today's Co-Move | Expected | Residual (σ) |
Flag ANY pair where today's residual > 1.5σ from the regression.

## Statistical Anomalies
For each anomaly:
- SYMBOL: return = X%, 20D μ = Y%, 20D σ = Z% → move = Wσ, p-value ≈ V
- Include volume anomalies: vol = Xk vs 20D avg = Yk → ratio = Z
- Include correlation breaks: ρ(A,B) 20D = X, today co-move implies ρ ≈ Y → break

## Microstructure
- Breadth: advancers/decliners ratio, % above 20DMA, % above 50DMA
- Dispersion: cross-sectional σ of returns today vs 20D avg dispersion
- Momentum: % of universe with positive 5D returns, % with positive 20D returns

{"User questions: " + "; ".join(context.get("user_questions", [])) if context.get("user_questions") else ""}

RULES: No prose. Every line = data. Be exhaustive — cover ALL asset classes. Use >, <, ≈, →, ↑, ↓, Δ, σ, ρ freely. Use 2+ pages if needed.""",

        "what_it_means": f"""Market data:
{market_summary}

Macro: {json.dumps(context.get("macro_data"), default=str)}
Debate regime: {context.get("debate_regime", "")}
Conflicts: {json.dumps(context.get("debate_conflicts", []), default=str)}
Risk warnings: {json.dumps(context.get("risk_warnings", []), default=str)}
Signals: {json.dumps(context.get("top_signals"), default=str)}

OUTPUT FORMAT — senior quant PM style. Formulaic, detailed, technical. NO paragraphs. Be THOROUGH.

## Regime Classification
- State: X (confidence: Y%)
- Composite score = w₁·f(VIX) + w₂·f(credit) + w₃·f(curve) + w₄·f(momentum) + w₅·f(dispersion)
- Show each component:
  - Vol component: VIX={{}}, percentile=X%, VIX/VXV={{}}, term structure={{contango|backwardation}}
  - Credit component: HY OAS={{}}, IG OAS={{}}, HY-IG={{}}, vs 1Y range [min, max]
  - Curve component: 2s10s={{}}, real rate={{}}, fed funds terminal implied={{}}
  - Momentum component: SPY 5D={{}}, 20D={{}}, breadth=X%
  - Dispersion: cross-sectional σ={{}}, vs 20D avg={{}}
- Regime transition probability: P(current→risk-off) ≈ X%, P(current→risk-on) ≈ Y%

## Divergence Matrix (DETAILED)
| Pair | Spread/Ratio | 20D Avg | 60D Avg | Z(20D) | Z(60D) | Mean-Revert Target | Half-Life (days) |
Cover at minimum 8-10 pairs:
- Equity vs Vol (SPX vs VIX)
- Credit vs Equity (HYG vs SPY)
- Crypto vs Tech (BTC proxies vs QQQ)
- Growth vs Value (QQQ/IWM or similar)
- Commodities vs USD (GLD vs DXY)
- EM vs DM (EEM vs SPY)
- Duration vs Credit (TLT vs HYG)
- Gold vs Real Rates (GLD vs TIP)
- Energy vs Broad (XLE vs SPY)
- Vol term structure (VIX vs VXZ)

## Factor Decomposition
| Factor | Proxy | 1D | 5D | 20D | Signal | Conviction (1-10) | Sizing (% of risk budget) |
- Momentum (high vs low momentum quintile)
- Value (high vs low P/E quintile)
- Quality (high vs low ROE)
- Size (IWM vs SPY)
- Low Vol (min vol vs market)
- Growth (QQQ vs IWM)
- Credit (HYG vs SHY)
- Carry (high yield vs short duration)

## Risk Decomposition
- Portfolio VaR(95%, 1D) estimate: $X (assuming $1M notional)
- Max sector concentration risk: sector X at Y% weight
- Tail risk: P(SPY < -2% | current regime) ≈ X%
- Correlation risk: avg pairwise ρ = X (high/low vs historical)
- Liquidity: bid-ask proxy (high vol ETFs vs normal)

## Highest Conviction Trades (DETAILED)
For each (minimum 5):
| # | Symbol | Dir | Entry | Target | Stop | R/R | Timeframe | Catalyst | Factor Exposure | Edge Decay |
Include:
- Expected return = X%, expected vol = Y%, Sharpe ≈ Z
- Position sizing: Kelly fraction f* = (μ/σ²) → suggested X% of capital
- Correlation to existing book: ρ ≈ X (diversifying/concentrating?)
- Key risk: what invalidates the thesis, expressed as a specific level/event

## Scenario Analysis
| Scenario | P(%) | SPY Δ | VIX Δ | HYG Δ | GLD Δ | BTC Δ | Portfolio Δ |
- Base case, bull case, bear case, tail risk
- For each: trigger condition, expected timeline, key indicators to watch

RULES: No prose. Every line = data or formula. Use ≈, →, ↑, ↓, Δ, σ, ρ, ∂, Σ, ∫ freely.
Be EXHAUSTIVE — this is for a PM who reads math, not English. Use as many pages as needed.""",
    }

    prompt = prompts.get(section, f"Write analysis for: {section}")
    system = (
        "You are a senior quant portfolio manager at a $5B systematic macro fund. "
        "You think in covariance matrices, factor loadings, and regime probabilities. "
        "Your communication style: formulas > tables > bullets > never paragraphs. "
        "Every output line MUST contain numbers, Greek letters, or mathematical operators. "
        "You use: σ, ρ, Δ, μ, β, α, Σ, θ, γ, ≈, →, ↑, ↓, ∝, ≫, ≪ naturally. "
        "Express uncertainty as probability ranges, not words. "
        "Show your work: intermediate calculations, not just conclusions. "
        "Think: Citadel/DE Shaw internal research note, not sell-side."
    )

    try:
        return await llm_client.generate(prompt, system=system, temperature=0.15)
    except Exception as e:
        logger.warning(f"LLM commentary failed for {section}: {e}")
        return f"*Analysis unavailable: {e}*"


# ---------------------------------------------------------------------------
# Main report generation
# ---------------------------------------------------------------------------

async def generate_report(
    as_of: Optional[date] = None,
    llm_client=None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate the daily market report.

    1. Resolve the latest trading day (handles weekends/holidays)
    2. Fetch full market dashboard (205 symbols, grouped)
    3. Fetch macro, narrative, debate, signals, events, news
    4. Generate rich LLM commentary with full market context
    5. Assemble template context
    6. Generate PDF

    Returns dict with report path and summary.
    """
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()

    logger.info(f"Generating report for {as_of}")

    # Resolve latest trading day for OHLCV lookups
    trading_day = await _resolve_latest_trading_day(as_of)
    if trading_day != as_of:
        logger.info(f"Using trading day {trading_day} (as_of={as_of} has no data)")

    # ---- Section 1: Market Dashboard ----
    market_groups, top_gainers, top_losers = await _fetch_market_dashboard(trading_day)
    macro_data = await _fetch_macro_data()

    # Build backward-compat market_snapshot
    market_snapshot = await _fetch_market_snapshot(as_of, market_groups, macro_data)

    # ---- Correlation Matrix ----
    try:
        correlation_matrix = await _compute_correlation_matrix(trading_day)
    except Exception as e:
        logger.warning(f"Correlation matrix computation failed: {e}")
        correlation_matrix = {"labels": [], "matrix": [], "notable_pairs": []}

    # ---- Remaining data fetches ----
    narrative_state = await _fetch_narrative_state(as_of)
    debate_payload = await _fetch_debate_results(as_of)
    portfolio_data = await _fetch_portfolio_state(as_of)
    track_record = await _fetch_track_record()
    signals_summary = await _fetch_signals_summary(as_of)
    forward_events = await _fetch_forward_events(as_of)

    # Optional data sources (may not have data yet)
    try:
        news_headlines = await _fetch_news_headlines(as_of)
    except Exception as e:
        logger.debug(f"News fetch failed: {e}")
        news_headlines = []

    try:
        insider_analyst = await _fetch_insider_and_analyst(as_of)
    except Exception as e:
        logger.debug(f"Insider/analyst fetch failed: {e}")
        insider_analyst = {"insider_trades": [], "analyst_recs": []}

    # Extract debate synthesis
    synthesis = debate_payload.get("synthesis", {}) if debate_payload else {}
    debate_r1 = debate_payload.get("round1", {}) if debate_payload else {}

    # Read feedback gaps from Mira
    from src.mira.push import read_feedback_gaps
    user_gaps = read_feedback_gaps()
    if user_gaps:
        logger.info(f"Addressing {len(user_gaps)} user feedback gaps")

    # ---- Section 3: LLM Commentary ----
    # Build rich market text for LLM
    market_text = _build_llm_market_summary(market_groups, top_gainers, top_losers, macro_data)

    commentary_context = {
        "market_text": market_text,
        "narrative": narrative_state,
        "macro_data": macro_data,
        "regime": synthesis.get("regime_consensus", ""),
        "signals_count": len(signals_summary),
        "portfolio_summary": portfolio_data.get("summary"),
        "top_signals": signals_summary[:5],
        "news": news_headlines[:10],
        "insider_analyst": insider_analyst,
        "user_questions": [g["question"] for g in user_gaps] if user_gaps else [],
    }

    what_happened = ""
    what_it_means = ""

    if llm_client:
        what_happened = await _generate_llm_commentary(
            llm_client, "what_happened", commentary_context, as_of,
        )
        what_it_means = await _generate_llm_commentary(
            llm_client, "what_it_means", {
                **commentary_context,
                "debate_regime": synthesis.get("regime_consensus"),
                "debate_conflicts": synthesis.get("key_conflicts", []),
                "risk_warnings": synthesis.get("risk_warnings", []),
            }, as_of,
        )
    else:
        if synthesis.get("regime_consensus"):
            what_happened = f"**Regime:** {synthesis['regime_consensus']}"
        what_it_means = ""

    # ---- Debate participants ----
    debate_participants = None
    llm_stats = debate_payload.get("llm_stats", {}) if debate_payload else {}
    if debate_r1 and llm_stats:
        available = list(llm_stats.keys())
        role_preference = {
            "macro": ["openai", "claude", "deepseek", "gemini"],
            "micro": ["deepseek", "openai", "gemini", "claude"],
            "crowd": ["gemini", "deepseek", "openai", "claude"],
        }
        debate_participants = {}
        used = set()
        for role, prefs in role_preference.items():
            for pref in prefs:
                if pref in available and pref not in used:
                    debate_participants[role] = pref
                    used.add(pref)
                    break
            if role not in debate_participants:
                debate_participants[role] = available[0] if available else role

    # Cover data
    portfolio_total = None
    if portfolio_data.get("summary"):
        portfolio_total = f"${portfolio_data['summary']['total_value']:,.0f}"

    regime_raw = synthesis.get("regime_consensus", "") or ""
    # Extract a short regime label (Bullish/Bearish/Neutral/Risk-Off etc.)
    regime = None
    if regime_raw:
        rl = regime_raw.lower()
        if any(w in rl for w in ("bull", "risk-on", "risk on", "rally")):
            regime = "Bullish"
        elif any(w in rl for w in ("bear", "risk-off", "risk off", "sell-off", "selloff", "crash")):
            regime = "Bearish"
        elif any(w in rl for w in ("volatile", "volatil", "uncertain", "choppy", "dispersion")):
            regime = "Volatile"
        elif any(w in rl for w in ("neutral", "range", "sideways", "mixed")):
            regime = "Neutral"
        else:
            # Fallback: first 20 chars
            regime = regime_raw[:20].strip()

    # ---- Assemble template context ----
    sections = {
        # Cover
        "regime": regime,
        "portfolio_total": portfolio_total,
        "trading_day": trading_day.isoformat(),

        # Correlation Matrix
        "correlation_matrix": correlation_matrix,

        # Section 1: Market Dashboard (NEW)
        "market_groups": market_groups,
        "top_gainers": top_gainers,
        "top_losers": top_losers,
        "macro_data": [
            {"name": m["name"], "value": m["value"]} for m in macro_data
        ],

        # Section 1 (backward compat): What Happened
        "market_snapshot": market_snapshot,
        "what_happened": what_happened,
        "narrative_state": narrative_state,

        # Section 2: What It Means
        "what_it_means": what_it_means,
        "regime_detail": None,
        "signals_summary": signals_summary,
        "debate_conflicts": synthesis.get("key_conflicts", []),

        # Section 3: What To Do
        "consensus_trades": synthesis.get("consensus_trades", []),
        "contrarian_trades": synthesis.get("contrarian_trades", []),
        "portfolio_actions": synthesis.get("portfolio_actions", []),
        "risk_warnings": synthesis.get("risk_warnings", []),

        # Section 4: Portfolio & Track Record
        "portfolio_positions": portfolio_data.get("positions", []),
        "portfolio_summary": portfolio_data.get("summary"),
        "track_record": track_record,

        # Section 5: Forward Calendar
        "forward_events": forward_events,

        # Additional context
        "news_headlines": news_headlines,
        "insider_trades": insider_analyst.get("insider_trades", []),
        "analyst_recs": insider_analyst.get("analyst_recs", []),

        # Appendix
        "debate_participants": debate_participants,
    }

    # Generate PDF
    pdf_path = await generate_pdf(sections, output_path=output_path)

    # Store report metadata
    await _store_report_metadata(as_of, pdf_path, sections)

    logger.info(f"Report generated: {pdf_path}")

    return {
        "as_of": as_of.isoformat(),
        "trading_day": trading_day.isoformat(),
        "status": "success",
        "pdf_path": pdf_path,
        "regime": regime,
        "portfolio_total": portfolio_total,
        "symbols_covered": len(ALL_SYMBOLS),
        "groups_covered": len(market_groups),
        "top_gainers": [g["symbol"] for g in top_gainers],
        "top_losers": [l["symbol"] for l in top_losers],
        "consensus_trades": synthesis.get("consensus_trades", []),
        "contrarian_trades": synthesis.get("contrarian_trades", []),
        "portfolio_actions": synthesis.get("portfolio_actions", []),
        "risk_warnings": synthesis.get("risk_warnings", []),
        "portfolio_positions": len(portfolio_data.get("positions", [])),
        "signals": len(signals_summary),
        "forward_events": len(forward_events),
        "news_headlines": len(news_headlines),
        "user_questions": [g["question"] for g in user_gaps] if user_gaps else [],
    }


async def _store_report_metadata(
    as_of: date, pdf_path: str, sections: Dict[str, Any],
) -> None:
    """Store report metadata in DB."""
    summary = {
        "regime": sections.get("regime", ""),
        "consensus_trades": len(sections.get("consensus_trades", [])),
        "contrarian_trades": len(sections.get("contrarian_trades", [])),
        "portfolio_total": sections.get("portfolio_total", ""),
        "signals": len(sections.get("signals_summary", [])),
        "symbols_covered": len(ALL_SYMBOLS),
        "groups_covered": len(sections.get("market_groups", [])),
    }

    try:
        async with engine.begin() as conn:
            await conn.execute(text("""
                INSERT INTO report.daily
                  (date, pdf_path, summary, regime, new_recommendations, active_recommendations)
                VALUES (:date, :pdf_path, :summary, :regime, :new_recs, :active_recs)
                ON CONFLICT (date) DO UPDATE SET
                  pdf_path = EXCLUDED.pdf_path,
                  summary = EXCLUDED.summary,
                  regime = EXCLUDED.regime,
                  new_recommendations = EXCLUDED.new_recommendations,
                  active_recommendations = EXCLUDED.active_recommendations
            """), {
                "date": as_of,
                "pdf_path": pdf_path,
                "summary": json.dumps(summary),
                "regime": sections.get("regime", ""),
                "new_recs": summary["consensus_trades"] + summary["contrarian_trades"],
                "active_recs": len(sections.get("portfolio_positions", [])),
            })
    except Exception as e:
        logger.warning(f"Failed to store report metadata: {e}")
