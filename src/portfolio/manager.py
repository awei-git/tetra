"""Portfolio daily mark-to-market + recommendation tracker.

Handles:
1. Position mark-to-market (update prices, PnL, weights)
2. Portfolio snapshots (daily total value, returns)
3. Recommendation tracking (mark open recs against market)
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from src.db.session import engine

logger = logging.getLogger(__name__)
UTC = timezone.utc


async def update_positions(as_of: Optional[date] = None) -> Dict[str, Any]:
    """Mark-to-market all positions with latest prices."""
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()

    async with engine.begin() as conn:
        # Get latest prices for all held symbols
        result = await conn.execute(text("""
            WITH latest AS (
                SELECT symbol, close,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS rn
                FROM market.ohlcv
                WHERE timestamp::date <= :date
            )
            SELECT symbol, close FROM latest WHERE rn = 1
        """), {"date": as_of})
        prices = {r.symbol: float(r.close) for r in result.fetchall()}

        # Get current positions
        result = await conn.execute(text("SELECT symbol, shares, avg_cost FROM portfolio.positions"))
        positions = result.fetchall()

        if not positions:
            return {"as_of": as_of.isoformat(), "status": "no_positions"}

        # Calculate totals for weight computation
        total_value = 0.0
        updated = []
        for pos in positions:
            price = prices.get(pos.symbol)
            if price is None:
                continue
            mv = float(pos.shares) * price
            total_value += mv
            updated.append({
                "symbol": pos.symbol,
                "shares": float(pos.shares),
                "avg_cost": float(pos.avg_cost),
                "current_price": price,
                "market_value": mv,
                "pnl": mv - float(pos.shares) * float(pos.avg_cost),
            })

        # Get cash
        result = await conn.execute(text("SELECT amount FROM portfolio.cash ORDER BY id DESC LIMIT 1"))
        cash_row = result.fetchone()
        cash = float(cash_row.amount) if cash_row else 0.0
        total_value += cash

        # Update positions with weights
        for u in updated:
            u["weight"] = u["market_value"] / total_value if total_value > 0 else 0
            await conn.execute(text("""
                UPDATE portfolio.positions SET
                  current_price = :price,
                  market_value = :mv,
                  unrealized_pnl = :pnl,
                  weight = :weight,
                  updated_at = NOW()
                WHERE symbol = :symbol
            """), {
                "symbol": u["symbol"],
                "price": u["current_price"],
                "mv": u["market_value"],
                "pnl": u["pnl"],
                "weight": u["weight"],
            })

        # Save snapshot
        prev_result = await conn.execute(text("""
            SELECT total_value FROM portfolio.snapshots
            WHERE date < :date ORDER BY date DESC LIMIT 1
        """), {"date": as_of})
        prev = prev_result.fetchone()
        prev_value = float(prev.total_value) if prev else total_value

        daily_return = (total_value / prev_value - 1) if prev_value > 0 else 0.0

        # Cumulative return from first snapshot
        first_result = await conn.execute(text("""
            SELECT total_value FROM portfolio.snapshots
            ORDER BY date ASC LIMIT 1
        """))
        first = first_result.fetchone()
        first_value = float(first.total_value) if first else total_value
        cum_return = (total_value / first_value - 1) if first_value > 0 else 0.0

        positions_json = json.dumps([
            {"symbol": u["symbol"], "shares": u["shares"],
             "price": u["current_price"], "value": round(u["market_value"], 2),
             "weight": round(u["weight"] * 100, 2), "pnl": round(u["pnl"], 2)}
            for u in updated
        ])

        await conn.execute(text("""
            INSERT INTO portfolio.snapshots
              (date, total_value, cash, invested, daily_return, cumulative_return, positions)
            VALUES (:date, :total, :cash, :invested, :daily_ret, :cum_ret,
                    CAST(:positions AS JSONB))
            ON CONFLICT (date) DO UPDATE SET
              total_value = EXCLUDED.total_value,
              cash = EXCLUDED.cash,
              invested = EXCLUDED.invested,
              daily_return = EXCLUDED.daily_return,
              cumulative_return = EXCLUDED.cumulative_return,
              positions = EXCLUDED.positions
        """), {
            "date": as_of,
            "total": total_value,
            "cash": cash,
            "invested": total_value - cash,
            "daily_ret": daily_return,
            "cum_ret": cum_return,
            "positions": positions_json,
        })

    logger.info(
        f"Portfolio update: total=${total_value:,.0f}, "
        f"daily={daily_return:+.2%}, positions={len(updated)}"
    )

    return {
        "as_of": as_of.isoformat(),
        "status": "success",
        "total_value": round(total_value, 2),
        "cash": round(cash, 2),
        "invested": round(total_value - cash, 2),
        "daily_return": round(daily_return, 4),
        "cumulative_return": round(cum_return, 4),
        "positions": [
            {"symbol": u["symbol"],
             "price": u["current_price"],
             "value": round(u["market_value"], 0),
             "weight": round(u["weight"] * 100, 1),
             "pnl": round(u["pnl"], 0)}
            for u in sorted(updated, key=lambda x: -x["market_value"])
        ],
    }


async def update_recommendation_tracker(as_of: Optional[date] = None) -> Dict[str, Any]:
    """Mark open recommendations against current prices."""
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()

    async with engine.begin() as conn:
        # Get open recs
        result = await conn.execute(text("""
            SELECT id, symbol, direction, entry_price, target_price, stop_loss,
                   created_date, confidence, method
            FROM tracker.recommendations
            WHERE status = 'open'
        """))
        open_recs = result.fetchall()

        if not open_recs:
            return {"as_of": as_of.isoformat(), "status": "no_open_recs"}

        # Get latest prices
        symbols = list(set(r.symbol for r in open_recs))
        result = await conn.execute(text("""
            WITH latest AS (
                SELECT symbol, close,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS rn
                FROM market.ohlcv
                WHERE symbol = ANY(:symbols) AND timestamp::date <= :date
            )
            SELECT symbol, close FROM latest WHERE rn = 1
        """), {"symbols": symbols, "date": as_of})
        prices = {r.symbol: float(r.close) for r in result.fetchall()}

        updated = 0
        closed = 0
        for rec in open_recs:
            price = prices.get(rec.symbol)
            if price is None:
                continue

            entry = float(rec.entry_price)
            multiplier = 1.0 if rec.direction == "long" else -1.0
            pnl_pct = multiplier * (price / entry - 1)

            # Check max favorable / adverse
            max_fav = pnl_pct if pnl_pct > 0 else 0
            max_adv = pnl_pct if pnl_pct < 0 else 0

            # Mark daily
            await conn.execute(text("""
                INSERT INTO tracker.daily_marks (rec_id, date, price, unrealized_pnl)
                VALUES (:rec_id, :date, :price, :pnl)
                ON CONFLICT (rec_id, date) DO UPDATE SET
                  price = EXCLUDED.price, unrealized_pnl = EXCLUDED.unrealized_pnl
            """), {"rec_id": rec.id, "date": as_of, "price": price, "pnl": pnl_pct})

            # Check stop loss / target
            new_status = "open"
            if rec.target_price and multiplier * (price - float(rec.target_price)) >= 0:
                new_status = "hit_target"
            elif rec.stop_loss and multiplier * (float(rec.stop_loss) - price) >= 0:
                new_status = "hit_stop"
            elif (as_of - rec.created_date).days > 30:
                new_status = "expired"

            if new_status != "open":
                await conn.execute(text("""
                    UPDATE tracker.recommendations SET
                      status = :status, closed_date = :date,
                      closed_price = :price, realized_pnl = :pnl
                    WHERE id = :id
                """), {"status": new_status, "date": as_of,
                       "price": price, "pnl": pnl_pct, "id": rec.id})
                closed += 1
            else:
                await conn.execute(text("""
                    UPDATE tracker.recommendations SET
                      max_favorable = GREATEST(COALESCE(max_favorable, 0), :fav),
                      max_adverse = LEAST(COALESCE(max_adverse, 0), :adv)
                    WHERE id = :id
                """), {"fav": max_fav, "adv": max_adv, "id": rec.id})

            updated += 1

    logger.info(f"Tracker update: {updated} recs marked, {closed} closed")
    return {
        "as_of": as_of.isoformat(),
        "status": "success",
        "open_recs": len(open_recs),
        "updated": updated,
        "closed": closed,
    }


async def create_recommendations_from_debate(
    as_of: date, debate_result: Dict[str, Any],
) -> int:
    """Convert debate consensus/contrarian trades into tracked recommendations."""
    synthesis = debate_result  # Already the synthesis dict

    trades = []
    for t in synthesis.get("consensus_trades", []):
        trades.append({**t, "method": "debate_consensus"})
    for t in synthesis.get("contrarian_trades", []):
        trades.append({**t, "method": "debate_contrarian"})

    if not trades:
        return 0

    # Get current prices for entry
    symbols = list(set(t["symbol"] for t in trades))
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            WITH latest AS (
                SELECT symbol, close,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS rn
                FROM market.ohlcv
                WHERE symbol = ANY(:symbols) AND timestamp::date <= :date
            )
            SELECT symbol, close FROM latest WHERE rn = 1
        """), {"symbols": symbols, "date": as_of})
        prices = {r.symbol: float(r.close) for r in result.fetchall()}

        created = 0
        for t in trades:
            symbol = t["symbol"]
            price = prices.get(symbol)
            if not price:
                continue

            direction = t.get("direction", "long")
            confidence = t.get("confidence", "medium")

            # Set target/stop based on confidence
            if confidence == "high":
                target_mult = 1.08 if direction == "long" else 0.92
                stop_mult = 0.95 if direction == "long" else 1.05
            else:
                target_mult = 1.05 if direction == "long" else 0.95
                stop_mult = 0.97 if direction == "long" else 1.03

            await conn.execute(text("""
                INSERT INTO tracker.recommendations
                  (created_date, symbol, direction, entry_price, target_price,
                   stop_loss, status, confidence, method, thesis, risk_factors)
                VALUES (:date, :symbol, :direction, :entry, :target,
                        :stop, 'open', :confidence, :method, :thesis, :risk)
            """), {
                "date": as_of,
                "symbol": symbol,
                "direction": direction,
                "entry": price,
                "target": price * target_mult,
                "stop": price * stop_mult,
                "confidence": confidence,
                "method": t["method"],
                "thesis": t.get("combined_thesis", t.get("thesis", "")),
                "risk": t.get("risk", ""),
            })
            created += 1

    logger.info(f"Created {created} tracked recommendations from debate")
    return created


async def run_portfolio_update(as_of: Optional[date] = None) -> Dict[str, Any]:
    """Run full portfolio update cycle."""
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()

    logger.info(f"Running portfolio update for {as_of}")

    positions = await update_positions(as_of)
    tracker = await update_recommendation_tracker(as_of)

    return {
        "as_of": as_of.isoformat(),
        "status": "success",
        "positions": positions,
        "tracker": tracker,
    }
