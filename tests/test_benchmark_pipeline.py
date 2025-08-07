#!/usr/bin/env python3
"""Test the benchmark pipeline with sample data."""

import asyncio
import asyncpg
from datetime import datetime, timedelta
import json
import random
import pytest


async def populate_test_backtest_data():
    """Populate the database with test backtest data."""
    # Connect to database
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="tetra_user",
        password="tetra_password",
        database="tetra"
    )
    
    try:
        run_date = datetime.now()
        
        # Define test strategies
        strategies = [
            ("Buy and Hold", "passive", 0.12, 1.2, -0.15),
            ("Golden Cross", "trend_following", 0.18, 1.5, -0.22),
            ("Mean Reversion", "mean_reversion", 0.15, 1.3, -0.18),
            ("Momentum", "trend_following", 0.22, 1.8, -0.25),
            ("RSI Strategy", "mean_reversion", 0.10, 0.9, -0.12),
        ]
        
        print("Inserting test backtest results...")
        
        # Insert backtest results
        for i, (name, category, total_return, sharpe, max_dd) in enumerate(strategies):
            # Generate some random metrics
            win_rate = random.uniform(0.4, 0.7)
            trades = random.randint(50, 200)
            volatility = random.uniform(0.15, 0.25)
            
            metadata = {
                "sortino_ratio": sharpe * 1.2,
                "calmar_ratio": abs(total_return / max_dd),
                "consistency_score": random.uniform(0.6, 0.9),
                "positive_months_pct": random.uniform(0.5, 0.7)
            }
            
            query = """
                INSERT INTO strategies.backtest_results (
                    strategy_name, run_date, backtest_start_date, backtest_end_date,
                    universe, initial_capital, final_value, total_return,
                    annualized_return, sharpe_ratio, max_drawdown, volatility,
                    win_rate, total_trades, metadata, created_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
                ) RETURNING id
            """
            
            result_id = await conn.fetchval(query,
                name, run_date,
                run_date - timedelta(days=90), run_date,  # 3 month backtest
                "core", 100000,
                100000 * (1 + total_return), total_return,
                total_return * 4,  # Annualized
                sharpe, max_dd, volatility,
                win_rate, trades,
                json.dumps(metadata), run_date
            )
            
            print(f"  ✓ {name}: return={total_return:.1%}, sharpe={sharpe:.2f}")
            
            # Insert ranking
            await conn.execute("""
                INSERT INTO strategies.strategy_rankings (
                    run_date, strategy_name, rank_by_sharpe, rank_by_return,
                    rank_by_consistency, composite_score, category, overall_rank,
                    created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
                run_date, name,
                i + 1, i + 1, i + 1,
                (5 - i) * 20,  # Composite score
                category, i + 1, run_date
            )
            
            # Insert some sample equity curve data
            dates = []
            values = []
            value = 100000
            for d in range(90):
                date = (run_date - timedelta(days=90-d)).strftime("%Y-%m-%d")
                value *= (1 + random.uniform(-0.02, 0.025))  # Daily returns
                dates.append(date)
                values.append(round(value, 2))
            
            await conn.execute("""
                INSERT INTO strategies.equity_curves (
                    backtest_id, strategy_name, dates, values, created_at
                ) VALUES ($1, $2, $3, $4, $5)
            """,
                result_id, name,
                json.dumps(dates), json.dumps(values),
                run_date
            )
        
        # Insert summary
        await conn.execute("""
            INSERT INTO strategies.backtest_summary (
                run_date, total_strategies, successful_strategies,
                avg_return, avg_sharpe, avg_max_drawdown,
                best_return, worst_return, best_sharpe,
                execution_time, metadata, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """,
            run_date, len(strategies), len(strategies),
            0.15, 1.35, -0.18,
            0.22, 0.10, 1.8,
            5.2, json.dumps({"test": True}), run_date
        )
        
        print("\n✅ Test data populated successfully!")
        print("Check the strategies tab at http://localhost:5189/strategies")
        
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_populate_benchmark_data():
    """Test populating benchmark data."""
    await populate_test_backtest_data()


if __name__ == "__main__":
    # Run directly for manual testing
    asyncio.run(populate_test_backtest_data())