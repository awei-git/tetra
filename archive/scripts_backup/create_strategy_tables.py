#!/usr/bin/env python3
"""Create strategy backtest tables in the database."""

import asyncio
import asyncpg
import os
from datetime import datetime

async def create_strategy_tables():
    """Create all strategy-related tables."""
    # Get database URL from environment or use default
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://tetra_user:tetra_password@localhost/tetra"
    )
    
    # Parse connection string
    # Format: postgresql://user:password@host:port/database
    import urllib.parse
    parsed = urllib.parse.urlparse(database_url)
    
    # Connect to database
    conn = await asyncpg.connect(
        host=parsed.hostname or 'localhost',
        port=parsed.port or 5432,
        user=parsed.username or 'postgres',
        password=parsed.password or 'postgres',
        database=parsed.path[1:] if parsed.path else 'tetra'  # Remove leading /
    )
    
    try:
        # Create strategies schema
        await conn.execute("CREATE SCHEMA IF NOT EXISTS strategies")
        print("✓ Created strategies schema")
        
        # Create backtest_results table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS strategies.backtest_results (
                id SERIAL PRIMARY KEY,
                strategy_name VARCHAR NOT NULL,
                run_date TIMESTAMP NOT NULL,
                backtest_start_date DATE NOT NULL,
                backtest_end_date DATE NOT NULL,
                universe VARCHAR NOT NULL,
                initial_capital FLOAT NOT NULL,
                final_value FLOAT,
                total_return FLOAT,
                annualized_return FLOAT,
                sharpe_ratio FLOAT,
                max_drawdown FLOAT,
                volatility FLOAT,
                win_rate FLOAT,
                total_trades INTEGER,
                metadata JSONB,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
        
        # Create index
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy_date 
            ON strategies.backtest_results(strategy_name, run_date)
        """)
        print("✓ Created backtest_results table")
        
        # Create strategy_rankings table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS strategies.strategy_rankings (
                id SERIAL PRIMARY KEY,
                run_date TIMESTAMP NOT NULL,
                strategy_name VARCHAR NOT NULL,
                rank_by_sharpe INTEGER,
                rank_by_return INTEGER,
                rank_by_consistency INTEGER,
                composite_score FLOAT,
                category VARCHAR,
                overall_rank INTEGER,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
        
        # Create index
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_strategy_rankings_run_date 
            ON strategies.strategy_rankings(run_date)
        """)
        print("✓ Created strategy_rankings table")
        
        # Create backtest_summary table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS strategies.backtest_summary (
                id SERIAL PRIMARY KEY,
                run_date TIMESTAMP NOT NULL,
                total_strategies INTEGER NOT NULL,
                successful_strategies INTEGER NOT NULL,
                avg_return FLOAT,
                avg_sharpe FLOAT,
                avg_max_drawdown FLOAT,
                best_return FLOAT,
                worst_return FLOAT,
                best_sharpe FLOAT,
                execution_time FLOAT,
                metadata JSONB,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
        print("✓ Created backtest_summary table")
        
        # Create strategy_metadata table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS strategies.strategy_metadata (
                id SERIAL PRIMARY KEY,
                strategy_name VARCHAR NOT NULL UNIQUE,
                category VARCHAR,
                description TEXT,
                last_backtest_date TIMESTAMP,
                last_sharpe_ratio FLOAT,
                last_total_return FLOAT,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP
            )
        """)
        print("✓ Created strategy_metadata table")
        
        # Create equity_curves table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS strategies.equity_curves (
                id SERIAL PRIMARY KEY,
                backtest_id INTEGER NOT NULL,
                strategy_name VARCHAR NOT NULL,
                dates JSONB NOT NULL,
                values JSONB NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
        
        # Create index
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_equity_curves_backtest_id 
            ON strategies.equity_curves(backtest_id)
        """)
        print("✓ Created equity_curves table")
        
        print("\n✅ All strategy tables created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        raise
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(create_strategy_tables())