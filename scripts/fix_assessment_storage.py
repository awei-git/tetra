#!/usr/bin/env python3
"""
Fix assessment storage database tables.
"""

import asyncio
import asyncpg
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = 'postgresql://tetra_user:tetra_password@localhost:5432/tetra'

async def fix_tables():
    """Fix the assessment storage tables."""
    conn = None
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        # Create strategies schema if not exists
        await conn.execute("CREATE SCHEMA IF NOT EXISTS strategies")
        
        # Drop the problematic index if it exists
        try:
            await conn.execute("DROP INDEX IF EXISTS strategies.idx_backtest_strategy")
            logger.info("Dropped old index")
        except Exception as e:
            logger.warning(f"Could not drop index: {e}")
        
        # Ensure backtest_results table exists with correct columns
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS strategies.backtest_results (
                backtest_id SERIAL PRIMARY KEY,
                strategy_name VARCHAR(100),
                scenario_name VARCHAR(255),
                symbol VARCHAR(20),
                initial_capital DECIMAL(12,2),
                final_value DECIMAL(12,2),
                total_return DECIMAL(10,6),
                annualized_return DECIMAL(10,6),
                volatility DECIMAL(10,6),
                sharpe_ratio DECIMAL(8,4),
                max_drawdown DECIMAL(10,6),
                win_rate DECIMAL(5,4),
                total_trades INTEGER,
                metadata JSONB,
                backtest_start_date DATE,
                backtest_end_date DATE,
                run_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Created/verified backtest_results table")
        
        # Add missing columns if they don't exist
        await conn.execute("""
            ALTER TABLE strategies.backtest_results 
            ADD COLUMN IF NOT EXISTS symbol VARCHAR(20)
        """)
        await conn.execute("""
            ALTER TABLE strategies.backtest_results 
            ADD COLUMN IF NOT EXISTS scenario_name VARCHAR(255)
        """)
        
        # Create correct index on strategy_name
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_backtest_strategy 
            ON strategies.backtest_results(strategy_name)
        """)
        logger.info("Created index on strategy_name")
        
        # Create other required tables
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS strategies.strategy_metadata (
                strategy_id VARCHAR(100) PRIMARY KEY,
                strategy_name VARCHAR(200),
                category VARCHAR(50),
                description TEXT,
                parameters JSONB,
                comprehensive_metrics JSONB,
                current_assessment JSONB,
                projections JSONB,
                ranking_score DECIMAL(10,2),
                overall_rank INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Created/verified strategy_metadata table")
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS strategies.rankings (
                ranking_id SERIAL PRIMARY KEY,
                ranking_date DATE NOT NULL,
                ranking_type VARCHAR(50),
                rankings JSONB,
                scenarios_included TEXT[],
                weighting_scheme JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Created/verified rankings table")
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS strategies.assessment_summary (
                summary_id SERIAL PRIMARY KEY,
                run_date TIMESTAMP NOT NULL,
                total_strategies INTEGER,
                total_backtests INTEGER,
                successful_backtests INTEGER,
                failed_backtests INTEGER,
                top_strategies JSONB,
                summary_data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Created/verified assessment_summary table")
        
        logger.info("All tables fixed successfully!")
        
    except Exception as e:
        logger.error(f"Error fixing tables: {e}")
        raise
    finally:
        if conn:
            await conn.close()

if __name__ == "__main__":
    asyncio.run(fix_tables())