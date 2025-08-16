#!/usr/bin/env python3
"""
Update assessment database schema to include all required fields.
"""

import asyncio
import asyncpg
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = 'postgresql://tetra_user:tetra_password@localhost:5432/tetra'

async def update_schema():
    """Update the assessment schema with all required fields."""
    conn = None
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        # Drop and recreate the strategy_trades table with complete info
        await conn.execute("DROP TABLE IF EXISTS strategies.strategy_trades CASCADE")
        
        await conn.execute("""
            CREATE TABLE strategies.strategy_trades (
                trade_id SERIAL PRIMARY KEY,
                strategy_name VARCHAR(100),
                symbol VARCHAR(20),
                
                -- Current market data
                current_price DECIMAL(12,4),
                target_price DECIMAL(12,4),
                exit_price DECIMAL(12,4),
                stop_loss_price DECIMAL(12,4),
                
                -- Return windows
                return_2w DECIMAL(10,4),
                return_1m DECIMAL(10,4), 
                return_3m DECIMAL(10,4),
                
                -- Trade execution
                trade_type VARCHAR(20),  -- BUY, SELL, HOLD
                position_size DECIMAL(12,2),
                execution_instructions TEXT,
                signal_strength DECIMAL(5,2),
                
                -- Scenario analysis
                scenario_returns JSONB,  -- {scenario_name: return}
                scenario_prices JSONB,   -- {scenario_name: projected_price}
                
                -- Performance metrics
                expected_return DECIMAL(10,4),
                volatility DECIMAL(10,4),
                sharpe_ratio DECIMAL(8,4),
                max_drawdown DECIMAL(10,4),
                win_probability DECIMAL(5,4),
                
                -- Scoring
                composite_score DECIMAL(10,2),
                score_components JSONB,  -- breakdown of score calculation
                rank INTEGER,
                
                -- Metadata
                last_signal_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Created strategy_trades table")
        
        # Create index for faster queries
        await conn.execute("""
            CREATE INDEX idx_trades_strategy_symbol 
            ON strategies.strategy_trades(strategy_name, symbol)
        """)
        await conn.execute("""
            CREATE INDEX idx_trades_score 
            ON strategies.strategy_trades(composite_score DESC)
        """)
        
        # Update strategy_metadata to include scoring formula
        await conn.execute("""
            ALTER TABLE strategies.strategy_metadata 
            ADD COLUMN IF NOT EXISTS scoring_formula TEXT DEFAULT 
            'Score = (0.4 × Sharpe Ratio) + (0.3 × Return/Volatility) + (0.2 × Win Rate) + (0.1 × (1 - Max Drawdown))'
        """)
        
        # Create a table for scenario-specific results
        await conn.execute("DROP TABLE IF EXISTS strategies.scenario_results CASCADE")
        await conn.execute("""
            CREATE TABLE strategies.scenario_results (
                result_id SERIAL PRIMARY KEY,
                strategy_name VARCHAR(100),
                symbol VARCHAR(20),
                scenario_name VARCHAR(255),
                
                -- Scenario-specific metrics
                scenario_return DECIMAL(10,4),
                scenario_volatility DECIMAL(10,4),
                scenario_sharpe DECIMAL(8,4),
                scenario_max_dd DECIMAL(10,4),
                scenario_win_rate DECIMAL(5,4),
                
                -- Price projections
                start_price DECIMAL(12,4),
                end_price DECIMAL(12,4),
                high_price DECIMAL(12,4),
                low_price DECIMAL(12,4),
                
                -- Trade stats
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Created scenario_results table")
        
        logger.info("Schema update completed successfully!")
        
    except Exception as e:
        logger.error(f"Error updating schema: {e}")
        raise
    finally:
        if conn:
            await conn.close()

if __name__ == "__main__":
    asyncio.run(update_schema())