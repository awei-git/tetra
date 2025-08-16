"""Step 6: Store assessment results in the database."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import asyncpg
import os
from pathlib import Path

from src.pipelines.base import PipelineStep, PipelineContext

logger = logging.getLogger(__name__)


class DatabaseStorageStep(PipelineStep):
    """Store assessment results in the database for frontend display."""
    
    def __init__(self):
        super().__init__("DatabaseStorage")
        self.database_url = os.getenv(
            'DATABASE_URL',
            'postgresql://tetra_user:tetra_password@localhost:5432/tetra'
        )
    
    async def execute(self, context: PipelineContext) -> None:
        """Store assessment results in database."""
        logger.info("Storing assessment results in database")
        
        # Get data from context
        comprehensive_metrics = context.data.get('comprehensive_metrics', {})
        rankings = context.data.get('rankings', {})
        backtest_results = context.data.get('backtest_results', [])
        strategies = context.data.get('strategies', [])
        
        if not comprehensive_metrics:
            logger.warning("No comprehensive metrics to store")
            return
        
        conn = None
        try:
            # Connect to database
            conn = await asyncpg.connect(self.database_url)
            
            # Create tables if they don't exist
            await self._ensure_tables_exist(conn)
            
            # Store backtest results
            await self._store_backtest_results(conn, backtest_results, comprehensive_metrics)
            
            # Store strategy rankings
            await self._store_rankings(conn, rankings)
            
            # Store strategy metadata with comprehensive metrics
            await self._store_strategy_metadata(conn, strategies, comprehensive_metrics, rankings)
            
            # Store assessment summary
            await self._store_assessment_summary(conn, context)
            
            logger.info("Successfully stored assessment results in database")
            
        except Exception as e:
            logger.error(f"Failed to store results in database: {e}")
            raise
        finally:
            if conn:
                await conn.close()
    
    async def _ensure_tables_exist(self, conn: asyncpg.Connection) -> None:
        """Ensure required database tables exist."""
        # Create strategies schema if not exists
        await conn.execute("CREATE SCHEMA IF NOT EXISTS strategies")
        
        # Backtest results table already exists with correct schema
        # Just ensure symbol and scenario_name columns exist
        await conn.execute("""
            ALTER TABLE strategies.backtest_results 
            ADD COLUMN IF NOT EXISTS symbol VARCHAR(20)
        """)
        await conn.execute("""
            ALTER TABLE strategies.backtest_results 
            ADD COLUMN IF NOT EXISTS scenario_name VARCHAR(255)
        """)
        
        # Create rankings table
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
        
        # Create regime_performance table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS strategies.regime_performance (
                strategy_id VARCHAR(100),
                regime_type VARCHAR(50),
                total_return DECIMAL(10,6),
                sharpe_ratio DECIMAL(8,4),
                max_drawdown DECIMAL(10,6),
                win_rate DECIMAL(5,4),
                periods_count INTEGER,
                total_days INTEGER,
                regime_metrics JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (strategy_id, regime_type)
            )
        """)
        
        # Create strategy_metadata table for frontend display
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS strategies.strategy_metadata (
                strategy_id VARCHAR(100) PRIMARY KEY,
                strategy_name VARCHAR(200),
                category VARCHAR(50),
                description TEXT,
                parameters JSONB,
                comprehensive_metrics JSONB,  -- All 50+ metrics
                current_assessment JSONB,      -- Real-time signals
                projections JSONB,             -- Return projections
                ranking_score DECIMAL(10,2),
                overall_rank INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create assessment_summary table
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
        
        # Create indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_backtest_strategy 
            ON strategies.backtest_results(strategy_name)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_backtest_sharpe 
            ON strategies.backtest_results(sharpe_ratio DESC)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_strategy_rank 
            ON strategies.strategy_metadata(overall_rank)
        """)
    
    async def _store_backtest_results(
        self, 
        conn: asyncpg.Connection, 
        results: List,
        comprehensive_metrics: Dict
    ) -> None:
        """Store individual backtest results."""
        if not results:
            return
        
        # Clear existing results
        await conn.execute("DELETE FROM strategies.backtest_results")
        
        for result in results:
            strategy_metrics = comprehensive_metrics.get(result.strategy_name, {})
            
            await conn.execute("""
                INSERT INTO strategies.backtest_results (
                    strategy_name, scenario_name, symbol,
                    initial_capital, final_value,
                    total_return, annualized_return, volatility,
                    sharpe_ratio, max_drawdown,
                    win_rate, total_trades,
                    metadata, backtest_start_date, backtest_end_date,
                    run_date, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
            """,
                result.strategy_name,
                result.scenario_name,
                result.symbol,
                100000.0,  # initial_capital
                100000.0 * (1 + result.total_return),  # final_value
                result.total_return,
                result.annualized_return,
                result.volatility,
                result.sharpe_ratio,
                result.max_drawdown,
                result.win_rate,
                result.total_trades,
                json.dumps({
                    **strategy_metrics,
                    'profit_factor': result.profit_factor,
                    'equity_curve': result.equity_curve if result.equity_curve else [],
                    'trade_log': result.trade_log if result.trade_log else []
                }),
                result.metadata.get('start_date'),
                result.metadata.get('end_date'),
                datetime.now(),
                datetime.now()
            )
        
        logger.info(f"Stored {len(results)} backtest results")
    
    async def _store_rankings(self, conn: asyncpg.Connection, rankings: Dict) -> None:
        """Store strategy rankings."""
        if not rankings:
            return
        
        # Store overall rankings
        await conn.execute("""
            INSERT INTO strategies.rankings (
                ranking_date, ranking_type, rankings, weighting_scheme
            ) VALUES ($1, $2, $3, $4)
        """,
            datetime.now().date(),
            'overall',
            json.dumps(rankings.get('overall', [])),
            json.dumps({
                'sharpe_ratio': 0.3,
                'total_return': 0.2,
                'max_drawdown': 0.2,
                'consistency': 0.15,
                'regime_adaptability': 0.15
            })
        )
        
        # Store category rankings
        if rankings.get('by_category'):
            await conn.execute("""
                INSERT INTO strategies.rankings (
                    ranking_date, ranking_type, rankings
                ) VALUES ($1, $2, $3)
            """,
                datetime.now().date(),
                'by_category',
                json.dumps(rankings['by_category'])
            )
        
        # Store regime rankings
        if rankings.get('by_regime'):
            await conn.execute("""
                INSERT INTO strategies.rankings (
                    ranking_date, ranking_type, rankings
                ) VALUES ($1, $2, $3)
            """,
                datetime.now().date(),
                'by_regime',
                json.dumps(rankings['by_regime'])
            )
        
        logger.info("Stored strategy rankings")
    
    async def _store_strategy_metadata(
        self, 
        conn: asyncpg.Connection, 
        strategies: List[Dict],
        comprehensive_metrics: Dict,
        rankings: Dict
    ) -> None:
        """Store strategy metadata with comprehensive metrics for frontend."""
        if not strategies:
            return
        
        # Clear existing metadata
        await conn.execute("DELETE FROM strategies.strategy_metadata")
        
        # Get overall rankings for rank assignment
        overall_rankings = rankings.get('overall', [])
        rank_map = {r['name']: r['rank'] for r in overall_rankings}
        
        for strategy in strategies:
            strategy_name = strategy['name']
            metrics = comprehensive_metrics.get(strategy_name, {})
            
            # Prepare metadata structure for frontend
            metadata = {
                'symbol': 'MULTI',  # Assessment across multiple symbols
                'category': strategy.get('category', 'unknown'),
                'parameters': strategy.get('config', {}).get('parameters', {}),
                
                # All comprehensive metrics
                'total_return': metrics.get('total_return', 0),
                'annualized_return': metrics.get('annualized_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'sortino_ratio': metrics.get('sortino_ratio', 0),
                'calmar_ratio': metrics.get('calmar_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'avg_drawdown': metrics.get('avg_drawdown', 0),
                'volatility': metrics.get('volatility', 0),
                'downside_deviation': metrics.get('downside_deviation', 0),
                
                # Trade metrics
                'win_rate': metrics.get('win_rate', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'total_trades': metrics.get('total_trades', 0),
                'avg_win': metrics.get('avg_win', 0),
                'avg_loss': metrics.get('avg_loss', 0),
                'payoff_ratio': metrics.get('payoff_ratio', 0),
                'expectancy': metrics.get('expectancy', 0),
                'sqn': metrics.get('sqn', 0),
                'edge_ratio': metrics.get('edge_ratio', 0),
                'kelly_fraction': metrics.get('kelly_fraction', 0),
                
                # Risk metrics
                'var_95': metrics.get('var_95', 0),
                'var_99': metrics.get('var_99', 0),
                'cvar_95': metrics.get('cvar_95', 0),
                'cvar_99': metrics.get('cvar_99', 0),
                'omega_ratio': metrics.get('omega_ratio', 0),
                'ulcer_index': metrics.get('ulcer_index', 0),
                'tail_ratio': metrics.get('tail_ratio', 0),
                
                # Efficiency metrics
                'time_in_market': metrics.get('time_in_market', 0),
                'trade_frequency': metrics.get('trade_frequency', 0),
                'recovery_factor': metrics.get('recovery_factor', 0),
                'mar_ratio': metrics.get('mar_ratio', 0),
                
                # Robustness
                'return_stability': metrics.get('return_stability', 0),
                'consistency_score': metrics.get('consistency_score', 0),
                
                # Ranking
                'ranking_score': metrics.get('ranking_score', 0),
                'overall_rank': rank_map.get(strategy_name, 999)
            }
            
            # Current assessment (placeholder for real-time signals)
            current_assessment = metrics.get('current_assessment', {
                'current_signal': 'HOLD',
                'position_size': 0,
                'risk_metrics': {
                    'risk_per_trade': 0.02,
                    'position_risk': 0
                }
            })
            
            # Projections
            projections = metrics.get('projections', {
                '1w': 0,
                '2w': 0,
                '1m': 0,
                '3m': 0,
                '6m': 0,
                '1y': 0
            })
            
            await conn.execute("""
                INSERT INTO strategies.strategy_metadata (
                    strategy_id, strategy_name, category, description,
                    parameters, comprehensive_metrics, current_assessment,
                    projections, ranking_score, overall_rank,
                    created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
                strategy_name,
                strategy_name,
                strategy.get('category', 'unknown'),
                strategy.get('config', {}).get('description', ''),
                json.dumps(strategy.get('config', {}).get('parameters', {})),
                json.dumps(metadata),
                json.dumps(current_assessment),
                json.dumps(projections),
                metrics.get('ranking_score', 0),
                rank_map.get(strategy_name, 999),
                datetime.now(),
                datetime.now()
            )
        
        logger.info(f"Stored metadata for {len(strategies)} strategies")
    
    async def _store_assessment_summary(
        self, 
        conn: asyncpg.Connection, 
        context: PipelineContext
    ) -> None:
        """Store assessment run summary."""
        summary = context.data.get('backtest_summary', {})
        rankings = context.data.get('rankings', {})
        
        # Get top strategies
        top_strategies = rankings.get('overall', [])[:10]
        
        summary_data = {
            'execution_time': context.data.get('execution_time', 0),
            'scenarios_tested': len(context.data.get('scenarios', [])),
            'symbols_tested': len(context.data.get('symbols', [])),
            'strategies_tested': len(context.data.get('strategies', [])),
            'top_strategy': top_strategies[0] if top_strategies else None,
            'rankings': rankings
        }
        
        await conn.execute("""
            INSERT INTO strategies.assessment_summary (
                run_date, total_strategies, total_backtests,
                successful_backtests, failed_backtests,
                top_strategies, summary_data
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
            datetime.now(),
            len(context.data.get('strategies', [])),
            summary.get('total_backtests', 0),
            summary.get('successful_backtests', 0),
            summary.get('failed_backtests', 0),
            json.dumps(top_strategies),
            json.dumps(summary_data)
        )
        
        logger.info("Stored assessment summary")