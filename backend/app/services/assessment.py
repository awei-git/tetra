"""Service for assessment pipeline data and analysis."""

from typing import Dict, Any, List, Optional
from datetime import datetime, date, timedelta
import logging
import json
import asyncpg

logger = logging.getLogger(__name__)


class AssessmentService:
    """Service for accessing assessment pipeline results."""
    
    def __init__(self, db_connection: asyncpg.Connection):
        self.db = db_connection
    
    async def get_top_strategies_by_category(self, limit: int = 3) -> Dict[str, List[Dict]]:
        """Get top strategies in each category."""
        query = """
            WITH latest_run AS (
                SELECT MAX(run_date) as max_date FROM assessment.strategy_rankings
            ),
            ranked_by_category AS (
                SELECT 
                    strategy_name,
                    category,
                    overall_rank,
                    category_rank,
                    avg_score,
                    avg_return,
                    avg_sharpe,
                    avg_drawdown,
                    ROW_NUMBER() OVER (PARTITION BY category ORDER BY category_rank) as rank_in_cat
                FROM assessment.strategy_rankings
                WHERE run_date = (SELECT max_date FROM latest_run)
            )
            SELECT * FROM ranked_by_category
            WHERE rank_in_cat <= $1
            ORDER BY category, rank_in_cat
        """
        
        rows = await self.db.fetch(query, limit)
        
        # Group by category
        result = {}
        for row in rows:
            category = row['category'] or 'uncategorized'
            if category not in result:
                result[category] = []
            result[category].append(dict(row))
        
        return result
    
    async def get_top_performers_by_scenario(self, limit: int = 3) -> Dict[str, List[Dict]]:
        """Get top performing strategies for each scenario."""
        query = """
            WITH latest_run AS (
                SELECT MAX(run_date) as max_date FROM assessment.backtest_results
            ),
            ranked_by_scenario AS (
                SELECT 
                    strategy_name,
                    scenario_name,
                    scenario_type,
                    symbol,
                    total_return,
                    sharpe_ratio,
                    max_drawdown,
                    score,
                    ROW_NUMBER() OVER (PARTITION BY scenario_name ORDER BY score DESC) as rank_in_scenario
                FROM assessment.backtest_results
                WHERE run_date = (SELECT max_date FROM latest_run)
            )
            SELECT * FROM ranked_by_scenario
            WHERE rank_in_scenario <= $1
            ORDER BY scenario_name, rank_in_scenario
        """
        
        rows = await self.db.fetch(query, limit)
        
        # Group by scenario
        result = {}
        for row in rows:
            scenario = row['scenario_name']
            if scenario not in result:
                result[scenario] = []
            result[scenario].append(dict(row))
        
        return result
    
    async def get_strategy_symbol_performance(self, strategy_name: str) -> List[Dict]:
        """Get performance of a strategy across all symbols."""
        query = """
            WITH latest_run AS (
                SELECT MAX(run_date) as max_date FROM assessment.backtest_results
            )
            SELECT 
                symbol,
                AVG(total_return) as avg_return,
                AVG(sharpe_ratio) as avg_sharpe,
                AVG(max_drawdown) as avg_drawdown,
                AVG(score) as avg_score,
                COUNT(*) as scenario_count
            FROM assessment.backtest_results
            WHERE strategy_name = $1
            AND run_date = (SELECT max_date FROM latest_run)
            GROUP BY symbol
            ORDER BY avg_score DESC
        """
        
        rows = await self.db.fetch(query, strategy_name)
        return [dict(row) for row in rows]
    
    async def get_best_strategy_symbol_pairs(self, limit: int = 10) -> List[Dict]:
        """Get the best strategy-symbol combinations."""
        query = """
            WITH latest_run AS (
                SELECT MAX(run_date) as max_date FROM assessment.backtest_results
            ),
            aggregated AS (
                SELECT 
                    strategy_name,
                    symbol,
                    AVG(total_return) as avg_return,
                    AVG(sharpe_ratio) as avg_sharpe,
                    AVG(max_drawdown) as avg_drawdown,
                    AVG(score) as avg_score,
                    COUNT(*) as scenario_count
                FROM assessment.backtest_results
                WHERE run_date = (SELECT max_date FROM latest_run)
                GROUP BY strategy_name, symbol
            )
            SELECT 
                strategy_name,
                symbol,
                avg_return,
                avg_sharpe,
                avg_drawdown,
                avg_score,
                scenario_count
            FROM aggregated
            ORDER BY avg_score DESC
            LIMIT $1
        """
        
        rows = await self.db.fetch(query, limit)
        return [dict(row) for row in rows]
    
    async def get_scenario_analysis(self, scenario_name: str) -> Dict[str, Any]:
        """Get detailed analysis for a specific scenario."""
        query = """
            WITH latest_run AS (
                SELECT MAX(run_date) as max_date FROM assessment.backtest_results
            )
            SELECT 
                strategy_name,
                symbol,
                total_return,
                sharpe_ratio,
                max_drawdown,
                win_rate,
                total_trades,
                score
            FROM assessment.backtest_results
            WHERE scenario_name = $1
            AND run_date = (SELECT max_date FROM latest_run)
            ORDER BY score DESC
        """
        
        rows = await self.db.fetch(query, scenario_name)
        
        if not rows:
            return {'scenario': scenario_name, 'results': []}
        
        # Calculate statistics
        returns = [row['total_return'] for row in rows if row['total_return']]
        avg_return = sum(returns) / len(returns) if returns else 0
        best_return = max(returns) if returns else 0
        worst_return = min(returns) if returns else 0
        
        return {
            'scenario': scenario_name,
            'statistics': {
                'avg_return': avg_return,
                'best_return': best_return,
                'worst_return': worst_return,
                'strategies_tested': len(rows)
            },
            'top_performers': [dict(row) for row in rows[:5]],
            'worst_performers': [dict(row) for row in rows[-5:]]
        }
    
    async def get_regime_performance(self) -> Dict[str, Any]:
        """Get performance statistics by market regime."""
        query = """
            WITH latest_run AS (
                SELECT MAX(run_date) as max_date FROM assessment.backtest_results
            )
            SELECT 
                scenario_type,
                COUNT(DISTINCT strategy_name) as strategies_count,
                AVG(total_return) as avg_return,
                MIN(total_return) as min_return,
                MAX(total_return) as max_return,
                AVG(sharpe_ratio) as avg_sharpe,
                AVG(max_drawdown) as avg_drawdown
            FROM assessment.backtest_results
            WHERE run_date = (SELECT max_date FROM latest_run)
            AND scenario_type IS NOT NULL
            GROUP BY scenario_type
            ORDER BY scenario_type
        """
        
        rows = await self.db.fetch(query)
        
        result = {}
        for row in rows:
            result[row['scenario_type']] = {
                'strategies_count': row['strategies_count'],
                'avg_return': float(row['avg_return']) if row['avg_return'] else 0,
                'min_return': float(row['min_return']) if row['min_return'] else 0,
                'max_return': float(row['max_return']) if row['max_return'] else 0,
                'avg_sharpe': float(row['avg_sharpe']) if row['avg_sharpe'] else 0,
                'avg_drawdown': float(row['avg_drawdown']) if row['avg_drawdown'] else 0
            }
        
        return result
    
    async def get_assessment_summary(self) -> Dict[str, Any]:
        """Get overall assessment summary."""
        # Get latest run info
        latest_run = await self.db.fetchrow("""
            SELECT 
                run_date,
                COUNT(DISTINCT strategy_name) as strategies_count,
                COUNT(DISTINCT symbol) as symbols_count,
                COUNT(DISTINCT scenario_name) as scenarios_count,
                COUNT(*) as total_tests
            FROM assessment.backtest_results
            WHERE run_date = (SELECT MAX(run_date) FROM assessment.backtest_results)
            GROUP BY run_date
        """)
        
        if not latest_run:
            return {'has_data': False}
        
        # Get top overall strategy
        top_strategy = await self.db.fetchrow("""
            SELECT 
                strategy_name,
                category,
                avg_score,
                avg_return,
                avg_sharpe
            FROM assessment.strategy_rankings
            WHERE run_date = $1
            ORDER BY overall_rank
            LIMIT 1
        """, latest_run['run_date'])
        
        # Get best scenario performance
        best_scenario = await self.db.fetchrow("""
            SELECT 
                scenario_name,
                top_strategy,
                top_return
            FROM assessment.scenario_performance
            WHERE run_date = $1
            ORDER BY top_return DESC
            LIMIT 1
        """, latest_run['run_date'])
        
        return {
            'has_data': True,
            'run_date': latest_run['run_date'].isoformat(),
            'statistics': {
                'strategies_tested': latest_run['strategies_count'],
                'symbols_tested': latest_run['symbols_count'],
                'scenarios_tested': latest_run['scenarios_count'],
                'total_backtests': latest_run['total_tests']
            },
            'top_strategy': dict(top_strategy) if top_strategy else None,
            'best_scenario': dict(best_scenario) if best_scenario else None
        }