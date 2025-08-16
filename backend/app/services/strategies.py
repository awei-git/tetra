"""Service for strategy data and performance calculations."""

from typing import Dict, Any, List, Optional
from datetime import datetime, date, timedelta
import logging
import json
import numpy as np
from sqlalchemy import text

# Import these when implementing real-time performance calculation
# from src.simulators.historical.simulator import HistoricalSimulator
# from src.simulators.historical.market_data_provider import MarketDataProvider
# from src.strats import strategy_registry

logger = logging.getLogger(__name__)


class StrategyService:
    """Service for managing strategy data and performance."""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    async def get_strategy_trades(self, strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get detailed trade recommendations with all metrics."""
        query = """
            SELECT 
                strategy_name, symbol, current_price, target_price, exit_price, stop_loss_price,
                return_2w, return_1m, return_3m, trade_type, position_size, 
                execution_instructions, signal_strength, scenario_returns, scenario_prices,
                expected_return, volatility, sharpe_ratio, max_drawdown, win_probability,
                composite_score, score_components, rank, last_signal_date
            FROM strategies.strategy_trades
            ORDER BY composite_score DESC
            LIMIT 100
        """
        
        rows = await self.db.fetch(query)
        trades = []
        for row in rows:
            trades.append({
                "strategy": row["strategy_name"],
                "symbol": row["symbol"],
                "current_price": float(row["current_price"]),
                "target_price": float(row["target_price"]),
                "exit_price": float(row["exit_price"]),
                "stop_loss": float(row["stop_loss_price"]),
                "returns": {
                    "2W": float(row["return_2w"]),
                    "1M": float(row["return_1m"]),
                    "3M": float(row["return_3m"])
                },
                "trade_type": row["trade_type"],
                "position_size": float(row["position_size"]),
                "execution": row["execution_instructions"],
                "signal_strength": float(row["signal_strength"]),
                "scenarios": json.loads(row["scenario_returns"]) if row["scenario_returns"] else {},
                "scenario_prices": json.loads(row["scenario_prices"]) if row["scenario_prices"] else {},
                "metrics": {
                    "expected_return": float(row["expected_return"]),
                    "volatility": float(row["volatility"]),
                    "sharpe_ratio": float(row["sharpe_ratio"]),
                    "max_drawdown": float(row["max_drawdown"]),
                    "win_probability": float(row["win_probability"])
                },
                "score": float(row["composite_score"]),
                "score_breakdown": json.loads(row["score_components"]) if row["score_components"] else {},
                "rank": row["rank"],
                "last_signal": row["last_signal_date"].isoformat() if row["last_signal_date"] else None
            })
        
        return trades
    
    async def get_strategies_list(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of all strategies with their latest performance metrics."""
        # First check if we have data in the strategies schema (from new pipeline)
        strategies_check = await self.db.fetchval("""
            SELECT COUNT(*) FROM strategies.strategy_metadata
        """)
        
        if strategies_check > 0:
            # Use strategies.strategy_metadata table (simplified)
            if category and category != "":
                query = """
                    SELECT 
                        strategy_name,
                        category,
                        description,
                        comprehensive_metrics->>'total_return' as total_return,
                        comprehensive_metrics->>'sharpe_ratio' as sharpe_ratio,
                        comprehensive_metrics->>'max_drawdown' as max_drawdown,
                        comprehensive_metrics->>'volatility' as volatility,
                        comprehensive_metrics->>'win_rate' as win_rate,
                        comprehensive_metrics->>'total_trades' as total_trades,
                        current_assessment->>'current_signal' as current_signal,
                        overall_rank,
                        ranking_score as composite_score,
                        created_at as last_run
                    FROM strategies.strategy_metadata
                    WHERE category = $1
                    ORDER BY overall_rank
                """
                rows = await self.db.fetch(query, category)
            else:
                query = """
                    SELECT 
                        strategy_name,
                        category,
                        description,
                        comprehensive_metrics->>'total_return' as total_return,
                        comprehensive_metrics->>'sharpe_ratio' as sharpe_ratio,
                        comprehensive_metrics->>'max_drawdown' as max_drawdown,
                        comprehensive_metrics->>'volatility' as volatility,
                        comprehensive_metrics->>'win_rate' as win_rate,
                        comprehensive_metrics->>'total_trades' as total_trades,
                        current_assessment->>'current_signal' as current_signal,
                        overall_rank,
                        ranking_score as composite_score,
                        created_at as last_run
                    FROM strategies.strategy_metadata
                    ORDER BY overall_rank
                """
                rows = await self.db.fetch(query)
            
            strategies = []
            for row in rows:
                strategies.append({
                    "strategy_name": row["strategy_name"],
                    "category": row["category"],
                    "description": row["description"],
                    "total_return": float(row["total_return"]) if row["total_return"] else 0.0,
                    "annualized_return": float(row["total_return"]) if row["total_return"] else 0.0,
                    "sharpe_ratio": float(row["sharpe_ratio"]) if row["sharpe_ratio"] else 0.0,
                    "max_drawdown": float(row["max_drawdown"]) if row["max_drawdown"] else 0.0,
                    "volatility": float(row["volatility"]) if row["volatility"] else 0.0,
                    "win_rate": float(row["win_rate"]) if row["win_rate"] else 0.0,
                    "total_trades": int(row["total_trades"]) if row["total_trades"] else 0,
                    "current_signal": row["current_signal"],
                    "overall_rank": row["overall_rank"],
                    "composite_score": float(row["composite_score"]) if row["composite_score"] else 0.0,
                    "last_run": row["last_run"].isoformat() if row["last_run"] else None
                })
            
            return strategies
        
        elif False:  # Old assessment schema code (disabled)
            if category and category != "":
                query = f"""
                    WITH latest_run AS (
                        SELECT MAX(run_date) as max_date FROM assessment.backtest_results
                    ),
                    aggregated_results AS (
                        SELECT 
                            br.strategy_name,
                            br.strategy_category as category,
                            AVG(br.total_return) as total_return,
                            AVG(br.annual_return) as annualized_return,
                            AVG(br.sharpe_ratio) as sharpe_ratio,
                            AVG(br.max_drawdown) as max_drawdown,
                            AVG(br.volatility) as volatility,
                            AVG(br.win_rate) as win_rate,
                            AVG(br.total_trades) as total_trades,
                            AVG(br.score) as composite_score,
                            MAX(br.run_date) as last_run
                        FROM assessment.backtest_results br
                        WHERE br.run_date = (SELECT max_date FROM latest_run)
                        AND br.strategy_category = '{category}'
                        GROUP BY br.strategy_name, br.strategy_category
                    ),
                    ranked AS (
                        SELECT 
                            *,
                            ROW_NUMBER() OVER (ORDER BY composite_score DESC NULLS LAST) as overall_rank
                        FROM aggregated_results
                    )
                    SELECT * FROM ranked ORDER BY overall_rank
                """
            else:
                query = """
                    WITH latest_run AS (
                        SELECT MAX(run_date) as max_date FROM assessment.backtest_results
                    ),
                    aggregated_results AS (
                        SELECT 
                            br.strategy_name,
                            br.strategy_category as category,
                            AVG(br.total_return) as total_return,
                            AVG(br.annual_return) as annualized_return,
                            AVG(br.sharpe_ratio) as sharpe_ratio,
                            AVG(br.max_drawdown) as max_drawdown,
                            AVG(br.volatility) as volatility,
                            AVG(br.win_rate) as win_rate,
                            AVG(br.total_trades) as total_trades,
                            AVG(br.score) as composite_score,
                            MAX(br.run_date) as last_run
                        FROM assessment.backtest_results br
                        WHERE br.run_date = (SELECT max_date FROM latest_run)
                        GROUP BY br.strategy_name, br.strategy_category
                    ),
                    ranked AS (
                        SELECT 
                            *,
                            ROW_NUMBER() OVER (ORDER BY composite_score DESC NULLS LAST) as overall_rank
                        FROM aggregated_results
                    )
                    SELECT * FROM ranked ORDER BY overall_rank
                """
            rows = await self.db.fetch(query)
        else:
            # Fall back to strategies schema (old pipeline data)
            if category and category != "":
                query = f"""
                    WITH latest_results AS (
                        SELECT DISTINCT ON (strategy_name)
                            strategy_name,
                            run_date,
                            total_return,
                            annualized_return,
                            sharpe_ratio,
                            max_drawdown,
                            volatility,
                            win_rate,
                            total_trades,
                            metadata
                        FROM strategies.backtest_results
                        ORDER BY strategy_name, run_date DESC
                    ),
                    latest_rankings AS (
                        SELECT DISTINCT ON (strategy_name)
                            strategy_name,
                            overall_rank,
                            composite_score,
                            rank_by_sharpe,
                            rank_by_return,
                            rank_by_consistency,
                            category
                        FROM strategies.strategy_rankings
                        ORDER BY strategy_name, run_date DESC
                    )
                    SELECT 
                        lr.strategy_name,
                        COALESCE(lrank.category, 'unknown') as category,
                        NULL as description,
                        lr.total_return,
                        lr.annualized_return,
                        lr.sharpe_ratio,
                        lr.max_drawdown,
                        lr.volatility,
                        lr.win_rate,
                        lr.total_trades,
                        lr.run_date as last_run,
                        lr.metadata,
                        lrank.overall_rank,
                        lrank.composite_score
                    FROM latest_results lr
                    LEFT JOIN latest_rankings lrank ON lr.strategy_name = lrank.strategy_name
                    WHERE lrank.category = '{category}'
                    ORDER BY lrank.overall_rank ASC NULLS LAST
                """
                rows = await self.db.fetch(query)
            else:
                query = """
                    WITH latest_results AS (
                        SELECT DISTINCT ON (strategy_name)
                            strategy_name,
                            run_date,
                            total_return,
                            annualized_return,
                            sharpe_ratio,
                            max_drawdown,
                            volatility,
                            win_rate,
                            total_trades,
                            metadata
                        FROM strategies.backtest_results
                        ORDER BY strategy_name, run_date DESC
                    ),
                    latest_rankings AS (
                        SELECT DISTINCT ON (strategy_name)
                            strategy_name,
                            overall_rank,
                            composite_score,
                            rank_by_sharpe,
                            rank_by_return,
                            rank_by_consistency,
                            category
                        FROM strategies.strategy_rankings
                        ORDER BY strategy_name, run_date DESC
                    )
                    SELECT 
                        lr.strategy_name,
                        COALESCE(lrank.category, 'unknown') as category,
                        NULL as description,
                        lr.total_return,
                        lr.annualized_return,
                        lr.sharpe_ratio,
                        lr.max_drawdown,
                        lr.volatility,
                        lr.win_rate,
                        lr.total_trades,
                        lr.run_date as last_run,
                        lr.metadata,
                        lrank.overall_rank,
                        lrank.composite_score
                    FROM latest_results lr
                    LEFT JOIN latest_rankings lrank ON lr.strategy_name = lrank.strategy_name
                    ORDER BY lrank.overall_rank ASC NULLS LAST
                """
                rows = await self.db.fetch(query)
        
        strategies = []
        for row in rows:
            strategy = dict(row)
            # Parse metadata if it exists
            if strategy.get('metadata'):
                try:
                    metadata = json.loads(strategy['metadata']) if isinstance(strategy['metadata'], str) else strategy['metadata']
                    strategy.update({
                        'sortino_ratio': metadata.get('sortino_ratio'),
                        'calmar_ratio': metadata.get('calmar_ratio'),
                        'profit_factor': metadata.get('profit_factor'),
                        'expectancy': metadata.get('expectancy')
                    })
                except:
                    pass
            strategies.append(strategy)
        
        return strategies
    
    async def get_strategy_performance(
        self,
        strategy_name: str,
        symbol: str,
        window_size: int,
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        """Calculate strategy performance for a specific symbol and time period."""
        
        # Get stored equity curve data for this strategy
        equity_data = await self.get_equity_curve(strategy_name)
        
        if not equity_data or not equity_data.get('dates') or not equity_data.get('values'):
            # No stored equity curve, return error
            return {
                'strategy_name': strategy_name,
                'symbol': symbol,
                'window_size': window_size,
                'start_date': str(start_date),
                'end_date': str(end_date),
                'error': 'No historical performance data available for this strategy'
            }
        
        # Filter equity curve to requested date range
        dates = equity_data['dates']
        values = equity_data['values']
        
        # Convert dates to datetime objects for filtering
        filtered_dates = []
        filtered_values = []
        
        for i, date_str in enumerate(dates):
            try:
                if isinstance(date_str, str):
                    curve_date = datetime.fromisoformat(date_str.replace('Z', '+00:00')).date()
                else:
                    curve_date = date_str if isinstance(date_str, date) else datetime.fromisoformat(str(date_str)).date()
                
                if start_date <= curve_date <= end_date:
                    filtered_dates.append(date_str)
                    filtered_values.append(values[i])
            except:
                continue
        
        if not filtered_dates:
            return {
                'strategy_name': strategy_name,
                'symbol': symbol,
                'window_size': window_size,
                'start_date': str(start_date),
                'end_date': str(end_date),
                'error': 'No data available for the requested date range'
            }
        
        # Calculate metrics from actual data
        returns = []
        for i in range(1, len(filtered_values)):
            if filtered_values[i-1] != 0:
                returns.append((filtered_values[i] - filtered_values[i-1]) / filtered_values[i-1])
        
        # Calculate actual metrics
        total_return = (filtered_values[-1] - filtered_values[0]) / filtered_values[0] if filtered_values[0] != 0 else 0
        
        # Calculate Sharpe ratio (annualized)
        if returns and len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = np.sqrt(252) * mean_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        peak = filtered_values[0]
        max_drawdown = 0
        for value in filtered_values:
            if value > peak:
                peak = value
            drawdown = (value - peak) / peak if peak != 0 else 0
            if drawdown < max_drawdown:
                max_drawdown = drawdown
        
        # Calculate win rate
        winning_days = sum(1 for r in returns if r > 0)
        win_rate = winning_days / len(returns) if returns else 0
        
        results = {
            'equity_curve': {
                'dates': filtered_dates,
                'values': filtered_values
            },
            'metrics': {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            }
        }
        
        # Calculate rolling performance metrics
        equity_curve = results.get('equity_curve', {})
        if not equity_curve or not equity_curve.get('dates') or not equity_curve.get('values'):
            return {
                'strategy_name': strategy_name,
                'symbol': symbol,
                'window_size': window_size,
                'start_date': str(start_date),
                'end_date': str(end_date),
                'error': 'No data available for backtest'
            }
        
        # Use the already processed dates and values
        dates = equity_curve['dates']
        values = equity_curve['values']
        
        # Convert values to numpy array and calculate rolling returns
        values_array = np.array(values, dtype=float)
        returns = np.diff(values_array) / values_array[:-1]
        
        # Calculate rolling metrics (252-day windows)
        rolling_window = min(252, len(returns) // 4)
        
        rolling_returns = []
        rolling_sharpe = []
        rolling_drawdown = []
        
        for i in range(rolling_window, len(returns)):
            window_returns = returns[i-rolling_window:i]
            
            # Annualized return
            ann_return = (1 + np.mean(window_returns)) ** 252 - 1
            rolling_returns.append(ann_return)
            
            # Sharpe ratio
            if np.std(window_returns) > 0:
                sharpe = np.sqrt(252) * np.mean(window_returns) / np.std(window_returns)
            else:
                sharpe = 0
            rolling_sharpe.append(sharpe)
            
            # Max drawdown in window
            window_values = values_array[i-rolling_window:i+1]
            peak = np.maximum.accumulate(window_values)
            drawdown = (window_values - peak) / peak
            max_dd = np.min(drawdown)
            rolling_drawdown.append(max_dd)
        
        # Align dates with rolling metrics
        metric_dates = dates[rolling_window:]
        
        return {
            'strategy_name': strategy_name,
            'symbol': symbol,
            'window_size': window_size,
            'start_date': str(start_date),
            'end_date': str(end_date),
            'equity_curve': {
                'dates': [str(d) for d in dates],
                'values': values_array.tolist() if isinstance(values_array, np.ndarray) else list(values_array)
            },
            'rolling_metrics': {
                'dates': [str(d) for d in metric_dates],
                'returns': [float(r) for r in rolling_returns],
                'sharpe_ratios': [float(s) for s in rolling_sharpe],
                'max_drawdowns': [float(d) for d in rolling_drawdown]
            },
            'summary_metrics': results.get('metrics', {})
        }
    
    async def get_strategy_scenarios(
        self,
        strategy_name: str,
        symbol: str
    ) -> Dict[str, Any]:
        """Get strategy performance in different market scenarios."""
        
        # Define key market scenarios
        scenarios = {
            'covid_crash': {
                'name': 'COVID-19 Crash',
                'start': date(2020, 2, 1),
                'end': date(2020, 4, 30),
                'type': 'bear'
            },
            'covid_recovery': {
                'name': 'COVID Recovery',
                'start': date(2020, 4, 1),
                'end': date(2021, 1, 31),
                'type': 'bull'
            },
            'tech_bubble_burst': {
                'name': 'Tech Bubble Burst',
                'start': date(2022, 1, 1),
                'end': date(2022, 10, 31),
                'type': 'bear'
            },
            'financial_crisis': {
                'name': '2008 Financial Crisis',
                'start': date(2008, 1, 1),
                'end': date(2009, 3, 31),
                'type': 'bear'
            },
            'post_crisis_recovery': {
                'name': 'Post-Crisis Recovery',
                'start': date(2009, 3, 1),
                'end': date(2010, 12, 31),
                'type': 'bull'
            },
            'low_volatility': {
                'name': 'Low Volatility Period',
                'start': date(2017, 1, 1),
                'end': date(2017, 12, 31),
                'type': 'neutral'
            },
            'high_volatility': {
                'name': 'High Volatility Period',
                'start': date(2020, 3, 1),
                'end': date(2020, 5, 31),
                'type': 'volatile'
            }
        }
        
        # Get strategy performance for each scenario
        scenario_results = {}
        
        for scenario_key, scenario_info in scenarios.items():
            try:
                performance = await self.get_strategy_performance(
                    strategy_name=strategy_name,
                    symbol=symbol,
                    window_size=20,  # Shorter window for scenarios
                    start_date=scenario_info['start'],
                    end_date=scenario_info['end']
                )
                
                scenario_results[scenario_key] = {
                    'name': scenario_info['name'],
                    'type': scenario_info['type'],
                    'start_date': str(scenario_info['start']),
                    'end_date': str(scenario_info['end']),
                    'total_return': performance.get('summary_metrics', {}).get('total_return'),
                    'sharpe_ratio': performance.get('summary_metrics', {}).get('sharpe_ratio'),
                    'max_drawdown': performance.get('summary_metrics', {}).get('max_drawdown'),
                    'win_rate': performance.get('summary_metrics', {}).get('win_rate')
                }
            except Exception as e:
                logger.error(f"Error calculating scenario {scenario_key}: {e}")
                scenario_results[scenario_key] = {
                    'name': scenario_info['name'],
                    'type': scenario_info['type'],
                    'error': str(e)
                }
        
        # Calculate scenario statistics
        bear_returns = [r['total_return'] for r in scenario_results.values() 
                       if r.get('type') == 'bear' and r.get('total_return') is not None]
        bull_returns = [r['total_return'] for r in scenario_results.values() 
                       if r.get('type') == 'bull' and r.get('total_return') is not None]
        
        return {
            'strategy_name': strategy_name,
            'symbol': symbol,
            'scenarios': scenario_results,
            'summary': {
                'avg_bear_return': np.mean(bear_returns) if bear_returns else None,
                'avg_bull_return': np.mean(bull_returns) if bull_returns else None,
                'bear_market_resilience': len([r for r in bear_returns if r > 0]) / len(bear_returns) if bear_returns else 0,
                'consistency_score': 1 - np.std(bear_returns + bull_returns) if (bear_returns + bull_returns) else 0
            }
        }
    
    async def get_strategy_metrics(
        self,
        strategy_name: str,
        run_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get detailed metrics for a strategy from database."""
        
        # Check if we have assessment data
        assessment_check = await self.db.fetchval("""
            SELECT COUNT(*) FROM assessment.backtest_results
        """)
        
        if assessment_check > 0:
            # Use assessment schema
            if run_date:
                query = """
                    SELECT 
                        strategy_name,
                        strategy_category as category,
                        scenario_name,
                        symbol,
                        total_return,
                        annual_return as annualized_return,
                        sharpe_ratio,
                        sortino_ratio,
                        max_drawdown,
                        calmar_ratio,
                        win_rate,
                        profit_factor,
                        volatility,
                        score as composite_score,
                        total_trades,
                        run_date,
                        created_at
                    FROM assessment.backtest_results
                    WHERE strategy_name = $1
                    AND run_date = $2
                    ORDER BY score DESC
                    LIMIT 1
                """
                result = await self.db.fetchrow(query, strategy_name, run_date)
            else:
                query = """
                    SELECT 
                        strategy_name,
                        strategy_category as category,
                        scenario_name,
                        symbol,
                        total_return,
                        annual_return as annualized_return,
                        sharpe_ratio,
                        sortino_ratio,
                        max_drawdown,
                        calmar_ratio,
                        win_rate,
                        profit_factor,
                        volatility,
                        score as composite_score,
                        total_trades,
                        run_date,
                        created_at
                    FROM assessment.backtest_results
                    WHERE strategy_name = $1
                    ORDER BY run_date DESC, score DESC
                    LIMIT 1
                """
                result = await self.db.fetchrow(query, strategy_name)
            
            if result:
                return dict(result)
            return None
        else:
            query = """
                SELECT 
                    br.*,
                    sr.overall_rank,
                    sr.rank_by_sharpe,
                    sr.rank_by_return,
                    sr.rank_by_consistency,
                    sr.composite_score
                FROM strategies.backtest_results br
                LEFT JOIN strategies.strategy_rankings sr
                    ON br.strategy_name = sr.strategy_name
                    AND br.run_date = sr.run_date
                WHERE br.strategy_name = $1
                ORDER BY br.run_date DESC
                LIMIT 1
            """
            result = await self.db.fetchrow(query, strategy_name)
        
        if not result:
            return None
        
        metrics = dict(result)
        
        # Parse metadata
        if metrics.get('metadata'):
            try:
                metadata = json.loads(metrics['metadata']) if isinstance(metrics['metadata'], str) else metrics['metadata']
                metrics.update(metadata)
            except:
                pass
        
        return metrics
    
    async def get_equity_curve(
        self,
        strategy_name: str,
        backtest_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get stored equity curve data."""
        
        if backtest_id:
            query = """
                SELECT dates, values, created_at
                FROM strategies.equity_curves
                WHERE strategy_name = $1 AND backtest_id = $2
            """
            result = await self.db.fetchrow(query, strategy_name, backtest_id)
        else:
            query = """
                SELECT dates, values, created_at
                FROM strategies.equity_curves
                WHERE strategy_name = $1
                ORDER BY created_at DESC
                LIMIT 1
            """
            result = await self.db.fetchrow(query, strategy_name)
        
        if not result:
            return None
        
        return {
            'strategy_name': strategy_name,
            'dates': json.loads(result['dates']) if isinstance(result['dates'], str) else result['dates'],
            'values': json.loads(result['values']) if isinstance(result['values'], str) else result['values'],
            'created_at': result['created_at'].isoformat() if result['created_at'] else None
        }
    
    async def get_latest_rankings(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get latest strategy rankings."""
        
        query = """
            WITH latest_run AS (
                SELECT MAX(run_date) as run_date
                FROM strategies.strategy_rankings
            )
            SELECT 
                sr.*,
                sm.description,
                br.total_return,
                br.sharpe_ratio,
                br.max_drawdown,
                br.volatility,
                br.win_rate
            FROM strategies.strategy_rankings sr
            JOIN latest_run lr ON sr.run_date = lr.run_date
            LEFT JOIN strategies.strategy_metadata sm ON sr.strategy_name = sm.strategy_name
            LEFT JOIN strategies.backtest_results br 
                ON sr.strategy_name = br.strategy_name 
                AND sr.run_date = br.run_date
            WHERE ($1::text IS NULL OR sr.category = $1)
            ORDER BY sr.overall_rank ASC
        """
        
        rows = await self.db.fetch(query, category)
        return [dict(row) for row in rows]
    
    async def get_latest_summary(self) -> Dict[str, Any]:
        """Get latest backtest run summary."""
        
        query = """
            SELECT *
            FROM strategies.backtest_summary
            ORDER BY run_date DESC
            LIMIT 1
        """
        
        result = await self.db.fetchrow(query)
        
        if not result:
            return None
        
        summary = dict(result)
        
        # Parse metadata
        if summary.get('metadata'):
            try:
                metadata = json.loads(summary['metadata']) if isinstance(summary['metadata'], str) else summary['metadata']
                summary['metadata'] = metadata
            except:
                pass
        
        return summary