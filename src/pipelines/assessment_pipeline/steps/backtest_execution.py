"""
Optimized backtest execution with vectorized computation for short windows.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ProcessPoolExecutor

from src.pipelines.base import PipelineStep, PipelineContext

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    """Results for a specific time window."""
    window: str  # "2W", "1M", "3M"
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    volatility: float
    total_trades: int


@dataclass
class BacktestResult:
    """Container for backtest results."""
    strategy_name: str
    symbol: str
    scenario_name: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    equity_curve: List[float]
    trade_log: List[Dict]
    metadata: Dict[str, Any]
    error: Optional[str] = None


def run_single_backtest_worker(combination: Dict) -> BacktestResult:
    """
    Standalone function for multiprocessing execution.
    This function must be at module level to be picklable.
    """
    from src.pipelines.assessment_pipeline.steps.backtest_execution import BacktestExecutionStep
    backtest_step = BacktestExecutionStep()
    return backtest_step._run_single_backtest_sync(combination)


class BacktestExecutionStep(PipelineStep):
    """Execute backtests for all combinations of strategies, symbols, and scenarios."""
    
    def __init__(self, parallel_workers: int = 12):
        super().__init__("BacktestExecution")
        self.parallel_workers = parallel_workers
        self.backtest_results = []
    
    async def execute(self, context: PipelineContext) -> None:
        """Execute all backtests."""
        logger.info("Starting backtest execution")
        
        # Get data from context
        strategies = context.data.get('strategies', [])
        symbols = context.data.get('symbols', [])
        scenarios = context.data.get('scenarios', [])
        metrics_data = context.data.get('metrics_data', {})
        
        if not strategies:
            raise ValueError("No strategies found in context")
        if not symbols:
            raise ValueError("No symbols found in context")
        if not scenarios:
            raise ValueError("No scenarios found in context")
        
        # Create all combinations
        combinations = []
        for strategy in strategies:
            for symbol in symbols:
                for scenario in scenarios:
                    combinations.append({
                        'strategy': strategy,
                        'symbol': symbol,
                        'scenario': scenario,
                        'metrics': metrics_data.get(scenario['name'])
                    })
        
        logger.info(f"Running {len(combinations)} backtests")
        
        # Execute backtests (parallel for CPU-bound work)
        results = await self._run_backtests_parallel(combinations)
        
        # Separate successful and failed results
        successful_results = [r for r in results if r.error is None]
        failed_results = [r for r in results if r.error is not None]
        
        # Store results in context
        context.data['backtest_results'] = successful_results
        context.data['failed_backtests'] = failed_results
        
        # Calculate summary statistics
        summary = {
            'total_backtests': len(results),
            'successful_backtests': len(successful_results),
            'failed_backtests': len(failed_results),
            'success_rate': len(successful_results) / len(results) if results else 0
        }
        context.data['backtest_summary'] = summary
        
        logger.info(
            f"Completed {len(successful_results)} successful backtests, "
            f"{len(failed_results)} failed"
        )
    
    async def _run_backtests_parallel(self, combinations: List[Dict]) -> List[BacktestResult]:
        """Run backtests in parallel using ProcessPoolExecutor."""
        results = []
        
        # Run backtests in batches with reasonable size for system stability
        # Use smaller batch size to avoid overwhelming the system
        import os
        batch_size = min(200, os.cpu_count() * 10) if os.cpu_count() else 100
        for i in range(0, len(combinations), batch_size):
            batch = combinations[i:i + batch_size]
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(combinations) + batch_size - 1)//batch_size}")
        
        return results
    
    async def _process_batch(self, batch: List[Dict]) -> List[BacktestResult]:
        """Process a batch of backtests using multiprocessing for true parallelization."""
        loop = asyncio.get_event_loop()
        
        # Use ProcessPoolExecutor for CPU-bound parallel processing
        # Use 12 workers for balanced performance and stability
        import os
        num_workers = min(12, os.cpu_count() or 4, len(batch))  # Use up to 12 cores
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all backtests to the process pool
            futures = []
            for combination in batch:
                future = loop.run_in_executor(
                    executor,
                    run_single_backtest_worker,  # Use a standalone function for pickling
                    combination
                )
                futures.append(future)
            
            # Wait for all futures to complete
            results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                combo = batch[i]
                error_result = BacktestResult(
                    strategy_name=combo['strategy']['name'],
                    symbol=combo['symbol'],
                    scenario_name=combo['scenario']['name'],
                    total_return=0.0,
                    annualized_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    win_rate=0.0,
                    profit_factor=0.0,
                    total_trades=0,
                    equity_curve=[],
                    trade_log=[],
                    metadata={},
                    error=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _run_single_backtest_sync(self, combination: Dict) -> BacktestResult:
        """Synchronous version of _run_single_backtest for multiprocessing."""
        strategy_data = combination['strategy']
        symbol = combination['symbol']
        scenario = combination['scenario']
        metrics_df = combination['metrics']
        
        try:
            # Check if metrics exist for this scenario
            if metrics_df is None or metrics_df.empty:
                raise ValueError(f"No metrics available for scenario {scenario['name']}")
            
            # Get symbol-specific metrics
            symbol_metrics = metrics_df[metrics_df['symbol'] == symbol] if 'symbol' in metrics_df.columns else metrics_df
            
            if symbol_metrics.empty:
                raise ValueError(f"No metrics for symbol {symbol} in scenario {scenario['name']}")
            
            # Load strategy instance
            strategy_instance = strategy_data.get('instance')
            if not strategy_instance:
                # Create strategy instance if not provided
                from src.strategies.factory import StrategyFactory
                from src.strategies.implementations import buy_and_hold_strategy
                
                strategy_name = strategy_data['name']
                factory_name = strategy_name.replace('_strategy', '')
                if factory_name == 'buy_and_hold':
                    strategy_instance = buy_and_hold_strategy()
                else:
                    try:
                        factory = StrategyFactory()
                        strategy_instance = factory.create_strategy(strategy_name)
                    except:
                        strategy_instance = buy_and_hold_strategy()
            
            # Run the optimized backtest
            return self._execute_vectorized_backtest(
                strategy_data, strategy_instance, symbol, scenario, symbol_metrics
            )
            
        except Exception as e:
            logger.error(f"Backtest failed for {strategy_data['name']}-{symbol}-{scenario['name']}: {e}")
            return BacktestResult(
                strategy_name=strategy_data['name'],
                symbol=symbol,
                scenario_name=scenario['name'],
                total_return=0.0,
                annualized_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                equity_curve=[],
                trade_log=[],
                metadata={},
                error=str(e)
            )
    
    def _execute_vectorized_backtest(
        self, 
        strategy_data: Dict, 
        strategy_instance: Any,
        symbol: str, 
        scenario: Dict, 
        symbol_metrics: pd.DataFrame
    ) -> BacktestResult:
        """Execute vectorized backtest - call strategy ONCE and simulate trades."""
        try:
            initial_capital = 100000
            
            # Generate signals ONCE for all data
            signals_df = None
            if hasattr(strategy_instance, 'generate_signals'):
                signals_df = strategy_instance.generate_signals(symbol_metrics, symbol_metrics, None)
            
            # If no signals generated, use buy-and-hold
            if signals_df is None or signals_df.empty or 'signal' not in signals_df.columns:
                # Simple buy-and-hold
                signals = pd.Series([1] * len(symbol_metrics), index=symbol_metrics.index)
            else:
                signals = signals_df['signal']
            
            # Vectorized backtest calculation
            prices = symbol_metrics['close'].values if 'close' in symbol_metrics.columns else np.ones(len(symbol_metrics))
            returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else np.array([0])
            
            # Apply strategy signals (shift by 1 to avoid look-ahead bias)
            if len(signals) > 1:
                positions = signals.shift(1).fillna(0).values[1:]
            else:
                positions = np.array([0])
            
            # Ensure arrays are same length
            min_len = min(len(positions), len(returns))
            positions = positions[:min_len]
            returns = returns[:min_len]
            
            # Calculate strategy returns
            strategy_returns = positions * returns
            
            # Calculate cumulative equity curve
            equity_curve = initial_capital * np.cumprod(1 + strategy_returns)
            equity_curve = np.concatenate([[initial_capital], equity_curve])
            
            # Track trades
            trades = []
            position_changes = np.diff(np.concatenate([[0], positions]))
            trade_indices = np.where(position_changes != 0)[0]
            
            for idx in trade_indices:
                if idx < len(prices):
                    trade_type = 'BUY' if position_changes[idx] > 0 else 'SELL'
                    trades.append({
                        'type': trade_type,
                        'price': prices[idx],
                        'date': idx,
                        'position_change': position_changes[idx]
                    })
            
            # Calculate metrics
            final_value = equity_curve[-1] if len(equity_curve) > 0 else initial_capital
            total_return = (final_value - initial_capital) / initial_capital
            
            # Annualized return
            days = len(symbol_metrics)
            annualized_return = ((1 + total_return) ** (252 / days)) - 1 if days > 0 else 0
            
            # Volatility
            volatility = np.std(strategy_returns) * np.sqrt(252) if len(strategy_returns) > 0 else 0
            
            # Sharpe ratio
            risk_free_rate = 0.02
            excess_return = annualized_return - risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            
            # Max drawdown
            if len(equity_curve) > 0:
                running_max = np.maximum.accumulate(equity_curve)
                drawdown = (equity_curve - running_max) / running_max
                max_drawdown = np.min(drawdown)
            else:
                max_drawdown = 0
            
            # Win rate
            winning_returns = np.sum(strategy_returns > 0)
            total_trading_days = np.sum(positions != 0)
            win_rate = winning_returns / total_trading_days if total_trading_days > 0 else 0
            
            # Profit factor
            gross_profit = np.sum(strategy_returns[strategy_returns > 0])
            gross_loss = abs(np.sum(strategy_returns[strategy_returns < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            return BacktestResult(
                strategy_name=strategy_data['name'],
                symbol=symbol,
                scenario_name=scenario['name'],
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=len(trades),
                equity_curve=equity_curve.tolist(),
                trade_log=trades,
                metadata={
                    'scenario_type': scenario.get('type'),
                    'start_date': str(scenario.get('start_date')),
                    'end_date': str(scenario.get('end_date')),
                    'initial_capital': initial_capital,
                    'final_value': final_value
                },
                error=None
            )
            
        except Exception as e:
            logger.error(f"Vectorized backtest execution failed: {e}")
            raise