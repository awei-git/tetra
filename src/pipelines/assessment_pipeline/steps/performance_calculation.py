"""Step 4: Calculate comprehensive performance metrics for all backtests."""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

from src.pipelines.base import PipelineStep, PipelineContext

logger = logging.getLogger(__name__)


class PerformanceCalculationStep(PipelineStep):
    """Calculate comprehensive performance metrics for strategy assessment."""
    
    def __init__(self):
        super().__init__("PerformanceCalculation")
    
    async def execute(self, context: PipelineContext) -> None:
        """Calculate comprehensive performance metrics."""
        logger.info("Calculating comprehensive performance metrics")
        
        # Get backtest results from context
        backtest_results = context.data.get('backtest_results', [])
        if not backtest_results:
            logger.warning("No backtest results to process")
            return
        
        # Group results by strategy for aggregation
        results_by_strategy = self._group_by_strategy(backtest_results)
        
        # Calculate comprehensive metrics for each strategy
        comprehensive_metrics = {}
        for strategy_name, results in results_by_strategy.items():
            metrics = await self._calculate_strategy_metrics(strategy_name, results)
            comprehensive_metrics[strategy_name] = metrics
        
        # Calculate regime-specific performance
        regime_metrics = await self._calculate_regime_performance(backtest_results)
        
        # Add current assessment (real-time signals)
        current_assessments = await self._calculate_current_assessments(
            comprehensive_metrics, 
            context
        )
        
        # Add projections
        projections = await self._calculate_projections(comprehensive_metrics)
        
        # Combine all metrics
        for strategy_name in comprehensive_metrics:
            comprehensive_metrics[strategy_name].update({
                'regime_performance': regime_metrics.get(strategy_name, {}),
                'current_assessment': current_assessments.get(strategy_name, {}),
                'projections': projections.get(strategy_name, {})
            })
        
        # Store in context
        context.data['comprehensive_metrics'] = comprehensive_metrics
        
        logger.info(f"Calculated comprehensive metrics for {len(comprehensive_metrics)} strategies")
    
    def _group_by_strategy(self, results: List) -> Dict[str, List]:
        """Group backtest results by strategy name."""
        grouped = defaultdict(list)
        for result in results:
            grouped[result.strategy_name].append(result)
        return dict(grouped)
    
    async def _calculate_strategy_metrics(self, strategy_name: str, results: List) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a single strategy."""
        metrics = {}
        
        # Core Performance Metrics
        total_returns = [r.total_return for r in results if r.total_return != 0]
        annualized_returns = [r.annualized_return for r in results if r.annualized_return != 0]
        volatilities = [r.volatility for r in results if r.volatility > 0]
        sharpe_ratios = [r.sharpe_ratio for r in results if r.sharpe_ratio != 0]
        max_drawdowns = [r.max_drawdown for r in results]
        
        metrics['total_return'] = np.mean(total_returns) if total_returns else 0
        metrics['annualized_return'] = np.mean(annualized_returns) if annualized_returns else 0
        metrics['volatility'] = np.mean(volatilities) if volatilities else 0
        metrics['downside_deviation'] = self._calculate_downside_deviation(total_returns)
        metrics['sharpe_ratio'] = np.mean(sharpe_ratios) if sharpe_ratios else 0
        metrics['sortino_ratio'] = self._calculate_sortino_ratio(
            metrics['annualized_return'], 
            metrics['downside_deviation']
        )
        metrics['max_drawdown'] = min(max_drawdowns) if max_drawdowns else 0
        metrics['avg_drawdown'] = np.mean(max_drawdowns) if max_drawdowns else 0
        metrics['calmar_ratio'] = (
            metrics['annualized_return'] / abs(metrics['max_drawdown'])
            if metrics['max_drawdown'] != 0 else 0
        )
        
        # Trade Quality Metrics
        win_rates = [r.win_rate for r in results if r.total_trades > 0]
        profit_factors = [r.profit_factor for r in results if r.profit_factor > 0]
        total_trades = [r.total_trades for r in results]
        
        metrics['win_rate'] = np.mean(win_rates) if win_rates else 0
        metrics['profit_factor'] = np.mean(profit_factors) if profit_factors else 0
        metrics['total_trades'] = sum(total_trades)
        metrics['avg_trades_per_scenario'] = np.mean(total_trades) if total_trades else 0
        
        # Calculate additional trade metrics
        metrics['avg_win'] = self._calculate_avg_win(results)
        metrics['avg_loss'] = self._calculate_avg_loss(results)
        metrics['payoff_ratio'] = (
            abs(metrics['avg_win'] / metrics['avg_loss'])
            if metrics['avg_loss'] != 0 else 0
        )
        metrics['expectancy'] = (
            metrics['win_rate'] * metrics['avg_win'] +
            (1 - metrics['win_rate']) * metrics['avg_loss']
        )
        metrics['sqn'] = self._calculate_sqn(metrics['expectancy'], total_returns)
        metrics['edge_ratio'] = metrics['expectancy'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
        metrics['kelly_fraction'] = self._calculate_kelly_fraction(
            metrics['win_rate'], 
            metrics['payoff_ratio']
        )
        
        # Risk-Adjusted Performance
        metrics['var_95'] = np.percentile(total_returns, 5) if total_returns else 0
        metrics['var_99'] = np.percentile(total_returns, 1) if total_returns else 0
        metrics['cvar_95'] = self._calculate_cvar(total_returns, 0.05)
        metrics['cvar_99'] = self._calculate_cvar(total_returns, 0.01)
        metrics['omega_ratio'] = self._calculate_omega_ratio(total_returns)
        metrics['ulcer_index'] = self._calculate_ulcer_index(results)
        metrics['tail_ratio'] = self._calculate_tail_ratio(total_returns)
        
        # Timing & Efficiency Metrics
        metrics['time_in_market'] = self._calculate_time_in_market(results)
        metrics['trade_frequency'] = metrics['avg_trades_per_scenario'] * 252 / 365  # Annualized
        metrics['recovery_factor'] = (
            metrics['total_return'] / abs(metrics['max_drawdown'])
            if metrics['max_drawdown'] != 0 else 0
        )
        metrics['mar_ratio'] = metrics['calmar_ratio']  # Managed Account Ratio
        
        # Robustness & Stability
        metrics['return_stability'] = self._calculate_return_stability(results)
        metrics['consistency_score'] = self._calculate_consistency_score(results)
        
        # Ranking Score (composite)
        metrics['ranking_score'] = self._calculate_ranking_score(metrics)
        
        return metrics
    
    def _calculate_downside_deviation(self, returns: List[float]) -> float:
        """Calculate downside deviation (volatility of negative returns)."""
        if not returns:
            return 0
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return 0
        return np.std(negative_returns) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, annual_return: float, downside_dev: float) -> float:
        """Calculate Sortino ratio."""
        risk_free_rate = 0.02
        if downside_dev == 0:
            return 0
        return (annual_return - risk_free_rate) / downside_dev
    
    def _calculate_avg_win(self, results: List) -> float:
        """Calculate average winning trade return."""
        all_wins = []
        for result in results:
            if hasattr(result, 'trade_log'):
                for trade in result.trade_log:
                    if trade.get('pnl', 0) > 0:
                        all_wins.append(trade['pnl'])
        return np.mean(all_wins) if all_wins else 0
    
    def _calculate_avg_loss(self, results: List) -> float:
        """Calculate average losing trade return."""
        all_losses = []
        for result in results:
            if hasattr(result, 'trade_log'):
                for trade in result.trade_log:
                    if trade.get('pnl', 0) < 0:
                        all_losses.append(trade['pnl'])
        return np.mean(all_losses) if all_losses else 0
    
    def _calculate_sqn(self, expectancy: float, returns: List[float]) -> float:
        """Calculate System Quality Number (Van Tharp)."""
        if not returns or expectancy == 0:
            return 0
        std_dev = np.std(returns)
        if std_dev == 0:
            return 0
        return (expectancy / std_dev) * np.sqrt(len(returns))
    
    def _calculate_kelly_fraction(self, win_rate: float, payoff_ratio: float) -> float:
        """Calculate Kelly Criterion for optimal position sizing."""
        if payoff_ratio == 0:
            return 0
        kelly = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio
        return max(0, min(kelly, 0.25))  # Cap at 25% for safety
    
    def _calculate_cvar(self, returns: List[float], alpha: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if not returns:
            return 0
        cutoff = np.percentile(returns, alpha * 100)
        tail_returns = [r for r in returns if r <= cutoff]
        return np.mean(tail_returns) if tail_returns else 0
    
    def _calculate_omega_ratio(self, returns: List[float], threshold: float = 0) -> float:
        """Calculate Omega ratio."""
        if not returns:
            return 0
        gains = sum(max(0, r - threshold) for r in returns)
        losses = sum(max(0, threshold - r) for r in returns)
        return gains / losses if losses > 0 else 0
    
    def _calculate_ulcer_index(self, results: List) -> float:
        """Calculate Ulcer Index (measure of downside volatility)."""
        all_drawdowns = []
        for result in results:
            if hasattr(result, 'equity_curve') and result.equity_curve:
                equity = pd.Series(result.equity_curve)
                running_max = equity.expanding().max()
                drawdown = ((equity - running_max) / running_max) * 100
                all_drawdowns.extend(drawdown.tolist())
        
        if not all_drawdowns:
            return 0
        return np.sqrt(np.mean([d**2 for d in all_drawdowns]))
    
    def _calculate_tail_ratio(self, returns: List[float]) -> float:
        """Calculate tail ratio (right tail / left tail)."""
        if not returns:
            return 0
        right_tail = np.percentile(returns, 95)
        left_tail = abs(np.percentile(returns, 5))
        return right_tail / left_tail if left_tail > 0 else 0
    
    def _calculate_time_in_market(self, results: List) -> float:
        """Calculate percentage of time with open positions."""
        total_days = 0
        days_with_positions = 0
        
        for result in results:
            if hasattr(result, 'metadata'):
                scenario_days = result.metadata.get('trading_days', 252)
                total_days += scenario_days
                if result.total_trades > 0:
                    # Estimate based on trades
                    days_with_positions += scenario_days * 0.7  # Assume 70% time in market
        
        return days_with_positions / total_days if total_days > 0 else 0
    
    def _calculate_return_stability(self, results: List) -> float:
        """Calculate stability of returns across scenarios."""
        returns = [r.total_return for r in results]
        if len(returns) < 2:
            return 0
        return 1 - (np.std(returns) / (abs(np.mean(returns)) + 1e-10))
    
    def _calculate_consistency_score(self, results: List) -> float:
        """Calculate percentage of profitable scenarios."""
        if not results:
            return 0
        profitable = sum(1 for r in results if r.total_return > 0)
        return profitable / len(results)
    
    def _calculate_ranking_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite ranking score."""
        score = (
            metrics.get('sharpe_ratio', 0) * 30 +
            metrics.get('total_return', 0) * 100 +
            (1 / (1 + abs(metrics.get('max_drawdown', 0)))) * 20 +
            metrics.get('win_rate', 0) * 20 +
            min(metrics.get('profit_factor', 0), 3) * 10 +
            metrics.get('sqn', 0) * 5
        )
        return score
    
    async def _calculate_regime_performance(self, results: List) -> Dict[str, Dict]:
        """Calculate performance metrics by market regime."""
        regime_metrics = defaultdict(lambda: defaultdict(list))
        
        # Group results by regime type
        for result in results:
            if hasattr(result, 'metadata'):
                regime_type = result.metadata.get('scenario_type', 'unknown')
                strategy_name = result.strategy_name
                
                regime_metrics[strategy_name][regime_type].append({
                    'return': result.total_return,
                    'sharpe': result.sharpe_ratio,
                    'drawdown': result.max_drawdown,
                    'win_rate': result.win_rate
                })
        
        # Calculate aggregated regime metrics
        aggregated = {}
        for strategy_name, regimes in regime_metrics.items():
            aggregated[strategy_name] = {}
            for regime_type, metrics_list in regimes.items():
                if metrics_list:
                    aggregated[strategy_name][f'{regime_type}_return'] = np.mean(
                        [m['return'] for m in metrics_list]
                    )
                    aggregated[strategy_name][f'{regime_type}_sharpe'] = np.mean(
                        [m['sharpe'] for m in metrics_list]
                    )
        
        return aggregated
    
    async def _calculate_current_assessments(
        self, 
        comprehensive_metrics: Dict, 
        context: PipelineContext
    ) -> Dict[str, Dict]:
        """Calculate current real-time assessments for each strategy."""
        current_assessments = {}
        
        # Get latest market data if available
        symbols = context.data.get('symbols', [])
        
        for strategy_name in comprehensive_metrics:
            # For now, return placeholder structure
            # In production, this would calculate real signals from latest data
            current_assessments[strategy_name] = {
                'current_signal': 'HOLD',
                'position_size': 0,
                'risk_metrics': {
                    'risk_per_trade': 0.02,
                    'position_risk': 0
                }
            }
        
        return current_assessments
    
    async def _calculate_projections(self, comprehensive_metrics: Dict) -> Dict[str, Dict]:
        """Calculate return projections for different time horizons."""
        projections = {}
        
        for strategy_name, metrics in comprehensive_metrics.items():
            annual_return = metrics.get('annualized_return', 0)
            
            # Simple projection based on historical performance
            projections[strategy_name] = {
                '1w': annual_return / 52,
                '2w': annual_return / 26,
                '1m': annual_return / 12,
                '3m': annual_return / 4,
                '6m': annual_return / 2,
                '1y': annual_return
            }
        
        return projections