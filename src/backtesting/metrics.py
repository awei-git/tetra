"""Performance metrics calculation for backtesting."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class PerformanceReport:
    """Complete performance report for a backtest."""
    # Returns
    total_return: float
    annualized_return: float
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Portfolio metrics
    final_equity: float
    peak_equity: float
    lowest_equity: float
    
    # Time metrics
    time_in_market: float
    longest_winning_streak: int
    longest_losing_streak: int
    
    # Risk-adjusted metrics
    calmar_ratio: float
    omega_ratio: float
    
    # Additional data
    equity_curve: pd.Series
    returns: pd.Series
    trades: List[Any] = field(default_factory=list)
    positions: List[Any] = field(default_factory=list)
    
    # Optional fields
    benchmark_returns: Optional[pd.Series] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None
    correlation: Optional[float] = None
    information_ratio: Optional[float] = None
    
    # Strategy-specific metrics
    strategy_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Configuration
    config: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'returns': {
                'total_return': self.total_return,
                'annualized_return': self.annualized_return,
            },
            'risk': {
                'volatility': self.volatility,
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'max_drawdown': self.max_drawdown,
                'max_drawdown_duration': self.max_drawdown_duration,
            },
            'trades': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': self.win_rate,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'profit_factor': self.profit_factor,
            },
            'portfolio': {
                'final_equity': self.final_equity,
                'peak_equity': self.peak_equity,
                'lowest_equity': self.lowest_equity,
            },
            'time': {
                'time_in_market': self.time_in_market,
                'longest_winning_streak': self.longest_winning_streak,
                'longest_losing_streak': self.longest_losing_streak,
            },
            'risk_adjusted': {
                'calmar_ratio': self.calmar_ratio,
                'omega_ratio': self.omega_ratio,
            },
            'benchmark': {
                'beta': self.beta,
                'alpha': self.alpha,
                'correlation': self.correlation,
                'information_ratio': self.information_ratio,
            } if self.benchmark_returns is not None else None,
            'strategy_metrics': self.strategy_metrics,
        }


class MetricsCalculator:
    """Calculate performance metrics for backtesting."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.returns: List[float] = []
        self.equity_curve: List[float] = []
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        
    def update(self, returns: pd.Series):
        """Update metrics with new returns data."""
        self.returns.extend(returns.tolist())
        
        # Update max drawdown calculation
        if len(self.returns) > 0:
            cumulative_returns = (1 + pd.Series(self.returns)).cumprod()
            self.equity_curve = cumulative_returns.tolist()
            
            # Calculate running max drawdown
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            self.max_drawdown = abs(drawdown.min())
            self.peak_equity = running_max.max()
    
    def generate_report(self,
                       equity_curve: pd.Series,
                       trades: List[Any],
                       positions: List[Any],
                       benchmark: Optional[pd.DataFrame] = None,
                       initial_capital: float = 100000) -> PerformanceReport:
        """Generate complete performance report.
        
        Args:
            equity_curve: Equity curve series
            trades: List of completed trades
            positions: List of position snapshots
            benchmark: Optional benchmark data
            initial_capital: Initial capital amount
            
        Returns:
            Complete performance report
        """
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Basic return metrics
        total_return = (equity_curve.iloc[-1] / initial_capital) - 1
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Sharpe ratio
        excess_returns = returns - self.risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0
        
        # Drawdown metrics
        drawdown_info = self._calculate_drawdown_metrics(equity_curve)
        
        # Trade statistics
        trade_stats = self._calculate_trade_statistics(trades)
        
        # Time in market
        time_in_market = self._calculate_time_in_market(positions, equity_curve)
        
        # Streak analysis
        winning_streak, losing_streak = self._calculate_streaks(trades)
        
        # Calmar ratio
        calmar_ratio = annualized_return / drawdown_info['max_drawdown'] if drawdown_info['max_drawdown'] > 0 else 0
        
        # Omega ratio
        omega_ratio = self._calculate_omega_ratio(returns, self.risk_free_rate / 252)
        
        # Benchmark metrics if provided
        benchmark_metrics = {}
        if benchmark is not None:
            benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark)
        
        return PerformanceReport(
            # Returns
            total_return=total_return,
            annualized_return=annualized_return,
            
            # Risk metrics
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=drawdown_info['max_drawdown'],
            max_drawdown_duration=drawdown_info['max_duration'],
            
            # Trade statistics
            total_trades=trade_stats['total_trades'],
            winning_trades=trade_stats['winning_trades'],
            losing_trades=trade_stats['losing_trades'],
            win_rate=trade_stats['win_rate'],
            avg_win=trade_stats['avg_win'],
            avg_loss=trade_stats['avg_loss'],
            profit_factor=trade_stats['profit_factor'],
            
            # Portfolio metrics
            final_equity=equity_curve.iloc[-1],
            peak_equity=equity_curve.max(),
            lowest_equity=equity_curve.min(),
            
            # Time metrics
            time_in_market=time_in_market,
            longest_winning_streak=winning_streak,
            longest_losing_streak=losing_streak,
            
            # Risk-adjusted metrics
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            
            # Additional data
            equity_curve=equity_curve,
            returns=returns,
            trades=trades,
            positions=positions,
            
            # Benchmark metrics
            **benchmark_metrics
        )
    
    def _calculate_drawdown_metrics(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """Calculate drawdown-related metrics."""
        # Calculate drawdown series
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        
        # Max drawdown
        max_drawdown = abs(drawdown.min())
        
        # Drawdown duration
        is_drawdown = drawdown < 0
        drawdown_groups = is_drawdown.ne(is_drawdown.shift()).cumsum()
        
        max_duration = 0
        if is_drawdown.any():
            drawdown_lengths = is_drawdown.groupby(drawdown_groups).sum()
            max_duration = int(drawdown_lengths[drawdown_lengths > 0].max())
        
        return {
            'max_drawdown': max_drawdown,
            'max_duration': max_duration,
            'drawdown_series': drawdown
        }
    
    def _calculate_trade_statistics(self, trades: List[Any]) -> Dict[str, Any]:
        """Calculate trade-related statistics."""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        # Calculate P&L for each trade
        pnls = []
        for trade in trades:
            if hasattr(trade, 'pnl'):
                pnl = trade.pnl
            else:
                # Calculate P&L from trade attributes
                if trade.side.value == 'long':
                    pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                else:
                    pnl = (trade.entry_price - trade.exit_price) * trade.quantity
                pnl -= trade.commission + trade.slippage
            pnls.append(pnl)
        
        pnls = np.array(pnls)
        winning_trades = pnls[pnls > 0]
        losing_trades = pnls[pnls < 0]
        
        total_trades = len(trades)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        
        win_rate = num_winning / total_trades if total_trades > 0 else 0
        avg_win = winning_trades.mean() if num_winning > 0 else 0
        avg_loss = abs(losing_trades.mean()) if num_losing > 0 else 0
        
        gross_profit = winning_trades.sum() if num_winning > 0 else 0
        gross_loss = abs(losing_trades.sum()) if num_losing > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': num_winning,
            'losing_trades': num_losing,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def _calculate_time_in_market(self, positions: List[Any], equity_curve: pd.Series) -> float:
        """Calculate percentage of time with open positions."""
        if not positions or len(equity_curve) <= 1:
            return 0.0
        
        periods_with_positions = 0
        total_periods = len(equity_curve)
        
        for position_state in positions:
            if hasattr(position_state, 'positions') and position_state.positions:
                periods_with_positions += 1
        
        return periods_with_positions / total_periods if total_periods > 0 else 0.0
    
    def _calculate_streaks(self, trades: List[Any]) -> Tuple[int, int]:
        """Calculate longest winning and losing streaks."""
        if not trades:
            return 0, 0
        
        current_winning_streak = 0
        current_losing_streak = 0
        max_winning_streak = 0
        max_losing_streak = 0
        
        for trade in trades:
            if hasattr(trade, 'pnl'):
                pnl = trade.pnl
            else:
                # Calculate P&L from trade attributes
                if trade.side.value == 'long':
                    pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                else:
                    pnl = (trade.entry_price - trade.exit_price) * trade.quantity
                pnl -= trade.commission + trade.slippage
            
            if pnl > 0:
                current_winning_streak += 1
                current_losing_streak = 0
                max_winning_streak = max(max_winning_streak, current_winning_streak)
            else:
                current_losing_streak += 1
                current_winning_streak = 0
                max_losing_streak = max(max_losing_streak, current_losing_streak)
        
        return max_winning_streak, max_losing_streak
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float) -> float:
        """Calculate Omega ratio."""
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0].sum())
        
        return gains / losses if losses > 0 else float('inf')
    
    def _calculate_benchmark_metrics(self, returns: pd.Series, benchmark: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics relative to benchmark."""
        # Extract benchmark returns
        if 'returns' in benchmark.columns:
            benchmark_returns = benchmark['returns']
        else:
            # Calculate from price data
            if 'close' in benchmark.columns:
                benchmark_returns = benchmark['close'].pct_change().dropna()
            else:
                return {}
        
        # Align dates
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) < 20:  # Need sufficient data
            return {}
        
        # Beta and alpha (CAPM)
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = aligned_benchmark.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Annualized values
        strategy_annual_return = aligned_returns.mean() * 252
        benchmark_annual_return = aligned_benchmark.mean() * 252
        alpha = strategy_annual_return - (self.risk_free_rate + beta * (benchmark_annual_return - self.risk_free_rate))
        
        # Correlation
        correlation = aligned_returns.corr(aligned_benchmark)
        
        # Information ratio
        active_returns = aligned_returns - aligned_benchmark
        tracking_error = active_returns.std() * np.sqrt(252)
        information_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        return {
            'benchmark_returns': benchmark_returns,
            'beta': beta,
            'alpha': alpha,
            'correlation': correlation,
            'information_ratio': information_ratio
        }