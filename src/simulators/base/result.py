"""Simulation result classes."""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


@dataclass
class SimulationSnapshot:
    """Point-in-time portfolio snapshot."""
    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    positions: Dict[str, Dict[str, Any]]
    daily_return: Optional[float] = None
    cumulative_return: Optional[float] = None
    
    
@dataclass
class SimulationResult:
    """Complete simulation results with metrics."""
    
    # Basic info
    start_date: date
    end_date: date
    initial_value: float
    final_value: float
    
    # Return metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    volatility: float = 0.0
    
    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Other metrics
    exposure_time: float = 0.0  # Percentage of time with positions
    turnover: float = 0.0  # Annual turnover rate
    
    # Time series data
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    positions_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    snapshots: List[SimulationSnapshot] = field(default_factory=list)
    
    # Benchmark comparison
    benchmark_return: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    information_ratio: Optional[float] = None
    
    def calculate_metrics(self, risk_free_rate: float = 0.02) -> None:
        """
        Calculate all performance metrics from snapshots.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        if not self.snapshots:
            return
            
        # Build equity curve
        dates = [s.timestamp for s in self.snapshots]
        values = [s.total_value for s in self.snapshots]
        self.equity_curve = pd.Series(values, index=dates)
        
        # Calculate returns
        self.returns = self.equity_curve.pct_change().dropna()
        
        # Basic return metrics
        self.total_return = (self.final_value - self.initial_value) / self.initial_value
        
        # Annualized return
        days = (self.end_date - self.start_date).days
        if days > 0:
            years = days / 365.25
            self.annual_return = (1 + self.total_return) ** (1 / years) - 1
        
        # Risk metrics
        if len(self.returns) > 1:
            self.volatility = self.returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            excess_returns = self.returns - risk_free_rate / 252
            if self.volatility > 0:
                self.sharpe_ratio = np.sqrt(252) * excess_returns.mean() / self.returns.std()
            
            # Sortino ratio (downside deviation)
            downside_returns = self.returns[self.returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                if downside_std > 0:
                    self.sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std
            
            # Max drawdown
            self._calculate_drawdown()
        
        # Trading metrics
        self._calculate_trading_metrics()
        
        # Exposure time
        self._calculate_exposure()
        
    def _calculate_drawdown(self) -> None:
        """Calculate maximum drawdown and duration."""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        self.max_drawdown = drawdown.min()
        
        # Find drawdown duration
        if self.max_drawdown < 0:
            dd_start = drawdown.idxmin()
            dd_idx = drawdown.index.get_loc(dd_start)
            
            # Find recovery
            post_dd = cumulative.iloc[dd_idx:]
            recovery_mask = post_dd >= running_max.iloc[dd_idx]
            
            if recovery_mask.any():
                recovery_idx = recovery_mask.idxmax()
                self.max_drawdown_duration = (recovery_idx - dd_start).days
            else:
                # Still in drawdown
                self.max_drawdown_duration = (drawdown.index[-1] - dd_start).days
    
    def _calculate_trading_metrics(self) -> None:
        """Calculate win rate, profit factor, etc."""
        if not self.trades:
            return
            
        pnls = [t.get('pnl', 0) for t in self.trades if 'pnl' in t]
        if not pnls:
            return
            
        self.total_trades = len(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        
        if wins:
            self.avg_win = np.mean(wins)
            
        if losses:
            self.avg_loss = np.mean(losses)
            total_losses = abs(sum(losses))
            if total_losses > 0:
                self.profit_factor = sum(wins) / total_losses
        elif wins:
            self.profit_factor = float('inf')
    
    def _calculate_exposure(self) -> None:
        """Calculate percentage of time with positions."""
        if not self.snapshots:
            return
            
        days_with_positions = sum(
            1 for s in self.snapshots 
            if s.positions_value > 0
        )
        
        total_days = len(self.snapshots)
        if total_days > 0:
            self.exposure_time = days_with_positions / total_days
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_value': self.initial_value,
            'final_value': self.final_value,
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'volatility': self.volatility,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'exposure_time': self.exposure_time,
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""
Simulation Results
==================
Period: {self.start_date} to {self.end_date}
Initial Value: ${self.initial_value:,.2f}
Final Value: ${self.final_value:,.2f}

Returns
-------
Total Return: {self.total_return:.2%}
Annual Return: {self.annual_return:.2%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Max Drawdown: {self.max_drawdown:.2%}

Trading
-------
Total Trades: {self.total_trades}
Win Rate: {self.win_rate:.2%}
Profit Factor: {self.profit_factor:.2f}
"""