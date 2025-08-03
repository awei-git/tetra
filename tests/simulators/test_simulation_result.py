"""Tests for simulation result and metrics."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from src.simulators.base import SimulationResult, SimulationSnapshot


class TestSimulationResult:
    """Test SimulationResult class."""
    
    def test_simulation_result_initialization(self):
        """Test creating simulation result."""
        result = SimulationResult(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            initial_value=100000,
            final_value=110000
        )
        
        assert result.start_date == date(2023, 1, 1)
        assert result.end_date == date(2023, 12, 31)
        assert result.initial_value == 100000
        assert result.final_value == 110000
    
    def test_calculate_basic_metrics(self):
        """Test basic metric calculations."""
        result = SimulationResult(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            initial_value=100000,
            final_value=110000
        )
        
        # Create daily snapshots
        dates = pd.date_range(
            start=datetime(2023, 1, 1),
            end=datetime(2023, 12, 31),
            freq='D'
        )
        
        # Simulate linear growth
        for i, date in enumerate(dates):
            value = 100000 + (10000 * i / len(dates))
            snapshot = SimulationSnapshot(
                timestamp=date,
                total_value=value,
                cash=50000,
                positions_value=value - 50000,
                positions={}
            )
            result.snapshots.append(snapshot)
        
        # Calculate metrics
        result.calculate_metrics()
        
        assert result.total_return == 0.1  # 10% return
        assert result.annual_return == 0.1  # 1 year period
        assert len(result.equity_curve) == len(dates)
        assert len(result.returns) == len(dates) - 1
    
    def test_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        result = SimulationResult(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            initial_value=100000,
            final_value=95000
        )
        
        # Create snapshots with drawdown
        values = [100000, 105000, 110000, 100000, 95000, 98000, 100000]
        dates = pd.date_range(
            start=datetime(2023, 1, 1),
            periods=len(values),
            freq='D'
        )
        
        for date, value in zip(dates, values):
            snapshot = SimulationSnapshot(
                timestamp=date,
                total_value=value,
                cash=value,
                positions_value=0,
                positions={}
            )
            result.snapshots.append(snapshot)
        
        result.calculate_metrics()
        
        # Max drawdown should be from 110k to 95k = -13.6%
        assert result.max_drawdown < 0
        assert abs(result.max_drawdown - (-15000/110000)) < 0.001
    
    def test_trading_metrics(self):
        """Test trading metrics calculation."""
        result = SimulationResult(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            initial_value=100000,
            final_value=105000
        )
        
        # Add some trades
        result.trades = [
            {'pnl': 1000},
            {'pnl': 500},
            {'pnl': -300},
            {'pnl': 800},
            {'pnl': -200},
            {'pnl': 1500},
        ]
        
        result._calculate_trading_metrics()
        
        assert result.total_trades == 6
        assert result.winning_trades == 4
        assert result.losing_trades == 2
        assert result.win_rate == 4/6
        assert result.avg_win == (1000 + 500 + 800 + 1500) / 4
        assert result.avg_loss == (-300 + -200) / 2
        assert result.profit_factor == 3800 / 500  # 7.6
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        result = SimulationResult(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            initial_value=100000,
            final_value=110000
        )
        
        # Create more realistic returns
        np.random.seed(42)
        dates = pd.date_range(
            start=datetime(2023, 1, 1),
            end=datetime(2023, 12, 31),
            freq='B'  # Business days
        )
        
        # Generate returns with 15% annual vol, 10% drift
        daily_returns = np.random.normal(0.10/252, 0.15/np.sqrt(252), len(dates))
        values = [100000]
        
        for ret in daily_returns[1:]:
            values.append(values[-1] * (1 + ret))
        
        for date, value in zip(dates, values):
            snapshot = SimulationSnapshot(
                timestamp=date,
                total_value=value,
                cash=0,
                positions_value=value,
                positions={}
            )
            result.snapshots.append(snapshot)
        
        result.final_value = values[-1]
        result.calculate_metrics(risk_free_rate=0.02)
        
        # Sharpe should be positive with positive returns
        assert result.sharpe_ratio > 0
        assert result.volatility > 0
    
    def test_result_serialization(self):
        """Test converting result to dict."""
        result = SimulationResult(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            initial_value=100000,
            final_value=110000
        )
        
        result.total_return = 0.1
        result.sharpe_ratio = 1.5
        result.max_drawdown = -0.05
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['total_return'] == 0.1
        assert result_dict['sharpe_ratio'] == 1.5
        assert result_dict['max_drawdown'] == -0.05
        assert 'start_date' in result_dict
        assert 'end_date' in result_dict


class TestSimulationSnapshot:
    """Test SimulationSnapshot class."""
    
    def test_snapshot_creation(self):
        """Test creating portfolio snapshot."""
        snapshot = SimulationSnapshot(
            timestamp=datetime.now(),
            total_value=105000,
            cash=20000,
            positions_value=85000,
            positions={
                'AAPL': {
                    'quantity': 100,
                    'market_value': 15000,
                    'unrealized_pnl': 500
                }
            }
        )
        
        assert snapshot.total_value == 105000
        assert snapshot.cash == 20000
        assert snapshot.positions_value == 85000
        assert 'AAPL' in snapshot.positions