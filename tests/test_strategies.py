"""Tests for trading strategies."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo

from src.strats.base import (
    BaseStrategy, PositionSide, Trade, Position, StrategyState
)
from src.strats.event_based import (
    EventBasedStrategy, EventTrigger, EventType, EventImpact,
    MarketEvent, EarningsStrategy, FOMCStrategy
)
from src.strats.signal_based import (
    SignalBasedStrategy, SignalCondition, SignalRule, ConditionOperator,
    MomentumStrategy, MeanReversionStrategy, MLStrategy
)
from src.strats.time_based import (
    TimeBasedStrategy, TradingWindow, TradingSchedule, SessionType,
    IntradayStrategy, OvernightStrategy
)
from src.strats.composite import (
    CompositeStrategy, StrategyWeight, CombinationMode,
    ConservativeComposite, AggressiveComposite
)


class TestBaseStrategy:
    """Test base strategy functionality."""
    
    def test_position_tracking(self):
        """Test position tracking."""
        strategy = MockStrategy("Test", initial_capital=100000)
        
        # Execute a trade
        trade = strategy.execute_trade(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=100,
            price=150.0,
            timestamp=datetime.now()
        )
        
        assert trade is not None
        assert "AAPL" in strategy.state.positions
        assert strategy.state.positions["AAPL"].quantity == 100
        assert strategy.state.cash < 100000
    
    def test_risk_limits(self):
        """Test risk limit checks."""
        strategy = MockStrategy("Test")
        strategy.stop_loss = 0.02
        strategy.take_profit = 0.05
        
        # Create a position
        position = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=100,
            avg_price=100.0,
            current_price=100.0
        )
        
        # Test stop loss
        assert strategy.check_risk_limits(position, 97.0) == True  # 3% loss
        
        # Test take profit
        assert strategy.check_risk_limits(position, 106.0) == True  # 6% gain
        
        # Test normal price
        assert strategy.check_risk_limits(position, 101.0) == False
    
    def test_position_sizing(self):
        """Test position sizing calculation."""
        strategy = MockStrategy("Test", initial_capital=100000, position_size=0.1)
        
        # Test basic sizing
        size = strategy.calculate_position_size("AAPL", 150.0)
        assert size == 67  # 10% of 100k / 150
        
        # Test with volatility
        size = strategy.calculate_position_size("AAPL", 150.0, volatility=0.03)
        assert size < 67  # Should be reduced for high volatility


class TestEventBasedStrategy:
    """Test event-based strategies."""
    
    def test_event_trigger_evaluation(self):
        """Test event trigger evaluation."""
        trigger = EventTrigger(
            event_type=EventType.EARNINGS,
            impact=EventImpact.HIGH,
            pre_event_days=5,
            post_event_days=2
        )
        
        strategy = EventBasedStrategy(
            name="Test Event",
            event_triggers=[trigger]
        )
        
        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'close': np.random.randn(10) * 10 + 100
        }, index=dates)
        
        # Create event data
        events = pd.DataFrame([
            {
                'timestamp': dates[5],
                'event_type': 'earnings',
                'symbol': 'AAPL',
                'impact': 'high',
                'actual': 2.5,
                'expected': 2.3
            }
        ])
        
        signals = strategy.generate_signals(data, events=events)
        assert isinstance(signals, pd.DataFrame)
        assert 'signal' in signals.columns
    
    def test_earnings_strategy(self):
        """Test earnings-specific strategy."""
        strategy = EarningsStrategy()
        
        # Create test event
        event = MarketEvent(
            timestamp=datetime.now(),
            event_type=EventType.EARNINGS,
            symbol="AAPL",
            description="Q4 Earnings",
            impact=EventImpact.HIGH,
            actual_value=2.5,
            expected_value=2.3,
            previous_value=2.1
        )
        
        assert event.is_positive_surprise == True
        assert event.surprise > 0


class TestSignalBasedStrategy:
    """Test signal-based strategies."""
    
    def test_signal_condition(self):
        """Test signal condition evaluation."""
        # Test simple condition
        condition = SignalCondition(
            signal_name="rsi",
            operator=ConditionOperator.LESS_THAN,
            value=30
        )
        
        signals = pd.Series({'rsi': 25, 'macd': 0.5})
        assert condition.evaluate(signals) == True
        
        signals['rsi'] = 35
        assert condition.evaluate(signals) == False
        
        # Test between condition
        condition = SignalCondition(
            signal_name="rsi",
            operator=ConditionOperator.BETWEEN,
            value=[30, 70]
        )
        
        signals['rsi'] = 50
        assert condition.evaluate(signals) == True
        
        signals['rsi'] = 80
        assert condition.evaluate(signals) == False
    
    def test_signal_rule(self):
        """Test signal rule evaluation."""
        rule = SignalRule(
            name="oversold_bounce",
            entry_conditions=[
                SignalCondition("rsi", ConditionOperator.LESS_THAN, 30),
                SignalCondition("volume_ratio", ConditionOperator.GREATER_THAN, 1.5)
            ],
            exit_conditions=[
                SignalCondition("rsi", ConditionOperator.GREATER_THAN, 70)
            ],
            position_side=PositionSide.LONG,
            require_all=True
        )
        
        # Test entry
        signals = pd.Series({'rsi': 25, 'volume_ratio': 2.0})
        assert rule.check_entry(signals) == True
        
        signals['rsi'] = 35
        assert rule.check_entry(signals) == False  # RSI too high
        
        # Test exit
        signals['rsi'] = 75
        assert rule.check_exit(signals) == True
    
    def test_momentum_strategy(self):
        """Test momentum strategy."""
        strategy = MomentumStrategy()
        
        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        signals = pd.DataFrame({
            'rsi_14': np.linspace(40, 80, 50),
            'macd_signal': np.sin(np.linspace(0, 4*np.pi, 50)),
            'adx_14': np.ones(50) * 30,
            'returns_20': np.linspace(-0.1, 0.1, 50)
        }, index=dates)
        
        data = pd.DataFrame({
            'close': np.linspace(100, 110, 50)
        }, index=dates)
        
        output = strategy.generate_signals(data, signals)
        assert output is not None
        assert len(output) == len(data)


class TestTimeBasedStrategy:
    """Test time-based strategies."""
    
    def test_trading_window(self):
        """Test trading window logic."""
        window = TradingWindow(
            start_time=time(9, 30),
            end_time=time(16, 0),
            days_of_week=[0, 1, 2, 3, 4],  # Mon-Fri
            timezone="America/New_York"
        )
        
        # Test during market hours
        market_time = datetime(2024, 1, 2, 14, 30, tzinfo=ZoneInfo("America/New_York"))  # Tuesday 2:30 PM
        assert window.is_active(market_time) == True
        
        # Test outside market hours
        after_hours = datetime(2024, 1, 2, 17, 0, tzinfo=ZoneInfo("America/New_York"))  # 5 PM
        assert window.is_active(after_hours) == False
        
        # Test weekend
        weekend = datetime(2024, 1, 6, 14, 30, tzinfo=ZoneInfo("America/New_York"))  # Saturday
        assert window.is_active(weekend) == False
    
    def test_trading_schedule(self):
        """Test trading schedule."""
        windows = [
            TradingWindow(
                start_time=time(9, 30),
                end_time=time(11, 30),
                session_type=SessionType.REGULAR
            ),
            TradingWindow(
                start_time=time(14, 0),
                end_time=time(16, 0),
                session_type=SessionType.REGULAR
            )
        ]
        
        schedule = TradingSchedule(
            windows=windows,
            max_trades_per_day=5,
            force_close_end_of_day=True
        )
        
        # Test active window detection
        morning = datetime(2024, 1, 2, 10, 30, tzinfo=ZoneInfo("America/New_York"))
        window = schedule.get_active_window(morning)
        assert window is not None
        assert window.session_type == SessionType.REGULAR
        
        # Test force close
        eod = datetime(2024, 1, 2, 15, 50, tzinfo=ZoneInfo("America/New_York"))
        assert schedule.should_force_close(eod) == True
    
    def test_intraday_strategy(self):
        """Test intraday strategy."""
        strategy = IntradayStrategy()
        
        # Create test data
        dates = pd.date_range(
            start='2024-01-02 09:00', 
            end='2024-01-02 16:30', 
            freq='30min',
            tz='America/New_York'
        )
        
        data = pd.DataFrame({
            'close': np.random.randn(len(dates)) * 2 + 100
        }, index=dates)
        
        signals = strategy.generate_signals(data)
        assert isinstance(signals, pd.DataFrame)
        assert 'window' in signals.columns


class TestCompositeStrategy:
    """Test composite strategies."""
    
    def test_strategy_combination(self):
        """Test combining multiple strategies."""
        # Create mock strategies
        strat1 = MockStrategy("Strat1")
        strat2 = MockStrategy("Strat2")
        
        weights = [
            StrategyWeight(strat1, weight=1.0),
            StrategyWeight(strat2, weight=0.5)
        ]
        
        composite = CompositeStrategy(
            name="Test Composite",
            strategies=weights,
            combination_mode=CombinationMode.WEIGHTED
        )
        
        # Test signal generation
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({'close': np.ones(10) * 100}, index=dates)
        
        signals = composite.generate_signals(data)
        assert isinstance(signals, pd.DataFrame)
    
    def test_unanimous_combination(self):
        """Test unanimous combination mode."""
        composite = ConservativeComposite()
        
        # All strategies must agree
        votes = [
            (PositionSide.LONG, 1.0),
            (PositionSide.LONG, 1.0),
            (PositionSide.LONG, 1.0)
        ]
        
        # In real implementation, this would be tested through should_enter
        assert composite.combination_mode == CombinationMode.UNANIMOUS
    
    def test_adaptive_weights(self):
        """Test adaptive weight adjustment."""
        composite = AggressiveComposite()
        
        # Record some performance
        composite.record_trade_result("Momentum Strategy", 0.02)
        composite.record_trade_result("Momentum Strategy", -0.01)
        composite.record_trade_result("Momentum Strategy", 0.03)
        
        assert "Momentum Strategy" in composite.strategy_performance
        assert len(composite.strategy_performance["Momentum Strategy"]) == 3


# Mock strategy for testing
class MockStrategy(BaseStrategy):
    """Mock strategy for testing base functionality."""
    
    def generate_signals(self, data, signals=None, events=None):
        output = pd.DataFrame(index=data.index)
        output['signal'] = 0
        output['position_size'] = 0
        return output
    
    def should_enter(self, symbol, timestamp, data, signals=None, events=None):
        return False, PositionSide.FLAT, 0.0
    
    def should_exit(self, position, timestamp, data, signals=None, events=None):
        return False


if __name__ == "__main__":
    pytest.main([__file__])