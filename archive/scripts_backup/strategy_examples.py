#!/usr/bin/env python3
"""Examples of using different trading strategies."""

import sys
from pathlib import Path
from datetime import datetime, time
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.strats.event_based import (
    EventBasedStrategy, EventTrigger, EventType, EventImpact,
    EarningsStrategy, FOMCStrategy
)
from src.strats.signal_based import (
    SignalBasedStrategy, SignalCondition, SignalRule, ConditionOperator,
    PositionSide, MomentumStrategy, MeanReversionStrategy
)
from src.strats.time_based import (
    TimeBasedStrategy, TradingWindow, TradingSchedule, SessionType,
    IntradayStrategy
)
from src.strats.composite import (
    CompositeStrategy, StrategyWeight, CombinationMode
)


def example_event_based_strategy():
    """Example of creating an event-based strategy."""
    print("\n=== Event-Based Strategy Example ===")
    
    # Define event triggers
    earnings_trigger = EventTrigger(
        event_type=EventType.EARNINGS,
        impact=EventImpact.HIGH,
        pre_event_days=5,
        post_event_days=3,
        entry_conditions={
            'iv_threshold': 0.3,
            'volume_spike': 1.5
        }
    )
    
    fomc_trigger = EventTrigger(
        event_type=EventType.FOMC,
        impact=EventImpact.HIGH,
        pre_event_days=2,
        post_event_days=5
    )
    
    # Create strategy
    strategy = EventBasedStrategy(
        name="Multi-Event Strategy",
        event_triggers=[earnings_trigger, fomc_trigger],
        initial_capital=100000,
        position_size=0.1
    )
    
    print(f"Created strategy: {strategy.name}")
    print(f"Event types: {[t.event_type.value for t in [earnings_trigger, fomc_trigger]]}")
    
    # Simulate some data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(30) * 2 + 100,
        'high': np.random.randn(30) * 2 + 102,
        'low': np.random.randn(30) * 2 + 98,
        'close': np.random.randn(30) * 2 + 100,
        'volume': np.random.randint(1000000, 5000000, 30)
    }, index=dates)
    
    # Create event data
    events = pd.DataFrame([
        {
            'timestamp': dates[10],
            'event_type': 'earnings',
            'symbol': 'AAPL',
            'impact': 'high',
            'expected': 2.3,
            'actual': 2.5
        },
        {
            'timestamp': dates[20],
            'event_type': 'fomc',
            'symbol': None,
            'impact': 'high',
            'expected': 0.25,
            'actual': 0.50
        }
    ])
    
    # Generate signals
    signals = strategy.generate_signals(data, events=events)
    print(f"\nGenerated {len(signals[signals['signal'] != 0])} trading signals")


def example_signal_based_strategy():
    """Example of creating a signal-based strategy."""
    print("\n=== Signal-Based Strategy Example ===")
    
    # Define custom trading rules
    rules = [
        # Trend following rule
        SignalRule(
            name="trend_follow",
            entry_conditions=[
                SignalCondition("sma_50", ConditionOperator.GREATER_THAN, "sma_200"),
                SignalCondition("rsi_14", ConditionOperator.BETWEEN, [40, 60]),
                SignalCondition("adx_14", ConditionOperator.GREATER_THAN, 25)
            ],
            exit_conditions=[
                SignalCondition("sma_50", ConditionOperator.LESS_THAN, "sma_200")
            ],
            position_side=PositionSide.LONG,
            stop_loss=0.02,
            take_profit=0.05
        ),
        # Mean reversion rule
        SignalRule(
            name="mean_revert",
            entry_conditions=[
                SignalCondition("rsi_14", ConditionOperator.LESS_THAN, 20),
                SignalCondition("bb_position", ConditionOperator.LESS_THAN, -2)
            ],
            exit_conditions=[
                SignalCondition("rsi_14", ConditionOperator.GREATER_THAN, 50)
            ],
            position_side=PositionSide.LONG,
            position_size_factor=0.5,
            time_limit=10
        )
    ]
    
    # Create strategy
    strategy = SignalBasedStrategy(
        name="Multi-Signal Strategy",
        signal_rules=rules,
        confirmation_required=1,
        initial_capital=100000
    )
    
    print(f"Created strategy: {strategy.name}")
    print(f"Number of rules: {len(strategy.signal_rules)}")
    for rule in rules:
        print(f"  - {rule.name}: {len(rule.entry_conditions)} entry conditions")
    
    # Use predefined momentum strategy
    momentum = MomentumStrategy()
    print(f"\nPredefined strategy: {momentum.name}")


def example_time_based_strategy():
    """Example of creating a time-based strategy."""
    print("\n=== Time-Based Strategy Example ===")
    
    # Define custom trading windows
    windows = [
        # Morning session
        TradingWindow(
            start_time=time(9, 30),
            end_time=time(11, 0),
            session_type=SessionType.REGULAR
        ),
        # Power hour
        TradingWindow(
            start_time=time(15, 0),
            end_time=time(15, 45),
            session_type=SessionType.REGULAR
        ),
        # Asian session for FX
        TradingWindow(
            start_time=time(19, 0),
            end_time=time(3, 0),
            session_type=SessionType.ASIAN
        )
    ]
    
    # Create schedule
    schedule = TradingSchedule(
        windows=windows,
        max_trades_per_day=10,
        max_trades_per_window=3,
        force_close_end_of_day=True
    )
    
    # Create strategy
    strategy = TimeBasedStrategy(
        name="Multi-Session Strategy",
        trading_schedule=schedule,
        initial_capital=100000
    )
    
    print(f"Created strategy: {strategy.name}")
    print(f"Trading windows: {len(schedule.windows)}")
    for window in windows:
        print(f"  - {window.session_type.value}: {window.start_time} - {window.end_time}")
    
    # Use predefined intraday strategy
    intraday = IntradayStrategy()
    print(f"\nPredefined strategy: {intraday.name}")


def example_composite_strategy():
    """Example of creating a composite strategy."""
    print("\n=== Composite Strategy Example ===")
    
    # Create individual strategies
    momentum = MomentumStrategy()
    mean_reversion = MeanReversionStrategy()
    intraday = IntradayStrategy()
    
    # Define weights and combine
    weights = [
        StrategyWeight(momentum, weight=1.5, min_confidence=0.6),
        StrategyWeight(mean_reversion, weight=1.0, min_confidence=0.7),
        StrategyWeight(intraday, weight=0.8, min_confidence=0.5)
    ]
    
    # Create composite with different modes
    conservative = CompositeStrategy(
        name="Conservative Multi-Strategy",
        strategies=weights,
        combination_mode=CombinationMode.UNANIMOUS,
        min_agreement=0.8
    )
    
    aggressive = CompositeStrategy(
        name="Aggressive Multi-Strategy",
        strategies=weights,
        combination_mode=CombinationMode.WEIGHTED,
        min_agreement=0.4,
        adapt_weights=True
    )
    
    print(f"Created conservative strategy: {conservative.name}")
    print(f"  Mode: {conservative.combination_mode.value}")
    print(f"  Min agreement: {conservative.min_agreement}")
    
    print(f"\nCreated aggressive strategy: {aggressive.name}")
    print(f"  Mode: {aggressive.combination_mode.value}")
    print(f"  Adaptive weights: {aggressive.adapt_weights}")


def example_custom_strategy():
    """Example of creating a fully custom strategy."""
    print("\n=== Custom Strategy Example ===")
    
    # Combine all types of strategies
    earnings_trigger = EventTrigger(
        event_type=EventType.EARNINGS,
        impact=EventImpact.HIGH,
        pre_event_days=3,
        post_event_days=2
    )
    
    event_strategy = EventBasedStrategy(
        name="Earnings Player",
        event_triggers=[earnings_trigger]
    )
    
    signal_rules = [
        SignalRule(
            name="breakout",
            entry_conditions=[
                SignalCondition("price", ConditionOperator.CROSSES_ABOVE, "resistance"),
                SignalCondition("volume", ConditionOperator.GREATER_THAN, "avg_volume_20")
            ],
            exit_conditions=[
                SignalCondition("price", ConditionOperator.CROSSES_BELOW, "support")
            ],
            position_side=PositionSide.LONG
        )
    ]
    
    signal_strategy = SignalBasedStrategy(
        name="Breakout Hunter",
        signal_rules=signal_rules
    )
    
    morning_window = TradingWindow(
        start_time=time(9, 30),
        end_time=time(10, 30),
        session_type=SessionType.REGULAR
    )
    
    time_strategy = TimeBasedStrategy(
        name="Morning Trader",
        trading_schedule=TradingSchedule([morning_window])
    )
    
    # Combine all strategies
    hybrid = CompositeStrategy(
        name="Hybrid Master Strategy",
        strategies=[
            StrategyWeight(event_strategy, weight=1.0),
            StrategyWeight(signal_strategy, weight=1.5),
            StrategyWeight(time_strategy, weight=0.8)
        ],
        combination_mode=CombinationMode.ADAPTIVE,
        adapt_weights=True,
        min_agreement=0.5
    )
    
    print(f"Created hybrid strategy: {hybrid.name}")
    print(f"Components: {len(hybrid.strategies)} strategies")
    for sw in hybrid.strategies:
        print(f"  - {sw.strategy.name} (weight: {sw.weight})")


def main():
    """Run all examples."""
    print("Trading Strategy Examples")
    print("=" * 50)
    
    example_event_based_strategy()
    example_signal_based_strategy()
    example_time_based_strategy()
    example_composite_strategy()
    example_custom_strategy()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()