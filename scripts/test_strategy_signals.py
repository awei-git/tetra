#!/usr/bin/env python3
"""Test strategy signal generation to debug why no trades are happening."""

import asyncio
from datetime import datetime, date
import sys
sys.path.append('/Users/angwei/Repos/tetra')

from src.strats.benchmark import buy_and_hold_strategy
from src.simulators.portfolio import Portfolio
import pandas as pd

async def test_strategy():
    """Test buy_and_hold strategy signal generation."""
    
    # Create strategy
    strategy = buy_and_hold_strategy()
    strategy.set_symbols(['SPY'])
    
    # Create mock portfolio
    portfolio = Portfolio(initial_cash=100000.0)
    
    # Mock market data
    market_data = {
        'SPY': {
            'open': 450.0,
            'high': 451.0,
            'low': 449.0,
            'close': 450.5,
            'volume': 1000000
        }
    }
    
    # Mock historical data
    dates = pd.date_range(end=date.today(), periods=252, freq='D')
    hist_data = pd.DataFrame({
        'open': [440 + i * 0.1 for i in range(252)],
        'high': [441 + i * 0.1 for i in range(252)],
        'low': [439 + i * 0.1 for i in range(252)],
        'close': [440 + i * 0.1 for i in range(252)],
        'volume': [1000000] * 252
    }, index=dates)
    
    historical_data = {'SPY': hist_data}
    
    # Test signal generation
    print(f"Strategy universe: {strategy.universe}")
    print(f"Strategy rules: {len(strategy.signal_rules)}")
    print(f"Portfolio positions: {portfolio.positions}")
    
    try:
        signals = strategy.generate_signals(
            market_data,
            portfolio,
            datetime.now(),
            historical_data
        )
        print(f"\nGenerated signals: {signals}")
        
        if signals:
            print(f"Number of signals: {len(signals)}")
            for signal in signals:
                print(f"  - {signal}")
        else:
            print("No signals generated!")
            
            # Debug indicators
            from src.strats.signal_based import SignalBasedStrategy
            if isinstance(strategy, SignalBasedStrategy):
                indicators = strategy._calculate_required_indicators(hist_data)
                print(f"\nCalculated indicators: {indicators}")
                
                # Check each rule
                for rule in strategy.signal_rules:
                    print(f"\nRule: {rule.name}")
                    print(f"Entry conditions: {len(rule.entry_conditions)}")
                    for cond in rule.entry_conditions:
                        print(f"  - {cond.signal_name} {cond.operator.value} {cond.value}")
                        
                    # Test conditions
                    current_indicators = pd.Series(indicators)
                    entry_check = rule.check_entry(current_indicators)
                    print(f"Entry conditions met: {entry_check}")
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_strategy())