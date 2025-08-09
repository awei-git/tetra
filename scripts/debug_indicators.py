#!/usr/bin/env python3
"""Debug indicator calculations."""

import asyncio
from datetime import date
import pandas as pd
import sys
sys.path.append('/Users/angwei/Repos/tetra')

from src.strats.benchmark import golden_cross_strategy, macd_crossover_strategy
from src.simulators.historical.market_replay import MarketReplay

async def debug_indicators():
    """Debug indicator calculations."""
    
    # Load some market data
    replay = MarketReplay()
    await replay.load_data(['SPY'], date(2025, 6, 1), date(2025, 8, 7))
    
    # Get full OHLCV data
    df = await replay._load_symbol_data('SPY', date(2025, 6, 1), date(2025, 8, 7))
    print(f"Loaded {len(df)} days of SPY data")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Last 5 rows:\n{df.tail()}")
    
    # Test golden cross strategy
    strategy = golden_cross_strategy()
    strategy.set_symbols(['SPY'])
    
    # Calculate indicators
    indicators = strategy._calculate_required_indicators(df)
    print(f"\nCalculated indicators: {list(indicators.keys())}")
    for name, value in indicators.items():
        print(f"  {name}: {value} (type: {type(value).__name__})")
    
    # Check what the conditions are expecting
    for rule in strategy.signal_rules:
        print(f"\nRule: {rule.name}")
        for cond in rule.entry_conditions:
            print(f"  Entry condition: {cond.signal_name} {cond.operator.value} {cond.value}")
            print(f"    Value type: {type(cond.value)}")
            print(f"    signal_name in indicators: {cond.signal_name in indicators}")
            if isinstance(cond.value, str):
                print(f"    value in indicators: {cond.value in indicators}")
        for cond in rule.exit_conditions:
            print(f"  Exit condition: {cond.signal_name} {cond.operator.value} {cond.value}")
    
    # Try to evaluate conditions
    print("\nTrying to evaluate conditions...")
    current_indicators = pd.Series(indicators)
    print(f"Current indicators as Series:\n{current_indicators}")
    
    for rule in strategy.signal_rules:
        print(f"\nChecking rule: {rule.name}")
        try:
            entry = rule.check_entry(current_indicators)
            print(f"  Entry conditions met: {entry}")
        except Exception as e:
            print(f"  Entry check failed: {e}")
    
    # Test MACD strategy too
    print("\n" + "="*50)
    print("Testing MACD strategy...")
    
    macd_strat = macd_crossover_strategy()
    macd_strat.set_symbols(['SPY'])
    
    # Calculate indicators
    macd_indicators = macd_strat._calculate_required_indicators(df)
    print(f"\nCalculated indicators: {list(macd_indicators.keys())}")
    for name, value in macd_indicators.items():
        print(f"  {name}: {value} (type: {type(value).__name__})")

if __name__ == "__main__":
    asyncio.run(debug_indicators())