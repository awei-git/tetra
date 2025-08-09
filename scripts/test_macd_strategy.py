#!/usr/bin/env python3
"""Test MACD strategy indicator issues."""

import asyncio
from datetime import date
import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/angwei/Repos/tetra')

from src.strats.benchmark import macd_crossover_strategy
from src.simulators.historical.market_replay import MarketReplay

async def test_macd():
    """Test MACD strategy."""
    
    # Load some market data
    replay = MarketReplay()
    await replay.load_data(['AAPL'], date(2025, 6, 1), date(2025, 8, 7))
    
    # Get full OHLCV data
    df = await replay._load_symbol_data('AAPL', date(2025, 6, 1), date(2025, 8, 7))
    print(f"Loaded {len(df)} days of AAPL data")
    
    # Test MACD strategy
    strategy = macd_crossover_strategy()
    strategy.set_symbols(['AAPL'])
    
    # Calculate indicators
    indicators = strategy._calculate_required_indicators(df)
    print(f"\nCalculated indicators:")
    for name, value in indicators.items():
        print(f"  {name}: {value} (type: {type(value).__name__})")
    
    # Try to evaluate conditions
    print(f"\nStrategy rules:")
    for rule in strategy.signal_rules:
        print(f"\nRule: {rule.name}")
        
        # Create a pandas Series from indicators
        signals = pd.Series(indicators)
        print(f"Signals Series:\n{signals}")
        
        # Try to check entry conditions
        for i, cond in enumerate(rule.entry_conditions):
            print(f"\n  Entry condition {i+1}: {cond.signal_name} {cond.operator.value} {cond.value}")
            print(f"    signal_name type: {type(cond.signal_name)}")
            print(f"    value type: {type(cond.value)}")
            print(f"    signal_name in signals: {cond.signal_name in signals}")
            
            if cond.signal_name in signals:
                signal_val = signals[cond.signal_name]
                print(f"    signal value: {signal_val} (type: {type(signal_val)})")
                
                # Check if value is in signals
                if isinstance(cond.value, str):
                    print(f"    value in signals: {cond.value in signals}")
                    if cond.value in signals:
                        compare_val = signals[cond.value]
                        print(f"    compare value: {compare_val} (type: {type(compare_val)})")
            
            try:
                result = cond.evaluate(signals)
                print(f"    Result: {result}")
            except Exception as e:
                print(f"    Error: {e}")
                print(f"    Error type: {type(e).__name__}")
                
                # Debug numpy types
                if cond.signal_name in signals:
                    sv = signals[cond.signal_name]
                    print(f"    Signal value numpy type: {sv.dtype if hasattr(sv, 'dtype') else 'N/A'}")

if __name__ == "__main__":
    asyncio.run(test_macd())