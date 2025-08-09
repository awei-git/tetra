#!/usr/bin/env python3
"""Debug indicator calculation."""

import asyncio
from datetime import date
import pandas as pd
import sys
sys.path.append('/Users/angwei/Repos/tetra')

from src.strats.benchmark import golden_cross_strategy
from src.simulators.historical.market_replay import MarketReplay

async def debug_calc():
    """Debug indicator calculation."""
    
    # Load some market data
    replay = MarketReplay()
    await replay.load_data(['SPY'], date(2025, 6, 1), date(2025, 8, 7))
    
    # Get full OHLCV data
    df = await replay._load_symbol_data('SPY', date(2025, 6, 1), date(2025, 8, 7))
    print(f"Loaded {len(df)} days of SPY data")
    
    # Test golden cross strategy
    strategy = golden_cross_strategy()
    strategy.set_symbols(['SPY'])
    
    # Get required indicators
    required_indicators = set()
    for rule in strategy.signal_rules:
        for condition in rule.entry_conditions + rule.exit_conditions:
            required_indicators.add(condition.signal_name)
            if isinstance(condition.value, str):
                if any(prefix in condition.value for prefix in ['sma_', 'ema_', 'rsi_', 'volume_', 'bb_', 'macd', 'highest_', 'lowest_', 'atr_', 'adx_', 'returns_', 'donchian_', 'vwap']):
                    required_indicators.add(condition.value)
    
    print(f"\nRequired indicators: {required_indicators}")
    
    # Calculate each required indicator manually
    indicators = {}
    hist_df = df.copy()
    hist_df.columns = hist_df.columns.str.lower()
    
    for indicator_name in required_indicators:
        print(f"\nCalculating {indicator_name}...")
        try:
            if 'sma_' in indicator_name:
                period = int(indicator_name.split('_')[1])
                if len(hist_df) >= period and 'close' in hist_df.columns:
                    indicators[indicator_name] = hist_df['close'].rolling(period).mean().iloc[-1]
                    print(f"  Result: {indicators[indicator_name]}")
            elif indicator_name.startswith('volume_sma_'):
                print(f"  Detected volume_sma pattern")
                if 'volume' in hist_df.columns:
                    remaining = indicator_name[11:]  # Remove 'volume_sma_'
                    print(f"  Remaining: '{remaining}'")
                    period = 20  # default
                    
                    if remaining.isdigit():
                        period = int(remaining)
                        print(f"  Period: {period}")
                    
                    if len(hist_df) >= period:
                        volume_sma = hist_df['volume'].rolling(period).mean().iloc[-1]
                        indicators[indicator_name] = volume_sma
                        print(f"  Result: {indicators[indicator_name]}")
            elif indicator_name in ['close', 'open', 'high', 'low', 'volume']:
                if indicator_name in hist_df.columns:
                    indicators[indicator_name] = hist_df[indicator_name].iloc[-1]
                    print(f"  Result: {indicators[indicator_name]}")
        except Exception as e:
            print(f"  Failed: {e}")
    
    print(f"\nFinal indicators: {list(indicators.keys())}")

if __name__ == "__main__":
    asyncio.run(debug_calc())