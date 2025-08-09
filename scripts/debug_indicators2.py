#!/usr/bin/env python3
"""Debug required indicators."""

import sys
sys.path.append('/Users/angwei/Repos/tetra')

from src.strats.benchmark import golden_cross_strategy

# Test golden cross strategy
strategy = golden_cross_strategy()

# Extract required indicators
required_indicators = set()
for rule in strategy.signal_rules:
    for condition in rule.entry_conditions + rule.exit_conditions:
        required_indicators.add(condition.signal_name)
        print(f"Condition: {condition.signal_name} {condition.operator.value} {condition.value}")
        print(f"  Value is string: {isinstance(condition.value, str)}")
        
        # Add string values that might be indicators
        if isinstance(condition.value, str):
            # Check if it looks like an indicator name
            if any(prefix in condition.value for prefix in ['sma_', 'ema_', 'rsi_', 'volume_', 'bb_', 'macd', 'highest_', 'lowest_', 'atr_', 'adx_', 'returns_', 'donchian_', 'vwap']):
                print(f"  Adding {condition.value} as required indicator")
                required_indicators.add(condition.value)

print(f"\nRequired indicators: {required_indicators}")