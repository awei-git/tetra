#!/usr/bin/env python3
"""Test condition evaluation with string references."""

import pandas as pd
import sys
sys.path.append('/Users/angwei/Repos/tetra')

from src.strats.signal_based import SignalCondition, ConditionOperator

# Create test indicators
indicators = {
    'close': 100.0,
    'sma_200': 95.0,
    'rsi_14': 25.0
}

signals = pd.Series(indicators)

# Test condition: close > sma_200
cond = SignalCondition("close", ConditionOperator.GREATER_THAN, "sma_200")
print(f"Condition: close > sma_200")
print(f"  close = {signals['close']}")
print(f"  sma_200 = {signals['sma_200']}")

try:
    result = cond.evaluate(signals)
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Error: {e}")
    print(f"  Error type: {type(e)}")

# Debug the issue
print(f"\nDebugging:")
print(f"  cond.value = '{cond.value}' (type: {type(cond.value)})")
print(f"  cond.value in signals: {cond.value in signals}")

# Test with numeric value
cond2 = SignalCondition("rsi_14", ConditionOperator.LESS_THAN, 30)
print(f"\nCondition: rsi_14 < 30")
print(f"  rsi_14 = {signals['rsi_14']}")
try:
    result = cond2.evaluate(signals)
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Error: {e}")