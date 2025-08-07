#!/usr/bin/env python3
"""Test Portfolio class to debug empty attribute error"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Test both Portfolio classes
print("Testing Portfolio classes...")

# Test simulators portfolio
try:
    from src.simulators.portfolio import Portfolio as SimPortfolio
    p1 = SimPortfolio(initial_cash=100000)
    print(f"Simulator Portfolio - has empty: {hasattr(p1, 'empty')}")
    print(f"Simulator Portfolio - get_symbols: {p1.get_symbols()}")
except Exception as e:
    print(f"Error with Simulator Portfolio: {e}")

# Test backtesting portfolio  
try:
    from src.backtesting.portfolio import Portfolio as BacktestPortfolio
    p2 = BacktestPortfolio(initial_cash=100000)
    print(f"Backtest Portfolio - has empty: {hasattr(p2, 'empty')}")
    if hasattr(p2, 'get_symbols'):
        print(f"Backtest Portfolio - get_symbols: {p2.get_symbols()}")
    else:
        print("Backtest Portfolio - no get_symbols method")
except Exception as e:
    print(f"Error with Backtest Portfolio: {e}")

# Check if Portfolio is a pandas object
import pandas as pd
print(f"\nIs SimPortfolio a DataFrame? {isinstance(p1, pd.DataFrame)}")
print(f"SimPortfolio type: {type(p1)}")
print(f"SimPortfolio attributes: {[a for a in dir(p1) if not a.startswith('_')][:10]}")