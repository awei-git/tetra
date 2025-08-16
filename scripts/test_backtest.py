"""Test backtest to debug issues."""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')

from src.simulators.historical.simulator import HistoricalSimulator
from src.strats.benchmark import buy_and_hold_strategy

async def test():
    """Test a simple backtest."""
    
    # Create simple test data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    data = pd.DataFrame({
        'close': np.random.uniform(100, 110, len(dates)),
        'open': np.random.uniform(100, 110, len(dates)),
        'high': np.random.uniform(105, 115, len(dates)),
        'low': np.random.uniform(95, 105, len(dates)),
        'volume': np.random.uniform(1000000, 2000000, len(dates))
    }, index=dates)
    
    # Create simulator
    simulator = HistoricalSimulator(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005
    )
    
    # Get strategy
    strategy = buy_and_hold_strategy()
    
    try:
        # Run backtest
        portfolio = await simulator.run(
            strategy=strategy,
            data=data,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        print("Backtest successful!")
        print(f"Portfolio value: {portfolio}")
        
    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())