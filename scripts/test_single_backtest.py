#!/usr/bin/env python3
"""Test a single backtest with detailed output."""

import asyncio
from datetime import date
import sys
sys.path.append('/Users/angwei/Repos/tetra')

from src.simulators.historical import HistoricalSimulator
from src.simulators.portfolio import Portfolio
from src.strats.benchmark import macd_crossover_strategy
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

async def test_single():
    """Test a single backtest."""
    
    # Create simulator
    sim = HistoricalSimulator()
    
    # Create portfolio
    portfolio = Portfolio(initial_cash=100000.0)
    
    # Create strategy
    strategy = macd_crossover_strategy()
    strategy.set_symbols(['AAPL'])
    
    # Set dates
    end_date = date(2025, 8, 7)
    start_date = date(2025, 7, 1)  # Shorter period for debugging
    
    print(f"Running backtest for {strategy.name} on AAPL from {start_date} to {end_date}")
    
    try:
        result = await sim.run_simulation(
            portfolio=portfolio,
            start_date=start_date,
            end_date=end_date,
            strategy=strategy
        )
        
        print(f"\n=== RESULTS ===")
        print(f"Final value: ${result.final_value:,.2f}")
        print(f"Total return: {result.total_return:.2%}")
        print(f"Total trades: {result.total_trades}")
        
    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_single())