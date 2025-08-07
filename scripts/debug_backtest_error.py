#!/usr/bin/env python3
"""Debug the backtest error with Portfolio.empty"""

import asyncio
import sys
from pathlib import Path
from datetime import date, timedelta

sys.path.append(str(Path(__file__).parent.parent))

from src.simulators.historical import HistoricalSimulator
from src.simulators.base import SimulationConfig
from src.simulators.portfolio import Portfolio
from src.strats.benchmark import buy_and_hold_strategy

async def test_backtest():
    """Test a simple backtest to find the error"""
    print("Setting up test backtest...")
    
    # Create config
    config = SimulationConfig(
        starting_cash=100000,
        commission_per_share=0.005,
        slippage_bps=5,
        benchmark_symbol="SPY"
    )
    
    # Create simulator
    simulator = HistoricalSimulator(config)
    
    # Create portfolio
    portfolio = Portfolio(initial_cash=100000)
    print(f"Portfolio created: {portfolio}")
    print(f"Portfolio type: {type(portfolio)}")
    print(f"Has empty attr: {hasattr(portfolio, 'empty')}")
    
    # Get strategy
    strategy = buy_and_hold_strategy()
    print(f"Strategy created: {strategy}")
    
    # Set dates
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    
    try:
        # Run simulation
        print(f"Running simulation from {start_date} to {end_date}...")
        result = await simulator.run_simulation(
            portfolio=portfolio,
            start_date=start_date,
            end_date=end_date,
            strategy=strategy
        )
        print(f"Simulation completed successfully!")
        print(f"Result: {result}")
    except AttributeError as e:
        print(f"AttributeError: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_backtest())