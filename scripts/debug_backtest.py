#!/usr/bin/env python3
"""Debug why strategies aren't executing trades in backtest."""

import asyncio
from datetime import date, timedelta
import sys
sys.path.append('/Users/angwei/Repos/tetra')

from src.simulators.historical import HistoricalSimulator
from src.simulators.portfolio import Portfolio
from src.strats.benchmark import buy_and_hold_strategy
from src.utils.logging import logger
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

async def debug_backtest():
    """Debug a simple backtest."""
    
    # Create simulator
    sim = HistoricalSimulator()
    
    # Create portfolio
    portfolio = Portfolio(initial_cash=100000.0)
    print(f"Initial portfolio: Cash=${portfolio.cash:,.2f}, Positions={len(portfolio.positions)}")
    
    # Create strategy
    strategy = buy_and_hold_strategy()
    strategy.set_symbols(['SPY'])
    
    # Set dates
    end_date = date(2025, 8, 7)
    start_date = date(2025, 5, 9)
    
    # Load data first
    print(f"\nLoading data for SPY from {start_date} to {end_date}")
    await sim.market_replay.load_data(['SPY'], start_date, end_date, preload=True)
    
    # Run simulation with verbose logging
    sim.config.verbose = True
    print("\nStarting simulation...")
    
    result = await sim.run_simulation(
        portfolio=portfolio,
        start_date=start_date,
        end_date=end_date,
        strategy=strategy
    )
    
    print(f"\n=== RESULTS ===")
    print(f"Final value: ${result.final_value:,.2f}")
    print(f"Initial value: ${result.initial_value:,.2f}")
    print(f"Total return: {result.total_return:.2%}")
    print(f"Total trades: {result.total_trades}")
    print(f"Number of snapshots: {len(result.snapshots)}")
    print(f"Portfolio positions: {list(portfolio.positions.keys())}")
    print(f"Portfolio cash: ${portfolio.cash:,.2f}")
    
    # Check if any signals were generated
    if hasattr(strategy, '_signal_history'):
        print(f"\nSignal history: {strategy._signal_history}")
    
    # Print first few transactions
    if portfolio.transactions:
        print(f"\nFirst 5 transactions:")
        for i, tx in enumerate(portfolio.transactions[:5]):
            print(f"  {i+1}. {tx.timestamp.date()}: {tx.transaction_type} {tx.quantity} {tx.symbol} @ ${tx.price:.2f}")
    else:
        print("\nNo transactions recorded!")
    
    # Check first few snapshots
    if result.snapshots:
        print(f"\nFirst 3 snapshots:")
        for i, snap in enumerate(result.snapshots[:3]):
            print(f"  {i+1}. {snap.timestamp.date()}: Value=${snap.total_value:,.2f}, Cash=${snap.cash:,.2f}, Positions={len(snap.positions)}")

if __name__ == "__main__":
    asyncio.run(debug_backtest())