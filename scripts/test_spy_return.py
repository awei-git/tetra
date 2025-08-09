#!/usr/bin/env python3
"""Test SPY returns specifically."""

import asyncio
from datetime import date
import sys
sys.path.append('/Users/angwei/Repos/tetra')

from src.simulators.historical import HistoricalSimulator
from src.simulators.portfolio import Portfolio
from src.strats.benchmark import buy_and_hold_strategy
from src.simulators.historical.market_replay import MarketReplay

async def test_spy():
    """Test SPY specifically."""
    
    # Test dates
    start_date = date(2025, 5, 9)
    end_date = date(2025, 8, 7)
    
    # First, check raw price data
    replay = MarketReplay()
    await replay.load_data(['SPY'], start_date, end_date)
    
    spy_data = await replay._load_symbol_data('SPY', start_date, end_date)
    print(f"SPY data points: {len(spy_data)}")
    print(f"First price: ${spy_data['close'].iloc[0]:.2f} on {spy_data.index[0].date()}")
    print(f"Last price: ${spy_data['close'].iloc[-1]:.2f} on {spy_data.index[-1].date()}")
    raw_return = (spy_data['close'].iloc[-1] - spy_data['close'].iloc[0]) / spy_data['close'].iloc[0]
    print(f"Raw price return: {raw_return:.2%}")
    
    print("\n" + "-"*60 + "\n")
    
    # Now test buy & hold strategy
    sim = HistoricalSimulator()
    portfolio = Portfolio(initial_cash=100000.0)
    strategy = buy_and_hold_strategy()
    strategy.set_symbols(['SPY'])
    
    print("Running buy & hold backtest...")
    result = await sim.run_simulation(
        portfolio=portfolio,
        start_date=start_date,
        end_date=end_date,
        strategy=strategy
    )
    
    print(f"\nBacktest Results:")
    print(f"Initial value: ${result.initial_value:,.2f}")
    print(f"Final value: ${result.final_value:,.2f}")
    print(f"Total return: {result.total_return:.2%}")
    print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
    print(f"Total trades: {result.total_trades}")
    
    # Check portfolio details
    print(f"\nPortfolio state:")
    print(f"Cash: ${portfolio.cash:,.2f}")
    print(f"Positions: {list(portfolio.positions.keys())}")
    if 'SPY' in portfolio.positions:
        pos = portfolio.positions['SPY']
        print(f"SPY shares: {pos.quantity}")
        print(f"Entry price: ${pos.entry_price:.2f}")
        print(f"Current value: ${pos.get_market_value(spy_data['close'].iloc[-1]):.2f}")

if __name__ == "__main__":
    asyncio.run(test_spy())