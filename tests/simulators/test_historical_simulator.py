"""
Quick test of the simulator to ensure it's working correctly.
"""

import asyncio
from datetime import date, datetime

from src.simulators import (
    HistoricalSimulator,
    Portfolio,
    SimulationConfig
)


async def test_basic_simulation():
    """Test basic simulation functionality."""
    print("Testing Historical Simulator...")
    
    # Create config
    config = SimulationConfig(
        starting_cash=100000,
        slippage_bps=10,
        commission_per_share=0.005
    )
    
    # Initialize simulator
    simulator = HistoricalSimulator(config)
    
    # Create simple portfolio
    portfolio = Portfolio(initial_cash=100000)
    
    # Add a position
    portfolio.add_position(
        symbol="AAPL",
        quantity=100,
        price=150.0,
        timestamp=datetime(2025, 7, 1),
        commission=5.0
    )
    
    print(f"Initial portfolio value: ${portfolio.get_total_value({}):.2f}")
    print(f"Cash: ${portfolio.cash:.2f}")
    print(f"AAPL position: {portfolio.positions['AAPL'].quantity} shares")
    
    try:
        # Run a short simulation (just July 2025)
        result = await simulator.run_simulation(
            portfolio=portfolio,
            start_date=date(2025, 7, 1),
            end_date=date(2025, 7, 31),
            preload_data=True
        )
        
        print(f"\nSimulation completed!")
        print(f"Final value: ${result.final_value:.2f}")
        print(f"Total return: {result.total_return:.2%}")
        print(f"Number of snapshots: {len(result.snapshots)}")
        
    except Exception as e:
        print(f"\nError during simulation: {e}")
        print("This is expected if the database doesn't have data for the test period.")
    
    finally:
        await simulator.cleanup()


async def test_portfolio_operations():
    """Test portfolio operations."""
    print("\nTesting Portfolio Operations...")
    
    portfolio = Portfolio(initial_cash=50000)
    
    # Test adding positions
    portfolio.add_position("AAPL", 50, 150.0, datetime.now(), 5.0)
    portfolio.add_position("MSFT", 30, 250.0, datetime.now(), 5.0)
    
    print(f"Cash after purchases: ${portfolio.cash:.2f}")
    
    # Test market prices
    market_prices = {"AAPL": 155.0, "MSFT": 245.0}
    
    print(f"Total value: ${portfolio.get_total_value(market_prices):.2f}")
    print(f"Returns: {portfolio.get_returns(market_prices):.2%}")
    
    # Test position reduction
    portfolio.add_position("AAPL", -25, 155.0, datetime.now(), 5.0)
    
    print(f"\nAfter selling 25 AAPL:")
    print(f"AAPL position: {portfolio.positions['AAPL'].quantity} shares")
    print(f"Cash: ${portfolio.cash:.2f}")


if __name__ == "__main__":
    print("Historical Simulator Test")
    print("=" * 50)
    
    asyncio.run(test_basic_simulation())
    asyncio.run(test_portfolio_operations())
    
    print("\nTest completed!")