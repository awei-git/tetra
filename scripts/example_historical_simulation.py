#!/usr/bin/env python3
"""
Example usage of the historical market simulator.

This script demonstrates how to:
1. Create a portfolio
2. Run a historical simulation
3. Simulate specific market events
4. Analyze performance results
"""

import asyncio
from datetime import date, datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.simulators import (
    HistoricalSimulator,
    Portfolio,
    SimulationConfig,
    EVENT_PERIODS
)


async def basic_simulation_example():
    """Run a basic historical simulation."""
    print("=== Basic Historical Simulation ===\n")
    
    # Create simulation config
    config = SimulationConfig(
        starting_cash=100000,
        slippage_bps=10,
        commission_per_share=0.005,
        include_dividends=True
    )
    
    # Initialize simulator
    simulator = HistoricalSimulator(config)
    
    # Create portfolio with initial positions
    portfolio = Portfolio(initial_cash=100000)
    
    # Add some initial positions
    initial_positions = [
        ("AAPL", 100, 150.0),  # 100 shares of AAPL at $150
        ("SPY", 50, 400.0),     # 50 shares of SPY at $400
        ("MSFT", 75, 250.0),    # 75 shares of MSFT at $250
    ]
    
    for symbol, quantity, price in initial_positions:
        portfolio.add_position(
            symbol=symbol,
            quantity=quantity,
            price=price,
            timestamp=datetime(2023, 1, 1),
            commission=config.calculate_commission(quantity, price)
        )
    
    print(f"Initial portfolio value: ${portfolio.get_total_value({}):.2f}")
    print(f"Cash remaining: ${portfolio.cash:.2f}")
    print("\nInitial positions:")
    for symbol, position in portfolio.positions.items():
        print(f"  {symbol}: {position.quantity} shares @ ${position.entry_price:.2f}")
    
    # Run simulation for 2023
    print("\nRunning simulation for 2023...")
    result = await simulator.run_simulation(
        portfolio=portfolio,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31)
    )
    
    # Print results
    print("\n" + result.summary())
    
    # Clean up
    await simulator.cleanup()


async def event_simulation_example():
    """Simulate specific market events."""
    print("\n=== Event-Based Simulation ===\n")
    
    # Create config for volatile events
    config = SimulationConfig(
        starting_cash=100000,
        slippage_bps=20,  # Higher slippage during volatility
        allow_short_selling=True
    )
    
    simulator = HistoricalSimulator(config)
    
    # Test different events
    events_to_test = ["covid_crash", "svb_collapse", "gme_squeeze"]
    
    for event_name in events_to_test:
        print(f"\n--- Simulating {event_name} ---")
        
        # Create fresh portfolio for each event
        portfolio = Portfolio(initial_cash=100000)
        
        # Add diversified positions
        positions = [
            ("SPY", 100, 300.0),
            ("QQQ", 50, 250.0),
            ("IWM", 75, 150.0),
        ]
        
        # For specific events, add relevant stocks
        if event_name == "svb_collapse":
            positions.append(("KRE", 100, 50.0))  # Regional banks ETF
        elif event_name == "gme_squeeze":
            positions.append(("GME", 10, 20.0))   # GameStop
            
        # Add positions (using event start date for entry)
        event = EVENT_PERIODS[event_name]
        for symbol, quantity, price in positions:
            try:
                portfolio.add_position(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    timestamp=datetime.combine(event.start_date, datetime.min.time()),
                    commission=5.0
                )
            except Exception as e:
                print(f"  Warning: Could not add {symbol}: {e}")
        
        # Run event simulation
        try:
            result = await simulator.simulate_event(
                event_name=event_name,
                portfolio=portfolio,
                context_days=30  # Include 30 days before event
            )
            
            print(f"  Period: {result.start_date} to {result.end_date}")
            print(f"  Total Return: {result.total_return:.2%}")
            print(f"  Max Drawdown: {result.max_drawdown:.2%}")
            print(f"  Volatility: {result.volatility:.2%}")
            
        except Exception as e:
            print(f"  Error simulating {event_name}: {e}")
    
    await simulator.cleanup()


async def portfolio_comparison_example():
    """Compare different portfolio strategies."""
    print("\n=== Portfolio Strategy Comparison ===\n")
    
    config = SimulationConfig(starting_cash=100000)
    simulator = HistoricalSimulator(config)
    
    # Define different portfolio strategies
    strategies = {
        "Conservative": [
            ("SPY", 300, 300.0),   # 60% stocks
            ("TLT", 200, 100.0),   # 40% bonds
        ],
        "Aggressive": [
            ("QQQ", 200, 250.0),   # 50% tech
            ("IWM", 200, 150.0),   # 30% small cap
            ("EEM", 100, 40.0),    # 20% emerging markets
        ],
        "Balanced": [
            ("SPY", 150, 300.0),   # 45% large cap
            ("TLT", 100, 100.0),   # 20% bonds
            ("QQQ", 100, 250.0),   # 25% tech
            ("GLD", 50, 150.0),    # 10% gold
        ]
    }
    
    # Simulate each strategy
    simulation_period = (date(2022, 1, 1), date(2023, 12, 31))
    
    for strategy_name, positions in strategies.items():
        print(f"\n--- {strategy_name} Portfolio ---")
        
        # Create portfolio
        portfolio = Portfolio(initial_cash=100000)
        
        # Add positions
        for symbol, quantity, price in positions:
            try:
                portfolio.add_position(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    timestamp=datetime(2022, 1, 1),
                    commission=5.0
                )
            except Exception as e:
                print(f"  Warning: Could not add {symbol}: {e}")
        
        # Run simulation
        try:
            result = await simulator.run_simulation(
                portfolio=portfolio,
                start_date=simulation_period[0],
                end_date=simulation_period[1]
            )
            
            print(f"  Total Return: {result.total_return:.2%}")
            print(f"  Annual Return: {result.annual_return:.2%}")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {result.max_drawdown:.2%}")
            
        except Exception as e:
            print(f"  Error simulating {strategy_name}: {e}")
    
    await simulator.cleanup()


async def main():
    """Run all examples."""
    try:
        # Note: These examples require market data in the database
        print("Historical Simulator Examples")
        print("=============================\n")
        
        # Uncomment the examples you want to run:
        
        # await basic_simulation_example()
        # await event_simulation_example()
        # await portfolio_comparison_example()
        
        print("\nNote: These examples require market data in your database.")
        print("Make sure you have run the data pipeline to populate historical data.")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. The database is running (docker-compose up)")
        print("2. Historical data has been loaded")
        print("3. The required symbols are available")


if __name__ == "__main__":
    asyncio.run(main())