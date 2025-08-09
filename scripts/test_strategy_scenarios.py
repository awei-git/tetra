#!/usr/bin/env python3
"""Test buy_and_hold strategy across different market scenarios."""

import asyncio
from datetime import date, timedelta
import sys
sys.path.append('/Users/angwei/Repos/tetra')

from src.simulators.historical import HistoricalSimulator
from src.simulators.portfolio import Portfolio
from src.strats.benchmark import buy_and_hold_strategy
from src.simulators.historical.event_periods import EVENT_PERIODS

async def test_scenario(event_name: str, symbol: str = 'SPY'):
    """Test strategy during a specific market event."""
    event = EVENT_PERIODS[event_name]
    
    # Create simulator and portfolio
    sim = HistoricalSimulator()
    portfolio = Portfolio(initial_cash=100000.0)
    
    # Create strategy
    strategy = buy_and_hold_strategy()
    strategy.set_symbols([symbol])
    
    # Add some context before event
    start_date = event.start_date - timedelta(days=30)
    end_date = event.end_date + timedelta(days=30)
    
    print(f"\nTesting {event.name} ({event.start_date} to {event.end_date})")
    print(f"Description: {event.description}")
    print(f"Volatility multiplier: {event.volatility_multiplier}x")
    
    try:
        result = await sim.run_simulation(
            portfolio=portfolio,
            start_date=start_date,
            end_date=end_date,
            strategy=strategy
        )
        
        # Find max drawdown during event period
        event_snapshots = [s for s in result.snapshots 
                          if event.start_date <= s.timestamp.date() <= event.end_date]
        
        if event_snapshots:
            event_values = [s.total_value for s in event_snapshots]
            max_value = max(event_values[:1] + [s.total_value for s in result.snapshots 
                                               if s.timestamp.date() < event.start_date])
            min_value = min(event_values)
            event_drawdown = (min_value - max_value) / max_value if max_value > 0 else 0
        else:
            event_drawdown = 0
        
        print(f"Results:")
        print(f"  Total return: {result.total_return:.2%}")
        print(f"  Max drawdown: {result.max_drawdown:.2%}")
        print(f"  Event drawdown: {event_drawdown:.2%}")
        print(f"  Sharpe ratio: {result.sharpe_ratio:.2f}")
        
        return {
            'event': event_name,
            'return': result.total_return,
            'max_dd': result.max_drawdown,
            'event_dd': event_drawdown,
            'sharpe': result.sharpe_ratio
        }
        
    except Exception as e:
        print(f"  Error: {e}")
        return None

async def main():
    """Test strategy across multiple scenarios."""
    print("Testing Buy & Hold Strategy Across Market Scenarios")
    print("=" * 60)
    
    # Test critical scenarios
    scenarios = [
        ('covid_crash', 'SPY'),        # Extreme volatility and crash
        ('financial_crisis', 'SPY'),   # Prolonged bear market
        ('dotcom_crash', 'QQQ'),       # Tech bubble burst
        ('fed_taper_2022', 'SPY'),     # Rising rates environment
        ('flash_crash', 'SPY'),        # Intraday volatility
        ('gme_squeeze', 'IWM'),        # Small-cap volatility
        ('svb_collapse', 'IWM'),       # Banking crisis impact on small-caps
    ]
    
    results = []
    for event_name, symbol in scenarios:
        result = await test_scenario(event_name, symbol)
        if result:
            results.append(result)
    
    # Summary statistics
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        
        returns = [r['return'] for r in results]
        max_dds = [r['max_dd'] for r in results]
        event_dds = [r['event_dd'] for r in results]
        
        print(f"Average return: {sum(returns)/len(returns):.2%}")
        print(f"Best scenario: {max(returns):.2%} ({results[returns.index(max(returns))]['event']})")
        print(f"Worst scenario: {min(returns):.2%} ({results[returns.index(min(returns))]['event']})")
        print(f"Average max drawdown: {sum(max_dds)/len(max_dds):.2%}")
        print(f"Worst drawdown: {min(max_dds):.2%} ({results[max_dds.index(min(max_dds))]['event']})")
        print(f"Win rate: {len([r for r in returns if r > 0])/len(returns):.1%}")

if __name__ == "__main__":
    asyncio.run(main())