#!/usr/bin/env python3
"""Quick test of comprehensive backtest framework."""

import asyncio
from datetime import date, timedelta
import sys
sys.path.append('/Users/angwei/Repos/tetra')

from src.pipelines.benchmark_pipeline.comprehensive_backtest import ComprehensiveBacktester

async def quick_test():
    """Run a quick test with limited scenarios."""
    tester = ComprehensiveBacktester()
    
    # Test just a few scenarios
    strategies = ["buy_and_hold", "golden_cross", "turtle_trading"]
    
    # Test 3 scenarios: bull, bear, and recent
    scenarios = [
        ("SPY", date(2020, 2, 20), date(2020, 4, 30), "covid_crash"),  # Bear
        ("SPY", date(2020, 3, 24), date(2021, 1, 8), "post_covid_rally"),  # Bull
        ("IWM", date(2025, 5, 9), date(2025, 8, 7), "recent_period"),  # Recent
    ]
    
    print("Running quick test with 3 strategies × 3 scenarios = 9 backtests")
    print("-" * 60)
    
    for strategy_name in strategies:
        for symbol, start_date, end_date, scenario_name in scenarios:
            result = await tester.run_single_backtest(
                strategy_name=strategy_name,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                window_name="test",
                scenario_name=scenario_name
            )
            
            if result.success:
                print(f"{strategy_name} on {symbol} during {scenario_name}:")
                print(f"  Return: {result.total_return*100:.1f}%")
                print(f"  Sharpe: {result.sharpe_ratio:.2f}")
                print(f"  Max DD: {result.max_drawdown*100:.1f}%")
                print(f"  Trades: {result.total_trades}")
            else:
                print(f"{strategy_name} on {symbol} during {scenario_name}: FAILED - {result.error}")
            print()
            tester.results.append(result)
    
    # Analyze results
    stats = tester.analyze_results()
    report = tester.generate_report(stats)
    print("\n" + "="*60)
    print(report)

if __name__ == "__main__":
    asyncio.run(quick_test())