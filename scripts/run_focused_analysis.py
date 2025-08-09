#!/usr/bin/env python3
"""Run focused strategy analysis on key symbols and strategies."""

import asyncio
import sys
sys.path.append('/Users/angwei/Repos/tetra')

from src.analysis.strategy_analyzer import StrategyAnalyzer, WindowSize, MarketScenario
from src.strats.benchmark import get_all_benchmarks
from datetime import date, timedelta
import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def main():
    """Run focused analysis on key strategy-symbol combinations."""
    
    print("="*80)
    print("FOCUSED STRATEGY PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Focus on key symbols and strategies
    symbols = ['SPY', 'QQQ']  # Major indices
    strategies = ['buy_and_hold', 'golden_cross', 'momentum_factor']  # Top strategies
    
    all_strategies = get_all_benchmarks()
    analyzer = StrategyAnalyzer(symbols, all_strategies)
    
    print("\n1. RECENT PERFORMANCE (Last 3 Months)")
    print("-"*40)
    
    # Test recent performance
    end_date = date(2025, 8, 7)
    start_date = end_date - timedelta(days=90)
    
    for strategy_name in strategies:
        for symbol in symbols:
            try:
                # Run single backtest
                metrics = await analyzer._run_single_backtest(
                    strategy_name,
                    all_strategies[strategy_name],
                    start_date,
                    end_date,
                    "3_months"
                )
                
                if metrics.success:
                    print(f"{strategy_name} on {symbol}: {metrics.total_return:.2%} return, "
                          f"{metrics.sharpe_ratio:.2f} Sharpe, {metrics.total_trades} trades")
                else:
                    print(f"{strategy_name} on {symbol}: FAILED - {metrics.error}")
                    
            except Exception as e:
                print(f"{strategy_name} on {symbol}: ERROR - {e}")
    
    print("\n2. ROLLING WINDOW ANALYSIS (Sample)")
    print("-"*40)
    
    # Test one strategy across rolling windows
    test_strategy = 'buy_and_hold'
    test_symbol = 'SPY'
    
    print(f"\nTesting {test_strategy} on {test_symbol} with 1-year rolling windows:")
    
    df = await analyzer.analyze_strategy_full_history(
        test_strategy,
        test_symbol,
        WindowSize.ONE_YEAR,
        start_year=2023,
        end_year=2025,
        step_days=90  # Quarterly steps
    )
    
    if not df.empty:
        print(f"  Periods tested: {len(df)}")
        print(f"  Average return: {df['total_return'].mean():.2%}")
        print(f"  Best period: {df['total_return'].max():.2%}")
        print(f"  Worst period: {df['total_return'].min():.2%}")
        print(f"  Win rate: {(df['total_return'] > 0).mean():.1%}")
    
    print("\n3. MARKET SCENARIO TESTING")
    print("-"*40)
    
    # Test under different scenarios
    scenarios = [MarketScenario.BULL_MARKET, MarketScenario.BEAR_MARKET, MarketScenario.HIGH_VOLATILITY]
    
    scenario_start = date(2024, 6, 1)
    scenario_end = date(2025, 6, 1)
    
    for strategy_name in strategies[:2]:  # Test first 2 strategies
        print(f"\n{strategy_name}:")
        results = await analyzer.test_market_scenarios(
            strategy_name,
            'SPY',
            scenario_start,
            scenario_end,
            scenarios
        )
        
        for result in results:
            print(f"  {result.scenario.value:15}: {result.metrics.total_return:.2%} return, "
                  f"{result.metrics.sharpe_ratio:.2f} Sharpe")
    
    print("\n4. QUICK RANKING")
    print("-"*40)
    
    # Collect recent results for ranking
    recent_results = []
    for strategy_name in strategies:
        for symbol in symbols:
            try:
                metrics = await analyzer._run_single_backtest(
                    strategy_name,
                    all_strategies[strategy_name],
                    start_date,
                    end_date,
                    "3_months"
                )
                if metrics.success:
                    recent_results.append(metrics.__dict__)
            except:
                pass
    
    if recent_results:
        df = pd.DataFrame(recent_results)
        rankings = analyzer.rank_strategies(df)
        
        print("\nStrategy Rankings (based on recent 3-month performance):")
        for idx, (strategy, row) in enumerate(rankings.iterrows()):
            if idx < 3:
                print(f"  #{int(row['rank'])}. {strategy}: Score={row['total_score']:.1f}, "
                      f"Avg Return={row['total_return_mean']*100:.1f}%")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())