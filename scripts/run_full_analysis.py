#!/usr/bin/env python3
"""Run comprehensive strategy analysis across all time periods and scenarios."""

import asyncio
import sys
sys.path.append('/Users/angwei/Repos/tetra')

from src.analysis.strategy_analyzer import run_comprehensive_analysis, generate_analysis_report
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def main():
    """Run full comprehensive analysis."""
    
    print("="*80)
    print("COMPREHENSIVE STRATEGY PERFORMANCE ANALYSIS")
    print("="*80)
    print("\nThis analysis will:")
    print("1. Test each strategy across rolling windows (2w, 3m, 1y)")
    print("2. Analyze performance across entire historical period")
    print("3. Test strategies under different market scenarios")
    print("4. Rank strategies by risk-adjusted performance")
    print("\nAnalyzing 5 strategies on 5 major assets...")
    print("Expected runtime: 3-5 minutes")
    print("-"*80)
    
    # Run analysis
    results = await run_comprehensive_analysis(
        symbols=['SPY', 'QQQ', 'IWM', 'GLD', 'TLT'],  # Stocks, Gold, Bonds
        strategies=['buy_and_hold', 'golden_cross', 'turtle_trading', 'rsi_reversion', 'momentum_factor'],
        start_year=2020,
        end_year=2025
    )
    
    # Generate report
    report = generate_analysis_report(results)
    print("\n" + report)
    
    # Save results
    results['time_series_results'].to_csv('/tmp/comprehensive_strategy_analysis.csv', index=False)
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nDetailed results saved to:")
    print("  /tmp/comprehensive_strategy_analysis.csv")
    print(f"\nTotal backtests run: {len(results['time_series_results'])}")
    print(f"Market scenarios tested: {len(results['scenario_results'])}")
    
    # Print quick summary
    rankings = results['rankings']
    if not rankings.empty:
        print("\nTop 3 Strategies:")
        for i, (strategy, row) in enumerate(rankings.head(3).iterrows()):
            print(f"  {i+1}. {strategy}: Score={row['total_score']:.1f}, "
                  f"Avg Return={row['total_return_mean']*100:.1f}%, "
                  f"Sharpe={row['sharpe_ratio_mean']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())