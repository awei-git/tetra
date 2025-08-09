#!/usr/bin/env python3
"""Run comprehensive strategy backtest."""

import asyncio
import sys
sys.path.append('/Users/angwei/Repos/tetra')

from src.pipelines.benchmark_pipeline.comprehensive_backtest import run_comprehensive_backtest

async def main():
    """Run the comprehensive backtest."""
    print("Starting Comprehensive Strategy Backtest")
    print("This will test strategies across multiple time windows and market scenarios...")
    print("Expected runtime: 5-10 minutes")
    print("-" * 60)
    
    report = await run_comprehensive_backtest()
    print(report)
    
    print("\n" + "-" * 60)
    print("Backtest complete! Results saved to /tmp/comprehensive_backtest_results.csv")

if __name__ == "__main__":
    asyncio.run(main())