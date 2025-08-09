#!/usr/bin/env python3
"""Test pipeline execution directly."""

import asyncio
from datetime import date
import sys
sys.path.append('/Users/angwei/Repos/tetra')

from src.pipelines.benchmark_pipeline.steps.strategy_backtest import StrategyBacktestStep
from src.pipelines.benchmark_pipeline.pipeline import PipelineContext
from src.simulators.historical import HistoricalSimulator
from src.strats.benchmark import buy_and_hold_strategy

async def test_pipeline_execution():
    """Test how the pipeline executes strategies."""
    
    # Create context like the pipeline does
    context = PipelineContext()
    context.data = {
        "start_date": date(2025, 5, 9),
        "end_date": date(2025, 8, 7),
        "strategies": {"buy_and_hold": buy_and_hold_strategy()},
        "symbols": ["SPY"],
        "simulator": HistoricalSimulator()
    }
    
    # Run the backtest step
    backtest_step = StrategyBacktestStep()
    await backtest_step.execute(context)
    
    # Check results
    results = context.data.get("backtest_results", {})
    print("Backtest Results:")
    for name, result in results.items():
        if result.get("status") == "success":
            print(f"\n{name}:")
            print(f"  Total Return: {result['total_return']:.2%}")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"  Final Value: ${result['final_value']:,.2f}")
        else:
            print(f"\n{name}: FAILED - {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(test_pipeline_execution())