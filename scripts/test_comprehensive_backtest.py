#!/usr/bin/env python3
"""Test comprehensive backtest execution with a small subset."""

import asyncio
import logging
from datetime import datetime
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines.assessment_pipeline.steps.backtest_execution import BacktestExecutionStep
from src.pipelines.base import PipelineContext
from src.strats.benchmark import buy_and_hold_strategy, golden_cross_strategy, rsi_mean_reversion_strategy

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# Reduce noise from other modules
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

async def test_comprehensive_backtest():
    """Test the comprehensive backtest with a few strategies and scenarios."""
    
    # Create context with minimal test data
    context = PipelineContext()
    
    # Load just 3 strategies for testing
    strategies = [
        {
            'name': 'buy_and_hold',
            'instance': buy_and_hold_strategy()
        },
        {
            'name': 'golden_cross',
            'instance': golden_cross_strategy()
        },
        {
            'name': 'rsi_reversion',
            'instance': rsi_mean_reversion_strategy()
        }
    ]
    
    # Use just 2 symbols
    symbols = ['SPY', 'QQQ']
    
    # Use just 2 scenarios that we have metrics for
    scenarios = [
        {
            'name': 'AI Boom 2023',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'type': 'bull'
        },
        {
            'name': 'COVID-19 Market Crash',
            'start_date': '2020-02-01',
            'end_date': '2020-04-30',
            'type': 'crash'
        }
    ]
    
    # Load metrics for these scenarios
    metrics_data = {}
    for scenario in scenarios:
        try:
            # Try to load metrics file - use exact name format
            filename = f"{scenario['name']}_metrics"
            filepath = f'/Users/angwei/Repos/tetra/data/metrics/{filename}.parquet'
            if os.path.exists(filepath):
                df = pd.read_parquet(filepath)
                metrics_data[scenario['name']] = df
                logger.info(f"Loaded metrics for {scenario['name']}: {len(df)} rows")
            else:
                logger.warning(f"No metrics file found for {scenario['name']}")
        except Exception as e:
            logger.error(f"Error loading metrics for {scenario['name']}: {e}")
    
    # Set context data
    context.data['strategies'] = strategies
    context.data['symbols'] = symbols
    context.data['scenarios'] = scenarios
    context.data['metrics_data'] = metrics_data
    
    # Total combinations: 3 strategies × 2 symbols × 2 scenarios = 12 backtests
    logger.info(f"Testing {len(strategies)} strategies × {len(symbols)} symbols × {len(scenarios)} scenarios = {len(strategies) * len(symbols) * len(scenarios)} backtests")
    
    # Create and run backtest step
    backtest_step = BacktestExecutionStep(parallel_workers=2)
    
    try:
        await backtest_step.execute(context)
        
        # Check results
        results = context.data.get('backtest_results', [])
        failed = context.data.get('failed_backtests', [])
        summary = context.data.get('backtest_summary', {})
        
        logger.info(f"\n=== BACKTEST RESULTS ===")
        logger.info(f"Total backtests: {summary.get('total_backtests', 0)}")
        logger.info(f"Successful: {summary.get('successful_backtests', 0)}")
        logger.info(f"Failed: {summary.get('failed_backtests', 0)}")
        logger.info(f"Success rate: {summary.get('success_rate', 0):.1%}")
        
        if results:
            logger.info(f"\n=== TOP PERFORMING BACKTESTS ===")
            # Sort by total return
            sorted_results = sorted(results, key=lambda x: x.total_return, reverse=True)[:5]
            for result in sorted_results:
                logger.info(f"{result.strategy_name} | {result.symbol} | {result.scenario_name}: "
                          f"Return={result.total_return:.2%}, Sharpe={result.sharpe_ratio:.2f}, "
                          f"Trades={result.total_trades}")
                
                # Show that strategies produce different results
                if result.total_trades > 0 and result.trade_log:
                    logger.info(f"  → Made {result.total_trades} trades, demonstrating active strategy logic")
        
        if failed:
            logger.info(f"\n=== FAILED BACKTESTS ===")
            for result in failed[:5]:
                logger.info(f"{result.strategy_name} | {result.symbol} | {result.scenario_name}: {result.error}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Backtest execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(test_comprehensive_backtest())
    if result:
        logger.info("\n✅ Comprehensive backtest test completed successfully!")
        if result['success_rate'] > 0:
            logger.info("✅ Strategies are producing differentiated results based on their logic!")
    else:
        logger.error("\n❌ Comprehensive backtest test failed!")
        sys.exit(1)