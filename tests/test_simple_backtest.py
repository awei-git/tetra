#!/usr/bin/env python3
"""Test simple backtest with a single strategy and symbol."""

import asyncio
from datetime import datetime, timedelta
from src.simulators.historical import HistoricalSimulator, SimulatorConfig
from src.simulators.portfolio import Portfolio
from src.strats.benchmark import buy_and_hold_strategy
from src.utils.logging import logger


async def test_simple_backtest():
    """Test a simple backtest with buy and hold on SPY."""
    
    # Setup dates
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)  # 30 days
    
    logger.info(f"Testing backtest from {start_date} to {end_date}")
    
    # Create simulator config
    config = SimulatorConfig(
        starting_cash=100000,
        commission_per_share=0.005,
        slippage_bps=5,
        benchmark_symbol="SPY",
        use_adjusted_prices=True,
        allow_fractional_shares=True,
        include_dividends=False,  # Disable dividends for now
        include_splits=False,     # Disable splits for now
        verbose=True  # Enable verbose logging
    )
    
    # Create simulator
    simulator = HistoricalSimulator(config)
    await simulator.initialize()
    
    # Create portfolio
    portfolio = Portfolio(
        initial_cash=config.starting_cash,
        base_currency=config.base_currency,
        allow_fractional=config.allow_fractional_shares
    )
    
    # Get strategy
    strategy = buy_and_hold_strategy()
    strategy.set_symbols(["SPY"])  # Set universe to just SPY
    
    logger.info(f"Running backtest for {strategy.name} on SPY")
    
    try:
        # Run simulation
        result = await simulator.run_simulation(
            portfolio=portfolio,
            start_date=start_date,
            end_date=end_date,
            strategy=strategy
        )
        
        # Print results
        logger.info(f"Backtest completed successfully!")
        logger.info(f"Initial value: ${result.initial_value:,.2f}")
        logger.info(f"Final value: ${result.final_value:,.2f}")
        logger.info(f"Total return: {result.total_return:.2%}")
        logger.info(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"Max drawdown: {result.max_drawdown:.2%}")
        logger.info(f"Total trades: {result.total_trades}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_simple_backtest())