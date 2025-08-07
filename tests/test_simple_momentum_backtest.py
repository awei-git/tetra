#!/usr/bin/env python3
"""Test simple momentum backtest with a single strategy and symbol."""

import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.simulators.historical import HistoricalSimulator, SimulatorConfig
from src.simulators.portfolio import Portfolio
from src.strats.signal_based import SignalBasedStrategy, SignalRule, SignalCondition, ConditionOperator, PositionSide
from src.utils.logging import logger


def create_simple_momentum_strategy():
    """Create a simple momentum strategy that buys on positive momentum signal."""
    rules = [
        SignalRule(
            name="simple_momentum",
            entry_conditions=[
                SignalCondition("momentum_signal", ConditionOperator.GREATER_THAN, 0),  # Positive momentum
                SignalCondition("volume", ConditionOperator.GREATER_THAN, 1000000)  # Minimum volume
            ],
            exit_conditions=[
                SignalCondition("momentum_signal", ConditionOperator.LESS_THAN, 0)  # Negative momentum
            ],
            position_side=PositionSide.LONG,
            position_size_factor=1.0
        )
    ]
    
    strategy = SignalBasedStrategy(
        name="Simple Momentum Strategy",
        signal_rules=rules,
        initial_capital=100000,
        position_size=0.95,  # Use 95% of capital
        max_positions=1
    )
    
    # Add a simple calculate_indicators method
    def calculate_indicators(historical_data):
        """Calculate SMA20 for the strategy."""
        # Get the first symbol's data
        if historical_data:
            symbol = list(historical_data.keys())[0]
            data = historical_data[symbol]
            
            # Create signals DataFrame from the price series
            # Assuming data is a price series, we'll create a DataFrame
            if isinstance(data, pd.Series):
                signals = pd.DataFrame(index=data.index)
                signals['close'] = data
                sma_20 = data.rolling(window=20).mean()
                # Create momentum signal: 1 when close > sma20, -1 when close < sma20
                signals['momentum_signal'] = 0
                signals.loc[data > sma_20, 'momentum_signal'] = 1
                signals.loc[data < sma_20, 'momentum_signal'] = -1
                signals['volume'] = 10000000  # Dummy volume for now
            else:
                # If it's already a DataFrame
                signals = pd.DataFrame(index=data.index)
                signals['close'] = data['close'] if 'close' in data else data
                sma_20 = signals['close'].rolling(window=20).mean()
                signals['momentum_signal'] = 0
                signals.loc[signals['close'] > sma_20, 'momentum_signal'] = 1
                signals.loc[signals['close'] < sma_20, 'momentum_signal'] = -1
                signals['volume'] = data['volume'] if 'volume' in data else 10000000
            
            return signals
            
        return pd.DataFrame()
    
    # Monkey patch the method
    strategy.calculate_indicators = calculate_indicators
    
    return strategy


async def test_momentum_backtest():
    """Test a simple momentum backtest."""
    
    # Setup dates
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=90)  # 90 days for enough data
    
    logger.info(f"Testing momentum backtest from {start_date} to {end_date}")
    
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
    strategy = create_simple_momentum_strategy()
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
        
        # Calculate annualized return properly
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (1 + result.total_return) ** (1 / years) - 1 if years > 0 else 0
        logger.info(f"Annualized return: {annualized_return:.2%}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_momentum_backtest())