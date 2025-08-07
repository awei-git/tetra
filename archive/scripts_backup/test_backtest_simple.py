"""Simple test script for backtesting engine without database dependency."""

import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.strats.base import BaseStrategy, PositionSide, Trade
from src.backtesting.portfolio import Portfolio
from src.backtesting.data_handler import DataHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockDataHandler(DataHandler):
    """Mock data handler that generates synthetic data."""
    
    def load_market_data(self, symbols, start_date, end_date, frequency="1d"):
        """Generate synthetic market data."""
        logger.info(f"Generating synthetic data for {symbols}")
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate data for each symbol
        data = []
        for date in dates:
            for symbol in symbols:
                # Simple synthetic price generation
                base_price = 100 + hash(symbol) % 50
                noise = np.random.randn() * 2
                price = base_price + noise + (date - start_date).days * 0.1
                
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'open': price - 1,
                    'high': price + 2,
                    'low': price - 2,
                    'close': price,
                    'volume': 1000000 + np.random.randint(-500000, 500000),
                    'adj_close': price
                })
        
        df = pd.DataFrame(data)
        df.set_index(['timestamp', 'symbol'], inplace=True)
        return df
    
    def load_event_data(self, symbols, start_date, end_date):
        """Return empty event data."""
        return pd.DataFrame()
    
    def get_historical_data(self, symbol, end_date, periods, frequency="1d"):
        """Generate historical data for a single symbol."""
        start_date = end_date - timedelta(days=periods + 10)
        data = self.load_market_data([symbol], start_date, end_date, frequency)
        if not data.empty and symbol in data.index.get_level_values('symbol'):
            return data.xs(symbol, level='symbol').tail(periods)
        return pd.DataFrame()


from src.strats.base import StrategyState

class SimpleTestStrategy(BaseStrategy):
    """Simple test strategy for verification."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="Simple Test Strategy",
            description="Buy and hold test strategy",
            *args,
            **kwargs
        )
        self.entry_count = 0
        self.max_entries = 3  # Only enter 3 positions for testing
        # Initialize state if not already done by parent
        if not hasattr(self, 'state'):
            self.state = StrategyState(
                timestamp=datetime.now(),
                cash=self.initial_capital,
                total_value=self.initial_capital
            )
    
    def generate_signals(self, data, events=None):
        """Generate signals - not used in this test."""
        return pd.DataFrame()
    
    def should_enter(self, symbol, timestamp, bar_data, signals, events):
        """Simple entry logic - buy if we haven't reached max entries."""
        if self.entry_count < self.max_entries:
            self.entry_count += 1
            return True, PositionSide.LONG, 0.3  # 30% position size
        return False, None, 0
    
    def should_exit(self, position, timestamp, bar_data, signals, events):
        """Simple exit logic - hold for at least 10 days then exit."""
        # Get entry time from metadata or first trade
        entry_time = None
        if hasattr(position, 'metadata') and 'entry_time' in position.metadata:
            entry_time = position.metadata['entry_time']
        elif hasattr(position, 'trades') and position.trades and len(position.trades) > 0:
            entry_time = position.trades[0].entry_time
        
        if entry_time:
            days_held = (timestamp - entry_time).days
            should_exit = days_held >= 10  # Changed to >= for faster exits
            if should_exit:
                logger.info(f"Exiting {position.symbol} after {days_held} days")
            return should_exit
        
        # Default: exit after some bars if no entry time found
        logger.warning(f"No entry time found for {position.symbol}, exiting")
        return True
    
    def calculate_metrics(self):
        """Return custom strategy metrics."""
        return {
            'total_entries': self.entry_count,
            'strategy_type': 'test'
        }


def test_backtesting_engine():
    """Test the backtesting engine with synthetic data."""
    
    # Create backtest configuration
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 3, 1),  # 2 months for quick test
        initial_capital=100000,
        commission=0.001,
        slippage=0.0001,
        max_positions=5,
        calculate_metrics_every=10,
        benchmark=None  # No benchmark for this test
    )
    
    # Create engine with mock data handler
    engine = BacktestEngine(config)
    engine.data_handler = MockDataHandler()  # Replace with mock
    
    # Define test symbols
    symbols = ['TEST1', 'TEST2', 'TEST3']
    
    # Run backtest
    logger.info("Starting test backtest...")
    report = engine.run(
        strategy=SimpleTestStrategy,
        symbols=symbols,
        signal_computer=None  # No signals for this test
    )
    
    # Print results
    print("\n" + "="*60)
    print("TEST BACKTEST RESULTS")
    print("="*60)
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"Final Equity: ${report.final_equity:,.2f}")
    print(f"Total Return: {report.total_return:.2%}")
    print(f"Total Trades: {report.total_trades}")
    print(f"Max Drawdown: {report.max_drawdown:.2%}")
    print(f"Strategy Metrics: {report.strategy_metrics}")
    
    # Verify basic functionality
    assert report.final_equity > 0, "Final equity should be positive"
    assert report.total_trades > 0, "Should have executed some trades"
    assert len(report.equity_curve) > 0, "Should have equity curve data"
    
    print("\n✅ Backtesting engine test passed!")
    
    return report


if __name__ == "__main__":
    try:
        report = test_backtesting_engine()
        logger.info("Test completed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise