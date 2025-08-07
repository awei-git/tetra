"""Simple backtest without signals to test database connection."""

import logging
from datetime import datetime, timedelta
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.strats.base import BaseStrategy, PositionSide

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BuyAndHoldStrategy(BaseStrategy):
    """Simple buy and hold strategy."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="Buy and Hold",
            description="Buy on day 1 and hold until end",
            *args,
            **kwargs
        )
        self.bought = set()  # Track which symbols we've bought
        
    def generate_signals(self, data, events=None):
        """Generate signals - not used."""
        return pd.DataFrame()
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events):
        """Buy once for each symbol."""
        if symbol not in self.bought and len(self.bought) < 3:
            self.bought.add(symbol)
            logger.info(f"Buying {symbol} at {timestamp}")
            return True, PositionSide.LONG, 0.3  # 30% position size
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events):
        """Never exit - hold until end."""
        return False


class SimpleTimedStrategy(BaseStrategy):
    """Strategy that buys after 10 days and sells after 30 days."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="Simple Timed",
            description="Buy after 10 days, sell after 30 days",
            *args,
            **kwargs
        )
        self.start_date = None
        self.positions_entered = {}
        
    def generate_signals(self, data, events=None):
        """Generate signals - not used."""
        return pd.DataFrame()
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events):
        """Enter after 10 days."""
        if self.start_date is None:
            self.start_date = timestamp
            
        days_elapsed = (timestamp - self.start_date).days
        
        if symbol not in self.positions_entered and days_elapsed >= 10:
            self.positions_entered[symbol] = timestamp
            logger.info(f"Entering {symbol} at {timestamp} (day {days_elapsed})")
            return True, PositionSide.LONG, 0.2  # 20% position size
            
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events):
        """Exit after holding for 30 days."""
        if position.symbol in self.positions_entered:
            entry_time = self.positions_entered[position.symbol]
            holding_days = (timestamp - entry_time).days
            
            if holding_days >= 30:
                logger.info(f"Exiting {position.symbol} at {timestamp} after {holding_days} days")
                return True
                
        return False


def run_simple_backtest():
    """Run a simple backtest with database data."""
    
    # Use a shorter date range
    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),  # 3 months
        initial_capital=100000,
        commission=0.001,
        slippage=0.0001,
        max_positions=5,
        calculate_metrics_every=20,
        save_trades=True,
        benchmark=None
    )
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Create and run backtest
    engine = BacktestEngine(config=config)
    
    print("\n" + "="*60)
    print("Testing Buy and Hold Strategy")
    print("="*60)
    
    try:
        report1 = engine.run(
            strategy=BuyAndHoldStrategy,
            symbols=symbols,
            signal_computer=None  # No signals
        )
        
        print(f"Initial Capital: ${config.initial_capital:,.2f}")
        print(f"Final Equity: ${report1.final_equity:,.2f}")
        print(f"Total Return: {report1.total_return:.2%}")
        print(f"Max Drawdown: {report1.max_drawdown:.2%}")
        print(f"Total Trades: {report1.total_trades}")
        
    except Exception as e:
        logger.error(f"Buy and Hold backtest failed: {e}")
        report1 = None
    
    print("\n" + "="*60)
    print("Testing Simple Timed Strategy")
    print("="*60)
    
    # Reset engine for second test
    engine = BacktestEngine(config=config)
    
    try:
        report2 = engine.run(
            strategy=SimpleTimedStrategy,
            symbols=symbols,
            signal_computer=None  # No signals
        )
        
        print(f"Initial Capital: ${config.initial_capital:,.2f}")
        print(f"Final Equity: ${report2.final_equity:,.2f}")
        print(f"Total Return: {report2.total_return:.2%}")
        print(f"Max Drawdown: {report2.max_drawdown:.2%}")
        print(f"Total Trades: {report2.total_trades}")
        
    except Exception as e:
        logger.error(f"Simple Timed backtest failed: {e}")
        report2 = None
    
    print("="*60)
    
    return report1, report2


if __name__ == "__main__":
    logger.info("Starting backtests with real database data...")
    report1, report2 = run_simple_backtest()
    
    if report1 or report2:
        logger.info("Backtests completed successfully!")
        print("\n✅ Successfully connected to database and ran backtests with real market data!")
    else:
        logger.error("All backtests failed!")