"""Simple backtest using real database data."""

import logging
from datetime import datetime
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.strategies.base import BaseStrategy, PositionSide
from src.signals.base.signal_computer import SignalComputer
from src.signals.base.config import SignalConfig
from src.signals.technical.momentum import RSISignal
from src.signals.technical.trend import SMASignal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleRSIStrategy(BaseStrategy):
    """Simple RSI-based strategy."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="Simple RSI Strategy",
            description="Buy when RSI < 30, sell when RSI > 70",
            *args,
            **kwargs
        )
        self.trades_entered = 0
        self.max_trades = 10  # Limit number of trades for testing
        
    def generate_signals(self, data, events=None):
        """Generate signals - not used."""
        return pd.DataFrame()
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events):
        """Enter when RSI is oversold."""
        # Limit number of trades for testing
        if self.trades_entered >= self.max_trades:
            return False, None, 0
            
        # Check if we have signals
        if signals is not None and not signals.empty:
            rsi = signals.get('rsi', 50)
            if rsi < 30:  # Oversold
                self.trades_entered += 1
                logger.info(f"Entering {symbol} at {timestamp}, RSI={rsi:.2f}")
                return True, PositionSide.LONG, 0.1  # 10% position size
                
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events):
        """Exit when RSI is overbought or stop loss hit."""
        if signals is not None and not signals.empty:
            rsi = signals.get('rsi', 50)
            if rsi > 70:  # Overbought
                logger.info(f"Exiting {position.symbol} at {timestamp}, RSI={rsi:.2f}")
                return True
                
        # Simple stop loss
        if hasattr(position, 'current_price') and hasattr(position, 'avg_price'):
            pnl_pct = (position.current_price - position.avg_price) / position.avg_price
            if pnl_pct < -0.05:  # 5% stop loss
                logger.info(f"Stop loss hit for {position.symbol} at {timestamp}, PnL={pnl_pct:.2%}")
                return True
                
        return False


def run_simple_backtest():
    """Run a simple backtest with database data."""
    
    # Use a shorter date range for testing
    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),  # 6 months
        initial_capital=100000,
        commission=0.001,
        slippage=0.0001,
        max_positions=3,
        calculate_metrics_every=20,
        save_trades=True,
        benchmark=None
    )
    
    # Create signal computer with just RSI and SMA
    signal_config = SignalConfig()
    signal_computer = SignalComputer(config=signal_config)
    
    # Register signals
    signal_computer.register_signal(RSISignal(config=signal_config, period=14))
    signal_computer.register_signal(SMASignal(config=signal_config, period=20))
    signal_computer.register_signal(SMASignal(config=signal_config, period=50))
    
    # Use just a few symbols for testing
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Create and run backtest
    engine = BacktestEngine(config=config)
    
    logger.info(f"Running backtest for {symbols} from {config.start_date} to {config.end_date}")
    
    try:
        report = engine.run(
            strategy=SimpleRSIStrategy,
            symbols=symbols,
            signal_computer=signal_computer
        )
        
        # Print results
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Initial Capital: ${config.initial_capital:,.2f}")
        print(f"Final Equity: ${report.final_equity:,.2f}")
        print(f"Total Return: {report.total_return:.2%}")
        print(f"Annualized Return: {report.annualized_return:.2%}")
        print(f"Sharpe Ratio: {report.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {report.max_drawdown:.2%}")
        print(f"Total Trades: {report.total_trades}")
        print(f"Win Rate: {report.win_rate:.2%}")
        print("="*60)
        
        return report
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    report = run_simple_backtest()
    if report:
        logger.info("Backtest completed successfully!")
    else:
        logger.error("Backtest failed!")