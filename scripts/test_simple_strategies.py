"""Test simple trading strategies that work with current backtesting system."""

import logging
from datetime import datetime
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.strategies.base import BaseStrategy, PositionSide

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleTrendFollowing(BaseStrategy):
    """Follow simple price trends."""
    
    def __init__(self, trend_days: int = 5, *args, **kwargs):
        super().__init__(
            name=f"Simple Trend Following ({trend_days}d)",
            description="Buy on uptrend, sell on downtrend",
            *args,
            **kwargs
        )
        self.trend_days = trend_days
        self.positions = set()
        self.entry_prices = {}
        
    def generate_signals(self, data, events=None):
        """Generate signals - not used."""
        return pd.DataFrame()
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events):
        """Enter on uptrend."""
        if symbol in self.positions:
            return False, None, 0
            
        # Simple trend: if day of month > 15, uptrend
        if timestamp.day > 15 and timestamp.day < 25:
            self.positions.add(symbol)
            self.entry_prices[symbol] = bar_data['close']
            logger.info(f"[Trend] Entering {symbol} on uptrend")
            return True, PositionSide.LONG, 0.2
            
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events):
        """Exit on downtrend or stop loss."""
        # Exit if trend reverses (simplified)
        if timestamp.day >= 25:
            self.positions.discard(position.symbol)
            logger.info(f"[Trend] Exiting {position.symbol} - trend reversal")
            return True
            
        # Stop loss at 3%
        if bar_data['close'] < position.avg_price * 0.97:
            self.positions.discard(position.symbol)
            logger.info(f"[Trend] Stop loss for {position.symbol}")
            return True
            
        return False


class SimpleRangeTrading(BaseStrategy):
    """Trade within a range."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="Simple Range Trading",
            description="Buy at range bottom, sell at range top",
            *args,
            **kwargs
        )
        self.positions = set()
        
    def generate_signals(self, data, events=None):
        """Generate signals - not used."""
        return pd.DataFrame()
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events):
        """Enter at range bottom."""
        if symbol in self.positions:
            return False, None, 0
            
        # Simulate range based on symbol hash
        symbol_factor = hash(symbol) % 10 / 10.0
        
        # Buy in first week of month (simulated range bottom)
        if timestamp.day <= 7:
            self.positions.add(symbol)
            logger.info(f"[Range] Buying {symbol} at range bottom")
            return True, PositionSide.LONG, 0.15
            
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events):
        """Exit at range top."""
        # Sell in third week (simulated range top)
        if 14 <= timestamp.day <= 21:
            self.positions.discard(position.symbol)
            logger.info(f"[Range] Selling {position.symbol} at range top")
            return True
            
        # Emergency stop at 5%
        if bar_data['close'] < position.avg_price * 0.95:
            self.positions.discard(position.symbol)
            logger.info(f"[Range] Emergency stop for {position.symbol}")
            return True
            
        return False


class SimpleDollarCostAverage(BaseStrategy):
    """Dollar cost averaging strategy."""
    
    def __init__(self, buy_day: int = 1, *args, **kwargs):
        super().__init__(
            name=f"Dollar Cost Average (day {buy_day})",
            description="Buy fixed amount on specific day each month",
            *args,
            **kwargs
        )
        self.buy_day = buy_day
        self.last_buy_month = {}
        
    def generate_signals(self, data, events=None):
        """Generate signals - not used."""
        return pd.DataFrame()
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events):
        """Enter on specific day of month."""
        # Check if it's the buy day
        if timestamp.day != self.buy_day:
            return False, None, 0
            
        # Check if we already bought this month
        month_key = f"{timestamp.year}-{timestamp.month}"
        if symbol in self.last_buy_month and self.last_buy_month[symbol] == month_key:
            return False, None, 0
            
        # Buy fixed allocation
        self.last_buy_month[symbol] = month_key
        logger.info(f"[DCA] Monthly purchase of {symbol}")
        return True, PositionSide.LONG, 0.1  # 10% each time
        
    def should_exit(self, position, timestamp, bar_data, signals, events):
        """Never exit - hold forever."""
        return False


class SimpleRotation(BaseStrategy):
    """Rotate between best performing assets."""
    
    def __init__(self, hold_days: int = 30, top_n: int = 2, *args, **kwargs):
        super().__init__(
            name=f"Simple Rotation (top {top_n})",
            description=f"Hold top {top_n} performers for {hold_days} days",
            *args,
            **kwargs
        )
        self.hold_days = hold_days
        self.top_n = top_n
        self.entry_dates = {}
        self.rankings = {}
        
    def generate_signals(self, data, events=None):
        """Generate signals - not used."""
        return pd.DataFrame()
        
    def calculate_score(self, symbol, timestamp):
        """Calculate performance score for ranking."""
        # Simple scoring based on symbol and date
        base = hash(symbol) % 100
        date_factor = (timestamp.day + timestamp.month) % 20
        return base + date_factor
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events):
        """Enter top ranked symbols."""
        # Update rankings
        self.rankings[symbol] = self.calculate_score(symbol, timestamp)
        
        # Check if already holding
        if symbol in self.entry_dates:
            return False, None, 0
            
        # Get current top N
        sorted_symbols = sorted(self.rankings.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [s[0] for s in sorted_symbols[:self.top_n]]
        
        if symbol in top_symbols:
            self.entry_dates[symbol] = timestamp
            logger.info(f"[Rotation] Entering {symbol} - rank {top_symbols.index(symbol) + 1}")
            return True, PositionSide.LONG, 0.4 / self.top_n
            
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events):
        """Exit after hold period or if no longer top ranked."""
        if position.symbol not in self.entry_dates:
            return False
            
        # Check hold period
        days_held = (timestamp - self.entry_dates[position.symbol]).days
        if days_held >= self.hold_days:
            del self.entry_dates[position.symbol]
            logger.info(f"[Rotation] Exiting {position.symbol} - hold period complete")
            return True
            
        # Update ranking and check if still top N
        self.rankings[position.symbol] = self.calculate_score(position.symbol, timestamp)
        sorted_symbols = sorted(self.rankings.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [s[0] for s in sorted_symbols[:self.top_n]]
        
        if position.symbol not in top_symbols:
            del self.entry_dates[position.symbol]
            logger.info(f"[Rotation] Exiting {position.symbol} - no longer top ranked")
            return True
            
        return False


def main():
    """Run simple strategy tests."""
    
    # Test configuration
    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),  # 3 months
        initial_capital=100000,
        commission=0.001,
        slippage=0.0001,
        max_positions=5,
        calculate_metrics_every=10,
        benchmark=None  # Disable benchmark
    )
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    # Test strategies
    strategies = [
        (SimpleTrendFollowing, "Trend Following (5d)", {'trend_days': 5}),
        (SimpleRangeTrading, "Range Trading", {}),
        (SimpleDollarCostAverage, "Dollar Cost Average", {'buy_day': 1}),
        (SimpleRotation, "Simple Rotation (Top 2)", {'hold_days': 30, 'top_n': 2}),
    ]
    
    results = []
    
    for strategy_class, name, kwargs in strategies:
        print(f"\n{'='*60}")
        print(f"Testing {name}")
        print(f"{'='*60}")
        
        engine = BacktestEngine(config=config)
        
        try:
            if kwargs:
                report = engine.run(
                    strategy=lambda *args, **kw: strategy_class(*args, **kwargs, **kw),
                    symbols=symbols,
                    signal_computer=None
                )
            else:
                report = engine.run(
                    strategy=strategy_class,
                    symbols=symbols,
                    signal_computer=None
                )
            
            print(f"Initial Capital: ${config.initial_capital:,.2f}")
            print(f"Final Equity: ${report.final_equity:,.2f}")
            print(f"Total Return: {report.total_return:.2%}")
            print(f"Max Drawdown: {report.max_drawdown:.2%}")
            print(f"Total Trades: {report.total_trades}")
            print(f"Win Rate: {report.win_rate:.2%}")
            if report.sharpe_ratio is not None:
                print(f"Sharpe Ratio: {report.sharpe_ratio:.2f}")
            
            results.append({
                'strategy': name,
                'return': report.total_return,
                'sharpe': report.sharpe_ratio,
                'drawdown': report.max_drawdown,
                'trades': report.total_trades,
                'win_rate': report.win_rate
            })
            
        except Exception as e:
            logger.error(f"Failed to test {name}: {e}")
            results.append({'strategy': name, 'error': str(e)})
    
    # Summary
    print(f"\n{'='*80}")
    print("STRATEGY PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Strategy':<25} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10} {'Trades':>10} {'WinRate':>10}")
    print(f"{'-'*80}")
    
    for r in results:
        if 'error' not in r:
            sharpe_str = f"{r['sharpe']:>10.2f}" if r['sharpe'] is not None else "       N/A"
            print(f"{r['strategy']:<25} {r['return']:>9.2%} {sharpe_str} "
                  f"{r['drawdown']:>9.2%} {r['trades']:>10d} {r['win_rate']:>9.2%}")
        else:
            print(f"{r['strategy']:<25} ERROR: {r['error']}")
    
    # Best performer
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x['return'])
        print(f"\nBest Strategy: {best['strategy']} with {best['return']:.2%} return")


if __name__ == "__main__":
    logger.info("Starting simple strategy tests...")
    main()
    logger.info("Simple strategy tests completed!")