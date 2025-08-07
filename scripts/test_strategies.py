"""Test different trading strategies with backtesting."""

import logging
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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


class MomentumStrategy(BaseStrategy):
    """Simple momentum strategy based on price changes."""
    
    def __init__(self, lookback_days: int = 20, momentum_threshold: float = 0.05, *args, **kwargs):
        super().__init__(
            name="Momentum Strategy",
            description=f"Buy when {lookback_days}-day momentum > {momentum_threshold}",
            *args,
            **kwargs
        )
        self.lookback_days = lookback_days
        self.momentum_threshold = momentum_threshold
        self.holdings = set()
        
    def generate_signals(self, data, events=None):
        """Generate signals - not used in this strategy."""
        return pd.DataFrame()
        
    def calculate_momentum(self, symbol: str, timestamp: datetime, bar_data: Dict) -> float:
        """Calculate momentum as percentage change over lookback period."""
        # In a real implementation, we'd use historical data
        # For testing, simulate momentum based on current price
        current_price = bar_data.get('close', 0)
        # Simple simulation: use day of month to create varying momentum
        day_factor = timestamp.day / 30.0
        return (day_factor - 0.5) * 0.2  # Range from -10% to +10%
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events) -> Tuple[bool, Optional[PositionSide], float]:
        """Enter when momentum is positive and above threshold."""
        if symbol in self.holdings:
            return False, None, 0
            
        momentum = self.calculate_momentum(symbol, timestamp, bar_data)
        
        if momentum > self.momentum_threshold:
            self.holdings.add(symbol)
            logger.info(f"[Momentum] Buying {symbol} - momentum: {momentum:.2%}")
            return True, PositionSide.LONG, 0.2  # 20% position size
            
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events) -> bool:
        """Exit when momentum turns negative."""
        momentum = self.calculate_momentum(position.symbol, timestamp, bar_data)
        
        if momentum < -self.momentum_threshold:
            self.holdings.discard(position.symbol)
            logger.info(f"[Momentum] Selling {position.symbol} - momentum: {momentum:.2%}")
            return True
            
        return False


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy - buy oversold, sell overbought."""
    
    def __init__(self, zscore_threshold: float = 2.0, holding_period: int = 5, *args, **kwargs):
        super().__init__(
            name="Mean Reversion Strategy",
            description=f"Trade when price deviates {zscore_threshold} std from mean",
            *args,
            **kwargs
        )
        self.zscore_threshold = zscore_threshold
        self.holding_period = holding_period
        self.entry_dates = {}
        
    def generate_signals(self, data, events=None):
        """Generate signals - not used in this strategy."""
        return pd.DataFrame()
        
    def calculate_zscore(self, symbol: str, timestamp: datetime, bar_data: Dict) -> float:
        """Calculate z-score of current price."""
        # Simulate z-score based on date patterns
        day = timestamp.day
        month = timestamp.month
        
        # Create cyclic pattern
        zscore = np.sin(day * 0.2) * 2.5 + np.cos(month * 0.5) * 1.5
        return zscore
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events) -> Tuple[bool, Optional[PositionSide], float]:
        """Enter when price is extreme (oversold or overbought)."""
        if symbol in self.entry_dates:
            return False, None, 0
            
        zscore = self.calculate_zscore(symbol, timestamp, bar_data)
        
        # Buy when oversold
        if zscore < -self.zscore_threshold:
            self.entry_dates[symbol] = timestamp
            logger.info(f"[MeanReversion] Buying {symbol} - oversold z-score: {zscore:.2f}")
            return True, PositionSide.LONG, 0.15
            
        # Short when overbought (if shorting enabled)
        elif zscore > self.zscore_threshold and self.allow_short:
            self.entry_dates[symbol] = timestamp
            logger.info(f"[MeanReversion] Shorting {symbol} - overbought z-score: {zscore:.2f}")
            return True, PositionSide.SHORT, 0.15
            
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events) -> bool:
        """Exit after holding period or when z-score normalizes."""
        if position.symbol not in self.entry_dates:
            return False
            
        # Check holding period
        days_held = (timestamp - self.entry_dates[position.symbol]).days
        if days_held >= self.holding_period:
            del self.entry_dates[position.symbol]
            logger.info(f"[MeanReversion] Exiting {position.symbol} - holding period reached")
            return True
            
        # Check if z-score normalized
        zscore = self.calculate_zscore(position.symbol, timestamp, bar_data)
        if abs(zscore) < 0.5:  # Near mean
            del self.entry_dates[position.symbol]
            logger.info(f"[MeanReversion] Exiting {position.symbol} - z-score normalized: {zscore:.2f}")
            return True
            
        return False


class BreakoutStrategy(BaseStrategy):
    """Breakout strategy - trade on price breaking key levels."""
    
    def __init__(self, lookback: int = 20, breakout_factor: float = 1.02, *args, **kwargs):
        super().__init__(
            name="Breakout Strategy",
            description=f"Buy on {lookback}-day high breakout",
            *args,
            **kwargs
        )
        self.lookback = lookback
        self.breakout_factor = breakout_factor
        self.positions_taken = set()
        self.breakout_levels = {}
        
    def generate_signals(self, data, events=None):
        """Generate signals - not used in this strategy."""
        return pd.DataFrame()
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events) -> Tuple[bool, Optional[PositionSide], float]:
        """Enter on breakout above resistance."""
        if symbol in self.positions_taken:
            return False, None, 0
            
        current_price = bar_data.get('close', 0)
        high_price = bar_data.get('high', current_price)
        
        # Simulate resistance level
        resistance = current_price * (1 + (timestamp.day % 10) * 0.01)
        
        if high_price > resistance * self.breakout_factor:
            self.positions_taken.add(symbol)
            self.breakout_levels[symbol] = resistance
            logger.info(f"[Breakout] Buying {symbol} - broke resistance at {resistance:.2f}")
            return True, PositionSide.LONG, 0.25
            
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events) -> bool:
        """Exit on stop loss or target."""
        current_price = bar_data.get('close', 0)
        entry_level = self.breakout_levels.get(position.symbol, position.entry_price)
        
        # Stop loss at 2% below breakout
        stop_loss = entry_level * 0.98
        # Target at 5% above breakout
        target = entry_level * 1.05
        
        if current_price <= stop_loss:
            self.positions_taken.discard(position.symbol)
            logger.info(f"[Breakout] Stop loss hit for {position.symbol} at {current_price:.2f}")
            return True
            
        if current_price >= target:
            self.positions_taken.discard(position.symbol)
            logger.info(f"[Breakout] Target reached for {position.symbol} at {current_price:.2f}")
            return True
            
        return False


class VolumeBasedStrategy(BaseStrategy):
    """Trade based on volume patterns."""
    
    def __init__(self, volume_multiplier: float = 2.0, *args, **kwargs):
        super().__init__(
            name="Volume Based Strategy",
            description="Trade on unusual volume spikes",
            *args,
            **kwargs
        )
        self.volume_multiplier = volume_multiplier
        self.positions = set()
        self.avg_volumes = {}
        
    def generate_signals(self, data, events=None):
        """Generate signals - not used in this strategy."""
        return pd.DataFrame()
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events) -> Tuple[bool, Optional[PositionSide], float]:
        """Enter on high volume breakout."""
        if symbol in self.positions:
            return False, None, 0
            
        current_volume = bar_data.get('volume', 0)
        
        # Initialize average volume
        if symbol not in self.avg_volumes:
            self.avg_volumes[symbol] = current_volume
            
        # Update moving average
        self.avg_volumes[symbol] = 0.95 * self.avg_volumes[symbol] + 0.05 * current_volume
        
        # Check for volume spike
        if current_volume > self.avg_volumes[symbol] * self.volume_multiplier:
            close = bar_data.get('close', 0)
            open_price = bar_data.get('open', close)
            
            # Buy if price is up on high volume
            if close > open_price:
                self.positions.add(symbol)
                logger.info(f"[Volume] Buying {symbol} - volume spike: {current_volume/self.avg_volumes[symbol]:.1f}x average")
                return True, PositionSide.LONG, 0.2
                
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events) -> bool:
        """Exit when volume dries up or after fixed period."""
        current_volume = bar_data.get('volume', 0)
        avg_volume = self.avg_volumes.get(position.symbol, current_volume)
        
        # Exit if volume drops below 50% of average
        if current_volume < avg_volume * 0.5:
            self.positions.discard(position.symbol)
            logger.info(f"[Volume] Exiting {position.symbol} - low volume")
            return True
            
        # Exit after 10 days
        if (timestamp - position.entry_time).days >= 10:
            self.positions.discard(position.symbol)
            logger.info(f"[Volume] Exiting {position.symbol} - time limit")
            return True
            
        return False


def run_strategy_test(strategy_class, strategy_name: str, config: BacktestConfig, 
                     symbols: List[str], **strategy_kwargs) -> Dict:
    """Run a single strategy test and return results."""
    print(f"\n{'='*60}")
    print(f"Testing {strategy_name}")
    print(f"{'='*60}")
    
    engine = BacktestEngine(config=config)
    
    try:
        # Create strategy instance with kwargs
        if strategy_kwargs:
            report = engine.run(
                strategy=lambda *args, **kw: strategy_class(*args, **strategy_kwargs, **kw),
                symbols=symbols,
                signal_computer=None
            )
        else:
            report = engine.run(
                strategy=strategy_class,
                symbols=symbols,
                signal_computer=None
            )
        
        # Display results
        print(f"Initial Capital: ${config.initial_capital:,.2f}")
        print(f"Final Equity: ${report.final_equity:,.2f}")
        print(f"Total Return: {report.total_return:.2%}")
        print(f"Annualized Return: {report.annualized_return:.2%}")
        print(f"Sharpe Ratio: {report.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {report.max_drawdown:.2%}")
        print(f"Win Rate: {report.win_rate:.2%}")
        print(f"Total Trades: {report.total_trades}")
        
        if report.total_trades > 0:
            print(f"Avg Win: {report.avg_win:.2%}")
            print(f"Avg Loss: {report.avg_loss:.2%}")
            print(f"Profit Factor: {report.profit_factor:.2f}")
        
        return {
            'strategy': strategy_name,
            'total_return': report.total_return,
            'sharpe_ratio': report.sharpe_ratio,
            'max_drawdown': report.max_drawdown,
            'total_trades': report.total_trades,
            'win_rate': report.win_rate
        }
        
    except Exception as e:
        logger.error(f"{strategy_name} test failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'strategy': strategy_name,
            'error': str(e)
        }


def test_time_based_strategies():
    """Test time-based trading strategies."""
    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),
        initial_capital=100000,
        commission=0.001,
        slippage=0.0001,
        max_positions=5
    )
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Test morning breakout strategy
    morning_window = TradingWindow(
        start_time=time(9, 30),
        end_time=time(10, 30),
        session_type=SessionType.REGULAR
    )
    
    results = []
    
    # This would need proper implementation of TimeBasedStrategy
    # For now, test our custom strategies
    

def main():
    """Run all strategy tests."""
    
    # Test configuration
    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),
        initial_capital=100000,
        commission=0.001,
        slippage=0.0001,
        max_positions=5,
        calculate_metrics_every=20
    )
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Collect all results
    all_results = []
    
    # Test Momentum Strategy
    results = run_strategy_test(
        MomentumStrategy,
        "Momentum Strategy (20d, 5%)",
        config,
        symbols,
        lookback_days=20,
        momentum_threshold=0.05
    )
    all_results.append(results)
    
    # Test Mean Reversion Strategy
    results = run_strategy_test(
        MeanReversionStrategy,
        "Mean Reversion (2.0 std)",
        config,
        symbols,
        zscore_threshold=2.0,
        holding_period=5
    )
    all_results.append(results)
    
    # Test Breakout Strategy
    results = run_strategy_test(
        BreakoutStrategy,
        "Breakout Strategy (20d)",
        config,
        symbols,
        lookback=20,
        breakout_factor=1.02
    )
    all_results.append(results)
    
    # Test Volume Based Strategy
    results = run_strategy_test(
        VolumeBasedStrategy,
        "Volume Based Strategy (2x)",
        config,
        symbols,
        volume_multiplier=2.0
    )
    all_results.append(results)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Strategy':<30} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10} {'Trades':>10} {'WinRate':>10}")
    print(f"{'-'*80}")
    
    for result in all_results:
        if 'error' not in result:
            print(f"{result['strategy']:<30} "
                  f"{result['total_return']:>9.2%} "
                  f"{result['sharpe_ratio']:>10.2f} "
                  f"{result['max_drawdown']:>9.2%} "
                  f"{result['total_trades']:>10d} "
                  f"{result['win_rate']:>9.2%}")
        else:
            print(f"{result['strategy']:<30} ERROR: {result['error']}")
    
    # Find best strategy
    valid_results = [r for r in all_results if 'error' not in r and r['total_return'] is not None]
    if valid_results:
        best_return = max(valid_results, key=lambda x: x['total_return'])
        best_sharpe = max(valid_results, key=lambda x: x['sharpe_ratio'] if x['sharpe_ratio'] is not None else -999)
        
        print(f"\n{'='*80}")
        print(f"Best Return: {best_return['strategy']} ({best_return['total_return']:.2%})")
        print(f"Best Sharpe: {best_sharpe['strategy']} ({best_sharpe['sharpe_ratio']:.2f})")
        print(f"{'='*80}")


if __name__ == "__main__":
    logger.info("Starting comprehensive strategy tests...")
    main()
    logger.info("Strategy tests completed!")