"""Test technical indicator-based trading strategies."""

import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.strats.base import BaseStrategy, PositionSide
from src.signals.technical import TechnicalIndicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MovingAverageCrossStrategy(BaseStrategy):
    """Classic moving average crossover strategy."""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30, *args, **kwargs):
        super().__init__(
            name=f"MA Cross ({fast_period}/{slow_period})",
            description=f"Buy when {fast_period}-MA crosses above {slow_period}-MA",
            *args,
            **kwargs
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.positions = {}
        self.indicators = TechnicalIndicators()
        
    def calculate_signals(self, symbol: str, timestamp: datetime, historical_prices: pd.Series) -> Dict:
        """Calculate moving averages and crossover signals."""
        if len(historical_prices) < self.slow_period:
            return {'signal': 0}
            
        # Calculate moving averages
        fast_ma = self.indicators.sma(historical_prices, self.fast_period)
        slow_ma = self.indicators.sma(historical_prices, self.slow_period)
        
        if fast_ma.empty or slow_ma.empty:
            return {'signal': 0}
            
        # Get current and previous values
        curr_fast = fast_ma.iloc[-1]
        curr_slow = slow_ma.iloc[-1]
        prev_fast = fast_ma.iloc[-2] if len(fast_ma) > 1 else curr_fast
        prev_slow = slow_ma.iloc[-2] if len(slow_ma) > 1 else curr_slow
        
        # Check for crossover
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return {'signal': 1, 'fast_ma': curr_fast, 'slow_ma': curr_slow}  # Golden cross
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            return {'signal': -1, 'fast_ma': curr_fast, 'slow_ma': curr_slow}  # Death cross
            
        return {'signal': 0, 'fast_ma': curr_fast, 'slow_ma': curr_slow}
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events) -> Tuple[bool, Optional[PositionSide], float]:
        """Enter on golden cross."""
        if symbol in self.positions:
            return False, None, 0
            
        # Simulate historical prices for testing
        prices = pd.Series([bar_data['close']] * self.slow_period)
        # Add some variation
        for i in range(len(prices)):
            prices.iloc[i] *= (1 + np.sin(i * 0.1) * 0.02)
            
        signals = self.calculate_signals(symbol, timestamp, prices)
        
        if signals['signal'] == 1:
            self.positions[symbol] = timestamp
            logger.info(f"[MA Cross] Golden cross for {symbol} - Fast MA: {signals.get('fast_ma', 0):.2f}, Slow MA: {signals.get('slow_ma', 0):.2f}")
            return True, PositionSide.LONG, 0.2
            
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events) -> bool:
        """Exit on death cross or stop loss."""
        # Simulate historical prices
        prices = pd.Series([bar_data['close']] * self.slow_period)
        for i in range(len(prices)):
            prices.iloc[i] *= (1 + np.sin(i * 0.1 + timestamp.day) * 0.02)
            
        signals = self.calculate_signals(position.symbol, timestamp, prices)
        
        if signals['signal'] == -1:
            self.positions.pop(position.symbol, None)
            logger.info(f"[MA Cross] Death cross for {position.symbol}")
            return True
            
        # Stop loss at 5%
        if bar_data['close'] < position.entry_price * 0.95:
            self.positions.pop(position.symbol, None)
            logger.info(f"[MA Cross] Stop loss for {position.symbol}")
            return True
            
        return False


class RSIMeanReversionStrategy(BaseStrategy):
    """Trade based on RSI overbought/oversold levels."""
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70, *args, **kwargs):
        super().__init__(
            name=f"RSI Mean Reversion ({oversold}/{overbought})",
            description="Buy oversold, sell overbought based on RSI",
            *args,
            **kwargs
        )
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.positions = {}
        self.indicators = TechnicalIndicators()
        
    def calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate current RSI value."""
        if len(prices) < self.rsi_period + 1:
            return 50  # Neutral
            
        rsi_series = self.indicators.rsi(prices, self.rsi_period)
        return rsi_series.iloc[-1] if not rsi_series.empty else 50
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events) -> Tuple[bool, Optional[PositionSide], float]:
        """Enter when RSI indicates oversold."""
        if symbol in self.positions:
            return False, None, 0
            
        # Simulate price history with momentum
        prices = pd.Series([bar_data['close']] * (self.rsi_period + 5))
        momentum = (timestamp.day - 15) / 15.0  # -1 to 1
        for i in range(len(prices)):
            prices.iloc[i] *= (1 + momentum * 0.01 * (i - len(prices)/2))
            
        rsi = self.calculate_rsi(prices)
        
        if rsi < self.oversold:
            self.positions[symbol] = {'entry_time': timestamp, 'entry_rsi': rsi}
            logger.info(f"[RSI] Buying {symbol} - RSI: {rsi:.1f} (oversold)")
            return True, PositionSide.LONG, 0.15
            
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events) -> bool:
        """Exit when RSI normalizes or hits overbought."""
        # Simulate price history
        prices = pd.Series([bar_data['close']] * (self.rsi_period + 5))
        days_held = (timestamp - self.positions[position.symbol]['entry_time']).days
        
        # Simulate mean reversion
        for i in range(len(prices)):
            reversion_factor = 1 - (days_held * 0.02)  # Revert over time
            prices.iloc[i] *= (1 + reversion_factor * 0.01 * (i - len(prices)/2))
            
        rsi = self.calculate_rsi(prices)
        
        # Exit if RSI > 50 (normalized) or > overbought
        if rsi > 50 or rsi > self.overbought:
            self.positions.pop(position.symbol, None)
            logger.info(f"[RSI] Exiting {position.symbol} - RSI: {rsi:.1f}")
            return True
            
        # Time-based exit after 20 days
        if days_held > 20:
            self.positions.pop(position.symbol, None)
            logger.info(f"[RSI] Exiting {position.symbol} - time limit reached")
            return True
            
        return False


class BollingerBandStrategy(BaseStrategy):
    """Trade based on Bollinger Band breakouts and mean reversion."""
    
    def __init__(self, bb_period: int = 20, bb_std: float = 2.0, *args, **kwargs):
        super().__init__(
            name=f"Bollinger Bands ({bb_period}, {bb_std}σ)",
            description="Trade when price touches Bollinger Bands",
            *args,
            **kwargs
        )
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.positions = {}
        self.indicators = TechnicalIndicators()
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events) -> Tuple[bool, Optional[PositionSide], float]:
        """Enter when price touches lower band (buy) or upper band (short)."""
        if symbol in self.positions:
            return False, None, 0
            
        # Simulate price history with volatility
        prices = pd.Series([bar_data['close']] * self.bb_period)
        volatility = 0.02 * (1 + timestamp.day / 30.0)  # Increasing volatility
        
        for i in range(len(prices)):
            prices.iloc[i] *= (1 + np.random.normal(0, volatility))
            
        # Calculate Bollinger Bands
        bb_data = self.indicators.bollinger_bands(prices, self.bb_period, self.bb_std)
        
        if bb_data.empty:
            return False, None, 0
            
        current_price = bar_data['close']
        upper_band = bb_data['upper'].iloc[-1]
        lower_band = bb_data['lower'].iloc[-1]
        middle_band = bb_data['middle'].iloc[-1]
        
        # Buy at lower band
        if current_price <= lower_band:
            self.positions[symbol] = {
                'entry_time': timestamp,
                'entry_band': 'lower',
                'target': middle_band
            }
            logger.info(f"[BB] Buying {symbol} at lower band - Price: {current_price:.2f}, Lower: {lower_band:.2f}")
            return True, PositionSide.LONG, 0.2
            
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events) -> bool:
        """Exit at middle band (target) or opposite band (stop)."""
        # Simulate current bands
        prices = pd.Series([bar_data['close']] * self.bb_period)
        days_held = (timestamp - self.positions[position.symbol]['entry_time']).days
        
        # Adjust for mean reversion
        for i in range(len(prices)):
            prices.iloc[i] *= (1 + (days_held * 0.01) * (i - len(prices)/2) / len(prices))
            
        bb_data = self.indicators.bollinger_bands(prices, self.bb_period, self.bb_std)
        
        if bb_data.empty:
            return False
            
        current_price = bar_data['close']
        middle_band = bb_data['middle'].iloc[-1]
        
        # Exit at target (middle band)
        if current_price >= middle_band:
            self.positions.pop(position.symbol, None)
            logger.info(f"[BB] Target reached for {position.symbol} at middle band")
            return True
            
        # Stop loss at 3%
        if current_price < position.entry_price * 0.97:
            self.positions.pop(position.symbol, None)
            logger.info(f"[BB] Stop loss for {position.symbol}")
            return True
            
        return False


class MACDStrategy(BaseStrategy):
    """Trade based on MACD crossovers and divergences."""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, *args, **kwargs):
        super().__init__(
            name=f"MACD ({fast}/{slow}/{signal})",
            description="Trade on MACD signal line crossovers",
            *args,
            **kwargs
        )
        self.fast_period = fast
        self.slow_period = slow
        self.signal_period = signal
        self.positions = {}
        self.indicators = TechnicalIndicators()
        
    def calculate_macd_signal(self, prices: pd.Series) -> int:
        """Calculate MACD crossover signal."""
        if len(prices) < self.slow_period + self.signal_period:
            return 0
            
        macd_data = self.indicators.macd(prices, self.fast_period, self.slow_period, self.signal_period)
        
        if macd_data.empty or len(macd_data) < 2:
            return 0
            
        # Check for crossover
        curr_macd = macd_data['macd'].iloc[-1]
        curr_signal = macd_data['signal'].iloc[-1]
        prev_macd = macd_data['macd'].iloc[-2]
        prev_signal = macd_data['signal'].iloc[-2]
        
        # Bullish crossover
        if prev_macd <= prev_signal and curr_macd > curr_signal:
            return 1
        # Bearish crossover
        elif prev_macd >= prev_signal and curr_macd < curr_signal:
            return -1
            
        return 0
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events) -> Tuple[bool, Optional[PositionSide], float]:
        """Enter on bullish MACD crossover."""
        if symbol in self.positions:
            return False, None, 0
            
        # Simulate trending price data
        prices = pd.Series([bar_data['close']] * (self.slow_period + self.signal_period + 10))
        trend = (timestamp.day - 15) / 30.0  # -0.5 to 0.5
        
        for i in range(len(prices)):
            prices.iloc[i] *= (1 + trend * 0.001 * i)
            
        signal = self.calculate_macd_signal(prices)
        
        if signal == 1:
            self.positions[symbol] = timestamp
            logger.info(f"[MACD] Bullish crossover for {symbol}")
            return True, PositionSide.LONG, 0.25
            
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events) -> bool:
        """Exit on bearish crossover or stop loss."""
        # Simulate price data
        prices = pd.Series([bar_data['close']] * (self.slow_period + self.signal_period + 10))
        days_held = (timestamp - self.positions[position.symbol]).days
        
        # Simulate trend reversal over time
        for i in range(len(prices)):
            reversal = -days_held * 0.001
            prices.iloc[i] *= (1 + reversal * i)
            
        signal = self.calculate_macd_signal(prices)
        
        if signal == -1:
            self.positions.pop(position.symbol, None)
            logger.info(f"[MACD] Bearish crossover for {position.symbol}")
            return True
            
        # Trailing stop at 5%
        if bar_data['close'] < position.entry_price * 0.95:
            self.positions.pop(position.symbol, None)
            logger.info(f"[MACD] Trailing stop for {position.symbol}")
            return True
            
        return False


class CompositeIndicatorStrategy(BaseStrategy):
    """Combine multiple indicators for stronger signals."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="Composite Indicator Strategy",
            description="Combine RSI, MACD, and BB for confirmation",
            *args,
            **kwargs
        )
        self.positions = {}
        self.indicators = TechnicalIndicators()
        
    def calculate_composite_score(self, prices: pd.Series) -> float:
        """Calculate composite score from multiple indicators."""
        score = 0.0
        
        # RSI component
        if len(prices) >= 15:
            rsi = self.indicators.rsi(prices, 14)
            if not rsi.empty:
                rsi_val = rsi.iloc[-1]
                if rsi_val < 30:
                    score += 1.0
                elif rsi_val > 70:
                    score -= 1.0
                    
        # MACD component
        if len(prices) >= 35:
            macd_data = self.indicators.macd(prices, 12, 26, 9)
            if not macd_data.empty and len(macd_data) >= 2:
                if macd_data['macd'].iloc[-1] > macd_data['signal'].iloc[-1]:
                    score += 0.5
                else:
                    score -= 0.5
                    
        # Bollinger Band component
        if len(prices) >= 20:
            bb_data = self.indicators.bollinger_bands(prices, 20, 2)
            if not bb_data.empty:
                current_price = prices.iloc[-1]
                lower = bb_data['lower'].iloc[-1]
                upper = bb_data['upper'].iloc[-1]
                middle = bb_data['middle'].iloc[-1]
                
                # Normalize position within bands
                band_position = (current_price - lower) / (upper - lower)
                if band_position < 0.2:
                    score += 0.5
                elif band_position > 0.8:
                    score -= 0.5
                    
        return score
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events) -> Tuple[bool, Optional[PositionSide], float]:
        """Enter when composite score is strongly positive."""
        if symbol in self.positions:
            return False, None, 0
            
        # Simulate price history
        prices = pd.Series([bar_data['close']] * 40)
        
        # Add realistic price movement
        for i in range(len(prices)):
            trend = np.sin(i * 0.1) * 0.02
            noise = np.random.normal(0, 0.005)
            prices.iloc[i] *= (1 + trend + noise)
            
        score = self.calculate_composite_score(prices)
        
        if score >= 1.5:  # Strong buy signal
            self.positions[symbol] = {'entry_time': timestamp, 'entry_score': score}
            logger.info(f"[Composite] Strong buy signal for {symbol} - Score: {score:.2f}")
            return True, PositionSide.LONG, 0.3  # Higher conviction = larger position
            
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events) -> bool:
        """Exit when score turns negative or time limit."""
        # Simulate price history
        prices = pd.Series([bar_data['close']] * 40)
        days_held = (timestamp - self.positions[position.symbol]['entry_time']).days
        
        # Simulate changing market conditions
        for i in range(len(prices)):
            mean_reversion = -days_held * 0.002
            prices.iloc[i] *= (1 + mean_reversion * (i - 20))
            
        score = self.calculate_composite_score(prices)
        
        if score <= -0.5:  # Sell signal
            self.positions.pop(position.symbol, None)
            logger.info(f"[Composite] Sell signal for {position.symbol} - Score: {score:.2f}")
            return True
            
        # Take profit at 10%
        if bar_data['close'] > position.entry_price * 1.10:
            self.positions.pop(position.symbol, None)
            logger.info(f"[Composite] Take profit for {position.symbol}")
            return True
            
        # Stop loss at 5%
        if bar_data['close'] < position.entry_price * 0.95:
            self.positions.pop(position.symbol, None)
            logger.info(f"[Composite] Stop loss for {position.symbol}")
            return True
            
        return False


def run_technical_strategy_tests():
    """Run all technical indicator strategy tests."""
    
    # Test configuration for shorter period
    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),  # 3 months
        initial_capital=100000,
        commission=0.001,
        slippage=0.0001,
        max_positions=5,
        calculate_metrics_every=10
    )
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    # Test each strategy
    strategies = [
        (MovingAverageCrossStrategy, "MA Cross (10/30)", {'fast_period': 10, 'slow_period': 30}),
        (MovingAverageCrossStrategy, "MA Cross (20/50)", {'fast_period': 20, 'slow_period': 50}),
        (RSIMeanReversionStrategy, "RSI Mean Reversion (30/70)", {'oversold': 30, 'overbought': 70}),
        (RSIMeanReversionStrategy, "RSI Mean Reversion (20/80)", {'oversold': 20, 'overbought': 80}),
        (BollingerBandStrategy, "Bollinger Bands (20, 2σ)", {'bb_period': 20, 'bb_std': 2.0}),
        (BollingerBandStrategy, "Bollinger Bands (20, 1.5σ)", {'bb_period': 20, 'bb_std': 1.5}),
        (MACDStrategy, "MACD (12/26/9)", {'fast': 12, 'slow': 26, 'signal': 9}),
        (CompositeIndicatorStrategy, "Composite Indicator", {}),
    ]
    
    results = []
    
    for strategy_class, name, kwargs in strategies:
        print(f"\n{'='*60}")
        print(f"Testing {name}")
        print(f"{'='*60}")
        
        engine = BacktestEngine(config=config)
        
        try:
            # Create strategy with parameters
            if kwargs:
                report = engine.run(
                    strategy=lambda *args, **kw: strategy_class(**kwargs, *args, **kw),
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
            print(f"Final Equity: ${report.final_equity:,.2f}")
            print(f"Total Return: {report.total_return:.2%}")
            print(f"Sharpe Ratio: {report.sharpe_ratio:.2f}")
            print(f"Max Drawdown: {report.max_drawdown:.2%}")
            print(f"Total Trades: {report.total_trades}")
            
            results.append({
                'strategy': name,
                'return': report.total_return,
                'sharpe': report.sharpe_ratio,
                'drawdown': report.max_drawdown,
                'trades': report.total_trades
            })
            
        except Exception as e:
            logger.error(f"Failed to test {name}: {e}")
            results.append({
                'strategy': name,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*80}")
    print("TECHNICAL STRATEGY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Strategy':<30} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10} {'Trades':>10}")
    print(f"{'-'*80}")
    
    for r in results:
        if 'error' not in r:
            print(f"{r['strategy']:<30} {r['return']:>9.2%} {r['sharpe']:>10.2f} "
                  f"{r['drawdown']:>9.2%} {r['trades']:>10d}")
    
    # Best performers
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best_return = max(valid_results, key=lambda x: x['return'])
        best_sharpe = max(valid_results, key=lambda x: x['sharpe'] if x['sharpe'] else -999)
        
        print(f"\nBest Return: {best_return['strategy']} ({best_return['return']:.2%})")
        print(f"Best Risk-Adjusted: {best_sharpe['strategy']} (Sharpe: {best_sharpe['sharpe']:.2f})")


if __name__ == "__main__":
    logger.info("Starting technical indicator strategy tests...")
    run_technical_strategy_tests()
    logger.info("Technical strategy tests completed!")