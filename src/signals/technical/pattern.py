"""Pattern recognition technical indicators."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from ..base import BaseSignal, SignalType
from ..base.config import SignalConfig


class PivotPointsSignal(BaseSignal):
    """Pivot Points signal with support and resistance levels."""
    
    def __init__(self, config: SignalConfig, 
                 levels: Optional[int] = None):
        super().__init__(config)
        self.levels = levels or config.pivot_support_resistance_levels
    
    @property
    def name(self) -> str:
        return "PivotPoints"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.PATTERN
    
    @property
    def description(self) -> str:
        return f"Pivot Points with {self.levels} support/resistance levels"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['high', 'low', 'close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, 2)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate pivot point
        pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
        
        # Calculate range
        range_hl = high.shift(1) - low.shift(1)
        
        # Calculate nearest support/resistance level distance
        result = pd.Series(index=data.index, dtype=float)
        
        for i in range(1, len(close)):
            if pd.isna(pivot.iloc[i]):
                continue
                
            # Calculate support and resistance levels
            levels = [pivot.iloc[i]]  # Include pivot as a level
            
            for level in range(1, self.levels + 1):
                # Resistance levels
                r1 = pivot.iloc[i] + level * range_hl.iloc[i]
                levels.append(r1)
                
                # Support levels
                s1 = pivot.iloc[i] - level * range_hl.iloc[i]
                levels.append(s1)
            
            # Find nearest level to current price
            current_price = close.iloc[i]
            distances = [abs(current_price - level) for level in levels]
            min_distance = min(distances)
            nearest_level = levels[distances.index(min_distance)]
            
            # Return signed distance as percentage
            result.iloc[i] = ((current_price - nearest_level) / current_price) * 100
        
        return result
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'levels': self.levels}


class FibonacciRetracementSignal(BaseSignal):
    """Fibonacci Retracement signal."""
    
    def __init__(self, config: SignalConfig, lookback: int = 50):
        super().__init__(config)
        self.lookback = lookback
        self.fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    
    @property
    def name(self) -> str:
        return "FibonacciRetracement"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.PATTERN
    
    @property
    def description(self) -> str:
        return f"Fibonacci Retracement with {self.lookback} period lookback"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['high', 'low', 'close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.lookback)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        result = pd.Series(index=data.index, dtype=float)
        
        for i in range(self.lookback, len(close)):
            # Find swing high and low in lookback period
            window_high = high.iloc[i-self.lookback:i]
            window_low = low.iloc[i-self.lookback:i]
            
            swing_high = window_high.max()
            swing_low = window_low.min()
            
            # Calculate Fibonacci levels
            diff = swing_high - swing_low
            fib_levels_price = []
            
            for level in self.fib_levels:
                fib_price = swing_high - (diff * level)
                fib_levels_price.append(fib_price)
            
            # Find nearest Fibonacci level
            current_price = close.iloc[i]
            distances = [abs(current_price - level) for level in fib_levels_price]
            min_distance = min(distances)
            nearest_fib_idx = distances.index(min_distance)
            nearest_fib_level = fib_levels_price[nearest_fib_idx]
            
            # Return distance to nearest Fibonacci level as percentage
            result.iloc[i] = ((current_price - nearest_fib_level) / current_price) * 100
        
        return result
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'lookback': self.lookback,
            'fib_levels': self.fib_levels
        }


class SupportResistanceSignal(BaseSignal):
    """Dynamic Support and Resistance signal."""
    
    def __init__(self, config: SignalConfig, 
                 lookback: int = 50,
                 min_touches: int = 3,
                 tolerance: float = 0.02):
        super().__init__(config)
        self.lookback = lookback
        self.min_touches = min_touches
        self.tolerance = tolerance
    
    @property
    def name(self) -> str:
        return "SupportResistance"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.PATTERN
    
    @property
    def description(self) -> str:
        return f"Dynamic Support/Resistance with {self.lookback} lookback"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['high', 'low', 'close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.lookback)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        result = pd.Series(index=data.index, dtype=float)
        
        for i in range(self.lookback, len(close)):
            # Get window data
            window_high = high.iloc[i-self.lookback:i]
            window_low = low.iloc[i-self.lookback:i]
            window_close = close.iloc[i-self.lookback:i]
            
            # Find potential support/resistance levels
            levels = []
            
            # Check each price level in the window
            all_prices = pd.concat([window_high, window_low, window_close]).unique()
            
            for level in all_prices:
                touches = 0
                
                # Count how many times price touched this level
                for j in range(len(window_high)):
                    high_touch = abs(window_high.iloc[j] - level) / level <= self.tolerance
                    low_touch = abs(window_low.iloc[j] - level) / level <= self.tolerance
                    close_touch = abs(window_close.iloc[j] - level) / level <= self.tolerance
                    
                    if high_touch or low_touch or close_touch:
                        touches += 1
                
                if touches >= self.min_touches:
                    levels.append(level)
            
            if len(levels) > 0:
                # Find nearest level
                current_price = close.iloc[i]
                distances = [abs(current_price - level) for level in levels]
                min_distance = min(distances)
                nearest_level = levels[distances.index(min_distance)]
                
                # Calculate strength based on number of touches
                strength = len([l for l in levels if abs(l - nearest_level) / nearest_level <= self.tolerance])
                
                # Return signed distance weighted by strength
                distance_pct = ((current_price - nearest_level) / current_price) * 100
                result.iloc[i] = distance_pct * (1 + strength / 10)  # Weight by strength
            else:
                result.iloc[i] = 0
        
        return result
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'lookback': self.lookback,
            'min_touches': self.min_touches,
            'tolerance': self.tolerance
        }


class CandlePatternSignal(BaseSignal):
    """Candlestick pattern recognition signal."""
    
    def __init__(self, config: SignalConfig):
        super().__init__(config)
        self.patterns = {
            'doji': self._detect_doji,
            'hammer': self._detect_hammer,
            'shooting_star': self._detect_shooting_star,
            'engulfing': self._detect_engulfing,
            'harami': self._detect_harami,
            'morning_star': self._detect_morning_star,
            'evening_star': self._detect_evening_star
        }
    
    @property
    def name(self) -> str:
        return "CandlePattern"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.PATTERN
    
    @property
    def description(self) -> str:
        return "Candlestick pattern recognition"
    
    @property
    def dependencies(self) -> List[str]:
        return ['open', 'high', 'low', 'close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['open', 'high', 'low', 'close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, 5)  # Need at least 5 candles for some patterns
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        result = pd.Series(0, index=data.index, dtype=float)
        
        # Detect each pattern and aggregate signals
        for pattern_name, detector in self.patterns.items():
            pattern_signal = detector(data)
            result += pattern_signal
        
        # Normalize to -100 to 100 range
        result = result.clip(-100, 100)
        
        return result
    
    def _detect_doji(self, data: pd.DataFrame) -> pd.Series:
        """Detect doji pattern."""
        open_price = data['open']
        close = data['close']
        high = data['high']
        low = data['low']
        
        body = abs(close - open_price)
        range_hl = high - low
        
        # Doji: very small body relative to range
        doji = (body / range_hl < 0.1) & (range_hl > 0)
        
        return doji.astype(float) * 10  # Neutral signal
    
    def _detect_hammer(self, data: pd.DataFrame) -> pd.Series:
        """Detect hammer pattern (bullish)."""
        open_price = data['open']
        close = data['close']
        high = data['high']
        low = data['low']
        
        body = abs(close - open_price)
        upper_shadow = high - pd.concat([open_price, close], axis=1).max(axis=1)
        lower_shadow = pd.concat([open_price, close], axis=1).min(axis=1) - low
        
        # Hammer: small body at top, long lower shadow
        hammer = (lower_shadow > 2 * body) & (upper_shadow < 0.1 * body) & (body > 0)
        
        return hammer.astype(float) * 20  # Bullish signal
    
    def _detect_shooting_star(self, data: pd.DataFrame) -> pd.Series:
        """Detect shooting star pattern (bearish)."""
        open_price = data['open']
        close = data['close']
        high = data['high']
        low = data['low']
        
        body = abs(close - open_price)
        upper_shadow = high - pd.concat([open_price, close], axis=1).max(axis=1)
        lower_shadow = pd.concat([open_price, close], axis=1).min(axis=1) - low
        
        # Shooting star: small body at bottom, long upper shadow
        shooting_star = (upper_shadow > 2 * body) & (lower_shadow < 0.1 * body) & (body > 0)
        
        return shooting_star.astype(float) * -20  # Bearish signal
    
    def _detect_engulfing(self, data: pd.DataFrame) -> pd.Series:
        """Detect engulfing patterns."""
        open_price = data['open']
        close = data['close']
        
        prev_open = open_price.shift(1)
        prev_close = close.shift(1)
        
        # Bullish engulfing
        bullish_engulfing = (
            (prev_close < prev_open) &  # Previous candle is bearish
            (close > open_price) &       # Current candle is bullish
            (open_price <= prev_close) & # Opens below or at previous close
            (close >= prev_open)         # Closes above or at previous open
        )
        
        # Bearish engulfing
        bearish_engulfing = (
            (prev_close > prev_open) &  # Previous candle is bullish
            (close < open_price) &       # Current candle is bearish
            (open_price >= prev_close) & # Opens above or at previous close
            (close <= prev_open)         # Closes below or at previous open
        )
        
        signal = pd.Series(0, index=data.index, dtype=float)
        signal[bullish_engulfing] = 30   # Strong bullish
        signal[bearish_engulfing] = -30  # Strong bearish
        
        return signal
    
    def _detect_harami(self, data: pd.DataFrame) -> pd.Series:
        """Detect harami patterns."""
        open_price = data['open']
        close = data['close']
        
        prev_open = open_price.shift(1)
        prev_close = close.shift(1)
        
        # Current candle body inside previous candle body
        inside_bar = (
            (open_price > pd.concat([prev_open, prev_close], axis=1).min(axis=1)) &
            (open_price < pd.concat([prev_open, prev_close], axis=1).max(axis=1)) &
            (close > pd.concat([prev_open, prev_close], axis=1).min(axis=1)) &
            (close < pd.concat([prev_open, prev_close], axis=1).max(axis=1))
        )
        
        # Bullish harami (after downtrend)
        bullish_harami = inside_bar & (prev_close < prev_open)
        
        # Bearish harami (after uptrend)
        bearish_harami = inside_bar & (prev_close > prev_open)
        
        signal = pd.Series(0, index=data.index, dtype=float)
        signal[bullish_harami] = 15   # Moderate bullish
        signal[bearish_harami] = -15  # Moderate bearish
        
        return signal
    
    def _detect_morning_star(self, data: pd.DataFrame) -> pd.Series:
        """Detect morning star pattern (bullish)."""
        open_price = data['open']
        close = data['close']
        
        # Three-candle pattern
        first_close = close.shift(2)
        first_open = open_price.shift(2)
        second_close = close.shift(1)
        second_open = open_price.shift(1)
        
        morning_star = (
            (first_close < first_open) &  # First candle is bearish
            (abs(second_close - second_open) < 0.1 * abs(first_close - first_open)) &  # Second is small
            (close > open_price) &  # Third candle is bullish
            (close > (first_close + first_open) / 2)  # Third closes above first's midpoint
        )
        
        return morning_star.astype(float) * 40  # Strong bullish signal
    
    def _detect_evening_star(self, data: pd.DataFrame) -> pd.Series:
        """Detect evening star pattern (bearish)."""
        open_price = data['open']
        close = data['close']
        
        # Three-candle pattern
        first_close = close.shift(2)
        first_open = open_price.shift(2)
        second_close = close.shift(1)
        second_open = open_price.shift(1)
        
        evening_star = (
            (first_close > first_open) &  # First candle is bullish
            (abs(second_close - second_open) < 0.1 * abs(first_close - first_open)) &  # Second is small
            (close < open_price) &  # Third candle is bearish
            (close < (first_close + first_open) / 2)  # Third closes below first's midpoint
        )
        
        return evening_star.astype(float) * -40  # Strong bearish signal
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'patterns': list(self.patterns.keys())}