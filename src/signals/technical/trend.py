"""Trend following technical indicators."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from ..base import BaseSignal, SignalType
from ..base.config import SignalConfig


class SMASignal(BaseSignal):
    """Simple Moving Average signal."""
    
    def __init__(self, config: SignalConfig, period: int = 20):
        super().__init__(config)
        self.period = period
    
    @property
    def name(self) -> str:
        return f"SMA_{self.period}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.TREND
    
    @property
    def description(self) -> str:
        return f"Simple Moving Average with period {self.period}"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.period)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        return data['close'].rolling(window=self.period).mean()
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'period': self.period}


class EMASignal(BaseSignal):
    """Exponential Moving Average signal."""
    
    def __init__(self, config: SignalConfig, period: int = 20):
        super().__init__(config)
        self.period = period
    
    @property
    def name(self) -> str:
        return f"EMA_{self.period}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.TREND
    
    @property
    def description(self) -> str:
        return f"Exponential Moving Average with period {self.period}"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.period)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        return data['close'].ewm(span=self.period, adjust=False).mean()
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'period': self.period}


class WMASignal(BaseSignal):
    """Weighted Moving Average signal."""
    
    def __init__(self, config: SignalConfig, period: int = 20):
        super().__init__(config)
        self.period = period
    
    @property
    def name(self) -> str:
        return f"WMA_{self.period}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.TREND
    
    @property
    def description(self) -> str:
        return f"Weighted Moving Average with period {self.period}"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.period)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        weights = np.arange(1, self.period + 1)
        wma = data['close'].rolling(self.period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
        return wma
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'period': self.period}


class MACDSignal(BaseSignal):
    """MACD (Moving Average Convergence Divergence) signal."""
    
    def __init__(self, config: SignalConfig, 
                 fast_period: Optional[int] = None,
                 slow_period: Optional[int] = None,
                 signal_period: Optional[int] = None):
        super().__init__(config)
        self.fast_period = fast_period or config.macd_fast
        self.slow_period = slow_period or config.macd_slow
        self.signal_period = signal_period or config.macd_signal
    
    @property
    def name(self) -> str:
        return "MACD"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.TREND
    
    @property
    def description(self) -> str:
        return f"MACD ({self.fast_period},{self.slow_period},{self.signal_period})"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.slow_period + self.signal_period)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        ema_fast = data['close'].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = data['close'].ewm(span=self.slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        # Return the histogram as the main signal
        return histogram
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period
        }


class ADXSignal(BaseSignal):
    """Average Directional Index signal."""
    
    def __init__(self, config: SignalConfig, period: Optional[int] = None):
        super().__init__(config)
        self.period = period or config.adx_period
    
    @property
    def name(self) -> str:
        return "ADX"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.TREND
    
    @property
    def description(self) -> str:
        return f"Average Directional Index with period {self.period}"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['high', 'low', 'close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.period * 2)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate directional movements
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = pd.Series(0.0, index=data.index)
        minus_dm = pd.Series(0.0, index=data.index)
        
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
        
        # Calculate smoothed values
        atr = tr.ewm(span=self.period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=self.period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=self.period, adjust=False).mean() / atr)
        
        # Calculate ADX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.ewm(span=self.period, adjust=False).mean()
        
        return adx
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'period': self.period}


class ParabolicSARSignal(BaseSignal):
    """Parabolic SAR (Stop and Reverse) signal."""
    
    def __init__(self, config: SignalConfig, 
                 initial_af: float = 0.02,
                 max_af: float = 0.2,
                 increment: float = 0.02):
        super().__init__(config)
        self.initial_af = initial_af
        self.max_af = max_af
        self.increment = increment
    
    @property
    def name(self) -> str:
        return "PSAR"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.TREND
    
    @property
    def description(self) -> str:
        return "Parabolic SAR trend following indicator"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['high', 'low', 'close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, 2)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        sar = np.zeros_like(close)
        af = self.initial_af
        ep = high[0]
        sar[0] = low[0]
        trend = 1  # 1 for uptrend, -1 for downtrend
        
        for i in range(1, len(close)):
            if trend == 1:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                if low[i] <= sar[i]:
                    trend = -1
                    sar[i] = ep
                    ep = low[i]
                    af = self.initial_af
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + self.increment, self.max_af)
            else:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                if high[i] >= sar[i]:
                    trend = 1
                    sar[i] = ep
                    ep = high[i]
                    af = self.initial_af
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + self.increment, self.max_af)
        
        return pd.Series(sar, index=data.index)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'initial_af': self.initial_af,
            'max_af': self.max_af,
            'increment': self.increment
        }


class IchimokuSignal(BaseSignal):
    """Ichimoku Cloud signal."""
    
    def __init__(self, config: SignalConfig,
                 tenkan_period: int = 9,
                 kijun_period: int = 26,
                 senkou_b_period: int = 52):
        super().__init__(config)
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
    
    @property
    def name(self) -> str:
        return "Ichimoku"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.TREND
    
    @property
    def description(self) -> str:
        return "Ichimoku Cloud trend system"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['high', 'low', 'close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.senkou_b_period + self.kijun_period)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        
        # Tenkan-sen (Conversion Line)
        tenkan = (high.rolling(self.tenkan_period).max() + 
                 low.rolling(self.tenkan_period).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun = (high.rolling(self.kijun_period).max() + 
                low.rolling(self.kijun_period).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_a = (tenkan + kijun) / 2
        
        # Senkou Span B (Leading Span B)
        senkou_b = (high.rolling(self.senkou_b_period).max() + 
                   low.rolling(self.senkou_b_period).min()) / 2
        
        # Chikou Span (Lagging Span) - close shifted back
        chikou = data['close'].shift(-self.kijun_period)
        
        # Return cloud width as main signal (positive = bullish, negative = bearish)
        cloud_width = senkou_a - senkou_b
        
        return cloud_width.shift(self.kijun_period)  # Shift forward as per Ichimoku rules
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'tenkan_period': self.tenkan_period,
            'kijun_period': self.kijun_period,
            'senkou_b_period': self.senkou_b_period
        }


class SupertrendSignal(BaseSignal):
    """Supertrend indicator signal."""
    
    def __init__(self, config: SignalConfig,
                 period: int = 7,
                 multiplier: float = 3.0):
        super().__init__(config)
        self.period = period
        self.multiplier = multiplier
    
    @property
    def name(self) -> str:
        return "Supertrend"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.TREND
    
    @property
    def description(self) -> str:
        return f"Supertrend indicator with period {self.period} and multiplier {self.multiplier}"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['high', 'low', 'close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.period)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate ATR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.period).mean()
        
        # Calculate basic bands
        hl_avg = (high + low) / 2
        up_band = hl_avg + (self.multiplier * atr)
        dn_band = hl_avg - (self.multiplier * atr)
        
        # Initialize supertrend
        supertrend = pd.Series(index=data.index, dtype=float)
        direction = pd.Series(index=data.index, dtype=float)
        
        for i in range(self.period, len(close)):
            if i == self.period:
                if close.iloc[i] <= up_band.iloc[i]:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = up_band.iloc[i]
                else:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = dn_band.iloc[i]
            else:
                # Uptrend
                if direction.iloc[i-1] == -1:
                    if close.iloc[i] <= up_band.iloc[i]:
                        direction.iloc[i] = -1
                        supertrend.iloc[i] = min(up_band.iloc[i], supertrend.iloc[i-1])
                    else:
                        direction.iloc[i] = 1
                        supertrend.iloc[i] = dn_band.iloc[i]
                # Downtrend
                else:
                    if close.iloc[i] >= dn_band.iloc[i]:
                        direction.iloc[i] = 1
                        supertrend.iloc[i] = max(dn_band.iloc[i], supertrend.iloc[i-1])
                    else:
                        direction.iloc[i] = -1
                        supertrend.iloc[i] = up_band.iloc[i]
        
        # Return direction: 1 for buy, -1 for sell
        return direction
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'period': self.period,
            'multiplier': self.multiplier
        }