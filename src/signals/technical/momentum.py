"""Momentum technical indicators."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from ..base import BaseSignal, SignalType
from ..base.config import SignalConfig


class RSISignal(BaseSignal):
    """Relative Strength Index signal."""
    
    def __init__(self, config: SignalConfig, period: Optional[int] = None):
        super().__init__(config)
        self.period = period or config.rsi_period
    
    @property
    def name(self) -> str:
        return f"RSI_{self.period}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.MOMENTUM
    
    @property
    def description(self) -> str:
        return f"Relative Strength Index with period {self.period}"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.period + 1)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        delta = close.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        # Handle division by zero
        rs = avg_gain / avg_loss
        rs = rs.fillna(0)
        rs = rs.replace([np.inf, -np.inf], 0)
        
        rsi = 100 - (100 / (1 + rs))
        
        # Use EMA for smoothing after initial period
        for i in range(self.period, len(close)):
            if i == self.period:
                continue
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (self.period - 1) + gain.iloc[i]) / self.period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (self.period - 1) + loss.iloc[i]) / self.period
            
            if avg_loss.iloc[i] != 0:
                rs_val = avg_gain.iloc[i] / avg_loss.iloc[i]
                rsi.iloc[i] = 100 - (100 / (1 + rs_val))
            else:
                rsi.iloc[i] = 100
        
        return rsi
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'period': self.period}


class StochasticSignal(BaseSignal):
    """Stochastic Oscillator signal."""
    
    def __init__(self, config: SignalConfig, 
                 period: Optional[int] = None,
                 smooth_k: Optional[int] = None,
                 smooth_d: Optional[int] = None):
        super().__init__(config)
        self.period = period or config.stoch_period
        self.smooth_k = smooth_k or config.stoch_smooth_k
        self.smooth_d = smooth_d or config.stoch_smooth_d
    
    @property
    def name(self) -> str:
        return "Stochastic"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.MOMENTUM
    
    @property
    def description(self) -> str:
        return f"Stochastic Oscillator ({self.period},{self.smooth_k},{self.smooth_d})"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['high', 'low', 'close'])
        if not is_valid:
            return False, msg
        
        min_length = self.period + self.smooth_k + self.smooth_d
        return self._validate_data_length(data, min_length)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate %K
        lowest_low = low.rolling(window=self.period).min()
        highest_high = high.rolling(window=self.period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_percent = k_percent.fillna(50)  # Fill NaN with neutral value
        
        # Smooth %K to get %K
        k_smooth = k_percent.rolling(window=self.smooth_k).mean()
        
        # Calculate %D (signal line)
        d_percent = k_smooth.rolling(window=self.smooth_d).mean()
        
        # Return %K - %D as the signal
        return k_smooth - d_percent
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'period': self.period,
            'smooth_k': self.smooth_k,
            'smooth_d': self.smooth_d
        }


class CCISignal(BaseSignal):
    """Commodity Channel Index signal."""
    
    def __init__(self, config: SignalConfig, period: Optional[int] = None):
        super().__init__(config)
        self.period = period or config.cci_period
    
    @property
    def name(self) -> str:
        return "CCI"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.MOMENTUM
    
    @property
    def description(self) -> str:
        return f"Commodity Channel Index with period {self.period}"
    
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
        
        # Typical Price
        tp = (high + low + close) / 3
        
        # Simple Moving Average of TP
        sma = tp.rolling(window=self.period).mean()
        
        # Mean Deviation
        mad = tp.rolling(window=self.period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        
        # CCI
        cci = (tp - sma) / (0.015 * mad)
        
        return cci
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'period': self.period}


class ROCSignal(BaseSignal):
    """Rate of Change signal."""
    
    def __init__(self, config: SignalConfig, period: Optional[int] = None):
        super().__init__(config)
        self.period = period or config.roc_period
    
    @property
    def name(self) -> str:
        return f"ROC_{self.period}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.MOMENTUM
    
    @property
    def description(self) -> str:
        return f"Rate of Change with period {self.period}"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.period + 1)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        roc = ((close - close.shift(self.period)) / close.shift(self.period)) * 100
        return roc
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'period': self.period}


class WilliamsRSignal(BaseSignal):
    """Williams %R signal."""
    
    def __init__(self, config: SignalConfig, period: Optional[int] = None):
        super().__init__(config)
        self.period = period or config.williams_r_period
    
    @property
    def name(self) -> str:
        return "WilliamsR"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.MOMENTUM
    
    @property
    def description(self) -> str:
        return f"Williams %R with period {self.period}"
    
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
        
        highest_high = high.rolling(window=self.period).max()
        lowest_low = low.rolling(window=self.period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'period': self.period}


class MomentumSignal(BaseSignal):
    """Simple Momentum signal."""
    
    def __init__(self, config: SignalConfig, period: int = 10):
        super().__init__(config)
        self.period = period
    
    @property
    def name(self) -> str:
        return f"Momentum_{self.period}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.MOMENTUM
    
    @property
    def description(self) -> str:
        return f"Simple Momentum with period {self.period}"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.period + 1)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        momentum = close - close.shift(self.period)
        return momentum
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'period': self.period}


class TSISignal(BaseSignal):
    """True Strength Index signal."""
    
    def __init__(self, config: SignalConfig, 
                 fast_period: int = 13,
                 slow_period: int = 25):
        super().__init__(config)
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    @property
    def name(self) -> str:
        return "TSI"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.MOMENTUM
    
    @property
    def description(self) -> str:
        return f"True Strength Index ({self.fast_period},{self.slow_period})"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.slow_period + self.fast_period)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # Calculate price changes
        momentum = close.diff()
        
        # Double smoothed momentum
        ema_momentum_slow = momentum.ewm(span=self.slow_period, adjust=False).mean()
        ema_momentum_fast = ema_momentum_slow.ewm(span=self.fast_period, adjust=False).mean()
        
        # Double smoothed absolute momentum
        abs_momentum = momentum.abs()
        ema_abs_momentum_slow = abs_momentum.ewm(span=self.slow_period, adjust=False).mean()
        ema_abs_momentum_fast = ema_abs_momentum_slow.ewm(span=self.fast_period, adjust=False).mean()
        
        # TSI
        tsi = 100 * (ema_momentum_fast / ema_abs_momentum_fast)
        
        return tsi
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period
        }


class UltimateOscillatorSignal(BaseSignal):
    """Ultimate Oscillator signal."""
    
    def __init__(self, config: SignalConfig,
                 period1: int = 7,
                 period2: int = 14,
                 period3: int = 28):
        super().__init__(config)
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
    
    @property
    def name(self) -> str:
        return "UltimateOsc"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.MOMENTUM
    
    @property
    def description(self) -> str:
        return f"Ultimate Oscillator ({self.period1},{self.period2},{self.period3})"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['high', 'low', 'close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.period3 + 1)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate Buying Pressure
        bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
        
        # Calculate True Range
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        # Calculate averages for each period
        avg1 = bp.rolling(self.period1).sum() / tr.rolling(self.period1).sum()
        avg2 = bp.rolling(self.period2).sum() / tr.rolling(self.period2).sum()
        avg3 = bp.rolling(self.period3).sum() / tr.rolling(self.period3).sum()
        
        # Calculate Ultimate Oscillator
        uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
        
        return uo
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'period1': self.period1,
            'period2': self.period2,
            'period3': self.period3
        }