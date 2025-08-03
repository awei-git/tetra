"""Volatility technical indicators."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from ..base import BaseSignal, SignalType
from ..base.config import SignalConfig


class BollingerBandsSignal(BaseSignal):
    """Bollinger Bands signal."""
    
    def __init__(self, config: SignalConfig, 
                 period: Optional[int] = None,
                 std_dev: Optional[float] = None):
        super().__init__(config)
        self.period = period or config.bb_period
        self.std_dev = std_dev or config.bb_std
    
    @property
    def name(self) -> str:
        return "BollingerBands"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.VOLATILITY
    
    @property
    def description(self) -> str:
        return f"Bollinger Bands with period {self.period} and {self.std_dev} std dev"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.period)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # Calculate middle band (SMA)
        middle = close.rolling(window=self.period).mean()
        
        # Calculate standard deviation
        std = close.rolling(window=self.period).std()
        
        # Calculate bands
        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)
        
        # Return band width as percentage of middle band
        band_width = ((upper - lower) / middle) * 100
        
        return band_width
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'period': self.period,
            'std_dev': self.std_dev
        }


class ATRSignal(BaseSignal):
    """Average True Range signal."""
    
    def __init__(self, config: SignalConfig, period: Optional[int] = None):
        super().__init__(config)
        self.period = period or config.atr_period
    
    @property
    def name(self) -> str:
        return f"ATR_{self.period}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.VOLATILITY
    
    @property
    def description(self) -> str:
        return f"Average True Range with period {self.period}"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['high', 'low', 'close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.period + 1)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR using EMA
        atr = tr.ewm(span=self.period, adjust=False).mean()
        
        return atr
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'period': self.period}


class KeltnerChannelSignal(BaseSignal):
    """Keltner Channel signal."""
    
    def __init__(self, config: SignalConfig,
                 ema_period: int = 20,
                 atr_period: int = 10,
                 multiplier: float = 2.0):
        super().__init__(config)
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier
    
    @property
    def name(self) -> str:
        return "KeltnerChannel"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.VOLATILITY
    
    @property
    def description(self) -> str:
        return f"Keltner Channel with EMA {self.ema_period}, ATR {self.atr_period}"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['high', 'low', 'close'])
        if not is_valid:
            return False, msg
        
        min_length = max(self.ema_period, self.atr_period) + 1
        return self._validate_data_length(data, min_length)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate middle line (EMA of typical price)
        typical_price = (high + low + close) / 3
        middle = typical_price.ewm(span=self.ema_period, adjust=False).mean()
        
        # Calculate ATR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=self.atr_period, adjust=False).mean()
        
        # Calculate bands
        upper = middle + (atr * self.multiplier)
        lower = middle - (atr * self.multiplier)
        
        # Return channel width as percentage
        channel_width = ((upper - lower) / middle) * 100
        
        return channel_width
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'ema_period': self.ema_period,
            'atr_period': self.atr_period,
            'multiplier': self.multiplier
        }


class DonchianChannelSignal(BaseSignal):
    """Donchian Channel signal."""
    
    def __init__(self, config: SignalConfig, period: int = 20):
        super().__init__(config)
        self.period = period
    
    @property
    def name(self) -> str:
        return f"DonchianChannel_{self.period}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.VOLATILITY
    
    @property
    def description(self) -> str:
        return f"Donchian Channel with period {self.period}"
    
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
        
        # Calculate channel
        upper = high.rolling(window=self.period).max()
        lower = low.rolling(window=self.period).min()
        middle = (upper + lower) / 2
        
        # Return position of close relative to channel (-1 to 1)
        position = (close - middle) / ((upper - lower) / 2)
        position = position.clip(-1, 1)
        
        return position
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'period': self.period}


class StandardDeviationSignal(BaseSignal):
    """Standard Deviation signal."""
    
    def __init__(self, config: SignalConfig, period: int = 20):
        super().__init__(config)
        self.period = period
    
    @property
    def name(self) -> str:
        return f"StdDev_{self.period}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.VOLATILITY
    
    @property
    def description(self) -> str:
        return f"Standard Deviation with period {self.period}"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.period)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        std_dev = close.rolling(window=self.period).std()
        
        # Normalize by mean for comparability
        mean = close.rolling(window=self.period).mean()
        normalized_std = (std_dev / mean) * 100
        
        return normalized_std
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'period': self.period}


class HistoricalVolatilitySignal(BaseSignal):
    """Historical Volatility (realized volatility) signal."""
    
    def __init__(self, config: SignalConfig, 
                 period: int = 20,
                 annualize: bool = True):
        super().__init__(config)
        self.period = period
        self.annualize = annualize
        self.trading_days = 252  # Standard trading days per year
    
    @property
    def name(self) -> str:
        return f"HV_{self.period}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.VOLATILITY
    
    @property
    def description(self) -> str:
        ann_str = "annualized" if self.annualize else "raw"
        return f"Historical Volatility ({ann_str}) with period {self.period}"
    
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
        
        # Calculate log returns
        log_returns = np.log(close / close.shift(1))
        
        # Calculate rolling standard deviation
        volatility = log_returns.rolling(window=self.period).std()
        
        # Annualize if requested
        if self.annualize:
            volatility = volatility * np.sqrt(self.trading_days)
        
        # Convert to percentage
        volatility = volatility * 100
        
        return volatility
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'period': self.period,
            'annualize': self.annualize
        }