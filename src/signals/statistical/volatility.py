"""Volatility-based statistical signals."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats

from ..base import BaseSignal, SignalType
from ..base.config import SignalConfig


class RollingVolatilitySignal(BaseSignal):
    """Rolling volatility (standard deviation) signal."""
    
    def __init__(self, config: SignalConfig, 
                 window: Optional[int] = None,
                 annualize: bool = True):
        super().__init__(config)
        self.window = window or config.volatility_window
        self.annualize = annualize
        self.trading_days = 252
    
    @property
    def name(self) -> str:
        ann_str = "_ann" if self.annualize else ""
        return f"RollingVol_{self.window}{ann_str}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.STATISTICAL
    
    @property
    def description(self) -> str:
        ann_str = "annualized" if self.annualize else "raw"
        return f"Rolling {ann_str} volatility over {self.window} periods"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.window + 1)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        returns = np.log(close / close.shift(1))
        
        # Calculate rolling volatility
        vol = returns.rolling(window=self.window).std()
        
        # Annualize if requested
        if self.annualize:
            vol = vol * np.sqrt(self.trading_days)
        
        return vol * 100  # Return as percentage
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'window': self.window,
            'annualize': self.annualize
        }


class GARCHVolatilitySignal(BaseSignal):
    """GARCH(1,1) volatility forecast signal."""
    
    def __init__(self, config: SignalConfig, 
                 lookback: int = 252,
                 forecast_horizon: int = 1):
        super().__init__(config)
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
    
    @property
    def name(self) -> str:
        return f"GARCH_{self.lookback}_{self.forecast_horizon}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.STATISTICAL
    
    @property
    def description(self) -> str:
        return f"GARCH(1,1) {self.forecast_horizon}-step volatility forecast"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.lookback + 1)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        returns = np.log(close / close.shift(1)) * 100  # Percentage returns
        
        result = pd.Series(index=data.index, dtype=float)
        
        # Simple GARCH(1,1) implementation
        for i in range(self.lookback, len(returns)):
            window_returns = returns.iloc[i-self.lookback:i].dropna()
            
            if len(window_returns) < self.lookback * 0.8:  # Need at least 80% data
                continue
            
            # Initialize parameters
            omega = 0.00001
            alpha = 0.1
            beta = 0.85
            
            # Calculate unconditional variance
            variance = window_returns.var()
            
            # Simple GARCH forecast
            garch_var = omega + alpha * window_returns.iloc[-1]**2 + beta * variance
            
            # Multi-step forecast if needed
            for _ in range(self.forecast_horizon - 1):
                garch_var = omega + (alpha + beta) * garch_var
            
            result.iloc[i] = np.sqrt(garch_var * 252)  # Annualized volatility
        
        return result
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'lookback': self.lookback,
            'forecast_horizon': self.forecast_horizon
        }


class RealizedVolatilitySignal(BaseSignal):
    """Realized volatility using high-frequency data."""
    
    def __init__(self, config: SignalConfig, 
                 window: int = 20,
                 frequency: str = 'daily'):
        super().__init__(config)
        self.window = window
        self.frequency = frequency
    
    @property
    def name(self) -> str:
        return f"RealizedVol_{self.window}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.STATISTICAL
    
    @property
    def description(self) -> str:
        return f"Realized volatility over {self.window} periods"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['high', 'low', 'close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.window)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Parkinson's volatility estimator
        hl_ratio = np.log(high / low)
        parkinson = hl_ratio**2 / (4 * np.log(2))
        
        # Rolling sum for realized volatility
        realized_var = parkinson.rolling(window=self.window).sum()
        realized_vol = np.sqrt(realized_var * 252 / self.window)
        
        return realized_vol * 100
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'window': self.window,
            'frequency': self.frequency
        }


class VolatilityRegimeSignal(BaseSignal):
    """Volatility regime classification signal."""
    
    def __init__(self, config: SignalConfig,
                 lookback: Optional[int] = None,
                 low_vol_threshold: float = 15,
                 high_vol_threshold: float = 25):
        super().__init__(config)
        self.lookback = lookback or config.regime_lookback
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_threshold = high_vol_threshold
    
    @property
    def name(self) -> str:
        return "VolRegime"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.REGIME
    
    @property
    def description(self) -> str:
        return "Volatility regime classification (low/medium/high)"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, 30)  # Need at least 30 days
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        returns = np.log(close / close.shift(1))
        
        # Calculate 20-day rolling volatility
        vol = returns.rolling(window=20).std() * np.sqrt(252) * 100
        
        # Classify regime
        regime = pd.Series(index=data.index, dtype=float)
        regime[vol <= self.low_vol_threshold] = -1  # Low volatility
        regime[(vol > self.low_vol_threshold) & (vol <= self.high_vol_threshold)] = 0  # Medium
        regime[vol > self.high_vol_threshold] = 1  # High volatility
        
        return regime
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'lookback': self.lookback,
            'low_vol_threshold': self.low_vol_threshold,
            'high_vol_threshold': self.high_vol_threshold
        }


class VolumeWeightedVolatilitySignal(BaseSignal):
    """Volume-weighted volatility signal."""
    
    def __init__(self, config: SignalConfig, window: int = 20):
        super().__init__(config)
        self.window = window
    
    @property
    def name(self) -> str:
        return f"VWVol_{self.window}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.STATISTICAL
    
    @property
    def description(self) -> str:
        return f"Volume-weighted volatility over {self.window} periods"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close', 'volume']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['close', 'volume'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.window + 1)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        returns = np.log(close / close.shift(1))
        
        # Volume-weighted volatility
        vw_vol = pd.Series(index=data.index, dtype=float)
        
        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i-self.window+1:i+1]
            window_volume = volume.iloc[i-self.window+1:i+1]
            
            # Normalize volumes
            norm_volume = window_volume / window_volume.sum()
            
            # Calculate volume-weighted mean return
            vw_mean = (window_returns * norm_volume).sum()
            
            # Calculate volume-weighted variance
            vw_var = (norm_volume * (window_returns - vw_mean)**2).sum()
            
            # Annualized volatility
            vw_vol.iloc[i] = np.sqrt(vw_var * 252) * 100
        
        return vw_vol
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'window': self.window}