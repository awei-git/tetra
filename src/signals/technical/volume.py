"""Volume-based technical indicators."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from ..base import BaseSignal, SignalType
from ..base.config import SignalConfig


class OBVSignal(BaseSignal):
    """On Balance Volume signal."""
    
    def __init__(self, config: SignalConfig, ema_period: Optional[int] = None):
        super().__init__(config)
        self.ema_period = ema_period or config.obv_ema_period
    
    @property
    def name(self) -> str:
        return "OBV"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.VOLUME
    
    @property
    def description(self) -> str:
        return f"On Balance Volume with EMA period {self.ema_period}"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close', 'volume']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['close', 'volume'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.ema_period + 1)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        # Calculate OBV
        obv = pd.Series(0, index=data.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        # Calculate signal line (EMA of OBV)
        obv_ema = obv.ewm(span=self.ema_period, adjust=False).mean()
        
        # Return OBV - Signal as the signal
        return obv - obv_ema
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'ema_period': self.ema_period}


class MFISignal(BaseSignal):
    """Money Flow Index signal."""
    
    def __init__(self, config: SignalConfig, period: Optional[int] = None):
        super().__init__(config)
        self.period = period or config.mfi_period
    
    @property
    def name(self) -> str:
        return "MFI"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.VOLUME
    
    @property
    def description(self) -> str:
        return f"Money Flow Index with period {self.period}"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close', 'volume']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['high', 'low', 'close', 'volume'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.period + 1)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate raw money flow
        money_flow = typical_price * volume
        
        # Determine positive and negative money flow
        positive_flow = pd.Series(0, index=data.index, dtype=float)
        negative_flow = pd.Series(0, index=data.index, dtype=float)
        
        tp_diff = typical_price.diff()
        
        positive_flow[tp_diff > 0] = money_flow[tp_diff > 0]
        negative_flow[tp_diff < 0] = money_flow[tp_diff < 0]
        
        # Calculate money flow ratio
        positive_mf = positive_flow.rolling(window=self.period).sum()
        negative_mf = negative_flow.rolling(window=self.period).sum()
        
        # Avoid division by zero
        mf_ratio = positive_mf / negative_mf.replace(0, 1e-10)
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + mf_ratio))
        
        return mfi
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'period': self.period}


class VWAPSignal(BaseSignal):
    """Volume Weighted Average Price signal."""
    
    def __init__(self, config: SignalConfig, period: Optional[int] = None):
        super().__init__(config)
        self.period = period or config.vwap_period
    
    @property
    def name(self) -> str:
        return "VWAP"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.VOLUME
    
    @property
    def description(self) -> str:
        return f"Volume Weighted Average Price with period {self.period}"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close', 'volume']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['high', 'low', 'close', 'volume'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.period)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate VWAP
        pv = typical_price * volume
        vwap = pv.rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()
        
        # Return price deviation from VWAP as percentage
        deviation = ((close - vwap) / vwap) * 100
        
        return deviation
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'period': self.period}


class ADLSignal(BaseSignal):
    """Accumulation/Distribution Line signal."""
    
    def __init__(self, config: SignalConfig, ema_period: int = 20):
        super().__init__(config)
        self.ema_period = ema_period
    
    @property
    def name(self) -> str:
        return "ADL"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.VOLUME
    
    @property
    def description(self) -> str:
        return "Accumulation/Distribution Line"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close', 'volume']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['high', 'low', 'close', 'volume'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.ema_period + 1)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Calculate Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)  # Handle division by zero when high == low
        
        # Calculate Money Flow Volume
        mfv = mfm * volume
        
        # Calculate ADL
        adl = mfv.cumsum()
        
        # Calculate signal line (EMA of ADL)
        adl_ema = adl.ewm(span=self.ema_period, adjust=False).mean()
        
        # Return normalized divergence
        divergence = (adl - adl_ema) / adl_ema.abs() * 100
        
        return divergence
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'ema_period': self.ema_period}


class CMFSignal(BaseSignal):
    """Chaikin Money Flow signal."""
    
    def __init__(self, config: SignalConfig, period: int = 21):
        super().__init__(config)
        self.period = period
    
    @property
    def name(self) -> str:
        return "CMF"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.VOLUME
    
    @property
    def description(self) -> str:
        return f"Chaikin Money Flow with period {self.period}"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close', 'volume']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['high', 'low', 'close', 'volume'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.period)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Calculate Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)  # Handle division by zero
        
        # Calculate Money Flow Volume
        mfv = mfm * volume
        
        # Calculate CMF
        cmf = mfv.rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()
        
        return cmf * 100  # Return as percentage
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'period': self.period}


class VolumeProfileSignal(BaseSignal):
    """Volume Profile signal - identifies high volume price levels."""
    
    def __init__(self, config: SignalConfig, 
                 period: int = 50,
                 num_bins: int = 20):
        super().__init__(config)
        self.period = period
        self.num_bins = num_bins
    
    @property
    def name(self) -> str:
        return "VolumeProfile"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.VOLUME
    
    @property
    def description(self) -> str:
        return f"Volume Profile with {self.period} period and {self.num_bins} bins"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'close', 'volume']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['high', 'low', 'close', 'volume'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.period)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        high = data['high']
        low = data['low']
        
        # Initialize result series
        result = pd.Series(index=data.index, dtype=float)
        
        for i in range(self.period, len(close)):
            # Get window data
            window_high = high.iloc[i-self.period:i]
            window_low = low.iloc[i-self.period:i]
            window_close = close.iloc[i-self.period:i]
            window_volume = volume.iloc[i-self.period:i]
            
            # Define price range and bins
            price_min = window_low.min()
            price_max = window_high.max()
            bins = np.linspace(price_min, price_max, self.num_bins + 1)
            
            # Calculate volume at each price level
            volume_profile = np.zeros(self.num_bins)
            
            for j in range(len(window_close)):
                # Find which bin the close price falls into
                bin_idx = np.searchsorted(bins[1:], window_close.iloc[j])
                volume_profile[bin_idx] = volume_profile[bin_idx] + window_volume.iloc[j]
            
            # Find the Point of Control (POC) - price level with highest volume
            poc_idx = np.argmax(volume_profile)
            poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
            
            # Calculate distance from current price to POC
            distance_to_poc = ((close.iloc[i] - poc_price) / poc_price) * 100
            
            result.iloc[i] = distance_to_poc
        
        return result
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'period': self.period,
            'num_bins': self.num_bins
        }


class EaseOfMovementSignal(BaseSignal):
    """Ease of Movement indicator."""
    
    def __init__(self, config: SignalConfig, 
                 period: int = 14,
                 smoothing: int = 14):
        super().__init__(config)
        self.period = period
        self.smoothing = smoothing
    
    @property
    def name(self) -> str:
        return "EOM"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.VOLUME
    
    @property
    def description(self) -> str:
        return f"Ease of Movement with period {self.period} and smoothing {self.smoothing}"
    
    @property
    def dependencies(self) -> List[str]:
        return ['high', 'low', 'volume']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['high', 'low', 'volume'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.period + self.smoothing)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Calculate distance moved
        distance = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        
        # Calculate EMV denominator (box ratio)
        box_ratio = (volume / 1000000) / (high - low)
        box_ratio = box_ratio.replace([np.inf, -np.inf], np.nan).fillna(1)
        
        # Calculate 1-period EMV
        emv_1 = distance / box_ratio
        
        # Smooth the EMV
        emv = emv_1.rolling(window=self.smoothing).mean()
        
        return emv
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'period': self.period,
            'smoothing': self.smoothing
        }