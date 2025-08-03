"""Returns-based statistical signals."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from ..base import BaseSignal, SignalType
from ..base.config import SignalConfig


class SimpleReturnsSignal(BaseSignal):
    """Simple returns signal."""
    
    def __init__(self, config: SignalConfig, period: int = 1):
        super().__init__(config)
        self.period = period
    
    @property
    def name(self) -> str:
        return f"SimpleReturns_{self.period}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.STATISTICAL
    
    @property
    def description(self) -> str:
        return f"Simple returns over {self.period} periods"
    
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
        returns = (close - close.shift(self.period)) / close.shift(self.period)
        return returns * 100  # Return as percentage
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'period': self.period}


class LogReturnsSignal(BaseSignal):
    """Log returns signal."""
    
    def __init__(self, config: SignalConfig, period: int = 1):
        super().__init__(config)
        self.period = period
    
    @property
    def name(self) -> str:
        return f"LogReturns_{self.period}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.STATISTICAL
    
    @property
    def description(self) -> str:
        return f"Log returns over {self.period} periods"
    
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
        log_returns = np.log(close / close.shift(self.period))
        return log_returns * 100  # Return as percentage
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'period': self.period}


class CumulativeReturnsSignal(BaseSignal):
    """Cumulative returns signal."""
    
    def __init__(self, config: SignalConfig, window: int = 252):
        super().__init__(config)
        self.window = window
    
    @property
    def name(self) -> str:
        return f"CumReturns_{self.window}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.STATISTICAL
    
    @property
    def description(self) -> str:
        return f"Cumulative returns over {self.window} periods"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.window)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # Calculate rolling cumulative returns
        cum_returns = pd.Series(index=data.index, dtype=float)
        
        for i in range(self.window, len(close)):
            start_price = close.iloc[i - self.window]
            end_price = close.iloc[i]
            cum_returns.iloc[i] = ((end_price / start_price) - 1) * 100
        
        return cum_returns
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'window': self.window}


class RollingReturnsSignal(BaseSignal):
    """Rolling returns with various statistics."""
    
    def __init__(self, config: SignalConfig, window: int = 20, statistic: str = 'mean'):
        super().__init__(config)
        self.window = window
        self.statistic = statistic
        self.valid_statistics = ['mean', 'median', 'std', 'skew', 'kurt']
        
        if statistic not in self.valid_statistics:
            raise ValueError(f"Invalid statistic: {statistic}. Must be one of {self.valid_statistics}")
    
    @property
    def name(self) -> str:
        return f"RollingReturns_{self.window}_{self.statistic}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.STATISTICAL
    
    @property
    def description(self) -> str:
        return f"Rolling {self.statistic} of returns over {self.window} periods"
    
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
        returns = close.pct_change()
        
        if self.statistic == 'mean':
            result = returns.rolling(window=self.window).mean()
        elif self.statistic == 'median':
            result = returns.rolling(window=self.window).median()
        elif self.statistic == 'std':
            result = returns.rolling(window=self.window).std()
        elif self.statistic == 'skew':
            result = returns.rolling(window=self.window).skew()
        elif self.statistic == 'kurt':
            result = returns.rolling(window=self.window).kurt()
        
        return result * 100  # Return as percentage
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'window': self.window,
            'statistic': self.statistic
        }


class DrawdownSignal(BaseSignal):
    """Current drawdown from rolling peak."""
    
    def __init__(self, config: SignalConfig, window: Optional[int] = None):
        super().__init__(config)
        self.window = window  # None means all-time high
    
    @property
    def name(self) -> str:
        window_str = f"_{self.window}" if self.window else "_ATH"
        return f"Drawdown{window_str}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.STATISTICAL
    
    @property
    def description(self) -> str:
        window_desc = f"{self.window}-period" if self.window else "all-time"
        return f"Drawdown from {window_desc} high"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['close'])
        if not is_valid:
            return False, msg
        
        min_length = self.window if self.window else 2
        return self._validate_data_length(data, min_length)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        if self.window:
            # Rolling window peak
            peak = close.rolling(window=self.window, min_periods=1).max()
        else:
            # All-time high (expanding window)
            peak = close.expanding(min_periods=1).max()
        
        # Calculate drawdown
        drawdown = (close - peak) / peak * 100
        
        return drawdown
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'window': self.window}


class MaxDrawdownSignal(BaseSignal):
    """Maximum drawdown over rolling window."""
    
    def __init__(self, config: SignalConfig, window: int = 252):
        super().__init__(config)
        self.window = window
    
    @property
    def name(self) -> str:
        return f"MaxDrawdown_{self.window}"
    
    @property
    def signal_type(self) -> SignalType:
        return SignalType.STATISTICAL
    
    @property
    def description(self) -> str:
        return f"Maximum drawdown over {self.window} periods"
    
    @property
    def dependencies(self) -> List[str]:
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        is_valid, msg = self._validate_required_columns(data, ['close'])
        if not is_valid:
            return False, msg
        
        return self._validate_data_length(data, self.window)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        max_dd = pd.Series(index=data.index, dtype=float)
        
        for i in range(self.window, len(close)):
            window_data = close.iloc[i-self.window+1:i+1]
            
            # Calculate running peak
            running_peak = window_data.expanding().max()
            
            # Calculate drawdowns
            drawdowns = (window_data - running_peak) / running_peak
            
            # Get maximum drawdown
            max_dd.iloc[i] = drawdowns.min() * 100
        
        return max_dd
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'window': self.window}