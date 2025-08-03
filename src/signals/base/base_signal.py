"""Base class for all signal computations."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import time

from .types import SignalMetadata, SignalType
from .config import SignalConfig


class BaseSignal(ABC):
    """Abstract base class for signal computation."""
    
    def __init__(self, config: SignalConfig):
        self.config = config
        self._cache = {}
        self._last_compute_time = None
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Signal name."""
        pass
    
    @property
    @abstractmethod
    def signal_type(self) -> SignalType:
        """Signal type."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Signal description."""
        pass
    
    @property
    @abstractmethod
    def dependencies(self) -> List[str]:
        """List of dependencies (other signals or data columns)."""
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """Validate input data.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Compute the signal.
        
        Args:
            data: Input data with OHLCV columns
            **kwargs: Additional parameters
            
        Returns:
            Computed signal series
        """
        pass
    
    def get_metadata(self, compute_time_ms: Optional[float] = None) -> SignalMetadata:
        """Get signal metadata."""
        return SignalMetadata(
            name=self.name,
            type=self.signal_type,
            description=self.description,
            parameters=self.get_parameters(),
            dependencies=self.dependencies,
            compute_time_ms=compute_time_ms
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get signal-specific parameters."""
        return {}
    
    def compute_with_metadata(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.Series, SignalMetadata]:
        """Compute signal and return with metadata."""
        start_time = time.time()
        
        # Validate data
        is_valid, error_msg = self.validate_data(data)
        if not is_valid:
            raise ValueError(f"Data validation failed for {self.name}: {error_msg}")
        
        # Compute signal
        signal = self.compute(data, **kwargs)
        
        # Calculate compute time
        compute_time_ms = (time.time() - start_time) * 1000
        
        # Get metadata
        metadata = self.get_metadata(compute_time_ms)
        
        return signal, metadata
    
    def _validate_required_columns(self, data: pd.DataFrame, required: List[str]) -> Tuple[bool, Optional[str]]:
        """Helper to validate required columns."""
        missing = [col for col in required if col not in data.columns]
        if missing:
            return False, f"Missing required columns: {missing}"
        return True, None
    
    def _validate_data_length(self, data: pd.DataFrame, min_length: int) -> Tuple[bool, Optional[str]]:
        """Helper to validate data length."""
        if len(data) < min_length:
            return False, f"Insufficient data: {len(data)} rows, need at least {min_length}"
        return True, None
    
    def _handle_missing_data(self, series: pd.Series) -> pd.Series:
        """Handle missing data based on config."""
        if self.config.handle_missing == "drop":
            return series.dropna()
        elif self.config.handle_missing == "interpolate":
            return series.interpolate()
        elif self.config.handle_missing == "forward_fill":
            return series.fillna(method='ffill')
        else:
            return series