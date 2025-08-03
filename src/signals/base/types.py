"""Type definitions for signals module."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
import numpy as np


class SignalType(Enum):
    """Types of signals."""
    # Technical
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    PATTERN = "pattern"
    
    # Statistical
    STATISTICAL = "statistical"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    REGIME = "regime"
    
    # ML
    ML_CLASSIFICATION = "ml_classification"
    ML_REGRESSION = "ml_regression"
    ML_CLUSTERING = "ml_clustering"
    ML_ANOMALY = "ml_anomaly"
    
    # Composite
    COMPOSITE = "composite"
    CUSTOM = "custom"


@dataclass
class SignalMetadata:
    """Metadata for a signal."""
    name: str
    type: SignalType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    compute_time_ms: Optional[float] = None
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'type': self.type.value,
            'description': self.description,
            'parameters': self.parameters,
            'dependencies': self.dependencies,
            'compute_time_ms': self.compute_time_ms,
            'version': self.version
        }


@dataclass
class SignalResult:
    """Container for computed signals."""
    data: pd.DataFrame
    metadata: Dict[str, SignalMetadata]
    compute_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    errors: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def signal_names(self) -> List[str]:
        """Get list of signal names."""
        return list(self.data.columns)
    
    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0
    
    def get_signal(self, name: str) -> Optional[pd.Series]:
        """Get a specific signal."""
        if name in self.data.columns:
            return self.data[name]
        return None
    
    def filter_by_type(self, signal_type: SignalType) -> pd.DataFrame:
        """Filter signals by type."""
        filtered_cols = [
            col for col, meta in self.metadata.items()
            if meta.type == signal_type
        ]
        return self.data[filtered_cols]
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_signals': len(self.signal_names),
            'signal_types': {
                t.value: sum(1 for m in self.metadata.values() if m.type == t)
                for t in SignalType
            },
            'compute_time': self.compute_time,
            'timestamp': self.timestamp,
            'has_errors': self.has_errors,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'data': self.data.to_dict(),
            'metadata': {k: v.to_dict() for k, v in self.metadata.items()},
            'compute_time': self.compute_time,
            'timestamp': self.timestamp.isoformat(),
            'errors': self.errors,
            'warnings': self.warnings
        }