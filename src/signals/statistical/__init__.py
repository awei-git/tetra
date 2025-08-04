"""Statistical analysis signals."""

from .returns import (
    SimpleReturnsSignal,
    LogReturnsSignal,
    CumulativeReturnsSignal,
    RollingReturnsSignal,
    DrawdownSignal,
    MaxDrawdownSignal,
)

from .volatility import (
    RollingVolatilitySignal,
    GARCHVolatilitySignal,
    RealizedVolatilitySignal,
    VolatilityRegimeSignal,
    VolumeWeightedVolatilitySignal,
)

# Placeholder for composite class
class StatisticalSignals:
    """Composite statistical signals."""
    pass

__all__ = [
    # Returns
    'SimpleReturnsSignal',
    'LogReturnsSignal',
    'CumulativeReturnsSignal',
    'RollingReturnsSignal',
    'DrawdownSignal',
    'MaxDrawdownSignal',
    
    # Volatility
    'RollingVolatilitySignal',
    'GARCHVolatilitySignal',
    'RealizedVolatilitySignal',
    'VolatilityRegimeSignal',
    'VolumeWeightedVolatilitySignal',
    
    # Composite
    'StatisticalSignals',
]