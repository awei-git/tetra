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

from .correlation import (
    RollingCorrelationSignal,
    BetaSignal,
    CorrelationMatrixSignal,
    DynamicCorrelationSignal,
    LeadLagCorrelationSignal,
)

from .distribution import (
    SkewnessSignal,
    KurtosisSignal,
    JarqueBeraSignal,
    QuantileSignal,
    TailRatioSignal,
    ValueAtRiskSignal,
    ConditionalValueAtRiskSignal,
)

from .regime import (
    MarketRegimeSignal,
    VolatilityRegimeSignal,
    TrendRegimeSignal,
    HMMRegimeSignal,
)

from .statistical_tests import (
    AutocorrelationSignal,
    StationaritySignal,
    CointegrationSignal,
    HurstExponentSignal,
    EntropySignal,
)

from .composite import StatisticalSignals

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
    
    # Correlation
    'RollingCorrelationSignal',
    'BetaSignal',
    'CorrelationMatrixSignal',
    'DynamicCorrelationSignal',
    'LeadLagCorrelationSignal',
    
    # Distribution
    'SkewnessSignal',
    'KurtosisSignal',
    'JarqueBeraSignal',
    'QuantileSignal',
    'TailRatioSignal',
    'ValueAtRiskSignal',
    'ConditionalValueAtRiskSignal',
    
    # Regime
    'MarketRegimeSignal',
    'VolatilityRegimeSignal',
    'TrendRegimeSignal',
    'HMMRegimeSignal',
    
    # Statistical Tests
    'AutocorrelationSignal',
    'StationaritySignal',
    'CointegrationSignal',
    'HurstExponentSignal',
    'EntropySignal',
    
    # Composite
    'StatisticalSignals',
]