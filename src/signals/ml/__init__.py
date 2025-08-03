"""Machine Learning based signals."""

from .base import MLSignalBase, MLModelManager

from .classification import (
    DirectionClassifierSignal,
    RegimeClassifierSignal,
    PatternClassifierSignal,
    SupportResistanceMLSignal,
)

from .regression import (
    PriceRegressionSignal,
    VolatilityRegressionSignal,
    ReturnsPredictionSignal,
    MultiFactorRegressionSignal,
)

from .clustering import (
    MarketStateClusteringSignal,
    PriceActionClusteringSignal,
    VolumeProfileClusteringSignal,
)

from .anomaly import (
    PriceAnomalySignal,
    VolumeAnomalySignal,
    VolatilityAnomalySignal,
    MultivarAnomalySignal,
)

from .ensemble import (
    EnsembleDirectionSignal,
    StackedMLSignal,
    VotingClassifierSignal,
)

from .deep_learning import (
    LSTMPredictionSignal,
    TransformerSignal,
    AutoencoderSignal,
)

from .composite import MLSignals

__all__ = [
    # Base
    'MLSignalBase',
    'MLModelManager',
    
    # Classification
    'DirectionClassifierSignal',
    'RegimeClassifierSignal',
    'PatternClassifierSignal',
    'SupportResistanceMLSignal',
    
    # Regression
    'PriceRegressionSignal',
    'VolatilityRegressionSignal',
    'ReturnsPredictionSignal',
    'MultiFactorRegressionSignal',
    
    # Clustering
    'MarketStateClusteringSignal',
    'PriceActionClusteringSignal',
    'VolumeProfileClusteringSignal',
    
    # Anomaly Detection
    'PriceAnomalySignal',
    'VolumeAnomalySignal',
    'VolatilityAnomalySignal',
    'MultivarAnomalySignal',
    
    # Ensemble
    'EnsembleDirectionSignal',
    'StackedMLSignal',
    'VotingClassifierSignal',
    
    # Deep Learning
    'LSTMPredictionSignal',
    'TransformerSignal',
    'AutoencoderSignal',
    
    # Composite
    'MLSignals',
]