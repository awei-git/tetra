"""Composite class that registers all ML signals."""

from typing import Optional, List

from ..base import SignalConfig
from ..technical import TechnicalSignals

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


class MLSignals(TechnicalSignals):
    """Composite class that provides all ML signals.
    
    Inherits from TechnicalSignals to provide both technical and ML signals.
    """
    
    def __init__(self, config: Optional[SignalConfig] = None):
        super().__init__(config)
        self._register_ml_signals()
    
    def _register_ml_signals(self):
        """Register all ML signals."""
        
        # Classification signals
        self.register_signal(DirectionClassifierSignal(self.config))
        self.register_signal(RegimeClassifierSignal(self.config))
        self.register_signal(PatternClassifierSignal(self.config))
        self.register_signal(SupportResistanceMLSignal(self.config))
        
        # Regression signals
        self.register_signal(PriceRegressionSignal(self.config))
        self.register_signal(VolatilityRegressionSignal(self.config))
        self.register_signal(ReturnsPredictionSignal(self.config))
        self.register_signal(MultiFactorRegressionSignal(self.config))
        
        # Clustering signals
        self.register_signal(MarketStateClusteringSignal(self.config))
        self.register_signal(PriceActionClusteringSignal(self.config))
        self.register_signal(VolumeProfileClusteringSignal(self.config))
        
        # Anomaly detection signals
        self.register_signal(PriceAnomalySignal(self.config))
        self.register_signal(VolumeAnomalySignal(self.config))
        self.register_signal(VolatilityAnomalySignal(self.config))
        self.register_signal(MultivarAnomalySignal(self.config))
        
        # Ensemble signals
        self.register_signal(EnsembleDirectionSignal(self.config))
        self.register_signal(StackedMLSignal(self.config))
        self.register_signal(VotingClassifierSignal(self.config))
        
        # Deep learning signals (placeholders)
        self.register_signal(LSTMPredictionSignal(self.config))
        self.register_signal(TransformerSignal(self.config))
        self.register_signal(AutoencoderSignal(self.config))
    
    def get_classification_signals(self) -> List[str]:
        """Get list of classification ML signals."""
        return [
            f"DirectionClassifier_{self.config.ml_prediction_horizon}",
            "RegimeClassifier_4",
            "PatternClassifier_20",
            "SupportResistanceML",
            f"EnsembleDirection_{self.config.ml_prediction_horizon}",
            "VotingRegime_3"
        ]
    
    def get_regression_signals(self) -> List[str]:
        """Get list of regression ML signals."""
        return [
            f"PriceRegression_{self.config.ml_prediction_horizon}",
            "VolatilityRegression_20",
            "ReturnsPrediction_mean",
            "MultiFactorRegression",
            "StackedML"
        ]
    
    def get_clustering_signals(self) -> List[str]:
        """Get list of clustering ML signals."""
        return [
            "MarketStateClustering_5",
            "PriceActionClustering_8",
            "VolumeProfileClustering_4"
        ]
    
    def get_anomaly_signals(self) -> List[str]:
        """Get list of anomaly detection signals."""
        return [
            "PriceAnomaly_isolation_forest",
            "VolumeAnomaly",
            "VolatilityAnomaly",
            "MultivarAnomaly"
        ]
    
    def get_deep_learning_signals(self) -> List[str]:
        """Get list of deep learning signals."""
        return [
            "LSTM_50",
            "Transformer",
            "Autoencoder"
        ]
    
    def get_all_ml_signals(self) -> List[str]:
        """Get list of all ML signals."""
        return (
            self.get_classification_signals() +
            self.get_regression_signals() +
            self.get_clustering_signals() +
            self.get_anomaly_signals() +
            self.get_deep_learning_signals()
        )
    
    def get_all_signals(self) -> List[str]:
        """Get list of all signals (technical + ML)."""
        return self.get_all_technical_signals() + self.get_all_ml_signals()