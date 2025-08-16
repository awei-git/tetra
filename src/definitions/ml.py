"""Machine Learning definitions and configurations for Tetra platform."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Literal
from enum import Enum


# ==================== ML MODEL CONFIGURATIONS ====================

class ModelType(Enum):
    """ML model types available in the platform."""
    # Tree-based
    RANDOM_FOREST = "random_forest"
    EXTRA_TREES = "extra_trees"
    GRADIENT_BOOSTING = "gradient_boosting"
    HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"
    ADA_BOOST = "ada_boost"
    
    # Boosting libraries
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    
    # Neural networks
    MLP = "mlp"
    LSTM = "lstm"
    GRU = "gru"
    
    # Linear models
    RIDGE = "ridge"
    ELASTIC_NET = "elastic_net"
    
    # Support Vector
    SVR = "svr"
    
    # Time series
    ARIMA = "arima"
    SARIMAX = "sarimax"
    EXP_SMOOTHING = "exp_smoothing"


@dataclass
class MLModelConfig:
    """Configuration for individual ML models."""
    
    name: str
    model_type: ModelType
    parameters: Dict[str, Any]
    
    # Training configuration
    min_training_samples: int = 60  # Minimum 60 days for meaningful ML models
    validation_split: float = 0.2
    time_series_splits: int = 5
    
    # Feature requirements
    required_features: List[str] = field(default_factory=list)
    optional_features: List[str] = field(default_factory=list)


# Default model configurations
ML_MODEL_CONFIGS = {
    "rf_regressor": MLModelConfig(
        name="rf_regressor",
        model_type=ModelType.RANDOM_FOREST,
        parameters={
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": 42,
            "n_jobs": -1
        }
    ),
    "extra_trees": MLModelConfig(
        name="extra_trees",
        model_type=ModelType.EXTRA_TREES,
        parameters={
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": 42,
            "n_jobs": -1
        }
    ),
    "gb_regressor": MLModelConfig(
        name="gb_regressor",
        model_type=ModelType.GRADIENT_BOOSTING,
        parameters={
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": 42
        }
    ),
    "hist_gb": MLModelConfig(
        name="hist_gb",
        model_type=ModelType.HIST_GRADIENT_BOOSTING,
        parameters={
            "max_iter": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "min_samples_leaf": 20,
            "random_state": 42
        }
    ),
    "ada_boost": MLModelConfig(
        name="ada_boost",
        model_type=ModelType.ADA_BOOST,
        parameters={
            "n_estimators": 100,
            "learning_rate": 1.0,
            "random_state": 42
        }
    ),
    "xgboost": MLModelConfig(
        name="xgboost",
        model_type=ModelType.XGBOOST,
        parameters={
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
    ),
    "lightgbm": MLModelConfig(
        name="lightgbm",
        model_type=ModelType.LIGHTGBM,
        parameters={
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbosity": -1
        }
    ),
    "catboost": MLModelConfig(
        name="catboost",
        model_type=ModelType.CATBOOST,
        parameters={
            "iterations": 200,
            "depth": 6,
            "learning_rate": 0.1,
            "random_seed": 42,
            "verbose": False
        }
    ),
    "mlp": MLModelConfig(
        name="mlp",
        model_type=ModelType.MLP,
        parameters={
            "hidden_layer_sizes": (100, 50, 25),
            "activation": "relu",
            "solver": "adam",
            "learning_rate": "adaptive",
            "max_iter": 500,
            "early_stopping": True,
            "random_state": 42
        }
    ),
    "svr": MLModelConfig(
        name="svr",
        model_type=ModelType.SVR,
        parameters={
            "kernel": "rbf",
            "C": 1.0,
            "epsilon": 0.01
        }
    ),
    "ridge": MLModelConfig(
        name="ridge",
        model_type=ModelType.RIDGE,
        parameters={
            "alpha": 1.0
        }
    ),
    "elastic_net": MLModelConfig(
        name="elastic_net",
        model_type=ModelType.ELASTIC_NET,
        parameters={
            "alpha": 0.1,
            "l1_ratio": 0.5
        }
    )
}


# ==================== ML FEATURE CONFIGURATIONS ====================

@dataclass
class MLFeatureConfig:
    """Configuration for ML feature engineering."""
    
    # Price-based features
    price_return_lags: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20, 60])
    price_ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    momentum_periods: List[int] = field(default_factory=lambda: [20, 60])
    
    # Volume features
    volume_ma_period: int = 20
    
    # Statistical features
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    
    # Technical indicators to use
    technical_features: List[str] = field(default_factory=lambda: [
        'rsi', 'macd', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_lower', 'bb_middle',
        'atr', 'adx', 'cci', 'mfi', 'roc', 'williams_r',
        'stochastic_k', 'stochastic_d', 'obv_normalized'
    ])
    
    # Lag features for autoregression
    ar_lags: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    
    # Time-based features
    include_time_features: bool = True
    include_seasonality: bool = True


# ==================== ML TRADING CONFIGURATIONS ====================

@dataclass
class MLTradingConfig:
    """Configuration for ML-based trading strategies."""
    
    # Prediction thresholds
    prediction_threshold: float = 0.002  # 0.2% minimum predicted return
    high_confidence_threshold: float = 0.005  # 0.5% for high confidence trades
    
    # Confidence thresholds
    min_confidence: float = 0.3  # Minimum confidence to trade
    normal_confidence: float = 0.6  # Normal confidence level
    high_confidence: float = 0.7  # High confidence level
    
    # Position sizing
    base_position_size: float = 0.1  # 10% base position
    max_position_size: float = 0.2  # 20% maximum position
    kelly_fraction: float = 0.25  # Use 25% of Kelly Criterion
    
    # Risk management
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    max_positions: int = 10  # Maximum concurrent positions
    
    # Signal generation
    signal_lookback: int = 20  # Lookback for dynamic thresholds
    signal_percentile_buy: float = 0.7  # Buy above 70th percentile
    signal_percentile_sell: float = 0.3  # Sell below 30th percentile
    
    # Anomaly detection
    anomaly_contamination: float = 0.1  # Expect 10% anomalies
    anomaly_threshold: float = 0.8  # Trade on anomaly score > 0.8
    anomaly_boost_factor: float = 1.5  # Boost position for anomalies


# ==================== ML METRICS CONFIGURATIONS ====================

@dataclass
class MLMetricsConfig:
    """Configuration for ML performance metrics."""
    
    # Metrics to calculate
    regression_metrics: List[str] = field(default_factory=lambda: [
        'mse', 'mae', 'rmse', 'r2', 'mape'
    ])
    
    classification_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1', 'auc_roc'
    ])
    
    trading_metrics: List[str] = field(default_factory=lambda: [
        'hit_rate', 'profit_factor', 'avg_win', 'avg_loss',
        'prediction_sharpe', 'directional_accuracy'
    ])
    
    # Evaluation settings
    test_size: float = 0.2  # Use 20% for testing
    min_test_samples: int = 20  # Minimum samples for test set
    
    # Cross-validation
    cv_splits: int = 5  # Time series CV splits
    cv_method: str = "TimeSeriesSplit"


# ==================== ML ENSEMBLE CONFIGURATIONS ====================

@dataclass
class MLEnsembleConfig:
    """Configuration for ensemble methods."""
    
    voting_method: Literal["weighted", "average", "majority"] = "weighted"
    
    # Model selection
    min_models_for_ensemble: int = 2
    max_models_for_ensemble: int = 10
    
    # Weighting schemes
    use_cv_weights: bool = True  # Use cross-validation scores for weighting
    use_feature_importance: bool = True  # Consider feature importance
    
    # Confidence calculation
    confidence_method: str = "variance"  # Use prediction variance for confidence


# ==================== ML OUTPUT CONFIGURATIONS ====================

@dataclass
class MLOutputConfig:
    """Configuration for ML model outputs and storage."""
    
    # Model storage
    models_dir: str = "output/ml_pipeline/models"
    save_models: bool = True
    model_format: str = "pkl"  # pickle format
    
    # Prediction columns
    prediction_columns: Dict[str, str] = field(default_factory=lambda: {
        "ensemble": "ml_ensemble_prediction",
        "signal": "ml_signal",
        "action": "ml_action",
        "confidence": "ml_signal_strength",
        "risk": "ml_risk_score",
        "anomaly": "ml_anomaly_score",
        "position_size": "ml_position_size"
    })
    
    # Metadata storage
    save_feature_importance: bool = True
    save_model_weights: bool = True
    save_calibration_params: bool = True


# ==================== DEFAULT CONFIGURATIONS ====================

# Default ML configuration for the platform
DEFAULT_ML_CONFIG = {
    "models": ML_MODEL_CONFIGS,
    "features": MLFeatureConfig(),
    "trading": MLTradingConfig(),
    "metrics": MLMetricsConfig(),
    "ensemble": MLEnsembleConfig(),
    "output": MLOutputConfig()
}


# ==================== ML STRATEGY CONFIGURATIONS ====================

@dataclass
class MLStrategyConfig:
    """Configuration for ML-based trading strategies."""
    
    name: str
    model_name: str  # Which model to use
    
    # Trading parameters
    prediction_threshold: float = 0.002
    confidence_threshold: float = 0.6
    position_size_pct: float = 0.1
    max_positions: int = 10
    
    # Risk management
    stop_loss: Optional[float] = 0.02
    take_profit: Optional[float] = 0.05
    
    # Model source
    model_path: Optional[str] = None
    use_ensemble: bool = False
    
    # Strategy-specific settings
    strategy_params: Dict[str, Any] = field(default_factory=dict)


# Predefined ML strategy configurations
ML_STRATEGY_CONFIGS = {
    "ml_basic": MLStrategyConfig(
        name="ML Basic",
        model_name="xgboost",
        prediction_threshold=0.002,
        confidence_threshold=0.6,
        position_size_pct=0.1
    ),
    "ml_high_confidence": MLStrategyConfig(
        name="ML High Confidence",
        model_name="lightgbm",
        prediction_threshold=0.005,
        confidence_threshold=0.7,
        position_size_pct=0.2,
        max_positions=5
    ),
    "ml_ensemble": MLStrategyConfig(
        name="ML Ensemble",
        model_name="ensemble",
        use_ensemble=True,
        prediction_threshold=0.002,
        confidence_threshold=0.6,
        position_size_pct=0.15
    ),
    "ml_anomaly": MLStrategyConfig(
        name="ML Anomaly",
        model_name="xgboost",
        prediction_threshold=0.001,
        confidence_threshold=0.5,
        position_size_pct=0.33,
        max_positions=3,
        strategy_params={
            "anomaly_threshold": 0.8,
            "use_anomaly_boost": True
        }
    ),
    "ml_multi_horizon": MLStrategyConfig(
        name="ML Multi-Horizon",
        model_name="ensemble",
        use_ensemble=True,
        prediction_threshold=0.003,
        confidence_threshold=0.6,
        position_size_pct=0.12,
        strategy_params={
            "horizons": [1, 5, 20],
            "weight_by_horizon": True
        }
    )
}