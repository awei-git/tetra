"""
ML-based trading strategies that use predictions from the ML pipeline.
These strategies can be used in the benchmark pipeline.

This module contains:
1. Core ML strategy classes (MLPredictionStrategy, EnsembleMLStrategy)
2. Functions to create ML benchmark strategies from trained models
3. Legacy signal-based ML strategies for compatibility
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
import logging
import asyncio
from pathlib import Path
import joblib

from .base import BaseStrategy, PositionSide

logger = logging.getLogger(__name__)


class MLPredictionStrategy(BaseStrategy):
    """
    Base strategy that uses ML predictions for trading decisions.
    
    This strategy loads pre-trained ML models and uses their predictions
    to generate trading signals. It can work with any model from the ML pipeline.
    """
    
    def __init__(self,
                 name: str = "ML Prediction Strategy",
                 model_path: Optional[str] = None,
                 model_name: str = 'xgboost_return_1d',
                 prediction_threshold: float = 0.002,  # 0.2% return threshold
                 confidence_threshold: float = 0.6,
                 position_size_pct: float = 0.1,
                 max_positions: int = 10,
                 **kwargs):
        """
        Initialize ML prediction strategy.
        
        Args:
            model_path: Path to saved model file
            model_name: Name of model to use
            prediction_threshold: Minimum predicted return to trade
            confidence_threshold: Minimum confidence to trade
            position_size_pct: Position size as % of portfolio
            max_positions: Maximum concurrent positions
        """
        super().__init__(name, **kwargs)
        
        self.model_path = model_path
        self.model_name = model_name
        self.prediction_threshold = prediction_threshold
        self.confidence_threshold = confidence_threshold
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        
        self.model = None
        self.predictions_cache = {}
        
        # Load model if path provided
        if model_path:
            self._load_model()
            
    def _load_model(self):
        """Load pre-trained model from disk."""
        try:
            path = Path(self.model_path)
            if path.exists():
                model_data = joblib.load(path)
                self.model = model_data.get('model')
                logger.info(f"Loaded ML model from {self.model_path}")
            else:
                logger.warning(f"Model file not found: {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            
    def generate_signals(self, data: pd.DataFrame, events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate trading signals based on ML predictions."""
        if self.model is None:
            logger.warning("No ML model loaded, returning empty signals")
            return pd.DataFrame()
            
        signals = pd.DataFrame(index=data.index)
        
        try:
            # Prepare features from data
            features = self.model.prepare_features(data)
            
            # Make predictions
            predictions = self.model.predict(features)
            
            # Create signals based on predictions
            signals['prediction'] = predictions
            signals['signal'] = 0
            
            # Long signals when prediction exceeds threshold
            signals.loc[predictions > self.prediction_threshold, 'signal'] = 1
            
            # Short signals when prediction is below negative threshold
            signals.loc[predictions < -self.prediction_threshold, 'signal'] = -1
            
            # Add confidence if model provides it
            if hasattr(self.model, 'predict_proba'):
                confidence = self.model.predict_proba(features)
                signals['confidence'] = confidence.max(axis=1)
                
                # Filter by confidence
                low_confidence = signals['confidence'] < self.confidence_threshold
                signals.loc[low_confidence, 'signal'] = 0
                
        except Exception as e:
            logger.error(f"Error generating ML signals: {e}")
            
        return signals
    
    def should_enter(self, symbol: str, timestamp: datetime, bar_data: Dict,
                    signals: pd.DataFrame, events: Optional[pd.DataFrame] = None) -> Tuple[bool, PositionSide, float]:
        """Determine if should enter position based on ML prediction."""
        
        if signals.empty or 'signal' not in signals.columns:
            return False, PositionSide.LONG, 0.0
            
        # Get latest signal
        try:
            latest_signal = signals['signal'].iloc[-1]
            
            if latest_signal == 1:  # Long signal
                return True, PositionSide.LONG, self.position_size_pct
            elif latest_signal == -1:  # Short signal
                return True, PositionSide.SHORT, self.position_size_pct
            else:
                return False, PositionSide.LONG, 0.0
                
        except Exception as e:
            logger.error(f"Error in should_enter: {e}")
            return False, PositionSide.LONG, 0.0
            
    def should_exit(self, position: Any, timestamp: datetime, bar_data: Dict,
                   signals: pd.DataFrame, events: Optional[pd.DataFrame] = None) -> bool:
        """Determine if should exit position."""
        
        if signals.empty or 'signal' not in signals.columns:
            return False
            
        try:
            latest_signal = signals['signal'].iloc[-1]
            
            # Exit long if signal turns non-positive
            if position.side == PositionSide.LONG and latest_signal <= 0:
                return True
                
            # Exit short if signal turns non-negative
            if position.side == PositionSide.SHORT and latest_signal >= 0:
                return True
                
            # Exit if prediction reverses significantly
            if 'prediction' in signals.columns:
                latest_prediction = signals['prediction'].iloc[-1]
                
                if position.side == PositionSide.LONG and latest_prediction < -self.prediction_threshold:
                    return True
                elif position.side == PositionSide.SHORT and latest_prediction > self.prediction_threshold:
                    return True
                    
        except Exception as e:
            logger.error(f"Error in should_exit: {e}")
            
        return False


class EnsembleMLStrategy(MLPredictionStrategy):
    """
    Strategy that uses ensemble predictions from multiple ML models.
    
    Combines predictions from different model types (statistical, ML, deep learning)
    to make more robust trading decisions.
    """
    
    def __init__(self,
                 models_dir: str = 'output/ml_pipeline/models',
                 model_names: Optional[List[str]] = None,
                 voting_method: str = 'weighted',  # 'majority', 'weighted', 'average'
                 **kwargs):
        """
        Initialize ensemble ML strategy.
        
        Args:
            models_dir: Directory containing saved models
            model_names: List of model names to use (None = use all)
            voting_method: How to combine predictions
            **kwargs: Additional parameters for base class
        """
        super().__init__(**kwargs)
        
        self.models_dir = Path(models_dir)
        self.model_names = model_names
        self.voting_method = voting_method
        self.models = {}
        self.model_weights = {}
        
        # Load all models
        self._load_models()
        
    def _load_models(self):
        """Load all models for ensemble."""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return
            
        # Get all model files or specified models
        if self.model_names:
            model_files = [self.models_dir / f"{name}.pkl" for name in self.model_names]
        else:
            model_files = list(self.models_dir.glob("*.pkl"))
            
        for model_file in model_files:
            if model_file.exists():
                try:
                    model_data = joblib.load(model_file)
                    model_name = model_file.stem
                    self.models[model_name] = model_data.get('model')
                    
                    # Use validation MAE as weight (inverse)
                    if 'calibration_params' in model_data:
                        val_metrics = model_data['calibration_params'].get('validation_metrics', {})
                        mae = val_metrics.get('mae', 1.0)
                        self.model_weights[model_name] = 1.0 / mae if mae > 0 else 0.0
                    else:
                        self.model_weights[model_name] = 1.0
                        
                    logger.info(f"Loaded model: {model_name}")
                    
                except Exception as e:
                    logger.error(f"Error loading {model_file}: {e}")
                    
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}
            
        logger.info(f"Loaded {len(self.models)} models for ensemble")
        
    def generate_signals(self, data: pd.DataFrame, events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate ensemble signals from multiple models."""
        
        if not self.models:
            logger.warning("No models loaded for ensemble")
            return pd.DataFrame()
            
        all_predictions = []
        all_signals = []
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                # Prepare features
                features = model.prepare_features(data)
                
                # Make prediction
                prediction = model.predict(features)
                
                # Convert to signal
                signal = np.zeros_like(prediction)
                signal[prediction > self.prediction_threshold] = 1
                signal[prediction < -self.prediction_threshold] = -1
                
                all_predictions.append(prediction)
                all_signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error getting prediction from {model_name}: {e}")
                continue
                
        if not all_predictions:
            return pd.DataFrame()
            
        # Combine predictions based on voting method
        signals = pd.DataFrame(index=data.index)
        
        if self.voting_method == 'majority':
            # Majority voting on signals
            all_signals = np.array(all_signals)
            signals['signal'] = np.sign(np.sum(all_signals, axis=0))
            
        elif self.voting_method == 'weighted':
            # Weighted average of predictions
            weighted_predictions = np.zeros(len(data))
            for i, (model_name, prediction) in enumerate(zip(self.models.keys(), all_predictions)):
                weight = self.model_weights.get(model_name, 1.0)
                weighted_predictions += prediction * weight
                
            signals['prediction'] = weighted_predictions
            signals['signal'] = 0
            signals.loc[weighted_predictions > self.prediction_threshold, 'signal'] = 1
            signals.loc[weighted_predictions < -self.prediction_threshold, 'signal'] = -1
            
        else:  # average
            # Simple average of predictions
            avg_predictions = np.mean(all_predictions, axis=0)
            signals['prediction'] = avg_predictions
            signals['signal'] = 0
            signals.loc[avg_predictions > self.prediction_threshold, 'signal'] = 1
            signals.loc[avg_predictions < -self.prediction_threshold, 'signal'] = -1
            
        # Add ensemble confidence (std of predictions)
        signals['confidence'] = 1.0 - np.std(all_predictions, axis=0)
        
        return signals


# ============================================================================
# BENCHMARK CREATION FUNCTIONS
# ============================================================================

def create_ml_benchmark_strategies(models_dir: str = 'output/ml_pipeline/models') -> List:
    """
    Create ML-based benchmark strategies using trained models.
    
    This function checks for available trained models and creates
    appropriate strategies based on what's available.
    
    Args:
        models_dir: Directory containing trained models
        
    Returns:
        List of ML strategy instances
    """
    strategies = []
    models_path = Path(models_dir)
    
    # Check if models directory exists
    if not models_path.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        logger.info("Run ML pipeline first to train models: python -m src.pipelines.ml_pipeline.main")
        return strategies
        
    # Get available model files
    model_files = list(models_path.glob("*.pkl"))
    
    if not model_files:
        logger.warning("No trained models found. Run ML pipeline first.")
        return strategies
        
    logger.info(f"Found {len(model_files)} trained models")
    
    # 1. Create individual model strategies for best performers
    best_models = {
        'xgboost_return_1d': 'ML XGBoost 1-Day',
        'lightgbm_return_1d': 'ML LightGBM 1-Day',
        'arima_return_1d': 'ML ARIMA 1-Day',
        'xgboost_return_5d': 'ML XGBoost 5-Day',
        'gru_return_1d': 'ML GRU 1-Day'
    }
    
    for model_name, strategy_name in best_models.items():
        model_path = models_path / f"{model_name}.pkl"
        if model_path.exists():
            strategy = MLPredictionStrategy(
                model_path=str(model_path),
                model_name=model_name,
                prediction_threshold=0.002,  # 0.2% minimum predicted return
                position_size_pct=0.1  # 10% position size
            )
            strategy.name = strategy_name
            strategies.append(strategy)
            logger.info(f"Created strategy: {strategy_name}")
            
    # 2. Create ensemble strategy using all available models
    if len(model_files) >= 2:
        ensemble_strategy = EnsembleMLStrategy(
            models_dir=models_dir,
            voting_method='weighted',
            prediction_threshold=0.002,
            position_size_pct=0.15  # Larger position for ensemble
        )
        ensemble_strategy.name = 'ML Ensemble'
        strategies.append(ensemble_strategy)
        logger.info("Created ensemble strategy")
        
    # 3. Create multi-horizon strategy (1-day + 5-day models)
    multi_horizon_models = []
    for horizon in [1, 5]:
        model_path = models_path / f"xgboost_return_{horizon}d.pkl"
        if model_path.exists():
            multi_horizon_models.append(f"xgboost_return_{horizon}d")
            
    if len(multi_horizon_models) >= 2:
        multi_horizon_strategy = EnsembleMLStrategy(
            models_dir=models_dir,
            model_names=multi_horizon_models,
            voting_method='average',
            prediction_threshold=0.003,  # Higher threshold for multi-horizon
            position_size_pct=0.12
        )
        multi_horizon_strategy.name = 'ML Multi-Horizon'
        strategies.append(multi_horizon_strategy)
        logger.info("Created multi-horizon strategy")
        
    # 4. Create high-confidence strategy (higher thresholds)
    if model_files:
        best_model = model_files[0]  # Use first available model
        high_conf_strategy = MLPredictionStrategy(
            model_path=str(best_model),
            model_name=best_model.stem,
            prediction_threshold=0.005,  # 0.5% threshold (high confidence)
            confidence_threshold=0.7,  # 70% confidence required
            position_size_pct=0.2  # Larger position for high confidence
        )
        high_conf_strategy.name = 'ML High Confidence'
        strategies.append(high_conf_strategy)
        logger.info("Created high confidence strategy")
        
    return strategies


def check_ml_models_available(models_dir: str = 'output/ml_pipeline/models') -> bool:
    """
    Check if ML models are available for creating strategies.
    
    Args:
        models_dir: Directory to check for models
        
    Returns:
        True if models are available, False otherwise
    """
    models_path = Path(models_dir)
    
    if not models_path.exists():
        return False
        
    model_files = list(models_path.glob("*.pkl"))
    return len(model_files) > 0


def get_ml_strategy_by_name(name: str, models_dir: str = 'output/ml_pipeline/models'):
    """
    Get a specific ML strategy by name.
    
    Args:
        name: Strategy name
        models_dir: Directory containing models
        
    Returns:
        Strategy instance or None
    """
    strategies = create_ml_benchmark_strategies(models_dir)
    
    for strategy in strategies:
        if strategy.name.lower().replace(' ', '_') == name.lower().replace(' ', '_'):
            return strategy
            
    return None


# ============================================================================
# LEGACY SIGNAL-BASED ML STRATEGIES
# ============================================================================

def create_ml_strategies() -> List:
    """
    Create legacy signal-based ML strategies for backward compatibility.
    These use signal conditions rather than actual ML models.
    
    Returns:
        List of signal-based ML strategy instances
    """
    from .signal_based import SignalBasedStrategy, SignalRule, SignalCondition, ConditionOperator, PositionSide
    
    strategies = []
    
    # Basic ML Strategy (signal-based simulation)
    ml_basic = SignalBasedStrategy(
        name="ML Basic",
        signal_rules=[
            SignalRule(
                name="ml_long_signal",
                entry_conditions=[
                    SignalCondition("ml_prediction", ConditionOperator.GREATER_THAN, 0.6),
                    SignalCondition("ml_confidence", ConditionOperator.GREATER_THAN, 0.7)
                ],
                exit_conditions=[
                    SignalCondition("ml_prediction", ConditionOperator.LESS_THAN, 0.5)
                ],
                position_side=PositionSide.LONG,
                position_size_factor=1.0,
                stop_loss=0.02
            )
        ],
        max_positions=5,
        commission=0.001,
        position_size=0.2
    )
    strategies.append(ml_basic)
    
    # ML Anomaly Detection Strategy
    ml_anomaly = SignalBasedStrategy(
        name="ML Anomaly",
        signal_rules=[
            SignalRule(
                name="anomaly_signal",
                entry_conditions=[
                    SignalCondition("ml_anomaly_score", ConditionOperator.GREATER_THAN, 0.8),
                    SignalCondition("volume", ConditionOperator.GREATER_THAN, "volume_sma_20")
                ],
                exit_conditions=[
                    SignalCondition("ml_anomaly_score", ConditionOperator.LESS_THAN, 0.3)
                ],
                position_side=PositionSide.LONG,
                position_size_factor=1.5
            )
        ],
        max_positions=3,
        position_size=0.33
    )
    strategies.append(ml_anomaly)
    
    # High Confidence ML Strategy
    ml_high_conf = SignalBasedStrategy(
        name="ML High Confidence",
        signal_rules=[
            SignalRule(
                name="high_confidence_long",
                entry_conditions=[
                    SignalCondition("ml_prediction", ConditionOperator.GREATER_THAN, 0.75),
                    SignalCondition("ml_confidence", ConditionOperator.GREATER_THAN, 0.85)
                ],
                exit_conditions=[
                    SignalCondition("ml_prediction", ConditionOperator.LESS_THAN, 0.6)
                ],
                position_side=PositionSide.LONG,
                position_size_factor=2.0,
                stop_loss=0.015
            )
        ],
        max_positions=2,
        position_size=0.5
    )
    strategies.append(ml_high_conf)
    
    # ML + Technical Combo Strategy
    ml_tech_combo = SignalBasedStrategy(
        name="ML Technical Combo",
        signal_rules=[
            SignalRule(
                name="ml_tech_long",
                entry_conditions=[
                    SignalCondition("ml_prediction", ConditionOperator.GREATER_THAN, 0.55),
                    SignalCondition("rsi", ConditionOperator.LESS_THAN, 30),
                    SignalCondition("close", ConditionOperator.GREATER_THAN, "sma_50")
                ],
                exit_conditions=[
                    SignalCondition("ml_prediction", ConditionOperator.LESS_THAN, 0.45),
                    SignalCondition("rsi", ConditionOperator.GREATER_THAN, 70)
                ],
                position_side=PositionSide.LONG,
                position_size_factor=1.2
            )
        ],
        max_positions=4,
        position_size=0.25
    )
    strategies.append(ml_tech_combo)
    
    return strategies