"""Base classes for ML signals."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from abc import abstractmethod
import pickle
import joblib
from pathlib import Path
from datetime import datetime
import logging

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from ..base import BaseSignal, SignalType
from ..base.config import SignalConfig


logger = logging.getLogger(__name__)


class MLModelManager:
    """Manages ML model lifecycle: training, saving, loading."""
    
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or Path("models")
        self.model_dir.mkdir(exist_ok=True)
        self._models: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def save_model(self, model_name: str, model: Any, metadata: Dict[str, Any]):
        """Save model and metadata."""
        model_path = self.model_dir / f"{model_name}.pkl"
        meta_path = self.model_dir / f"{model_name}_meta.pkl"
        
        # Add timestamp
        metadata['saved_at'] = datetime.now().isoformat()
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save metadata
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved model {model_name} to {model_path}")
    
    def load_model(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """Load model and metadata."""
        model_path = self.model_dir / f"{model_name}.pkl"
        meta_path = self.model_dir / f"{model_name}_meta.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found")
        
        # Load model
        model = joblib.load(model_path)
        
        # Load metadata
        metadata = {}
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
        
        logger.info(f"Loaded model {model_name} from {model_path}")
        return model, metadata
    
    def model_exists(self, model_name: str) -> bool:
        """Check if model exists."""
        model_path = self.model_dir / f"{model_name}.pkl"
        return model_path.exists()
    
    def list_models(self) -> List[str]:
        """List available models."""
        return [p.stem for p in self.model_dir.glob("*.pkl") if not p.stem.endswith("_meta")]


class MLSignalBase(BaseSignal):
    """Base class for ML-based signals."""
    
    def __init__(self, config: SignalConfig, 
                 feature_window: Optional[int] = None,
                 prediction_horizon: Optional[int] = None,
                 model_manager: Optional[MLModelManager] = None):
        super().__init__(config)
        self.feature_window = feature_window or config.ml_feature_window
        self.prediction_horizon = prediction_horizon or config.ml_prediction_horizon
        self.model_manager = model_manager or MLModelManager()
        
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        self.last_train_date = None
        self.train_metrics = {}
    
    @property
    def signal_type(self) -> SignalType:
        """Default to ML regression type."""
        return SignalType.ML_REGRESSION
    
    @abstractmethod
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features from raw data."""
        pass
    
    @abstractmethod
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create labels for training."""
        pass
    
    @abstractmethod
    def create_model(self) -> Any:
        """Create the ML model."""
        pass
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare data for training or prediction."""
        # Create features
        features = self.create_features(data)
        
        # Store feature names
        if not self.feature_names:
            self.feature_names = list(features.columns)
        
        # Create labels if training
        labels = None
        if self.is_training_mode(data):
            labels = self.create_labels(data)
            
            # Align features and labels
            valid_idx = features.index.intersection(labels.index)
            features = features.loc[valid_idx]
            labels = labels.loc[valid_idx]
        
        return features, labels
    
    def is_training_mode(self, data: pd.DataFrame) -> bool:
        """Determine if we should train or predict."""
        # Train if model doesn't exist
        if not self.is_trained:
            return True
        
        # Retrain based on frequency
        if self.last_train_date is not None:
            days_since_train = (data.index[-1] - self.last_train_date).days
            if days_since_train >= self.config.ml_retrain_frequency:
                return True
        
        return False
    
    def train(self, features: pd.DataFrame, labels: pd.Series):
        """Train the ML model."""
        # Handle missing values
        features = features.fillna(method='ffill').fillna(0)
        labels = labels.fillna(method='ffill')
        
        # Remove any remaining NaN
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid_idx]
        labels = labels[valid_idx]
        
        if len(features) < self.config.ml_min_train_samples:
            raise ValueError(f"Insufficient training samples: {len(features)}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, shuffle=False
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create and train model
        self.model = self.create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate metrics
        train_score = self.model.score(X_train_scaled, y_train)
        val_score = self.model.score(X_val_scaled, y_val)
        
        self.train_metrics = {
            'train_score': train_score,
            'val_score': val_score,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'feature_count': len(self.feature_names)
        }
        
        self.is_trained = True
        self.last_train_date = features.index[-1]
        
        logger.info(f"{self.name} trained with score: train={train_score:.3f}, val={val_score:.3f}")
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Make predictions using the trained model."""
        if not self.is_trained or self.model is None:
            raise ValueError(f"{self.name} model not trained")
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(0)
        
        # Scale features
        if self.scaler is None:
            raise ValueError("Scaler not initialized")
        
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        
        # Return as series with proper index
        return pd.Series(predictions, index=features.index)
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Compute ML signal."""
        # Prepare data
        features, labels = self.prepare_data(data)
        
        # Check if we need to train
        if self.is_training_mode(data) and labels is not None:
            self.train(features, labels)
        
        # Make predictions
        predictions = self.predict(features)
        
        # Return predictions
        return predictions
    
    def save(self, name: Optional[str] = None):
        """Save model to disk."""
        model_name = name or self.name
        
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        metadata = {
            'feature_names': self.feature_names,
            'feature_window': self.feature_window,
            'prediction_horizon': self.prediction_horizon,
            'last_train_date': self.last_train_date,
            'train_metrics': self.train_metrics,
            'scaler': self.scaler
        }
        
        self.model_manager.save_model(model_name, self.model, metadata)
    
    def load(self, name: Optional[str] = None):
        """Load model from disk."""
        model_name = name or self.name
        
        model, metadata = self.model_manager.load_model(model_name)
        
        self.model = model
        self.feature_names = metadata.get('feature_names', [])
        self.feature_window = metadata.get('feature_window', self.feature_window)
        self.prediction_horizon = metadata.get('prediction_horizon', self.prediction_horizon)
        self.last_train_date = metadata.get('last_train_date')
        self.train_metrics = metadata.get('train_metrics', {})
        self.scaler = metadata.get('scaler')
        self.is_trained = True
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance if available."""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return None
        
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)
    
    def get_parameters(self) -> Dict[str, Any]:
        params = {
            'feature_window': self.feature_window,
            'prediction_horizon': self.prediction_horizon,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names)
        }
        
        if self.is_trained:
            params.update(self.train_metrics)
        
        return params