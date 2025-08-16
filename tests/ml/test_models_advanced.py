"""Tests for advanced ML models."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.ml.models_advanced import (
    XGBoostModel, LightGBMModel, CatBoostModel,
    LSTMModel, TransformerModel, AdvancedEnsemble
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create target with some pattern
    y = pd.Series((X['feature_0'] + 0.5 * X['feature_1'] - 0.3 * X['feature_2'] > 0).astype(int))
    
    # Split data
    split = int(0.8 * n_samples)
    return {
        'X_train': X[:split],
        'y_train': y[:split],
        'X_val': X[split:],
        'y_val': y[split:]
    }


@pytest.fixture
def time_series_data():
    """Create time series data for LSTM/Transformer testing."""
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='D')
    n_features = 5
    
    # Create correlated time series
    base = np.cumsum(np.random.randn(len(dates)))
    X = pd.DataFrame({
        f'feature_{i}': base + np.random.randn(len(dates)) * (i + 1)
        for i in range(n_features)
    }, index=dates)
    
    # Create target based on future movement
    y = pd.Series((X['feature_0'].shift(-1) > X['feature_0']).astype(int), index=dates)
    
    # Remove last value (no future for it)
    X = X[:-1]
    y = y[:-1]
    
    split = int(0.8 * len(X))
    return {
        'X_train': X[:split],
        'y_train': y[:split],
        'X_val': X[split:],
        'y_val': y[split:]
    }


class TestXGBoostModel:
    """Test XGBoost model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = XGBoostModel(n_estimators=100, max_depth=3)
        assert model.params['n_estimators'] == 100
        assert model.params['max_depth'] == 3
        assert not model.is_trained
        
    def test_training(self, sample_data):
        """Test model training."""
        model = XGBoostModel(n_estimators=10)  # Small for fast testing
        
        result = model.train(
            sample_data['X_train'],
            sample_data['y_train'],
            sample_data['X_val'],
            sample_data['y_val']
        )
        
        assert model.is_trained
        assert 'train_accuracy' in result
        assert 'val_accuracy' in result
        assert result['train_accuracy'] > 0.5
        assert result['val_accuracy'] > 0.5
        
    def test_prediction(self, sample_data):
        """Test model prediction."""
        model = XGBoostModel(n_estimators=10)
        
        # Should raise error before training
        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict(sample_data['X_val'])
            
        # Train model
        model.train(
            sample_data['X_train'],
            sample_data['y_train'],
            sample_data['X_val'],
            sample_data['y_val']
        )
        
        # Test predictions
        predictions = model.predict(sample_data['X_val'])
        assert len(predictions) == len(sample_data['y_val'])
        assert all(p in [0, 1] for p in predictions)
        
        # Test probabilities
        probabilities = model.predict_proba(sample_data['X_val'])
        assert probabilities.shape == (len(sample_data['y_val']), 2)
        assert all(0 <= p <= 1 for p in probabilities.flatten())
        
    def test_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        model = XGBoostModel(n_estimators=10)
        
        model.train(
            sample_data['X_train'],
            sample_data['y_train']
        )
        
        importance = model.get_feature_importance(top_n=5)
        assert len(importance) <= 5
        assert all(isinstance(v, (int, float, np.number)) for v in importance.values())


class TestLightGBMModel:
    """Test LightGBM model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = LightGBMModel(num_leaves=31, learning_rate=0.05)
        assert model.params['num_leaves'] == 31
        assert model.params['learning_rate'] == 0.05
        assert not model.is_trained
        
    def test_training(self, sample_data):
        """Test model training."""
        model = LightGBMModel(n_estimators=10)
        
        result = model.train(
            sample_data['X_train'],
            sample_data['y_train'],
            sample_data['X_val'],
            sample_data['y_val']
        )
        
        assert model.is_trained
        assert 'train_accuracy' in result
        assert 'val_accuracy' in result
        
    def test_categorical_features(self, sample_data):
        """Test training with categorical features."""
        model = LightGBMModel(n_estimators=10)
        
        # Add categorical feature
        X_train = sample_data['X_train'].copy()
        X_train['cat_feature'] = np.random.choice(['A', 'B', 'C'], size=len(X_train))
        X_train['cat_feature'] = X_train['cat_feature'].astype('category')
        
        X_val = sample_data['X_val'].copy()
        X_val['cat_feature'] = np.random.choice(['A', 'B', 'C'], size=len(X_val))
        X_val['cat_feature'] = X_val['cat_feature'].astype('category')
        
        result = model.train(
            X_train,
            sample_data['y_train'],
            X_val,
            sample_data['y_val'],
            categorical_features=['cat_feature']
        )
        
        assert model.is_trained


class TestCatBoostModel:
    """Test CatBoost model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = CatBoostModel(iterations=100, depth=6)
        assert model.params['iterations'] == 100
        assert model.params['depth'] == 6
        assert not model.is_trained
        
    def test_training(self, sample_data):
        """Test model training."""
        model = CatBoostModel(iterations=10)
        
        result = model.train(
            sample_data['X_train'],
            sample_data['y_train'],
            sample_data['X_val'],
            sample_data['y_val']
        )
        
        assert model.is_trained
        assert 'train_accuracy' in result
        assert 'val_accuracy' in result
        assert result['train_accuracy'] > 0.5


class TestLSTMModel:
    """Test LSTM model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = LSTMModel(
            sequence_length=30,
            lstm_units=[64, 32],
            dropout_rate=0.2
        )
        assert model.sequence_length == 30
        assert model.lstm_units == [64, 32]
        assert model.dropout_rate == 0.2
        assert not model.is_trained
        
    def test_sequence_preparation(self, time_series_data):
        """Test sequence preparation."""
        model = LSTMModel(sequence_length=10)
        
        X_seq, y_seq = model.prepare_sequences(
            time_series_data['X_train'],
            time_series_data['y_train']
        )
        
        # Check shapes
        expected_samples = len(time_series_data['X_train']) - model.sequence_length
        assert X_seq.shape == (expected_samples, 10, 5)  # (samples, seq_len, features)
        assert y_seq.shape == (expected_samples,)
        
    @pytest.mark.slow
    def test_training(self, time_series_data):
        """Test model training."""
        model = LSTMModel(
            sequence_length=10,
            lstm_units=[16, 8],  # Small for testing
        )
        
        result = model.train(
            time_series_data['X_train'],
            time_series_data['y_train'],
            time_series_data['X_val'],
            time_series_data['y_val'],
            epochs=2,  # Very few epochs for testing
            batch_size=32
        )
        
        assert model.is_trained
        assert 'train_accuracy' in result
        assert 'train_loss' in result
        
    @pytest.mark.slow
    def test_prediction(self, time_series_data):
        """Test model prediction."""
        model = LSTMModel(
            sequence_length=10,
            lstm_units=[16, 8]
        )
        
        # Train first
        model.train(
            time_series_data['X_train'],
            time_series_data['y_train'],
            epochs=2,
            batch_size=32
        )
        
        # Test prediction
        predictions = model.predict(time_series_data['X_val'])
        assert len(predictions) > 0
        
        # Test probabilities
        probabilities = model.predict_proba(time_series_data['X_val'])
        assert len(probabilities) > 0
        assert all(0 <= p <= 1 for p in probabilities.flatten())


class TestAdvancedEnsemble:
    """Test ensemble model."""
    
    def test_initialization(self):
        """Test ensemble initialization."""
        # Create dummy models
        models = {
            'model1': XGBoostModel(n_estimators=10),
            'model2': LightGBMModel(n_estimators=10)
        }
        
        ensemble = AdvancedEnsemble(models)
        assert len(ensemble.models) == 2
        assert not ensemble.use_stacking
        
    def test_weighted_prediction(self, sample_data):
        """Test weighted ensemble prediction."""
        # Train individual models
        xgb = XGBoostModel(n_estimators=10)
        xgb.train(sample_data['X_train'], sample_data['y_train'])
        
        lgb = LightGBMModel(n_estimators=10)
        lgb.train(sample_data['X_train'], sample_data['y_train'])
        
        # Create ensemble
        ensemble = AdvancedEnsemble({'xgb': xgb, 'lgb': lgb})
        ensemble.weights = {'xgb': 0.6, 'lgb': 0.4}
        
        # Test prediction
        predictions = ensemble.predict(sample_data['X_val'])
        assert len(predictions) == len(sample_data['y_val'])
        
        probabilities = ensemble.predict_proba(sample_data['X_val'])
        assert probabilities.shape[0] == len(sample_data['y_val'])
        
    def test_stacking(self, sample_data):
        """Test stacking ensemble."""
        # Train individual models
        xgb = XGBoostModel(n_estimators=10)
        xgb.train(sample_data['X_train'], sample_data['y_train'])
        
        cb = CatBoostModel(iterations=10)
        cb.train(sample_data['X_train'], sample_data['y_train'])
        
        # Create ensemble and train stacking
        ensemble = AdvancedEnsemble({'xgb': xgb, 'cb': cb})
        ensemble.train_stacking_meta_model(
            sample_data['X_val'],
            sample_data['y_val']
        )
        
        assert ensemble.use_stacking
        assert ensemble.meta_model is not None
        
        # Test stacking prediction
        predictions = ensemble.predict(sample_data['X_val'])
        assert len(predictions) == len(sample_data['y_val'])