"""Tests for ML-based signals."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.signals.ml import (
    ClassificationSignal, RegressionSignal, AnomalyDetectionSignal,
    ClusteringSignal, EnsembleSignal, RandomForestClassifier,
    XGBoostRegressor, LSTMPredictor, IsolationForestAnomaly,
    KMeansCluster, VotingEnsemble
)
from src.signals.base.types import SignalType


class TestMLSignals:
    """Test suite for ML signals."""
    
    @pytest.fixture
    def feature_data(self):
        """Create feature data for ML models."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=500, freq='D')
        
        # Create features
        df = pd.DataFrame({
            'date': dates,
            'close': 100 + np.cumsum(np.random.randn(500) * 0.5),
            'volume': np.random.randint(1000000, 5000000, 500),
            'rsi': np.random.uniform(20, 80, 500),
            'sma_20': 100 + np.cumsum(np.random.randn(500) * 0.3),
            'bb_width': np.random.uniform(1, 5, 500)
        })
        
        # Add OHLV
        df['open'] = df['close'] * (1 + np.random.uniform(-0.01, 0.01, 500))
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.02, 500))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.02, 500))
        
        return df.set_index('date')
    
    @pytest.fixture
    def labeled_data(self, feature_data):
        """Create labeled data for supervised learning."""
        df = feature_data.copy()
        
        # Create labels based on future returns
        df['future_return'] = df['close'].pct_change(5).shift(-5)
        df['label'] = (df['future_return'] > 0.02).astype(int)
        
        return df.dropna()
    
    def test_classification_signal(self, labeled_data):
        """Test classification signal."""
        classifier = RandomForestClassifier(
            feature_columns=['rsi', 'sma_20', 'bb_width', 'volume'],
            label_column='label',
            lookback_window=20,
            n_estimators=10
        )
        
        # Split data
        train_data = labeled_data.iloc[:400]
        test_data = labeled_data.iloc[400:]
        
        # Train model
        classifier.train(train_data)
        assert classifier.model is not None
        assert classifier.is_trained
        
        # Make predictions
        predictions = classifier.compute(test_data)
        
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == len(test_data)
        assert predictions.name == 'rf_classifier_signal'
        
        # Predictions should be 0 or 1
        assert set(predictions.unique()).issubset({0, 1})
        
        # Get prediction probabilities
        probs = classifier.predict_proba(test_data)
        assert isinstance(probs, pd.DataFrame)
        assert probs.shape[1] == 2  # Binary classification
        assert (probs.sum(axis=1).round(5) == 1).all()  # Probabilities sum to 1
    
    def test_regression_signal(self, labeled_data):
        """Test regression signal."""
        regressor = XGBoostRegressor(
            feature_columns=['rsi', 'sma_20', 'bb_width'],
            target_column='future_return',
            lookback_window=20,
            n_estimators=10,
            max_depth=3
        )
        
        # Train model
        train_data = labeled_data.iloc[:400]
        regressor.train(train_data)
        
        # Make predictions
        test_data = labeled_data.iloc[400:]
        predictions = regressor.compute(test_data)
        
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == len(test_data)
        assert predictions.name == 'xgb_regressor_signal'
        
        # Predictions should be continuous values
        assert predictions.dtype == np.float64
        
        # Test feature importance
        importance = regressor.get_feature_importance()
        assert isinstance(importance, dict)
        assert all(feat in importance for feat in ['rsi', 'sma_20', 'bb_width'])
        assert all(imp >= 0 for imp in importance.values())
    
    def test_anomaly_detection(self, feature_data):
        """Test anomaly detection signal."""
        anomaly_detector = IsolationForestAnomaly(
            feature_columns=['close', 'volume', 'rsi'],
            contamination=0.1,
            n_estimators=50
        )
        
        # Fit and predict
        result = anomaly_detector.compute(feature_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'anomaly_score' in result.columns
        assert 'is_anomaly' in result.columns
        
        # Anomaly scores should be continuous
        assert result['anomaly_score'].dtype == np.float64
        
        # Is_anomaly should be binary
        assert set(result['is_anomaly'].unique()).issubset({0, 1})
        
        # Roughly 10% should be anomalies (contamination=0.1)
        anomaly_rate = result['is_anomaly'].mean()
        assert 0.05 <= anomaly_rate <= 0.15
    
    def test_clustering_signal(self, feature_data):
        """Test clustering signal."""
        clusterer = KMeansCluster(
            feature_columns=['rsi', 'bb_width', 'volume'],
            n_clusters=5,
            standardize=True
        )
        
        # Fit and predict
        result = clusterer.compute(feature_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'cluster' in result.columns
        assert 'distance_to_center' in result.columns
        
        # Should have 5 clusters (0-4)
        assert set(result['cluster'].unique()) == {0, 1, 2, 3, 4}
        
        # Distance should be non-negative
        assert (result['distance_to_center'] >= 0).all()
        
        # Get cluster centers
        centers = clusterer.get_cluster_centers()
        assert centers.shape == (5, 3)  # 5 clusters, 3 features
    
    def test_ensemble_signal(self, labeled_data):
        """Test ensemble signal."""
        # Create base models
        rf_model = RandomForestClassifier(
            feature_columns=['rsi', 'sma_20'],
            label_column='label',
            lookback_window=10
        )
        
        xgb_model = Mock()
        xgb_model.compute.return_value = pd.Series(
            np.random.randint(0, 2, len(labeled_data)),
            index=labeled_data.index
        )
        xgb_model.signal_type = SignalType.ML_CLASSIFICATION
        
        # Create ensemble
        ensemble = VotingEnsemble(
            models=[rf_model, xgb_model],
            voting='soft',
            weights=[0.6, 0.4]
        )
        
        # Train RF model
        train_data = labeled_data.iloc[:400]
        rf_model.train(train_data)
        
        # Compute ensemble predictions
        test_data = labeled_data.iloc[400:]
        result = ensemble.compute(test_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'ensemble_prediction' in result.columns
        assert 'ensemble_confidence' in result.columns
        
        # Predictions should be 0 or 1
        assert set(result['ensemble_prediction'].unique()).issubset({0, 1})
        
        # Confidence should be between 0 and 1
        assert (result['ensemble_confidence'] >= 0).all()
        assert (result['ensemble_confidence'] <= 1).all()
    
    def test_lstm_predictor(self, feature_data):
        """Test LSTM predictor (mock test due to TensorFlow dependency)."""
        with patch('src.signals.ml.deep_learning.Sequential') as mock_sequential:
            # Mock the model
            mock_model = Mock()
            mock_model.predict.return_value = np.random.randn(len(feature_data) - 60, 1)
            mock_sequential.return_value = mock_model
            
            lstm = LSTMPredictor(
                feature_columns=['close', 'volume', 'rsi'],
                sequence_length=60,
                lstm_units=50,
                epochs=1
            )
            
            # Train and predict
            lstm.train(feature_data)
            predictions = lstm.compute(feature_data)
            
            assert isinstance(predictions, pd.Series)
            assert len(predictions) == len(feature_data)
            assert predictions.name == 'lstm_prediction'
    
    def test_model_persistence(self, labeled_data, tmp_path):
        """Test model saving and loading."""
        classifier = RandomForestClassifier(
            feature_columns=['rsi', 'sma_20'],
            label_column='label'
        )
        
        # Train model
        classifier.train(labeled_data)
        original_predictions = classifier.compute(labeled_data)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        classifier.save_model(str(model_path))
        assert model_path.exists()
        
        # Create new instance and load model
        new_classifier = RandomForestClassifier(
            feature_columns=['rsi', 'sma_20'],
            label_column='label'
        )
        new_classifier.load_model(str(model_path))
        
        # Predictions should be identical
        new_predictions = new_classifier.compute(labeled_data)
        pd.testing.assert_series_equal(original_predictions, new_predictions)
    
    def test_incremental_learning(self, labeled_data):
        """Test incremental learning capability."""
        classifier = RandomForestClassifier(
            feature_columns=['rsi', 'sma_20'],
            label_column='label',
            incremental=True
        )
        
        # Initial training
        initial_data = labeled_data.iloc[:200]
        classifier.train(initial_data)
        initial_score = classifier.get_model_score(initial_data)
        
        # Incremental update
        new_data = labeled_data.iloc[200:300]
        classifier.update(new_data)
        
        # Model should still work
        test_data = labeled_data.iloc[300:]
        predictions = classifier.compute(test_data)
        assert len(predictions) == len(test_data)
        
        # Performance might improve with more data
        final_score = classifier.get_model_score(labeled_data.iloc[:300])
        assert final_score is not None
    
    def test_feature_engineering(self, feature_data):
        """Test automatic feature engineering."""
        from src.signals.ml.feature_engineering import FeatureEngineer
        
        engineer = FeatureEngineer(
            include_interactions=True,
            include_polynomial=True,
            include_lag_features=True,
            lag_periods=[1, 5, 10]
        )
        
        # Generate features
        engineered_features = engineer.transform(feature_data[['close', 'volume', 'rsi']])
        
        # Should have more features than original
        assert engineered_features.shape[1] > 3
        
        # Check for specific feature types
        feature_names = engineered_features.columns.tolist()
        
        # Lag features
        assert any('lag_1' in name for name in feature_names)
        assert any('lag_5' in name for name in feature_names)
        
        # Interaction features
        assert any('*' in name for name in feature_names)
        
        # Polynomial features
        assert any('^2' in name for name in feature_names)