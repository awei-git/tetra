"""Tests for integrated ML pipeline."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.ml.integrated_pipeline import IntegratedMLPipeline
from src.ml.comprehensive_config import get_comprehensive_config


@pytest.fixture
def simple_config():
    """Simple configuration for testing."""
    return {
        'model_dir': 'test_models/',
        'registry_dir': 'test_registry/',
        'models_to_use': ['xgboost', 'lightgbm'],
        'validation_params': {
            'train_period_days': 100,
            'val_period_days': 20,
            'test_period_days': 10,
            'step_days': 30,
            'min_train_samples': 50
        },
        'feature_engineering': {
            'use_all_features': False,
            'max_features': 50
        },
        'preprocessing': {
            'handle_missing': 'forward_fill',
            'remove_outliers': True,
            'scale_features': True,
            'scaling_method': 'standard'
        },
        'risk_params': {
            'max_position_size': 0.2,
            'max_portfolio_risk': 0.06,
            'confidence_level': 0.95,
            'lookback_days': 252
        },
        'model_params': {
            'xgboost': {
                'n_estimators': 10,
                'max_depth': 3
            },
            'lightgbm': {
                'n_estimators': 10,
                'num_leaves': 15
            }
        },
        'backtest_params': {
            'initial_capital': 100000,
            'position_sizing': 'equal_weight',
            'max_positions': 5,
            'transaction_cost': 0.001,
            'slippage': 0.0005
        }
    }


@pytest.fixture
def sample_market_data():
    """Create sample market data for multiple symbols."""
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    market_data = {}
    
    for symbol in symbols:
        base_price = 100 * (1 + np.random.rand())
        returns = np.random.randn(len(dates)) * 0.02
        price = base_price * np.exp(np.cumsum(returns))
        
        market_data[symbol] = pd.DataFrame({
            'open': price * (1 + np.random.randn(len(dates)) * 0.001),
            'high': price * (1 + np.abs(np.random.randn(len(dates)) * 0.005)),
            'low': price * (1 - np.abs(np.random.randn(len(dates)) * 0.005)),
            'close': price,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
    
    return market_data


class TestIntegratedMLPipeline:
    """Test integrated ML pipeline."""
    
    def test_initialization(self, simple_config):
        """Test pipeline initialization."""
        pipeline = IntegratedMLPipeline(config=simple_config)
        
        assert pipeline.config == simple_config
        assert pipeline.feature_engineer is not None
        assert pipeline.data_preprocessor is not None
        assert pipeline.validator is not None
        assert pipeline.risk_manager is not None
        
    @pytest.mark.asyncio
    async def test_train_pipeline_basic(self, simple_config, sample_market_data):
        """Test basic pipeline training."""
        pipeline = IntegratedMLPipeline(config=simple_config)
        
        # Use smaller data for faster testing
        small_market_data = {
            k: v.iloc[:200] for k, v in sample_market_data.items()
        }
        
        # Mock the model training to speed up test
        with patch.object(pipeline.models, 'create_xgboost_model') as mock_xgb, \
             patch.object(pipeline.models, 'create_lightgbm_model') as mock_lgb:
            
            # Create mock models
            mock_xgb_model = Mock()
            mock_xgb_model.is_trained = True
            mock_xgb_model.train.return_value = {'train_accuracy': 0.6, 'val_accuracy': 0.55}
            mock_xgb_model.predict.return_value = np.random.randint(0, 2, 100)
            mock_xgb_model.predict_proba.return_value = np.random.rand(100, 2)
            
            mock_lgb_model = Mock()
            mock_lgb_model.is_trained = True
            mock_lgb_model.train.return_value = {'train_accuracy': 0.62, 'val_accuracy': 0.56}
            mock_lgb_model.predict.return_value = np.random.randint(0, 2, 100)
            mock_lgb_model.predict_proba.return_value = np.random.rand(100, 2)
            
            mock_xgb.return_value = mock_xgb_model
            mock_lgb.return_value = mock_lgb_model
            
            # Train pipeline
            summary = await pipeline.train_pipeline(
                historical_data=small_market_data
            )
            
            assert 'training_complete' in summary
            assert 'models_trained' in summary
            assert len(summary['models_trained']) > 0
            
    def test_generate_predictions(self, simple_config):
        """Test prediction generation."""
        pipeline = IntegratedMLPipeline(config=simple_config)
        
        # Create mock trained model
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        pipeline.current_models = {'test_model': mock_model}
        
        # Create test data
        test_data = pd.DataFrame({
            'feature1': [1, 2],
            'feature2': [3, 4]
        })
        
        # Mock feature engineering
        with patch.object(pipeline, '_prepare_features_for_prediction') as mock_prep:
            mock_prep.return_value = test_data
            
            predictions = pipeline.generate_predictions(
                {'TEST': test_data},
                model_name='test_model'
            )
            
            assert 'TEST' in predictions
            assert predictions['TEST']['prediction'] == 1  # 0.7 > 0.5
            assert predictions['TEST']['confidence'] == 0.7
            
    def test_save_and_load_pipeline(self, simple_config, tmp_path):
        """Test saving and loading pipeline."""
        pipeline = IntegratedMLPipeline(config=simple_config)
        
        # Mock some trained state
        pipeline.feature_columns = ['feat1', 'feat2', 'feat3']
        pipeline.is_trained = True
        
        # Save pipeline
        save_path = tmp_path / "test_pipeline"
        pipeline.save_pipeline(str(save_path))
        
        # Check files exist
        assert (save_path / "config.json").exists()
        assert (save_path / "feature_columns.json").exists()
        
        # Load pipeline
        new_pipeline = IntegratedMLPipeline(config=simple_config)
        new_pipeline.load_pipeline(str(save_path))
        
        assert new_pipeline.feature_columns == pipeline.feature_columns
        
    @pytest.mark.asyncio
    async def test_feature_engineering_error_handling(self, simple_config):
        """Test error handling in feature engineering."""
        pipeline = IntegratedMLPipeline(config=simple_config)
        
        # Create data that will cause errors
        bad_data = {
            'BAD': pd.DataFrame({
                'open': [None, None, None],
                'high': [None, None, None],
                'low': [None, None, None],
                'close': [None, None, None],
                'volume': [0, 0, 0]
            })
        }
        
        # Should handle gracefully
        result = await pipeline._engineer_features_for_symbol(
            'BAD', bad_data['BAD'], bad_data
        )
        
        assert result == (None, None)  # Should return None on error
        
    def test_validation_windows_creation(self, simple_config):
        """Test walk-forward validation windows."""
        pipeline = IntegratedMLPipeline(config=simple_config)
        
        # Create date range
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        windows = pipeline.validator.create_windows(start_date, end_date)
        
        assert len(windows) > 0
        
        # Check window properties
        for window in windows:
            assert window.train_end > window.train_start
            assert window.val_end > window.val_start
            assert window.test_end > window.test_start
            assert window.val_start == window.train_end
            assert window.test_start == window.val_end
            
    def test_comprehensive_config_loading(self):
        """Test loading comprehensive configuration."""
        config = get_comprehensive_config()
        pipeline = IntegratedMLPipeline(config=config)
        
        assert 'xgboost' in pipeline.config['models_to_use']
        assert 'lstm' in pipeline.config['models_to_use']
        assert pipeline.config['feature_engineering']['max_features'] == 500
        
    @pytest.mark.asyncio
    async def test_monitoring_integration(self, simple_config, sample_market_data):
        """Test monitoring components."""
        pipeline = IntegratedMLPipeline(config=simple_config)
        
        # Mock a trained model
        mock_model = Mock()
        mock_model.is_trained = True
        mock_model.predict.return_value = np.array([0, 1, 0, 1])
        mock_model.predict_proba.return_value = np.random.rand(4, 2)
        
        pipeline.current_models = {'test': mock_model}
        pipeline.is_trained = True
        
        # Create predictions
        test_data = list(sample_market_data.values())[0].iloc[:4]
        
        with patch.object(pipeline, '_prepare_features_for_prediction') as mock_prep:
            mock_prep.return_value = pd.DataFrame(np.random.randn(4, 10))
            
            # Should track predictions
            predictions = pipeline.generate_predictions(
                {'TEST': test_data},
                model_name='test'
            )
            
            # Check monitoring recorded prediction
            assert len(pipeline.monitor.prediction_history) > 0