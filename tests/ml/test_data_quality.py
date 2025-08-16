"""Tests for data quality and preprocessing components."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.ml.data_quality import DataQualityChecker, DataPreprocessor, DataValidator


@pytest.fixture
def sample_data():
    """Create sample data with various quality issues."""
    np.random.seed(42)
    n_samples = 100
    
    # Create base data
    data = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
    })
    
    # Add some quality issues
    data.loc[10:15, 'feature1'] = np.nan  # Missing values
    data.loc[20, 'feature2'] = 100  # Outlier
    data.loc[30:35, 'feature3'] = 0  # Constant values
    
    # Create target
    target = pd.Series(np.random.randint(0, 2, n_samples))
    
    return data, target


@pytest.fixture
def ohlcv_data():
    """Create OHLCV data with quality issues."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    data = pd.DataFrame({
        'open': 100 + np.random.randn(100) * 2,
        'high': 102 + np.random.randn(100) * 2,
        'low': 98 + np.random.randn(100) * 2,
        'close': 100 + np.random.randn(100) * 2,
        'volume': np.random.randint(100000, 1000000, 100)
    }, index=dates)
    
    # Add quality issues
    data.loc[dates[10], 'high'] = 90  # High < Low
    data.loc[dates[20], 'volume'] = -1000  # Negative volume
    data.loc[dates[30]:dates[35], 'volume'] = 0  # Zero volume
    
    return data


class TestDataQualityChecker:
    """Test data quality checker."""
    
    def test_initialization(self):
        """Test initialization."""
        checker = DataQualityChecker()
        assert checker is not None
        
    def test_check_missing_values(self, sample_data):
        """Test missing value detection."""
        checker = DataQualityChecker()
        data, _ = sample_data
        
        report = checker._check_missing_values(data)
        
        assert report['has_missing'] == True
        assert report['missing_columns'] == ['feature1']
        assert report['missing_counts']['feature1'] == 6
        assert report['missing_percentage']['feature1'] == 6.0
        
    def test_check_outliers(self, sample_data):
        """Test outlier detection."""
        checker = DataQualityChecker()
        data, _ = sample_data
        
        report = checker._check_outliers(data, method='zscore', threshold=3)
        
        assert report['has_outliers'] == True
        assert 'feature2' in report['outlier_columns']
        assert report['outlier_counts']['feature2'] > 0
        
    def test_check_constant_columns(self, sample_data):
        """Test constant column detection."""
        checker = DataQualityChecker()
        data, _ = sample_data
        
        # Add a truly constant column
        data['constant_col'] = 1
        
        report = checker._check_constant_columns(data)
        
        assert report['has_constant'] == True
        assert 'constant_col' in report['constant_columns']
        
    def test_check_ohlc_consistency(self, ohlcv_data):
        """Test OHLC consistency checks."""
        checker = DataQualityChecker()
        
        report = checker._check_ohlc_consistency(ohlcv_data)
        
        assert report['is_consistent'] == False
        assert len(report['inconsistencies']) > 0
        
        # Check specific issues
        issues = report['inconsistencies']
        assert any('high_low_invalid' in str(issue) for issue in issues)
        
    def test_generate_report(self, sample_data):
        """Test full quality report generation."""
        checker = DataQualityChecker()
        data, target = sample_data
        
        report = checker.generate_report(data, target)
        
        assert 'summary' in report
        assert 'missing_values' in report
        assert 'outliers' in report
        assert 'constant_columns' in report
        assert 'target_distribution' in report
        assert report['issues_count'] > 0


class TestDataPreprocessor:
    """Test data preprocessing."""
    
    def test_initialization(self):
        """Test initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
        
    def test_handle_missing_forward_fill(self, sample_data):
        """Test forward fill for missing values."""
        preprocessor = DataPreprocessor()
        data, _ = sample_data
        
        filled = preprocessor._handle_missing(data.copy(), method='forward_fill')
        
        # Should have fewer missing values
        assert filled.isna().sum().sum() < data.isna().sum().sum()
        
    def test_handle_missing_interpolate(self, sample_data):
        """Test interpolation for missing values."""
        preprocessor = DataPreprocessor()
        data, _ = sample_data
        
        filled = preprocessor._handle_missing(data.copy(), method='interpolate')
        
        # Should have no missing values in the middle
        assert filled.iloc[10:15]['feature1'].isna().sum() == 0
        
    def test_remove_outliers_zscore(self, sample_data):
        """Test Z-score outlier removal."""
        preprocessor = DataPreprocessor()
        data, _ = sample_data
        
        cleaned, removed_count = preprocessor._remove_outliers(
            data.copy(), method='zscore', threshold=3
        )
        
        assert len(cleaned) <= len(data)
        assert removed_count >= 0
        
    def test_remove_outliers_iqr(self, sample_data):
        """Test IQR outlier removal."""
        preprocessor = DataPreprocessor()
        data, _ = sample_data
        
        cleaned, removed_count = preprocessor._remove_outliers(
            data.copy(), method='iqr'
        )
        
        assert len(cleaned) <= len(data)
        
    def test_scale_features_standard(self, sample_data):
        """Test standard scaling."""
        preprocessor = DataPreprocessor()
        data, _ = sample_data
        
        # Remove NaN first
        data_clean = data.dropna()
        
        scaled = preprocessor._scale_features(data_clean.copy(), method='standard')
        
        # Check approximate mean and std
        numeric_cols = scaled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert abs(scaled[col].mean()) < 0.1  # Close to 0
            assert abs(scaled[col].std() - 1) < 0.1  # Close to 1
            
    def test_scale_features_robust(self, sample_data):
        """Test robust scaling."""
        preprocessor = DataPreprocessor()
        data, _ = sample_data
        
        # Remove NaN first
        data_clean = data.dropna()
        
        scaled = preprocessor._scale_features(data_clean.copy(), method='robust')
        
        # Check that scaling was applied
        assert not scaled.equals(data_clean)
        
    def test_preprocess_features_complete(self, sample_data):
        """Test complete preprocessing pipeline."""
        preprocessor = DataPreprocessor()
        data, target = sample_data
        
        config = {
            'handle_missing': 'forward_fill',
            'remove_outliers': True,
            'outlier_method': 'zscore',
            'scale_features': True,
            'scaling_method': 'standard',
            'feature_selection': True,
            'n_features': 2
        }
        
        processed, summary = preprocessor.preprocess_features(
            data.copy(), target, config
        )
        
        # Check results
        assert processed.shape[0] <= data.shape[0]  # May remove outliers
        assert processed.shape[1] == 2  # Selected 2 features
        assert processed.isna().sum().sum() == 0  # No missing values
        
        # Check summary
        assert 'missing_handled' in summary
        assert 'outliers_removed' in summary
        assert 'scaling_applied' in summary
        assert 'features_selected' in summary


class TestDataValidator:
    """Test data validation."""
    
    def test_validate_training_data_valid(self):
        """Test validation of valid training data."""
        # Create valid data
        X = pd.DataFrame(np.random.randn(200, 10))
        y = pd.Series(np.random.randint(0, 2, 200))
        
        is_valid = DataValidator.validate_training_data(X, y)
        assert is_valid == True
        
    def test_validate_training_data_shape_mismatch(self):
        """Test validation with shape mismatch."""
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.random.randint(0, 2, 90))  # Different length
        
        is_valid = DataValidator.validate_training_data(X, y)
        assert is_valid == False
        
    def test_validate_training_data_too_few_samples(self):
        """Test validation with too few samples."""
        X = pd.DataFrame(np.random.randn(50, 10))  # Less than 100
        y = pd.Series(np.random.randint(0, 2, 50))
        
        is_valid = DataValidator.validate_training_data(X, y)
        assert is_valid == False
        
    def test_validate_training_data_constant_target(self):
        """Test validation with constant target."""
        X = pd.DataFrame(np.random.randn(200, 10))
        y = pd.Series([1] * 200)  # All same value
        
        is_valid = DataValidator.validate_training_data(X, y)
        assert is_valid == False
        
    def test_validate_prediction_data_valid(self):
        """Test validation of valid prediction data."""
        # Create training data to establish features
        X_train = pd.DataFrame(
            np.random.randn(100, 5),
            columns=['f1', 'f2', 'f3', 'f4', 'f5']
        )
        
        # Create prediction data with same features
        X_pred = pd.DataFrame(
            np.random.randn(20, 5),
            columns=['f1', 'f2', 'f3', 'f4', 'f5']
        )
        
        is_valid = DataValidator.validate_prediction_data(X_pred, X_train.columns)
        assert is_valid == True
        
    def test_validate_prediction_data_missing_features(self):
        """Test validation with missing features."""
        train_features = ['f1', 'f2', 'f3', 'f4', 'f5']
        
        X_pred = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['f1', 'f2', 'f3']  # Missing f4, f5
        )
        
        is_valid = DataValidator.validate_prediction_data(X_pred, train_features)
        assert is_valid == False
        
    def test_validate_prediction_data_extra_features(self):
        """Test validation with extra features."""
        train_features = ['f1', 'f2', 'f3']
        
        X_pred = pd.DataFrame(
            np.random.randn(20, 5),
            columns=['f1', 'f2', 'f3', 'f4', 'f5']  # Extra f4, f5
        )
        
        is_valid = DataValidator.validate_prediction_data(X_pred, train_features)
        assert is_valid == False