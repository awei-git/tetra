"""Tests for advanced feature engineering."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.ml.feature_engineering_advanced import AdvancedFeatureEngineer


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data."""
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')  # 1 year
    n = len(dates)
    
    # Generate realistic price data
    np.random.seed(42)
    returns = np.random.randn(n) * 0.02  # 2% daily volatility
    price = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': price * (1 + np.random.randn(n) * 0.001),
        'high': price * (1 + np.abs(np.random.randn(n) * 0.005)),
        'low': price * (1 - np.abs(np.random.randn(n) * 0.005)),
        'close': price,
        'volume': np.random.randint(1000000, 5000000, n)
    }, index=dates)
    
    return df


@pytest.fixture
def market_data(sample_ohlcv_data):
    """Create market data with multiple symbols."""
    base_df = sample_ohlcv_data
    
    # Create correlated market data
    market_data = {
        'TEST': base_df,
        'SPY': base_df * 1.5 + np.random.randn(len(base_df), 5) * 5,
        'XLF': base_df * 0.8 + np.random.randn(len(base_df), 5) * 3,
        'GLD': base_df * 1.2 + np.random.randn(len(base_df), 5) * 4,
    }
    
    return market_data


class TestAdvancedFeatureEngineer:
    """Test advanced feature engineering."""
    
    def test_initialization(self):
        """Test feature engineer initialization."""
        engineer = AdvancedFeatureEngineer()
        assert engineer is not None
        
    @pytest.mark.asyncio
    async def test_price_features(self, sample_ohlcv_data):
        """Test price-based features."""
        engineer = AdvancedFeatureEngineer()
        
        features = engineer._create_price_features(sample_ohlcv_data)
        
        # Check basic returns
        assert 'returns_1d' in features.columns
        assert 'returns_5d' in features.columns
        assert 'returns_20d' in features.columns
        
        # Check log returns (volatility proxy)
        assert 'log_returns_20d' in features.columns
        assert 'log_returns_60d' in features.columns
        
        # Check moving average ratios
        assert 'price_to_ma20' in features.columns
        assert 'price_to_ema21' in features.columns
        
        # Check price position
        assert 'price_position_20d' in features.columns
        
        # Verify no NaN in most features (except beginning due to rolling)
        assert features['returns_1d'].notna().sum() > len(features) * 0.9
        
    @pytest.mark.asyncio
    async def test_volume_features(self, sample_ohlcv_data):
        """Test volume-based features."""
        engineer = AdvancedFeatureEngineer()
        
        features = engineer._create_volume_features(sample_ohlcv_data)
        
        # Check volume features
        assert 'volume_ma20' in features.columns
        assert 'volume_ratio_20d' in features.columns
        assert 'volume_spike' in features.columns
        
        # Check OBV
        assert 'obv' in features.columns
        assert 'obv_ma20' in features.columns
        
        # Check VWAP
        assert 'vwap' in features.columns
        assert 'price_to_vwap' in features.columns
        
    @pytest.mark.asyncio 
    async def test_volatility_features(self, sample_ohlcv_data):
        """Test volatility features."""
        engineer = AdvancedFeatureEngineer()
        
        features = engineer._create_volatility_features(sample_ohlcv_data)
        
        # Check ATR
        assert 'atr_14' in features.columns
        
        # Check volatility features
        assert 'volatility_20d' in features.columns
        assert 'volatility_ratio_20d' in features.columns
        assert 'parkinson_vol_20d' in features.columns
        assert 'garman_klass_vol' in features.columns
        
        # Check other volatility measures
        assert 'realized_vol_proxy' in features.columns
        assert 'vol_term_structure' in features.columns
        
    @pytest.mark.asyncio
    async def test_momentum_features(self, sample_ohlcv_data):
        """Test momentum indicators."""
        engineer = AdvancedFeatureEngineer()
        
        features = engineer._create_advanced_momentum_features(sample_ohlcv_data)
        
        # Check RSI
        assert 'rsi_14' in features.columns
        
        # Check other momentum indicators
        assert 'mom_5d' in features.columns
        assert 'mom_10d' in features.columns
        assert 'price_acceleration' in features.columns
        
        # Check Sharpe-like ratios
        assert 'return_to_vol_20d' in features.columns
        
        # Check other momentum
        assert 'intraday_momentum' in features.columns
        assert 'overnight_momentum' in features.columns
        
    @pytest.mark.asyncio
    async def test_pattern_features(self, sample_ohlcv_data):
        """Test pattern recognition features."""
        engineer = AdvancedFeatureEngineer()
        
        features = engineer._create_pattern_features(sample_ohlcv_data)
        
        # Check candle patterns
        assert 'candle_body_ratio' in features.columns
        assert 'upper_shadow_ratio' in features.columns
        assert 'lower_shadow_ratio' in features.columns
        assert 'is_doji' in features.columns
        
        # Check trend patterns
        assert 'consecutive_up_days' in features.columns
        assert 'consecutive_down_days' in features.columns
        
        # Check high/low patterns
        assert 'new_high_10d' in features.columns
        assert 'new_low_10d' in features.columns
        
    @pytest.mark.asyncio
    async def test_microstructure_features(self, sample_ohlcv_data):
        """Test market microstructure features."""
        engineer = AdvancedFeatureEngineer()
        
        features = engineer._create_microstructure_features(sample_ohlcv_data)
        
        # Check liquidity proxies
        assert 'amihud_illiquidity' in features.columns
        assert 'roll_spread' in features.columns
        
        # Check price impact measures
        assert 'kyle_lambda' in features.columns
        
        # Check volume patterns
        assert 'trade_intensity' in features.columns
        
    @pytest.mark.asyncio
    async def test_cross_asset_features(self, sample_ohlcv_data, market_data):
        """Test cross-asset features."""
        engineer = AdvancedFeatureEngineer()
        
        features = engineer._create_cross_asset_features(
            sample_ohlcv_data, 
            'TEST',
            market_data
        )
        
        # Check market beta (if SPY available)
        if 'SPY' in market_data:
            assert 'beta_20d' in features.columns
            assert 'beta_60d' in features.columns
            assert 'relative_strength_spy' in features.columns
            
        # Check sector correlations
        assert any('corr_' in col for col in features.columns)
        
    @pytest.mark.asyncio
    async def test_create_all_features(self, sample_ohlcv_data, market_data):
        """Test complete feature creation."""
        engineer = AdvancedFeatureEngineer()
        
        features = await engineer.create_all_features(
            sample_ohlcv_data,
            'TEST',
            market_data
        )
        
        # Should create many features
        assert len(features.columns) > 100
        
        # Check different feature categories exist
        feature_cols = features.columns.tolist()
        assert any('returns' in col for col in feature_cols)
        assert any('volume' in col for col in feature_cols)
        assert any('volatility' in col for col in feature_cols)
        assert any('rsi' in col for col in feature_cols)
        
        # Check data quality
        # After initial window, should have mostly non-null values
        non_null_ratio = features.iloc[100:].notna().mean()
        assert non_null_ratio.mean() > 0.8
        
    @pytest.mark.asyncio
    async def test_feature_creation_with_limited_data(self):
        """Test feature creation with minimal data."""
        engineer = AdvancedFeatureEngineer()
        
        # Create very small dataset
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        small_df = pd.DataFrame({
            'open': 100 + np.random.randn(50),
            'high': 101 + np.random.randn(50),
            'low': 99 + np.random.randn(50),
            'close': 100 + np.random.randn(50),
            'volume': np.random.randint(1000, 5000, 50)
        }, index=dates)
        
        features = await engineer.create_all_features(small_df, 'TEST')
        
        # Should still create features
        assert len(features.columns) > 50
        
        # But many will be NaN due to rolling windows
        assert features.notna().any().any()  # At least some non-NaN values
        
    def test_feature_names_consistency(self):
        """Test that feature names follow consistent patterns."""
        engineer = AdvancedFeatureEngineer()
        
        # Test data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'open': 100 + np.random.randn(100),
            'high': 101 + np.random.randn(100),
            'low': 99 + np.random.randn(100),
            'close': 100 + np.random.randn(100),
            'volume': np.random.randint(1000, 5000, 100)
        }, index=dates)
        
        # Get different feature sets
        price_features = engineer._create_price_features(df)
        volume_features = engineer._create_volume_features(df)
        
        # Check naming conventions
        for col in price_features.columns:
            assert isinstance(col, str)
            assert not col.startswith('_')  # No private feature names
            
        for col in volume_features.columns:
            assert isinstance(col, str)
            assert 'volume' in col.lower() or 'obv' in col.lower() or col in ['vwap', 'price_to_vwap', 'dollar_volume', 'acc_dist']