"""Tests for base signal functionality."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.signals.base import BaseSignal, SignalComputer, SignalConfig
from src.signals.base.types import SignalType, SignalMetadata


class MockSignal(BaseSignal):
    """Mock signal for testing."""
    
    def __init__(self, name: str = "mock_signal"):
        super().__init__(
            name=name,
            description="Test signal",
            signal_type=SignalType.TECHNICAL
        )
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Simple computation for testing."""
        if 'close' not in data.columns:
            raise ValueError("close column required")
        return data['close'] * 2


class TestBaseSignal:
    """Test BaseSignal functionality."""
    
    def test_signal_creation(self):
        """Test signal creation and properties."""
        signal = MockSignal("test_signal")
        
        assert signal.name == "test_signal"
        assert signal.description == "Test signal"
        assert signal.signal_type == SignalType.TECHNICAL
        assert signal.dependencies == []
    
    def test_compute_method(self):
        """Test compute method."""
        signal = MockSignal()
        
        # Create test data
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        })
        
        result = signal.compute(data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        assert result.iloc[0] == 200
        assert result.iloc[-1] == 208
    
    def test_compute_with_metadata(self):
        """Test compute with metadata."""
        signal = MockSignal()
        
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        })
        
        result, metadata = signal.compute_with_metadata(data)
        
        assert isinstance(result, pd.Series)
        assert isinstance(metadata, SignalMetadata)
        assert metadata.signal_name == "mock_signal"
        assert metadata.compute_time > 0
        assert metadata.data_points == 5
    
    def test_validation(self):
        """Test data validation."""
        signal = MockSignal()
        
        # Test with missing column
        bad_data = pd.DataFrame({
            'open': [100, 101, 102]
        })
        
        with pytest.raises(ValueError):
            signal.compute(bad_data)
    
    def test_get_parameters(self):
        """Test parameter extraction."""
        signal = MockSignal()
        params = signal.get_parameters()
        
        assert isinstance(params, dict)
        assert 'name' in params
        assert 'signal_type' in params


class TestSignalComputer:
    """Test SignalComputer functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(99, 101, 100),
            'high': np.random.uniform(101, 103, 100),
            'low': np.random.uniform(97, 99, 100),
            'close': np.random.uniform(98, 102, 100),
            'volume': np.random.randint(1000000, 2000000, 100)
        }).set_index('date')
    
    @pytest.fixture
    def multi_symbol_data(self):
        """Create multi-symbol data."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        data_list = []
        for symbol in symbols:
            df = pd.DataFrame({
                'date': dates,
                'symbol': symbol,
                'open': np.random.uniform(99, 101, 50),
                'high': np.random.uniform(101, 103, 50),
                'low': np.random.uniform(97, 99, 50),
                'close': np.random.uniform(98, 102, 50),
                'volume': np.random.randint(1000000, 2000000, 50)
            })
            data_list.append(df)
        
        return pd.concat(data_list, ignore_index=True)
    
    def test_computer_initialization(self):
        """Test SignalComputer initialization."""
        computer = SignalComputer()
        
        assert isinstance(computer.config, SignalConfig)
        assert len(computer._signals) == 0
    
    def test_register_signal(self):
        """Test signal registration."""
        computer = SignalComputer()
        signal = MockSignal("test1")
        
        computer.register_signal(signal)
        
        assert "test1" in computer._signals
        assert computer._signals["test1"] == signal
    
    def test_register_multiple_signals(self):
        """Test registering multiple signals."""
        computer = SignalComputer()
        signals = [
            MockSignal("test1"),
            MockSignal("test2"),
            MockSignal("test3")
        ]
        
        computer.register_signals(signals)
        
        assert len(computer._signals) == 3
        assert all(f"test{i+1}" in computer._signals for i in range(3))
    
    def test_compute_single_symbol(self, sample_data):
        """Test computation for single symbol."""
        computer = SignalComputer()
        computer.register_signal(MockSignal("double_close"))
        
        result = computer.compute(sample_data, signal_names=["double_close"])
        
        assert "double_close" in result.data.columns
        assert len(result.data) == len(sample_data)
        assert result.errors == {}
        assert result.compute_time > 0
    
    def test_compute_multi_symbol(self, multi_symbol_data):
        """Test computation for multiple symbols."""
        computer = SignalComputer()
        computer.register_signal(MockSignal("double_close"))
        
        result = computer.compute(multi_symbol_data, signal_names=["double_close"])
        
        # Check results for each symbol
        for symbol in ['AAPL', 'GOOGL', 'MSFT']:
            col_name = f"{symbol}_double_close"
            assert col_name in result.data.columns
            assert col_name in result.metadata
    
    def test_list_signals(self):
        """Test listing registered signals."""
        computer = SignalComputer()
        computer.register_signals([
            MockSignal("test1"),
            MockSignal("test2")
        ])
        
        signals = computer.list_signals()
        
        assert len(signals) == 2
        assert all('name' in s for s in signals)
        assert all('type' in s for s in signals)
        assert all('description' in s for s in signals)
    
    def test_cache_functionality(self, sample_data):
        """Test caching functionality."""
        config = SignalConfig(cache_results=True, cache_ttl_seconds=60)
        computer = SignalComputer(config)
        computer.register_signal(MockSignal("cached_signal"))
        
        # First computation
        result1 = computer.compute(sample_data, signal_names=["cached_signal"])
        time1 = result1.compute_time
        
        # Second computation (should use cache)
        result2 = computer.compute(sample_data, signal_names=["cached_signal"])
        time2 = result2.compute_time
        
        # Cache should make second computation faster
        assert time2 < time1
        assert result1.data.equals(result2.data)
        
        # Clear cache and verify
        computer.clear_cache()
        result3 = computer.compute(sample_data, signal_names=["cached_signal"])
        # Without cache, should take similar time to first computation
        assert result3.compute_time >= time2


class TestSignalConfig:
    """Test SignalConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = SignalConfig()
        
        assert config.rsi_period == 14
        assert config.macd_fast == 12
        assert config.macd_slow == 26
        assert config.parallel_compute == True
        assert config.cache_results == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SignalConfig(
            rsi_period=20,
            parallel_compute=False,
            min_data_points=100
        )
        
        assert config.rsi_period == 20
        assert config.parallel_compute == False
        assert config.min_data_points == 100
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid min_data_points
        with pytest.raises(ValueError):
            config = SignalConfig(min_data_points=10)
            config.validate()
        
        # Test invalid MACD settings
        with pytest.raises(ValueError):
            config = SignalConfig(macd_fast=30, macd_slow=20)
            config.validate()
        
        # Test invalid max_missing_pct
        with pytest.raises(ValueError):
            config = SignalConfig(max_missing_pct=1.5)
            config.validate()
    
    def test_config_serialization(self):
        """Test config to/from dict."""
        config = SignalConfig(
            rsi_period=20,
            custom_params={'alpha': 0.5, 'beta': 0.3}
        )
        
        # To dict
        config_dict = config.to_dict()
        assert config_dict['rsi_period'] == 20
        assert config_dict['custom_params']['alpha'] == 0.5
        
        # From dict
        config2 = SignalConfig.from_dict(config_dict)
        assert config2.rsi_period == 20
        assert config2.custom_params['alpha'] == 0.5
    
    def test_get_param(self):
        """Test parameter retrieval."""
        config = SignalConfig(
            custom_params={'custom_value': 42}
        )
        
        # Standard param
        assert config.get_param('rsi_period') == 14
        
        # Custom param
        assert config.get_param('custom_value') == 42
        
        # Non-existent param with default
        assert config.get_param('missing', 'default') == 'default'