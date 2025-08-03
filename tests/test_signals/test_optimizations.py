"""Tests for signal computation optimizations."""

import pytest
import pandas as pd
import numpy as np
import time
from concurrent.futures import TimeoutError

from src.signals.optimizations import (
    BatchSignalComputer, VectorizedSignals, NumbaAcceleratedSignals,
    MemoryOptimizedComputer, LazySignalEvaluator
)
from src.signals.technical import RSI, SMA, EMA, MACD, BollingerBands
from src.signals.base import SignalConfig


class TestBatchComputation:
    """Test batch signal computation."""
    
    @pytest.fixture
    def multi_symbol_data(self):
        """Create multi-symbol test data."""
        np.random.seed(42)
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        data_frames = []
        for symbol in symbols:
            df = pd.DataFrame({
                'date': dates,
                'symbol': symbol,
                'open': 100 + np.random.randn(100).cumsum(),
                'high': 102 + np.random.randn(100).cumsum(),
                'low': 98 + np.random.randn(100).cumsum(),
                'close': 100 + np.random.randn(100).cumsum(),
                'volume': np.random.randint(1000000, 5000000, 100)
            })
            data_frames.append(df)
        
        return pd.concat(data_frames, ignore_index=True)
    
    def test_batch_computer_initialization(self):
        """Test batch computer initialization."""
        computer = BatchSignalComputer(n_processes=2)
        
        assert computer.n_processes == 2
        assert computer._process_pool is not None
        
        # Clean up
        computer.close()
    
    def test_batch_processing(self, multi_symbol_data):
        """Test batch processing of multiple symbols."""
        computer = BatchSignalComputer(n_processes=2)
        
        # Register signals
        computer.register_signals([
            RSI(period=14),
            SMA(period=20),
            EMA(period=12)
        ])
        
        # Process batch
        symbols = multi_symbol_data['symbol'].unique().tolist()
        results = computer.compute_batch(
            multi_symbol_data,
            symbols,
            chunk_size=2
        )
        
        # Check results
        assert len(results) == len(symbols)
        
        for symbol, result in results.items():
            assert 'rsi_14' in result.data.columns
            assert 'sma_20' in result.data.columns
            assert 'ema_12' in result.data.columns
            assert len(result.data) == 100
        
        # Clean up
        computer.close()
    
    def test_optimized_computation(self, multi_symbol_data):
        """Test optimized multi-symbol computation."""
        computer = BatchSignalComputer()
        
        # Register vectorizable signal
        computer.register_signals([
            RSI(period=14),
            SMA(period=20)
        ])
        
        # Mock vectorized computation
        with pytest.mock.patch.object(RSI, 'compute_vectorized', 
                                     return_value=pd.DataFrame()):
            result = computer.compute_optimized(multi_symbol_data)
            
            assert isinstance(result.data, pd.DataFrame)
            assert result.compute_time > 0


class TestVectorizedSignals:
    """Test vectorized signal implementations."""
    
    @pytest.fixture
    def price_matrix(self):
        """Create price matrix for multiple symbols."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        # Create price data
        data = {}
        for symbol in symbols:
            data[symbol] = 100 + np.random.randn(100).cumsum()
        
        return pd.DataFrame(data, index=dates)
    
    def test_rsi_vectorized(self, price_matrix):
        """Test vectorized RSI calculation."""
        rsi = VectorizedSignals.compute_rsi_vectorized(price_matrix, period=14)
        
        assert isinstance(rsi, pd.DataFrame)
        assert rsi.shape == price_matrix.shape
        assert rsi.columns.tolist() == price_matrix.columns.tolist()
        
        # RSI should be between 0 and 100
        valid_values = rsi.dropna()
        assert (valid_values >= 0).all().all()
        assert (valid_values <= 100).all().all()
    
    def test_bollinger_bands_vectorized(self, price_matrix):
        """Test vectorized Bollinger Bands."""
        bb = VectorizedSignals.compute_bollinger_bands_vectorized(
            price_matrix, period=20, std_dev=2.0
        )
        
        assert isinstance(bb, dict)
        assert all(key in bb for key in ['upper', 'middle', 'lower', 'bandwidth', 'percent_b'])
        
        # Check relationships
        assert (bb['upper'] > bb['middle']).all().all()
        assert (bb['middle'] > bb['lower']).all().all()
    
    def test_macd_vectorized(self, price_matrix):
        """Test vectorized MACD calculation."""
        macd = VectorizedSignals.compute_macd_vectorized(
            price_matrix, fast_period=12, slow_period=26, signal_period=9
        )
        
        assert isinstance(macd, dict)
        assert all(key in macd for key in ['macd', 'signal', 'histogram'])
        
        # MACD histogram = MACD - Signal
        expected_histogram = macd['macd'] - macd['signal']
        pd.testing.assert_frame_equal(macd['histogram'], expected_histogram)
    
    def test_correlation_matrix_vectorized(self, price_matrix):
        """Test vectorized correlation matrix."""
        returns = price_matrix.pct_change().dropna()
        
        corr_matrix = VectorizedSignals.compute_correlation_matrix_vectorized(
            returns, window=20
        )
        
        # Should return rolling correlations
        assert isinstance(corr_matrix, pd.DataFrame)
        
        # Check that correlations are between -1 and 1
        assert (corr_matrix >= -1).all().all()
        assert (corr_matrix <= 1).all().all()


class TestNumbaAcceleration:
    """Test Numba-accelerated signals."""
    
    def test_numba_rsi(self):
        """Test Numba RSI calculation."""
        prices = pd.Series(100 + np.random.randn(200).cumsum())
        
        # Calculate with Numba
        numba_rsi = NumbaAcceleratedSignals.rsi(prices, period=14)
        
        assert isinstance(numba_rsi, pd.Series)
        assert len(numba_rsi) == len(prices)
        
        # Compare with standard calculation (should be very close)
        from src.signals.technical import RSI
        standard_rsi = RSI(period=14).compute(pd.DataFrame({'close': prices}))
        
        # Allow small differences due to calculation methods
        pd.testing.assert_series_equal(
            numba_rsi.dropna(),
            standard_rsi.dropna(),
            check_names=False,
            atol=0.1
        )
    
    def test_numba_performance(self):
        """Test that Numba version is faster."""
        # Large dataset
        prices = pd.Series(100 + np.random.randn(10000).cumsum())
        
        # Time Numba version
        start = time.time()
        numba_result = NumbaAcceleratedSignals.rsi(prices, period=14)
        numba_time = time.time() - start
        
        # Time standard version
        from src.signals.technical import RSI
        start = time.time()
        standard_result = RSI(period=14).compute(pd.DataFrame({'close': prices}))
        standard_time = time.time() - start
        
        # Numba should be faster (after JIT compilation)
        # First run compiles, so test second run
        start = time.time()
        numba_result2 = NumbaAcceleratedSignals.rsi(prices, period=14)
        numba_time2 = time.time() - start
        
        # Second run should be significantly faster
        assert numba_time2 < standard_time


class TestMemoryOptimization:
    """Test memory-optimized computation."""
    
    @pytest.fixture
    def large_data(self):
        """Create large dataset for memory testing."""
        # 1 year of minute data
        dates = pd.date_range(start='2024-01-01', periods=100000, freq='1min')
        return pd.DataFrame({
            'date': dates,
            'open': 100 + np.random.randn(100000).cumsum() * 0.01,
            'high': 101 + np.random.randn(100000).cumsum() * 0.01,
            'low': 99 + np.random.randn(100000).cumsum() * 0.01,
            'close': 100 + np.random.randn(100000).cumsum() * 0.01,
            'volume': np.random.randint(1000, 10000, 100000)
        }).set_index('date')
    
    def test_chunked_computation(self, large_data):
        """Test chunked computation for large datasets."""
        computer = MemoryOptimizedComputer(chunk_size=10000)
        computer.register_signals([RSI(period=14), SMA(period=20)])
        
        # Process in chunks
        result = computer.compute_chunked(large_data, chunk_size=10000)
        
        assert len(result.data) == len(large_data)
        assert 'rsi_14' in result.data.columns
        assert 'sma_20' in result.data.columns
    
    def test_dtype_optimization(self):
        """Test data type optimization."""
        computer = MemoryOptimizedComputer(dtype_optimization=True)
        
        # Create data with different types
        data = pd.DataFrame({
            'small_int': np.array([1, 2, 3, 4, 5], dtype=np.int64),
            'large_float': np.array([1e10, 2e10, 3e10, 4e10, 5e10], dtype=np.float64),
            'small_float': np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64),
            'category': ['A', 'B', 'A', 'B', 'A']
        })
        
        # Optimize dtypes
        optimized = computer._optimize_dtypes(data)
        
        # Check optimizations
        assert optimized['small_int'].dtype == np.int8
        assert optimized['small_float'].dtype == np.float32
        assert optimized['category'].dtype.name == 'category'
        
        # Memory should be reduced
        original_memory = data.memory_usage(deep=True).sum()
        optimized_memory = optimized.memory_usage(deep=True).sum()
        assert optimized_memory < original_memory
    
    def test_memory_estimation(self, large_data):
        """Test memory usage estimation."""
        computer = MemoryOptimizedComputer()
        computer.register_signals([RSI(), SMA(), MACD()])
        
        estimates = computer.estimate_memory_usage(
            large_data[:1000],  # Use subset for testing
            signal_names=['rsi_14', 'sma_20', 'macd']
        )
        
        assert 'input_data' in estimates
        assert 'signals' in estimates
        assert 'total' in estimates
        assert estimates['total'] > estimates['input_data']


class TestLazyEvaluation:
    """Test lazy signal evaluation."""
    
    @pytest.fixture
    def evaluator_with_signals(self):
        """Create evaluator with registered signals."""
        evaluator = LazySignalEvaluator()
        evaluator.register_signals([
            RSI(period=14),
            SMA(period=20),
            EMA(period=12),
            MACD(),  # Depends on EMA
            BollingerBands()  # Depends on SMA
        ])
        return evaluator
    
    def test_lazy_dataframe_creation(self, evaluator_with_signals):
        """Test lazy DataFrame creation."""
        data = pd.DataFrame({
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(1000000, 5000000, 100)
        })
        
        lazy_df = evaluator_with_signals.create_lazy_dataframe(data)
        
        # Should have base columns plus signals
        expected_columns = list(data.columns) + list(evaluator_with_signals._signals.keys())
        assert lazy_df.columns == expected_columns
    
    def test_lazy_signal_access(self, evaluator_with_signals):
        """Test lazy signal computation on access."""
        data = pd.DataFrame({
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(1000000, 5000000, 100)
        })
        
        lazy_df = evaluator_with_signals.create_lazy_dataframe(data)
        
        # Access RSI - should compute on demand
        rsi_values = lazy_df['rsi_14']
        
        assert isinstance(rsi_values, pd.Series)
        assert len(rsi_values) == len(data)
        
        # Second access should use cache
        rsi_values2 = lazy_df['rsi_14']
        pd.testing.assert_series_equal(rsi_values, rsi_values2)
    
    def test_dependency_resolution(self, evaluator_with_signals):
        """Test automatic dependency resolution."""
        data = pd.DataFrame({
            'open': 99 + np.random.randn(100).cumsum(),
            'high': 101 + np.random.randn(100).cumsum(),
            'low': 97 + np.random.randn(100).cumsum(),
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(1000000, 5000000, 100)
        })
        
        # Get computation order for MACD (depends on close -> EMA)
        order = evaluator_with_signals.get_computation_order(['macd'])
        
        # Should compute dependencies first
        assert 'macd' in order
        
        # Compute MACD
        lazy_df = evaluator_with_signals.create_lazy_dataframe(data)
        macd_result = lazy_df['macd']
        
        assert isinstance(macd_result, pd.DataFrame)
        assert 'macd' in macd_result.columns
        assert 'macd_signal' in macd_result.columns
    
    def test_batch_computation_with_dependencies(self, evaluator_with_signals):
        """Test batch computation with dependency resolution."""
        data = pd.DataFrame({
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(1000000, 5000000, 100)
        })
        
        # Compute multiple signals with dependencies
        signals = evaluator_with_signals.compute_batch(
            ['rsi_14', 'macd', 'bb_20'],
            data
        )
        
        assert len(signals) == 3
        assert 'rsi_14' in signals
        assert 'macd' in signals
        assert 'bb_20' in signals