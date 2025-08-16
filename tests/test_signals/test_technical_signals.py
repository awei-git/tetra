"""Tests for technical signals."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.signals.base.config import SignalConfig
from src.signals.technical import (
    RSISignal, 
    SMASignal, 
    EMASignal, 
    MACDSignal, 
    BollingerBandsSignal,
    ATRSignal, 
    ADXSignal, 
    StochasticSignal, 
    CCISignal, 
    WilliamsRSignal,
    MFISignal, 
    OBVSignal, 
    VWAPSignal, 
    PivotPointsSignal,
    IchimokuSignal, 
    ParabolicSARSignal,
    FibonacciRetracementSignal,
    CandlePatternSignal
)

# Helper functions to create signals with the old API
def RSI(period=14):
    config = SignalConfig(rsi_period=period)
    return RSISignal(config, period=period)

def SMA(period=20):
    config = SignalConfig()
    return SMASignal(config, period=period)

def EMA(period=21):
    config = SignalConfig()
    return EMASignal(config, period=period)

def MACD(fast=12, slow=26, signal=9):
    config = SignalConfig(macd_fast=fast, macd_slow=slow, macd_signal=signal)
    return MACDSignal(config)

def BollingerBands(period=20, std=2.0):
    config = SignalConfig(bb_period=period, bb_std=std)
    return BollingerBandsSignal(config)

def ATR(period=14):
    config = SignalConfig(atr_period=period)
    return ATRSignal(config)

def ADX(period=14):
    config = SignalConfig(adx_period=period)
    return ADXSignal(config)

def Stochastic(period=14, smooth_k=3, smooth_d=3):
    config = SignalConfig(stoch_period=period, stoch_smooth_k=smooth_k, stoch_smooth_d=smooth_d)
    return StochasticSignal(config)

def CCI(period=20):
    config = SignalConfig(cci_period=period)
    return CCISignal(config)

def WilliamsR(period=14):
    config = SignalConfig(williams_r_period=period)
    return WilliamsRSignal(config)

def MFI(period=14):
    config = SignalConfig(mfi_period=period)
    return MFISignal(config)

def OBV():
    config = SignalConfig()
    return OBVSignal(config)

def VWAP(period=14):
    config = SignalConfig(vwap_period=period)
    return VWAPSignal(config)

def PivotPoints():
    config = SignalConfig()
    return PivotPointsSignal(config)

def IchimokuCloud():
    config = SignalConfig()
    return IchimokuSignal(config)

def ParabolicSAR():
    config = SignalConfig()
    return ParabolicSARSignal(config)

def FibonacciRetracements(lookback=50):
    config = SignalConfig()
    return FibonacciRetracementSignal(config)

def CandlePatterns():
    config = SignalConfig()
    return CandlePatternSignal(config)


class TestTechnicalSignals:
    """Test suite for technical indicators."""
    
    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
        
        # Generate realistic price data
        close_prices = 100
        closes = []
        for _ in range(200):
            change = np.random.normal(0, 1)
            close_prices *= (1 + change / 100)
            closes.append(close_prices)
        
        df = pd.DataFrame({
            'date': dates,
            'close': closes
        })
        
        # Generate OHLV from close
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0]) + np.random.uniform(-0.5, 0.5, 200)
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 1, 200)
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 1, 200)
        df['volume'] = np.random.randint(1000000, 5000000, 200)
        
        return df.set_index('date')
    
    def test_rsi(self, sample_ohlcv):
        """Test RSI calculation."""
        rsi = RSI(period=14)
        result = rsi.compute(sample_ohlcv)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)
        assert result.name == 'RSI_14' or 'rsi' in str(result.name).lower()
        
        # RSI should be between 0 and 100
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()
        
        # First 13 values should be NaN or 0 (implementation fills NaN with 0)
        assert (result.iloc[:13] == 0).all() or result.iloc[:13].isna().all()
        assert result.iloc[14:].notna().all()
    
    def test_sma(self, sample_ohlcv):
        """Test SMA calculation."""
        sma = SMA(period=20)
        result = sma.compute(sample_ohlcv)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)
        assert result.name == 'SMA_20'
        
        # Check calculation
        for i in range(20, len(result)):
            expected = sample_ohlcv['close'].iloc[i-19:i+1].mean()
            assert np.isclose(result.iloc[i], expected)
    
    def test_ema(self, sample_ohlcv):
        """Test EMA calculation."""
        ema = EMA(period=12)
        result = ema.compute(sample_ohlcv)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)
        assert result.name == 'EMA_12'
        
        # EMA should start from first value
        assert not pd.isna(result.iloc[0])
        
        # EMA should be smoother than price
        price_volatility = sample_ohlcv['close'].std()
        ema_volatility = result.std()
        assert ema_volatility < price_volatility
    
    def test_macd(self, sample_ohlcv):
        """Test MACD calculation."""
        macd = MACD(fast_period=12, slow_period=26, signal_period=9)
        result = macd.compute(sample_ohlcv)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv)
        assert 'macd' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_histogram' in result.columns
        
        # MACD = Fast EMA - Slow EMA
        fast_ema = sample_ohlcv['close'].ewm(span=12, adjust=False).mean()
        slow_ema = sample_ohlcv['close'].ewm(span=26, adjust=False).mean()
        expected_macd = fast_ema - slow_ema
        
        # Check MACD line calculation (allowing for small differences)
        pd.testing.assert_series_equal(
            result['macd'],
            expected_macd,
            check_names=False,
            atol=1e-5
        )
    
    def test_bollinger_bands(self, sample_ohlcv):
        """Test Bollinger Bands calculation."""
        bb = BollingerBands(period=20, std_dev=2.0)
        result = bb.compute(sample_ohlcv)
        
        assert isinstance(result, pd.DataFrame)
        assert 'bb_upper' in result.columns
        assert 'bb_middle' in result.columns
        assert 'bb_lower' in result.columns
        assert 'bb_bandwidth' in result.columns
        assert 'bb_percent' in result.columns
        
        # Check band relationships
        assert (result['bb_upper'] > result['bb_middle']).all()
        assert (result['bb_middle'] > result['bb_lower']).all()
        
        # Check middle band is SMA
        sma20 = sample_ohlcv['close'].rolling(20).mean()
        pd.testing.assert_series_equal(
            result['bb_middle'],
            sma20,
            check_names=False
        )
    
    def test_atr(self, sample_ohlcv):
        """Test ATR calculation."""
        atr = ATR(period=14)
        result = atr.compute(sample_ohlcv)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)
        assert result.name == 'ATR_14'
        
        # ATR should be positive
        valid_values = result.dropna()
        assert (valid_values > 0).all()
        
        # ATR should increase with volatility
        high_vol_period = sample_ohlcv.iloc[50:100]
        low_vol_period = sample_ohlcv.iloc[150:200]
        
        if high_vol_period['close'].std() > low_vol_period['close'].std():
            assert result.iloc[50:100].mean() > result.iloc[150:200].mean()
    
    def test_stochastic(self, sample_ohlcv):
        """Test Stochastic Oscillator calculation."""
        stoch = Stochastic(period=14, smooth_k=3, smooth_d=3)
        result = stoch.compute(sample_ohlcv)
        
        assert isinstance(result, pd.DataFrame)
        assert 'stoch_k' in result.columns
        assert 'stoch_d' in result.columns
        
        # Stochastic should be between 0 and 100
        for col in ['stoch_k', 'stoch_d']:
            valid_values = result[col].dropna()
            assert (valid_values >= 0).all()
            assert (valid_values <= 100).all()
        
        # %D should be smoother than %K
        k_volatility = result['stoch_k'].std()
        d_volatility = result['stoch_d'].std()
        assert d_volatility < k_volatility
    
    def test_mfi(self, sample_ohlcv):
        """Test Money Flow Index calculation."""
        mfi = MFI(period=14)
        result = mfi.compute(sample_ohlcv)
        
        assert isinstance(result, pd.Series)
        assert result.name == 'MFI_14'
        
        # MFI should be between 0 and 100
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()
    
    def test_vwap(self, sample_ohlcv):
        """Test VWAP calculation."""
        vwap = VWAP()
        result = vwap.compute(sample_ohlcv)
        
        assert isinstance(result, pd.Series)
        assert result.name == 'VWAP' or result.name == 'vwap'
        
        # VWAP should be close to typical price weighted by volume
        typical_price = (sample_ohlcv['high'] + sample_ohlcv['low'] + sample_ohlcv['close']) / 3
        cumulative_tpv = (typical_price * sample_ohlcv['volume']).cumsum()
        cumulative_volume = sample_ohlcv['volume'].cumsum()
        expected_vwap = cumulative_tpv / cumulative_volume
        
        pd.testing.assert_series_equal(
            result,
            expected_vwap,
            check_names=False,
            atol=1e-5
        )
    
    def test_pivot_points(self, sample_ohlcv):
        """Test Pivot Points calculation."""
        pp = PivotPoints()
        result = pp.compute(sample_ohlcv)
        
        assert isinstance(result, pd.DataFrame)
        required_columns = ['pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3']
        for col in required_columns:
            assert col in result.columns
        
        # Check relationships
        # R1 > Pivot > S1
        assert (result['r1'] > result['pivot']).all()
        assert (result['pivot'] > result['s1']).all()
        
        # R2 > R1, R3 > R2
        assert (result['r2'] > result['r1']).all()
        assert (result['r3'] > result['r2']).all()
        
        # S1 > S2 > S3
        assert (result['s1'] > result['s2']).all()
        assert (result['s2'] > result['s3']).all()
    
    def test_signal_dependencies(self):
        """Test that signals correctly declare dependencies."""
        # MACD depends on close prices
        macd = MACD()
        assert 'close' in macd.dependencies
        
        # ATR depends on high, low, close
        atr = ATR()
        assert all(col in atr.dependencies for col in ['high', 'low', 'close'])
        
        # MFI depends on OHLCV
        mfi = MFI()
        assert all(col in mfi.dependencies for col in ['high', 'low', 'close', 'volume'])
    
    def test_signal_parameters(self):
        """Test signal parameter extraction."""
        rsi = RSI(period=21)
        params = rsi.get_parameters()
        
        assert params['period'] == 21
        assert params['name'] == 'rsi_21'
        assert params['signal_type'] == 'momentum'
        
        # Test MACD parameters
        macd = MACD(fast_period=10, slow_period=20, signal_period=5)
        params = macd.get_parameters()
        
        assert params['fast_period'] == 10
        assert params['slow_period'] == 20
        assert params['signal_period'] == 5


class TestPatternRecognition:
    """Test pattern recognition signals."""
    
    @pytest.fixture
    def pattern_data(self):
        """Create data with specific patterns."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Create a simple trend with reversal
        prices = []
        for i in range(50):
            prices.append(100 + i * 0.5)  # Uptrend
        for i in range(50):
            prices.append(125 - i * 0.3)  # Downtrend
        
        df = pd.DataFrame({
            'date': dates,
            'close': prices
        })
        
        # Add OHLV
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['open', 'close']].max(axis=1) * 1.01
        df['low'] = df[['open', 'close']].min(axis=1) * 0.99
        df['volume'] = 1000000
        
        return df.set_index('date')
    
    def test_candle_patterns(self, pattern_data):
        """Test candlestick pattern recognition."""
        patterns = CandlePatterns()
        result = patterns.compute(pattern_data)
        
        assert isinstance(result, pd.DataFrame)
        
        # Should detect various patterns
        pattern_columns = [col for col in result.columns if col.startswith('pattern_')]
        assert len(pattern_columns) > 0
        
        # Patterns should be binary (0 or 1) or strength values
        for col in pattern_columns:
            assert result[col].isin([0, 1, -1, 100, -100]).all()
    
    def test_fibonacci_retracements(self, pattern_data):
        """Test Fibonacci retracement levels."""
        fib = FibonacciRetracements(lookback=50)
        result = fib.compute(pattern_data)
        
        assert isinstance(result, pd.DataFrame)
        
        # Should have standard Fibonacci levels
        expected_levels = ['fib_0', 'fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786', 'fib_1000']
        for level in expected_levels:
            assert level in result.columns
        
        # Levels should be ordered
        for i in range(len(pattern_data)):
            if not result.iloc[i].isna().any():
                assert result['fib_0'].iloc[i] < result['fib_236'].iloc[i]
                assert result['fib_236'].iloc[i] < result['fib_382'].iloc[i]
                assert result['fib_618'].iloc[i] < result['fib_1000'].iloc[i]