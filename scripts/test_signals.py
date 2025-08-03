#!/usr/bin/env python3
"""Test script for signal computation module."""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signals import SignalConfig, TechnicalSignals, SignalType


def generate_test_data(n_days=100):
    """Generate synthetic OHLCV data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Generate synthetic price data
    np.random.seed(42)
    close = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.02))
    
    # Generate OHLCV
    data = pd.DataFrame({
        'date': dates,
        'open': close * (1 + np.random.randn(n_days) * 0.001),
        'high': close * (1 + np.abs(np.random.randn(n_days) * 0.005)),
        'low': close * (1 - np.abs(np.random.randn(n_days) * 0.005)),
        'close': close,
        'volume': np.random.randint(1000000, 10000000, n_days)
    })
    
    data.set_index('date', inplace=True)
    
    return data


def test_technical_signals():
    """Test technical signal computation."""
    print("\n=== Testing Technical Signals ===")
    
    # Generate test data
    data = generate_test_data(200)
    print(f"\nGenerated test data: {len(data)} rows")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"\nFirst few rows:")
    print(data.head())
    
    # Create signal config
    config = SignalConfig(
        # Customize some parameters
        rsi_period=14,
        bb_period=20,
        ema_periods=[9, 21, 50],
        sma_periods=[20, 50]
    )
    
    # Create technical signals computer
    tech_signals = TechnicalSignals(config)
    
    # List available signals
    print("\n=== Available Technical Signals ===")
    all_signals = tech_signals.list_signals()
    print(f"Total signals available: {len(all_signals)}")
    
    # Group by type
    by_type = {}
    for sig in all_signals:
        sig_type = sig['type']
        if sig_type not in by_type:
            by_type[sig_type] = []
        by_type[sig_type].append(sig['name'])
    
    for sig_type, names in by_type.items():
        print(f"\n{sig_type}: {len(names)} signals")
        print(f"  {', '.join(names[:5])}{', ...' if len(names) > 5 else ''}")
    
    # Compute some specific signals
    print("\n=== Computing Specific Signals ===")
    specific_signals = ['RSI_14', 'MACD', 'BollingerBands', 'ATR_14', 'OBV']
    
    result = tech_signals.compute(
        data,
        signal_names=specific_signals
    )
    
    print(f"\nComputation completed in {result.compute_time:.2f} seconds")
    print(f"Computed {len(result.signal_names)} signals")
    print(f"\nSignal names: {result.signal_names}")
    
    # Show last few values
    print("\nLast 5 values for each signal:")
    print(result.data.tail())
    
    # Check for errors
    if result.has_errors:
        print(f"\nErrors encountered: {result.errors}")
    
    if result.warnings:
        print(f"\nWarnings: {result.warnings}")
    
    # Compute all trend signals
    print("\n=== Computing All Trend Signals ===")
    trend_signals = tech_signals.get_trend_signals()
    print(f"Computing {len(trend_signals)} trend signals...")
    
    trend_result = tech_signals.compute(
        data,
        signal_names=trend_signals
    )
    
    print(f"Completed in {trend_result.compute_time:.2f} seconds")
    
    # Show summary
    summary = trend_result.summary()
    print(f"\nSummary:")
    for key, value in summary.items():
        if key != 'signal_types':
            print(f"  {key}: {value}")
    
    return result


def test_multi_symbol_signals():
    """Test signal computation for multiple symbols."""
    print("\n=== Testing Multi-Symbol Signal Computation ===")
    
    # Generate data for multiple symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    all_data = []
    
    for i, symbol in enumerate(symbols):
        data = generate_test_data(100)
        data['symbol'] = symbol
        # Make prices different for each symbol
        data['close'] = data['close'] * (1 + i * 0.5)
        data['open'] = data['open'] * (1 + i * 0.5)
        data['high'] = data['high'] * (1 + i * 0.5)
        data['low'] = data['low'] * (1 + i * 0.5)
        all_data.append(data)
    
    # Combine all data
    combined_data = pd.concat(all_data).reset_index()
    print(f"\nCombined data shape: {combined_data.shape}")
    print(f"Symbols: {combined_data['symbol'].unique()}")
    
    # Create signal computer
    config = SignalConfig()
    tech_signals = TechnicalSignals(config)
    
    # Compute signals for all symbols
    signals_to_compute = ['RSI_14', 'MACD', 'ATR_14']
    
    result = tech_signals.compute(
        combined_data,
        signal_names=signals_to_compute
    )
    
    print(f"\nComputed signals for {len(symbols)} symbols")
    print(f"Total columns: {len(result.data.columns)}")
    print(f"\nColumn names:")
    for symbol in symbols:
        symbol_cols = [col for col in result.data.columns if col.startswith(symbol)]
        print(f"  {symbol}: {symbol_cols}")
    
    # Show sample values
    print("\nSample values (last 3 rows):")
    print(result.data.tail(3))
    
    return result


def test_signal_caching():
    """Test signal computation caching."""
    print("\n=== Testing Signal Caching ===")
    
    data = generate_test_data(100)
    
    # Create config with caching enabled
    config = SignalConfig(
        cache_results=True,
        cache_ttl_seconds=60
    )
    
    tech_signals = TechnicalSignals(config)
    
    # First computation
    print("\nFirst computation...")
    result1 = tech_signals.compute(data, signal_names=['RSI_14', 'MACD'])
    print(f"Time: {result1.compute_time:.3f} seconds")
    
    # Second computation (should be cached)
    print("\nSecond computation (cached)...")
    result2 = tech_signals.compute(data, signal_names=['RSI_14', 'MACD'])
    print(f"Time: {result2.compute_time:.3f} seconds")
    
    # Clear cache and compute again
    print("\nAfter clearing cache...")
    tech_signals.clear_cache()
    result3 = tech_signals.compute(data, signal_names=['RSI_14', 'MACD'])
    print(f"Time: {result3.compute_time:.3f} seconds")
    
    return result1


def main():
    """Run all tests."""
    print("Signal Module Test Script")
    print("=" * 50)
    
    # Test technical signals
    test_technical_signals()
    
    # Test multi-symbol computation
    test_multi_symbol_signals()
    
    # Test caching
    test_signal_caching()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")


if __name__ == "__main__":
    main()