#!/usr/bin/env python3
"""Test and demonstrate signal computation optimizations."""

import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.signals.base import SignalComputer, SignalConfig
from src.signals.optimizations import BatchSignalComputer, MemoryOptimizedComputer
from src.signals.technical import RSI, SMA, EMA, MACD, BollingerBands
from src.db.connection import get_db_engine
from sqlalchemy import text


def load_test_data(symbols: list, days: int = 365):
    """Load test data from database."""
    engine = get_db_engine()
    
    # Build query
    symbol_list = "','".join(symbols)
    query = f"""
        SELECT symbol, timestamp as date, open, high, low, close, volume
        FROM market_data.ohlcv
        WHERE symbol IN ('{symbol_list}')
        AND timeframe = '1d'
        AND timestamp >= CURRENT_DATE - INTERVAL '{days} days'
        ORDER BY symbol, timestamp
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    
    return df


def test_standard_computation(data: pd.DataFrame, symbols: list):
    """Test standard signal computation."""
    print("\n=== Standard Signal Computation ===")
    
    # Initialize computer and register signals
    computer = SignalComputer()
    computer.register_signals([
        RSI(period=14),
        SMA(period=20),
        EMA(period=12),
        MACD(),
        BollingerBands()
    ])
    
    start_time = time.time()
    
    # Compute for each symbol separately
    results = {}
    for symbol in symbols:
        symbol_data = data[data['symbol'] == symbol].set_index('date')
        result = computer.compute(symbol_data[['open', 'high', 'low', 'close', 'volume']])
        results[symbol] = result
    
    elapsed = time.time() - start_time
    
    print(f"Processed {len(symbols)} symbols in {elapsed:.2f} seconds")
    print(f"Average time per symbol: {elapsed/len(symbols):.3f} seconds")
    
    # Show sample results
    first_symbol = symbols[0]
    print(f"\nSample results for {first_symbol}:")
    print(results[first_symbol].data.tail())
    
    return results, elapsed


def test_batch_computation(data: pd.DataFrame, symbols: list):
    """Test batch signal computation."""
    print("\n=== Batch Signal Computation ===")
    
    # Initialize batch computer
    batch_computer = BatchSignalComputer(n_processes=4)
    batch_computer.register_signals([
        RSI(period=14),
        SMA(period=20),
        EMA(period=12),
        MACD(),
        BollingerBands()
    ])
    
    start_time = time.time()
    
    # Compute all symbols in batch
    results = batch_computer.compute_batch(
        data,
        symbols,
        chunk_size=10
    )
    
    elapsed = time.time() - start_time
    
    print(f"Processed {len(symbols)} symbols in {elapsed:.2f} seconds")
    print(f"Average time per symbol: {elapsed/len(symbols):.3f} seconds")
    
    # Show sample results
    first_symbol = symbols[0]
    print(f"\nSample results for {first_symbol}:")
    print(results[first_symbol].data.tail())
    
    return results, elapsed


def test_memory_optimized(data: pd.DataFrame, symbols: list):
    """Test memory-optimized computation."""
    print("\n=== Memory-Optimized Computation ===")
    
    # Initialize memory-optimized computer
    mem_computer = MemoryOptimizedComputer(
        chunk_size=5000,
        dtype_optimization=True
    )
    mem_computer.register_signals([
        RSI(period=14),
        SMA(period=20),
        EMA(period=12),
        MACD(),
        BollingerBands()
    ])
    
    # Estimate memory usage
    first_symbol_data = data[data['symbol'] == symbols[0]]
    estimates = mem_computer.estimate_memory_usage(first_symbol_data)
    
    print(f"Memory estimates:")
    print(f"  Input data: {estimates['input_data'] / 1024 / 1024:.2f} MB")
    print(f"  Signal data: {estimates['total_signals'] / 1024 / 1024:.2f} MB")
    print(f"  Total: {estimates['total'] / 1024 / 1024:.2f} MB")
    
    start_time = time.time()
    
    # Process with memory optimization
    results = {}
    for symbol in symbols[:5]:  # Test with first 5 symbols
        symbol_data = data[data['symbol'] == symbol].set_index('date')
        result = mem_computer.compute_chunked(
            symbol_data[['open', 'high', 'low', 'close', 'volume']],
            chunk_size=1000
        )
        results[symbol] = result
    
    elapsed = time.time() - start_time
    
    print(f"\nProcessed {len(results)} symbols in {elapsed:.2f} seconds")
    print(f"Average time per symbol: {elapsed/len(results):.3f} seconds")
    
    return results, elapsed


def test_lazy_evaluation(data: pd.DataFrame, symbol: str):
    """Test lazy evaluation."""
    print("\n=== Lazy Evaluation Test ===")
    
    # Initialize memory-optimized computer
    mem_computer = MemoryOptimizedComputer()
    mem_computer.register_signals([
        RSI(period=14),
        SMA(period=20),
        EMA(period=12),
        MACD(),
        BollingerBands()
    ])
    
    # Get data for single symbol
    symbol_data = data[data['symbol'] == symbol].set_index('date')
    
    # Create lazy result
    lazy_result = mem_computer.compute_lazy(
        symbol_data[['open', 'high', 'low', 'close', 'volume']]
    )
    
    print(f"Available signals: {lazy_result.available_signals}")
    
    # Time individual signal access
    start_time = time.time()
    rsi_values = lazy_result['rsi_14']
    rsi_time = time.time() - start_time
    print(f"\nRSI computation time: {rsi_time:.3f} seconds")
    print(f"RSI tail values:\n{rsi_values.tail()}")
    
    # Access another signal (should be fast if using same base calculations)
    start_time = time.time()
    sma_values = lazy_result['sma_20']
    sma_time = time.time() - start_time
    print(f"\nSMA computation time: {sma_time:.3f} seconds")
    print(f"SMA tail values:\n{sma_values.tail()}")


def compare_performance():
    """Compare performance of different computation methods."""
    print("=" * 60)
    print("Signal Computation Optimization Comparison")
    print("=" * 60)
    
    # Load test data
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 
               'TSLA', 'NVDA', 'JPM', 'JNJ', 'V',
               'WMT', 'PG', 'UNH', 'HD', 'DIS',
               'MA', 'PYPL', 'NFLX', 'ADBE', 'CRM']
    
    print(f"Loading data for {len(symbols)} symbols...")
    data = load_test_data(symbols)
    print(f"Loaded {len(data)} total records")
    
    # Test different methods
    results = {}
    
    # 1. Standard computation
    std_results, std_time = test_standard_computation(data, symbols[:10])
    results['standard'] = std_time
    
    # 2. Batch computation
    batch_results, batch_time = test_batch_computation(data, symbols[:10])
    results['batch'] = batch_time
    
    # 3. Memory-optimized computation
    mem_results, mem_time = test_memory_optimized(data, symbols)
    results['memory_optimized'] = mem_time
    
    # 4. Lazy evaluation
    test_lazy_evaluation(data, symbols[0])
    
    # Summary
    print("\n" + "=" * 60)
    print("Performance Summary (10 symbols):")
    print("=" * 60)
    print(f"Standard computation: {results['standard']:.2f} seconds")
    print(f"Batch computation: {results['batch']:.2f} seconds")
    print(f"Speedup: {results['standard'] / results['batch']:.2f}x")
    
    # Memory usage comparison
    print("\nMemory Optimization Benefits:")
    print("- Reduced memory footprint through dtype optimization")
    print("- Chunked processing for large datasets")
    print("- Lazy evaluation for on-demand computation")


if __name__ == "__main__":
    compare_performance()