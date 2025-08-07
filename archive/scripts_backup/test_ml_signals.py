#!/usr/bin/env python3
"""Test script for ML signal computation."""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signals import SignalConfig
from src.signals.ml import MLSignals


def generate_ml_test_data(n_days=500):
    """Generate synthetic OHLCV data for ML testing."""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Generate more realistic price data with trends and volatility regimes
    np.random.seed(42)
    
    # Create base price with trend changes
    trend = np.cumsum(np.random.randn(n_days) * 0.02)
    
    # Add volatility clustering
    vol_regime = np.zeros(n_days)
    regime_changes = np.random.choice(range(0, n_days), size=10, replace=False)
    regime_changes.sort()
    
    current_vol = 0.01
    for i in range(n_days):
        if i in regime_changes:
            current_vol = np.random.uniform(0.005, 0.03)
        vol_regime[i] = current_vol
    
    # Generate price with volatility regimes
    returns = np.random.randn(n_days) * vol_regime
    log_price = np.cumsum(returns) + trend
    close = 100 * np.exp(log_price)
    
    # Generate OHLCV
    data = pd.DataFrame({
        'date': dates,
        'open': close * (1 + np.random.randn(n_days) * 0.001),
        'high': close * (1 + np.abs(np.random.randn(n_days) * 0.005)),
        'low': close * (1 - np.abs(np.random.randn(n_days) * 0.005)),
        'close': close,
        'volume': np.random.randint(1000000, 10000000, n_days) * (1 + vol_regime * 10)
    })
    
    data.set_index('date', inplace=True)
    
    return data


def test_ml_classification():
    """Test ML classification signals."""
    print("\n=== Testing ML Classification Signals ===")
    
    # Generate test data
    data = generate_ml_test_data(1000)  # Need more data for ML
    print(f"\nGenerated test data: {len(data)} rows")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Create signal config
    config = SignalConfig(
        ml_feature_window=20,
        ml_prediction_horizon=5,
        ml_min_train_samples=500,
        ml_retrain_frequency=100
    )
    
    # Create ML signals computer
    ml_signals = MLSignals(config)
    
    # Test direction classifier
    print("\n=== Direction Classifier ===")
    result = ml_signals.compute(
        data,
        signal_names=[f"DirectionClassifier_{config.ml_prediction_horizon}"]
    )
    
    print(f"Computation completed in {result.compute_time:.2f} seconds")
    print(f"\nSignal statistics:")
    signal = result.data.iloc[:, 0]
    print(f"  Mean: {signal.mean():.2f}")
    print(f"  Std: {signal.std():.2f}")
    print(f"  Min: {signal.min():.2f}")
    print(f"  Max: {signal.max():.2f}")
    
    # Show recent predictions
    print("\nRecent predictions:")
    print(result.data.tail())
    
    return result


def test_ml_regression():
    """Test ML regression signals."""
    print("\n=== Testing ML Regression Signals ===")
    
    data = generate_ml_test_data(800)
    
    config = SignalConfig(
        ml_feature_window=30,
        ml_prediction_horizon=10
    )
    
    ml_signals = MLSignals(config)
    
    # Test multiple regression signals
    regression_signals = [
        f"PriceRegression_{config.ml_prediction_horizon}",
        "VolatilityRegression_20",
        "MultiFactorRegression"
    ]
    
    print(f"\nComputing {len(regression_signals)} regression signals...")
    result = ml_signals.compute(data, signal_names=regression_signals)
    
    print(f"\nComputation completed in {result.compute_time:.2f} seconds")
    
    # Show correlations between signals
    print("\nSignal correlations:")
    corr = result.data.corr()
    print(corr)
    
    # Show signal ranges
    print("\nSignal ranges:")
    for col in result.data.columns:
        print(f"  {col}: [{result.data[col].min():.2f}, {result.data[col].max():.2f}]")
    
    return result


def test_ml_anomaly():
    """Test ML anomaly detection signals."""
    print("\n=== Testing ML Anomaly Detection ===")
    
    # Generate data with some anomalies
    data = generate_ml_test_data(600)
    
    # Inject some anomalies
    anomaly_indices = np.random.choice(range(100, 500), size=10, replace=False)
    for idx in anomaly_indices:
        data.iloc[idx, 3] *= np.random.uniform(0.95, 1.05)  # Price spike
        data.iloc[idx, 4] *= np.random.uniform(2, 5)  # Volume spike
    
    config = SignalConfig()
    ml_signals = MLSignals(config)
    
    # Test anomaly signals
    anomaly_signals = [
        "PriceAnomaly_isolation_forest",
        "VolumeAnomaly",
        "MultivarAnomaly"
    ]
    
    result = ml_signals.compute(data, signal_names=anomaly_signals)
    
    print(f"\nComputation completed in {result.compute_time:.2f} seconds")
    
    # Find detected anomalies
    for signal_name in result.data.columns:
        signal = result.data[signal_name]
        threshold = signal.std() * 2
        anomalies = signal[signal.abs() > threshold]
        print(f"\n{signal_name} detected {len(anomalies)} anomalies")
        if len(anomalies) > 0:
            print(f"  Anomaly dates: {anomalies.index[:5].tolist()}...")
    
    return result


def test_ml_clustering():
    """Test ML clustering signals."""
    print("\n=== Testing ML Clustering Signals ===")
    
    data = generate_ml_test_data(700)
    
    config = SignalConfig()
    ml_signals = MLSignals(config)
    
    # Test clustering signals
    clustering_signals = [
        "MarketStateClustering_5",
        "PriceActionClustering_8"
    ]
    
    result = ml_signals.compute(data, signal_names=clustering_signals)
    
    print(f"\nComputation completed in {result.compute_time:.2f} seconds")
    
    # Analyze cluster assignments
    for signal_name in result.data.columns:
        signal = result.data[signal_name]
        print(f"\n{signal_name} statistics:")
        print(f"  Unique values: {signal.nunique()}")
        print(f"  Value counts:")
        print(signal.value_counts().head())
    
    return result


def test_ml_ensemble():
    """Test ML ensemble signals."""
    print("\n=== Testing ML Ensemble Signals ===")
    
    data = generate_ml_test_data(800)
    
    config = SignalConfig(
        ml_prediction_horizon=5
    )
    
    ml_signals = MLSignals(config)
    
    # Test ensemble signals
    ensemble_signals = [
        f"EnsembleDirection_{config.ml_prediction_horizon}",
        "StackedML",
        "VotingRegime_3"
    ]
    
    result = ml_signals.compute(data, signal_names=ensemble_signals)
    
    print(f"\nComputation completed in {result.compute_time:.2f} seconds")
    
    # Compare ensemble performance
    print("\nEnsemble signal comparison:")
    for signal_name in result.data.columns:
        signal = result.data[signal_name]
        print(f"\n{signal_name}:")
        print(f"  Mean: {signal.mean():.2f}")
        print(f"  Volatility: {signal.std():.2f}")
        print(f"  Sharpe: {signal.mean() / signal.std() * np.sqrt(252):.2f}")
    
    return result


def test_ml_with_technical():
    """Test combining ML and technical signals."""
    print("\n=== Testing ML + Technical Signal Combination ===")
    
    data = generate_ml_test_data(600)
    
    config = SignalConfig()
    ml_signals = MLSignals(config)
    
    # Combine ML and technical signals
    combined_signals = [
        # Technical
        "RSI_14",
        "MACD",
        "BollingerBands",
        # ML
        "DirectionClassifier_5",
        "PriceAnomaly_isolation_forest",
        "MarketStateClustering_5"
    ]
    
    result = ml_signals.compute(data, signal_names=combined_signals)
    
    print(f"\nComputation completed in {result.compute_time:.2f} seconds")
    print(f"Computed {len(result.signal_names)} signals")
    
    # Show correlation between technical and ML signals
    print("\nTechnical vs ML signal correlations:")
    tech_signals = [s for s in result.data.columns if not any(ml in s for ml in ['Classifier', 'Anomaly', 'Clustering'])]
    ml_sigs = [s for s in result.data.columns if any(ml in s for ml in ['Classifier', 'Anomaly', 'Clustering'])]
    
    for tech in tech_signals:
        for ml in ml_sigs:
            corr = result.data[tech].corr(result.data[ml])
            if abs(corr) > 0.3:
                print(f"  {tech} vs {ml}: {corr:.3f}")
    
    return result


def main():
    """Run all ML tests."""
    print("ML Signal Module Test Script")
    print("=" * 50)
    
    try:
        # Test classification
        test_ml_classification()
        
        # Test regression
        test_ml_regression()
        
        # Test anomaly detection
        test_ml_anomaly()
        
        # Test clustering
        test_ml_clustering()
        
        # Test ensemble
        test_ml_ensemble()
        
        # Test combined signals
        test_ml_with_technical()
        
        print("\n" + "=" * 50)
        print("All ML tests completed successfully!")
        
    except ImportError as e:
        print(f"\nNote: Some ML libraries may not be installed: {e}")
        print("Install with: pip install scikit-learn xgboost")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()