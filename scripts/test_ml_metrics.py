#!/usr/bin/env python3
"""Test ML metrics calculation in the metrics pipeline."""

import asyncio
import logging
from datetime import date, timedelta
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.metrics_pipeline.calculators.ml_metrics import MLMetricsCalculator
from src.simulators.utils.data_loader import DataLoader
from src.definitions.market_universe import MarketUniverse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_ml_metrics():
    """Test ML metrics calculation with real database data."""
    
    # Initialize components
    data_loader = DataLoader()
    ml_calculator = MLMetricsCalculator()
    
    # Get a few test symbols
    test_symbols = MarketUniverse.get_high_priority_symbols()[:3]
    logger.info(f"Testing with symbols: {test_symbols}")
    
    # Load data from database
    end_date = date.today()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    try:
        # Load OHLCV data
        logger.info(f"Loading data from {start_date} to {end_date}")
        data_batch = await data_loader.load_ohlcv_batch(
            symbols=test_symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        if not data_batch:
            logger.error("No data loaded from database")
            return
        
        # Test ML metrics for each symbol
        for symbol, data in data_batch.items():
            logger.info(f"\nProcessing {symbol}...")
            logger.info(f"Data shape: {data.shape}")
            
            # Add some basic technical indicators for ML features
            data['sma_20'] = data['close'].rolling(20).mean()
            data['sma_50'] = data['close'].rolling(50).mean()
            data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            
            # Calculate ML metrics
            ml_metrics = ml_calculator.calculate_ml_metrics(data, symbol)
            
            if ml_metrics.empty:
                logger.warning(f"No ML metrics generated for {symbol}")
                continue
            
            logger.info(f"ML metrics shape: {ml_metrics.shape}")
            logger.info(f"ML metrics columns: {list(ml_metrics.columns)[:10]}...")
            
            # Display sample predictions
            latest_metrics = ml_metrics.iloc[-1]
            logger.info(f"\nLatest ML predictions for {symbol}:")
            logger.info(f"  Signal: {latest_metrics.get('ml_signal', 'N/A')}")
            logger.info(f"  Action: {latest_metrics.get('ml_action', 'N/A')}")
            logger.info(f"  Expected Return: {latest_metrics.get('ml_expected_return', 0):.4f}")
            logger.info(f"  Position Size: {latest_metrics.get('ml_position_size', 0):.2%}")
            logger.info(f"  Risk Score: {latest_metrics.get('ml_risk_score', 0):.2f}")
            
            # Check model performance metrics
            if 'ml_accuracy' in latest_metrics:
                logger.info(f"\nModel Performance:")
                logger.info(f"  Accuracy: {latest_metrics.get('ml_accuracy', 0):.2%}")
                logger.info(f"  Hit Rate: {latest_metrics.get('ml_hit_rate', 0):.2%}")
                logger.info(f"  R²: {latest_metrics.get('ml_r2', 0):.3f}")
                
    except Exception as e:
        logger.error(f"Error testing ML metrics: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(test_ml_metrics())