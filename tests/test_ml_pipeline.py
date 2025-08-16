#!/usr/bin/env python
"""
Test script for the new time series ML pipeline.
"""

import asyncio
import logging
from datetime import date, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml import (
    ARIMAModel, 
    XGBoostTSModel, 
    ModelCalibrator,
    PredictionEngine,
    MLPipeline
)
from src.simulators.historical.market_replay import MarketReplay

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_basic_models():
    """Test individual model creation and basic functionality."""
    logger.info("Testing basic model creation...")
    
    # Test ARIMA
    arima = ARIMAModel(prediction_horizon=1, target_type='return')
    logger.info(f"✓ Created ARIMA model: {arima.name}")
    
    # Test XGBoost
    xgb = XGBoostTSModel(prediction_horizon=5, target_type='return')
    logger.info(f"✓ Created XGBoost model: {xgb.name}")
    
    return True


async def test_calibration():
    """Test model calibration with simulator."""
    logger.info("Testing model calibration...")
    
    # Initialize simulator
    simulator = MarketReplay()
    calibrator = ModelCalibrator(simulator)
    
    # Create a simple model
    model = XGBoostTSModel(prediction_horizon=1, target_type='return')
    
    # Define calibration period (last 3 months)
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=90)
    
    # Calibrate on SPY
    logger.info(f"Calibrating {model.name} on SPY from {start_date} to {end_date}")
    
    try:
        result = await calibrator.calibrate_historical(
            model, 
            symbols=['SPY'],
            start_date=start_date,
            end_date=end_date,
            validation_split=0.2
        )
        
        if model.is_fitted:
            logger.info(f"✓ Model calibrated successfully")
            logger.info(f"  Validation MAE: {result['metrics']['validation'].get('mae', 0):.4f}")
            return True
        else:
            logger.error("✗ Model calibration failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ Calibration error: {e}")
        return False


async def test_prediction():
    """Test prediction generation."""
    logger.info("Testing prediction generation...")
    
    # Initialize components
    simulator = MarketReplay()
    calibrator = ModelCalibrator(simulator)
    predictor = PredictionEngine(simulator)
    
    # Create and calibrate model
    model = XGBoostTSModel(prediction_horizon=1, target_type='return')
    
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=90)
    
    # Calibrate
    await calibrator.calibrate_historical(
        model,
        symbols=['SPY'],
        start_date=start_date,
        end_date=end_date
    )
    
    if not model.is_fitted:
        logger.error("✗ Model not fitted, cannot test prediction")
        return False
        
    # Add model to predictor
    predictor.add_model(model)
    
    # Make prediction
    prediction_date = date.today()
    
    try:
        result = await predictor.predict_single(
            model.name,
            'SPY',
            prediction_date
        )
        
        if result['prediction'] is not None:
            logger.info(f"✓ Generated prediction for SPY on {prediction_date}")
            logger.info(f"  Prediction: {result['prediction']:.4f}")
            return True
        else:
            logger.error("✗ Prediction failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ Prediction error: {e}")
        return False


async def test_full_pipeline():
    """Test complete ML pipeline."""
    logger.info("Testing full ML pipeline...")
    
    # Configure minimal pipeline for testing
    config = {
        'models': {
            'statistical': ['arima'],
            'ml': ['xgboost'],
            'deep': []  # Skip deep learning for speed
        },
        'target_types': ['return'],
        'prediction_horizons': [1],
        'symbols': ['SPY', 'QQQ']
    }
    
    # Create pipeline
    pipeline = MLPipeline(config)
    
    # Define dates
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=60)  # 2 months for testing
    test_date = date.today()
    
    logger.info(f"Running pipeline from {start_date} to {end_date}")
    
    try:
        # Run full pipeline
        results = await pipeline.run_full_pipeline(
            train_start=start_date,
            train_end=end_date,
            test_date=test_date,
            symbols=['SPY', 'QQQ']
        )
        
        # Check results
        if results.get('predictions'):
            logger.info("✓ Full pipeline completed successfully")
            logger.info(f"  Models created: {results['summary']['n_models_created']}")
            logger.info(f"  Models fitted: {results['summary']['n_models_fitted']}")
            
            if 'best_overall_model' in results['summary']:
                logger.info(f"  Best model: {results['summary']['best_overall_model']}")
                logger.info(f"  Best MAE: {results['summary']['best_mae']:.4f}")
                
            return True
        else:
            logger.error("✗ Pipeline completed but no predictions generated")
            return False
            
    except Exception as e:
        logger.error(f"✗ Pipeline error: {e}")
        return False


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TESTING NEW ML PIPELINE")
    print("="*60 + "\n")
    
    tests = [
        ("Basic Models", test_basic_models),
        ("Model Calibration", test_calibration),
        ("Prediction Generation", test_prediction),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
            
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
        
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed")
        
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)