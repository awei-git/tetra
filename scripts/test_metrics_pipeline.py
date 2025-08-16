#!/usr/bin/env python3
"""Test the metrics pipeline with ML metrics and scenario data."""

import asyncio
import logging
from pathlib import Path
import sys
import json
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.metrics_pipeline.pipeline import MetricsPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_metrics_pipeline():
    """Test the metrics pipeline execution."""
    
    logger.info("=" * 80)
    logger.info("TESTING METRICS PIPELINE")
    logger.info("=" * 80)
    
    # Check if scenarios exist
    scenarios_dir = Path('data/scenarios')
    metadata_file = scenarios_dir / 'scenario_metadata.json'
    timeseries_file = scenarios_dir / 'scenario_timeseries.parquet'
    
    if not metadata_file.exists():
        logger.warning("No scenarios found. Run scenarios pipeline first:")
        logger.warning("python -m src.pipelines.scenarios_pipeline.main")
        return
    
    # Load and inspect scenarios
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Found {metadata['total_scenarios']} scenarios")
    logger.info(f"Generated at: {metadata['generated_at']}")
    
    # Show first few scenarios
    for i, scenario in enumerate(metadata['scenarios'][:3]):
        logger.info(f"\nScenario {i+1}: {scenario['name']}")
        logger.info(f"  Type: {scenario['scenario_type']}")
        logger.info(f"  Period: {scenario['start_date']} to {scenario['end_date']}")
        logger.info(f"  Description: {scenario['description'][:100]}...")
    
    # Check timeseries data
    if timeseries_file.exists():
        df_timeseries = pd.read_parquet(timeseries_file)
        logger.info(f"\nTimeseries data: {len(df_timeseries)} records")
        logger.info(f"Unique scenarios: {df_timeseries['scenario_id'].nunique()}")
        logger.info(f"Unique symbols: {df_timeseries['symbol'].nunique()}")
        
        # Show sample data
        logger.info("\nSample timeseries data:")
        logger.info(df_timeseries.head())
    
    # Initialize pipeline
    logger.info("\n" + "=" * 80)
    logger.info("INITIALIZING METRICS PIPELINE")
    logger.info("=" * 80)
    
    config = {
        'parallel_workers': 2,  # Use fewer workers for testing
        'batch_size': 10,
        'storage_dir': 'data/metrics_test'
    }
    
    pipeline = MetricsPipeline(config)
    
    # Test with a subset of scenarios
    logger.info("\nRunning metrics pipeline on first 5 scenarios...")
    
    try:
        # Run pipeline with scenario filter
        result = await pipeline.run(
            scenario_filter=None,  # Process all scenarios
            symbols_filter=None,   # Process all symbols
            force_recalculate=True  # Force recalculation for testing
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("METRICS PIPELINE RESULTS")
        logger.info("=" * 80)
        
        logger.info(f"Execution time: {result['execution_time']}")
        logger.info(f"Total scenarios: {result['total_scenarios']}")
        logger.info(f"Completed: {result['completed']}")
        logger.info(f"Failed: {result['failed']}")
        logger.info(f"Skipped: {result['skipped']}")
        
        # Check output files
        metrics_dir = Path(result['storage_location'])
        if metrics_dir.exists():
            parquet_files = list(metrics_dir.glob('*.parquet'))
            logger.info(f"\nGenerated {len(parquet_files)} metric files")
            
            # Inspect first metric file
            if parquet_files:
                first_file = parquet_files[0]
                logger.info(f"\nInspecting: {first_file.name}")
                
                df_metrics = pd.read_parquet(first_file)
                logger.info(f"Shape: {df_metrics.shape}")
                logger.info(f"Date range: {df_metrics.index.min()} to {df_metrics.index.max()}")
                
                # Check for ML metrics
                ml_columns = [col for col in df_metrics.columns if 'ml_' in col.lower()]
                logger.info(f"\nML metric columns ({len(ml_columns)}):")
                for col in ml_columns[:10]:  # Show first 10
                    logger.info(f"  - {col}")
                
                # Check for technical indicators
                tech_columns = ['rsi', 'macd', 'bb_upper', 'bb_lower', 'atr', 'adx']
                available_tech = [col for col in tech_columns if col in df_metrics.columns]
                logger.info(f"\nTechnical indicators available: {available_tech}")
                
                # Show sample ML predictions
                if 'ml_ensemble_prediction' in df_metrics.columns:
                    logger.info("\nSample ML predictions:")
                    sample = df_metrics[['ml_ensemble_prediction', 'ml_signal', 'ml_signal_strength']].tail(5)
                    logger.info(sample)
                
                # Check anomaly scores
                if 'ml_anomaly_score' in df_metrics.columns:
                    logger.info("\nAnomaly detection:")
                    high_anomaly = df_metrics[df_metrics['ml_anomaly_score'] > 0.8]
                    logger.info(f"High anomaly periods: {len(high_anomaly)}")
                    if not high_anomaly.empty:
                        logger.info(f"Max anomaly score: {df_metrics['ml_anomaly_score'].max():.3f}")
        
        # Check for saved ML models
        models_dir = Path('output/ml_pipeline/models')
        if models_dir.exists():
            model_files = list(models_dir.glob('*.pkl'))
            logger.info(f"\n{len(model_files)} ML models saved")
            for model_file in model_files[:5]:  # Show first 5
                logger.info(f"  - {model_file.name}")
        
        # Summary statistics
        if result['results']:
            completed_results = [r for r in result['results'] if r['status'] == 'completed']
            if completed_results:
                logger.info("\n" + "=" * 80)
                logger.info("PROCESSING SUMMARY")
                logger.info("=" * 80)
                
                for res in completed_results[:3]:  # Show first 3
                    logger.info(f"\nScenario: {res['scenario']}")
                    logger.info(f"  Status: {res['status']}")
                    logger.info(f"  Metrics count: {res.get('metrics_count', 'N/A')}")
                    logger.info(f"  File: {res.get('file', 'N/A')}")
        
        logger.info("\n" + "=" * 80)
        logger.info("METRICS PIPELINE TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Metrics pipeline failed: {e}", exc_info=True)
        raise


async def verify_ml_metrics():
    """Verify ML metrics are being calculated correctly."""
    
    logger.info("\n" + "=" * 80)
    logger.info("VERIFYING ML METRICS")
    logger.info("=" * 80)
    
    metrics_dir = Path('data/metrics_test')
    if not metrics_dir.exists():
        metrics_dir = Path('data/metrics')
    
    if not metrics_dir.exists():
        logger.warning("No metrics directory found")
        return
    
    # Load a sample metrics file
    parquet_files = list(metrics_dir.glob('*.parquet'))
    if not parquet_files:
        logger.warning("No metric files found")
        return
    
    # Analyze ML metrics in detail
    for file in parquet_files[:2]:  # Check first 2 files
        logger.info(f"\nAnalyzing: {file.name}")
        df = pd.read_parquet(file)
        
        # Check ML predictions
        ml_pred_cols = [col for col in df.columns if '_prediction' in col]
        if ml_pred_cols:
            logger.info(f"Found {len(ml_pred_cols)} prediction columns")
            
            # Statistics for ensemble prediction
            if 'ml_ensemble_prediction' in df.columns:
                pred = df['ml_ensemble_prediction']
                logger.info(f"\nEnsemble Prediction Statistics:")
                logger.info(f"  Mean: {pred.mean():.6f}")
                logger.info(f"  Std: {pred.std():.6f}")
                logger.info(f"  Min: {pred.min():.6f}")
                logger.info(f"  Max: {pred.max():.6f}")
                logger.info(f"  % Positive: {(pred > 0).mean() * 100:.1f}%")
        
        # Check ML signals
        if 'ml_signal' in df.columns:
            signal_counts = df['ml_signal'].value_counts()
            logger.info(f"\nML Signal Distribution:")
            for signal, count in signal_counts.items():
                logger.info(f"  {signal}: {count} ({count/len(df)*100:.1f}%)")
        
        # Check ML performance metrics
        perf_metrics = ['ml_accuracy', 'ml_r2', 'ml_hit_rate', 'ml_profit_factor']
        available_perf = [col for col in perf_metrics if col in df.columns]
        if available_perf:
            logger.info(f"\nML Performance Metrics:")
            for metric in available_perf:
                value = df[metric].iloc[-1] if not df[metric].isna().all() else None
                if value is not None:
                    logger.info(f"  {metric}: {value:.4f}")
        
        # Check feature importance
        if 'ml_feature_importance' in df.columns:
            importance = df['ml_feature_importance'].iloc[-1]
            if importance and isinstance(importance, str):
                logger.info(f"\nFeature importance data available: {len(importance)} chars")


if __name__ == "__main__":
    logger.info("Starting metrics pipeline test...")
    
    # Run main test
    asyncio.run(test_metrics_pipeline())
    
    # Verify ML metrics
    asyncio.run(verify_ml_metrics())
    
    logger.info("\nTest complete!")