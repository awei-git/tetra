#!/usr/bin/env python3
"""Test the data pipeline with the new ingestion module."""

import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.data_pipeline.pipeline import DataPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_data_pipeline():
    """Test the data pipeline execution."""
    
    logger.info("=" * 80)
    logger.info("TESTING DATA PIPELINE WITH NEW INGESTION MODULE")
    logger.info("=" * 80)
    
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Test configuration
    config = {
        "mode": "daily",  # Daily update mode
        "days_back": 2,   # Look back 2 days for gaps
        "fetch_intraday": False,  # Skip intraday for testing
        "providers": {
            "market_data": "polygon",  # or "yfinance" for free data
            "economic_data": "fred",
            "news": "newsapi"
        }
    }
    
    logger.info(f"Configuration: {config}")
    
    try:
        # Run the pipeline
        logger.info("\nStarting data pipeline...")
        result = await pipeline.run(**config)
        
        logger.info("\n" + "=" * 80)
        logger.info("DATA PIPELINE RESULTS")
        logger.info("=" * 80)
        
        # Display results
        logger.info(f"Status: {result.get('status')}")
        logger.info(f"Execution time: {result.get('execution_time')}")
        
        # Market data results
        if 'market_data' in result:
            market = result['market_data']
            logger.info(f"\nMarket Data:")
            logger.info(f"  Symbols processed: {market.get('symbols_processed', 0)}")
            logger.info(f"  Total records: {market.get('total_records', 0)}")
            logger.info(f"  Errors: {market.get('errors', 0)}")
        
        # Economic data results
        if 'economic_data' in result:
            econ = result['economic_data']
            logger.info(f"\nEconomic Data:")
            logger.info(f"  Indicators processed: {len(econ.get('success', {}))}")
            logger.info(f"  Total records: {econ.get('total_records', 0)}")
            logger.info(f"  Failed: {len(econ.get('failed', {}))}")
        
        # News results
        if 'news' in result:
            news = result['news']
            logger.info(f"\nNews Data:")
            logger.info(f"  Articles fetched: {news.get('articles', 0)}")
            logger.info(f"  Sentiments calculated: {news.get('sentiments', 0)}")
            logger.info(f"  Errors: {news.get('errors', 0)}")
        
        # Event results
        if 'events' in result:
            events = result['events']
            logger.info(f"\nEvent Data:")
            logger.info(f"  Total events: {events.get('total_records', 0)}")
            for event_type, count in events.get('success', {}).items():
                logger.info(f"    {event_type}: {count}")
        
        logger.info("\n" + "=" * 80)
        logger.info("DATA PIPELINE TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Data pipeline failed: {e}", exc_info=True)
        raise


async def test_ingestion_module_directly():
    """Test the ingestion module directly."""
    
    logger.info("\n" + "=" * 80)
    logger.info("TESTING INGESTION MODULE DIRECTLY")
    logger.info("=" * 80)
    
    from src.ingestion.data_ingester import DataIngester
    
    # Test with a small set of symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL"]
    # Use a date range that definitely has data (not future dates)
    end_date = date(2024, 12, 31)  # Use end of 2024
    start_date = end_date - timedelta(days=7)
    
    # Test with YFinance (free provider)
    ingester = DataIngester(provider="yfinance")
    
    logger.info(f"\nTesting OHLCV ingestion for {test_symbols}")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    try:
        result = await ingester.ingest_ohlcv_batch(
            symbols=test_symbols,
            from_date=start_date,
            to_date=end_date,
            timeframe="1d"
        )
        
        logger.info(f"\nOHLCV Results:")
        logger.info(f"  Symbols processed: {result['symbols_processed']}")
        logger.info(f"  Total records: {result['total_records']}")
        logger.info(f"  Success: {list(result['success'].keys())}")
        logger.info(f"  Failed: {list(result['failed'].keys())}")
        
    except Exception as e:
        logger.error(f"OHLCV ingestion failed: {e}")
    
    # Test economic indicators with FRED
    logger.info("\nTesting economic indicator ingestion")
    
    ingester_fred = DataIngester(provider="fred")
    indicators = ["GDP", "CPI", "UNEMPLOYMENT"]
    
    try:
        result = await ingester_fred.ingest_economic_indicators(
            indicators=indicators,
            from_date=start_date,
            to_date=end_date
        )
        
        logger.info(f"\nEconomic Indicator Results:")
        logger.info(f"  Total records: {result['total_records']}")
        logger.info(f"  Success: {list(result['success'].keys())}")
        logger.info(f"  Failed: {list(result['failed'].keys())}")
        
    except Exception as e:
        logger.error(f"Economic indicator ingestion failed: {e}")
    
    await ingester.close()
    await ingester_fred.close()


if __name__ == "__main__":
    logger.info("Starting data pipeline tests...")
    
    # Test ingestion module directly first
    asyncio.run(test_ingestion_module_directly())
    
    # Then test full pipeline
    asyncio.run(test_data_pipeline())
    
    logger.info("\nAll tests complete!")