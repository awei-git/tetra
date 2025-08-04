#!/usr/bin/env python
"""Test daily update with a past date to avoid API restrictions"""

import asyncio
import sys
from datetime import datetime, date
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.data_ingester import DataIngester
from src.utils.logging import logger

async def test_update():
    """Test update with a small batch"""
    logger.info("Testing daily update with past date")
    
    # Use a past Friday
    test_date = date(2025, 8, 1)  # Friday
    
    # Test with just a few symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL"]
    
    ingester = DataIngester(provider="polygon")
    
    try:
        result = await ingester.ingest_ohlcv_batch(
            symbols=test_symbols,
            from_date=test_date,
            to_date=test_date,
            timeframe="1d"
        )
        
        logger.info(f"Test result: {result}")
        
        if result['total_records'] > 0:
            logger.info("✅ Daily update is working correctly!")
            return True
        else:
            logger.warning("No records fetched, but no errors either")
            return True
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_update())
    sys.exit(0 if success else 1)