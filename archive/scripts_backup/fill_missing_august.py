#!/usr/bin/env python
"""
Fill missing August data using YFinance
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.data_ingester import DataIngester
from src.utils.logging import logger
from src.db.base import get_session
from sqlalchemy import text


async def get_missing_symbols():
    """Get symbols missing August data"""
    async for db in get_session():
        query = """
        SELECT symbol 
        FROM (
            SELECT symbol, MAX(timestamp)::date as last_date 
            FROM market_data.ohlcv 
            GROUP BY symbol
        ) t 
        WHERE last_date < '2025-08-01'
        ORDER BY symbol
        """
        result = await db.execute(text(query))
        return [row.symbol for row in result.fetchall()]


async def fill_missing_data():
    """Fill missing August data using YFinance"""
    logger.info("=== Filling Missing August Data with YFinance ===")
    
    # Get symbols missing data
    missing_symbols = await get_missing_symbols()
    logger.info(f"Found {len(missing_symbols)} symbols missing August data")
    
    if not missing_symbols:
        logger.info("No missing symbols found!")
        return
    
    # Use YFinance which has better rate limits
    ingester = DataIngester(provider="yfinance")
    
    # Fetch last 15 days to ensure we get all missing data
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=15)
    
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Process in batches
    batch_size = 20  # YFinance can handle larger batches
    total_records = 0
    failed_symbols = []
    
    for i in range(0, len(missing_symbols), batch_size):
        batch = missing_symbols[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(missing_symbols) + batch_size - 1) // batch_size
        
        logger.info(f"\nBatch {batch_num}/{total_batches}: {batch}")
        
        try:
            # Fetch daily data
            result = await ingester.ingest_ohlcv_batch(
                symbols=batch,
                from_date=start_date,
                to_date=end_date,
                timeframe="1d"
            )
            
            records = result.get("total_records", 0)
            errors = result.get("errors", [])
            
            total_records += records
            
            if errors:
                logger.warning(f"Batch had {len(errors)} errors: {errors}")
                failed_symbols.extend(errors)
                
            logger.info(f"Batch complete: {records} records inserted")
            
            # Small delay between batches
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Batch failed: {e}")
            failed_symbols.extend(batch)
    
    # Summary
    logger.info("\n=== Fill Complete ===")
    logger.info(f"Total records: {total_records}")
    logger.info(f"Successful symbols: {len(missing_symbols) - len(failed_symbols)}")
    logger.info(f"Failed symbols: {len(failed_symbols)}")
    
    if failed_symbols:
        logger.warning(f"Failed: {failed_symbols}")
    
    return {
        "total_records": total_records,
        "symbols_processed": len(missing_symbols) - len(failed_symbols),
        "symbols_failed": len(failed_symbols)
    }


if __name__ == "__main__":
    try:
        result = asyncio.run(fill_missing_data())
        logger.info(f"Result: {result}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fill failed: {e}")
        sys.exit(1)