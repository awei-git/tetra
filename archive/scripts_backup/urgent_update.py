#!/usr/bin/env python
"""
Urgent update for high priority symbols
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.data_ingester import DataIngester
from src.utils.logging import logger


# High priority symbols to update immediately
URGENT_SYMBOLS = [
    'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    'VTI', 'VOO', 'IWM', 'DIA', 'ARKK', 'ARKQ', 'XLK', 'XLF', 'XLV', 'XLE',
    'GLD', 'SLV', 'USO', 'UNG', 'TLT', 'HYG', 'LQD', 'AGG',
    'BTC-USD', 'ETH-USD'
]


async def run_urgent_update():
    """Run urgent update for high priority symbols"""
    logger.info("=== Starting Urgent Update for High Priority Symbols ===")
    logger.info(f"Time: {datetime.now()}")
    logger.info(f"Updating {len(URGENT_SYMBOLS)} high priority symbols")
    
    # Initialize data ingester with polygon (primary provider)
    ingester = DataIngester(provider="polygon")
    
    # Fetch last 30 days to ensure we have complete recent data
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Process in smaller batches for urgent symbols
    batch_size = 10
    total_records = 0
    failed_symbols = []
    
    for i in range(0, len(URGENT_SYMBOLS), batch_size):
        batch = URGENT_SYMBOLS[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(URGENT_SYMBOLS) + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches}: {batch}")
        
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
    
    # Try yfinance for failed symbols
    if failed_symbols:
        logger.info(f"Attempting backup fetch with yfinance for {len(failed_symbols)} symbols")
        yf_ingester = DataIngester(provider="yfinance")
        
        try:
            result = await yf_ingester.ingest_ohlcv_batch(
                symbols=failed_symbols,
                from_date=start_date,
                to_date=end_date,
                timeframe="1d"
            )
            
            backup_records = result.get("total_records", 0)
            logger.info(f"Backup fetch complete: {backup_records} records")
            total_records += backup_records
            
        except Exception as e:
            logger.error(f"Backup fetch failed: {e}")
    
    # Summary
    logger.info("=== Urgent Update Complete ===")
    logger.info(f"Total records: {total_records}")
    logger.info(f"Successful symbols: {len(URGENT_SYMBOLS) - len(failed_symbols)}")
    logger.info(f"Failed symbols: {len(failed_symbols)}")
    
    return {
        "total_records": total_records,
        "symbols_processed": len(URGENT_SYMBOLS) - len(failed_symbols),
        "symbols_failed": len(failed_symbols),
        "date": end_date.isoformat()
    }


if __name__ == "__main__":
    try:
        result = asyncio.run(run_urgent_update())
        logger.info(f"Update result: {result}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Urgent update failed: {e}")
        sys.exit(1)