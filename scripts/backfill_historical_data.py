#!/usr/bin/env python
"""
Backfill historical market data for 2005-2015.
This gives us 20 years of data for comprehensive backtesting.
"""

import asyncio
import logging
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.data_ingester import DataIngester
from src.definitions.market_universe import MarketUniverse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def backfill_historical_data():
    """Backfill historical data from 2005 to 2015."""
    
    # Initialize components
    ingester = DataIngester()
    universe = MarketUniverse()
    
    # Get all symbols
    symbols = universe.get_all_symbols()
    logger.info(f"Backfilling data for {len(symbols)} symbols")
    
    # Set date range for backfill
    start_date = datetime(2005, 1, 1)
    end_date = datetime(2015, 12, 31)
    
    logger.info(f"Backfilling from {start_date.date()} to {end_date.date()}")
    
    # Track progress
    total_records = 0
    failed_symbols = []
    
    # Process each symbol
    for i, symbol in enumerate(symbols, 1):
        try:
            logger.info(f"[{i}/{len(symbols)}] Processing {symbol}...")
            
            # Ingest data for this symbol
            result = await ingester.ingest_all_data_for_symbol(
                symbol=symbol,
                from_date=start_date.date(),
                to_date=end_date.date()
            )
            
            # Extract OHLCV records from result
            records = result.get('ohlcv', [])
            
            if records:
                total_records += len(records)
                logger.info(f"  ✓ Ingested {len(records)} records for {symbol}")
            else:
                logger.warning(f"  ⚠ No data available for {symbol} in this period")
                
        except Exception as e:
            logger.error(f"  ✗ Failed to ingest {symbol}: {e}")
            failed_symbols.append(symbol)
            continue
        
        # Small delay to avoid rate limiting
        if i % 10 == 0:
            await asyncio.sleep(1)
    
    # Summary
    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info(f"Total records ingested: {total_records:,}")
    logger.info(f"Successful symbols: {len(symbols) - len(failed_symbols)}")
    logger.info(f"Failed symbols: {len(failed_symbols)}")
    
    if failed_symbols:
        logger.info(f"Failed symbols list: {failed_symbols}")
    
    logger.info(f"Data now covers: 2005-2025 (20 years)")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(backfill_historical_data())