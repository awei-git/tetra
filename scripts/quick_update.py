#!/usr/bin/env python
"""
Quick market data update script
Fetches only the latest data (1 day) for all tracked symbols
Much faster than full daily update
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.data_ingester import DataIngester
from src.data_definitions.market_universe import MarketUniverse
from src.simulators.utils.trading_calendar import TradingCalendar
from src.utils.logging import logger


async def run_quick_update():
    """Run quick market data update - only fetch latest day"""
    logger.info("=== Starting Quick Market Data Update ===")
    logger.info(f"Time: {datetime.now()}")
    
    # Initialize trading calendar
    calendar = TradingCalendar()
    today = datetime.now().date()
    
    # For daily bars, we need complete day data
    # If it's before market close (4 PM ET), get previous day
    # Otherwise, we can get today's data
    current_time = datetime.now()
    market_close_hour = 16  # 4 PM ET
    
    if calendar.is_trading_day(today) and current_time.hour >= market_close_hour + 1:
        # After market close, we can get today's complete data
        end_date = today
        logger.info(f"Market closed. Fetching today's complete data for: {end_date}")
    else:
        # Get previous trading day for complete data
        end_date = calendar.previous_trading_day(today)
        logger.info(f"Fetching previous trading day data for: {end_date}")
    
    # For quick update, also try to get more recent data if available
    # Try multiple days to get the most recent available data
    dates_to_try = [end_date]
    
    # If we're using previous day, also try today in case data is available
    if end_date != today and calendar.is_trading_day(today):
        dates_to_try.insert(0, today)
    
    start_date = end_date  # Default to the safe date
    
    # Get symbols to update
    symbols = MarketUniverse.get_all_symbols()
    logger.info(f"Updating {len(symbols)} symbols")
    
    # Initialize data ingester
    ingester = DataIngester(provider="polygon")
    
    # Process all symbols in one batch (faster with unlimited API)
    batch_size = 50  # Still use batches for database efficiency
    total_records = 0
    failed_symbols = []
    
    # Try to get the most recent data available
    successful_date = None
    
    for date_to_try in dates_to_try:
        logger.info(f"Attempting to fetch data for date: {date_to_try}")
        temp_records = 0
        temp_failed = []
        
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(symbols) + batch_size - 1) // batch_size
            
            if i == 0:  # Only log for first batch
                logger.info(f"Processing batch {batch_num}/{total_batches}: {len(batch_symbols)} symbols for {date_to_try}")
            
            try:
                result = await ingester.ingest_ohlcv_batch(
                    symbols=batch_symbols,
                    from_date=date_to_try,
                    to_date=date_to_try,
                    timeframe="1d"
                )
                
                records = result.get("total_records", 0)
                errors = result.get("errors", [])
                
                temp_records += records
                
                if errors:
                    temp_failed.extend(errors)
                
            except Exception as e:
                logger.error(f"Batch failed: {e}")
                temp_failed.extend(batch_symbols)
        
        # If we got data for most symbols, use this date
        if temp_records > len(symbols) * 0.8:  # 80% success rate
            total_records = temp_records
            failed_symbols = temp_failed
            successful_date = date_to_try
            logger.info(f"Successfully fetched data for {date_to_try}: {temp_records} records")
            break
        else:
            logger.warning(f"Insufficient data for {date_to_try}: only {temp_records} records")
    
    if successful_date is None:
        # Fallback to the default date if nothing worked
        logger.warning("Could not fetch recent data, using fallback date")
        successful_date = end_date
    
    # Summary
    logger.info("=== Quick Update Complete ===")
    logger.info(f"Total records: {total_records}")
    logger.info(f"Successful symbols: {len(symbols) - len(failed_symbols)}")
    logger.info(f"Failed symbols: {len(failed_symbols)}")
    
    if failed_symbols:
        logger.warning(f"Failed symbols: {failed_symbols[:10]}...")
    
    return {
        "status": "success" if len(failed_symbols) == 0 else "partial",
        "records_processed": total_records,
        "symbols_updated": len(symbols) - len(failed_symbols),
        "failed_symbols": failed_symbols,
        "date": str(successful_date) if successful_date else str(end_date),
        "dates_tried": [str(d) for d in dates_to_try]
    }


def main():
    """Main entry point"""
    return asyncio.run(run_quick_update())


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result["status"] == "success" else 1)