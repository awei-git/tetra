#!/usr/bin/env python
"""
Simple daily market data update script
Fetches the latest data for all tracked symbols
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.data_ingester import DataIngester
from src.data_definitions.market_universe import MarketUniverse
from src.simulators.utils.trading_calendar import TradingCalendar
from src.utils.logging import logger


async def run_daily_update():
    """Run daily market data update with gap detection and filling"""
    logger.info("=== Starting Daily Market Data Update ===")
    logger.info(f"Time: {datetime.now()}")
    
    # Initialize trading calendar
    calendar = TradingCalendar()
    today = datetime.now().date()
    
    # Check if today is a trading day
    if not calendar.is_trading_day(today):
        logger.info(f"Today ({today}) is not a trading day. Checking for previous trading day data...")
        end_date = calendar.previous_trading_day(today)
    else:
        # For free tier Polygon, we can't fetch current day data
        # So always fetch previous trading day
        end_date = calendar.previous_trading_day(today)
        logger.info(f"Using previous trading day ({end_date}) to avoid API restrictions")
    
    logger.info(f"Target date for data: {end_date}")
    
    # Get symbols to update
    symbols = MarketUniverse.get_all_symbols()
    logger.info(f"Updating {len(symbols)} symbols")
    
    # Initialize data ingester with polygon (primary provider)
    ingester = DataIngester(provider="polygon")
    
    # First, identify symbols with gaps or missing recent data
    from src.db.base import get_session
    from sqlalchemy import text
    
    async for db in get_session():
        # Find the latest date for each symbol
        query = """
        SELECT symbol, MAX(timestamp)::date as last_date
        FROM market_data.ohlcv
        WHERE symbol = ANY(:symbols)
        GROUP BY symbol
        """
        result = await db.execute(text(query), {"symbols": symbols})
        symbol_dates = {row.symbol: row.last_date for row in result.fetchall()}
    
    logger.info(f"Found {len(symbol_dates)} symbols with existing data")
    
    # Process in batches
    batch_size = 50
    total_records = 0
    failed_symbols = []
    
    # Group symbols by how much data they need
    symbols_by_range = {}
    for symbol in symbols:
        if symbol in symbol_dates:
            last_date = symbol_dates[symbol]
            # Count trading days, not calendar days
            trading_days_behind = calendar.count_trading_days(last_date, end_date) - 1
            
            # Group by trading days behind
            if trading_days_behind <= 0:
                range_key = "current"
                start_date = end_date
            elif trading_days_behind <= 3:
                range_key = "recent"
                start_date = calendar.next_trading_day(last_date)
            elif trading_days_behind <= 10:
                range_key = "week_old"
                start_date = calendar.next_trading_day(last_date)
            elif trading_days_behind <= 30:
                range_key = "month_old"
                # Get start date that's ~30 trading days back
                start_date = last_date
                for _ in range(min(30, trading_days_behind)):
                    start_date = calendar.previous_trading_day(start_date)
            else:
                range_key = "very_old"
                # Get start date that's ~90 trading days back
                start_date = end_date
                for _ in range(90):
                    start_date = calendar.previous_trading_day(start_date)
        else:
            # No data for this symbol - fetch 1 year of trading days
            range_key = "missing"
            start_date = end_date
            # Go back ~252 trading days (1 year)
            for _ in range(252):
                start_date = calendar.previous_trading_day(start_date)
        
        if range_key not in symbols_by_range:
            symbols_by_range[range_key] = []
        symbols_by_range[range_key].append((symbol, start_date))
    
    # Log the grouping
    for range_key, syms in symbols_by_range.items():
        logger.info(f"{range_key}: {len(syms)} symbols")
    
    # Process each group
    for range_key, symbol_list in symbols_by_range.items():
        if not symbol_list:
            continue
            
        logger.info(f"\n=== Processing {range_key} symbols ({len(symbol_list)} total) ===")
        
        # Process this group in batches
        for i in range(0, len(symbol_list), batch_size):
            batch_symbols_data = symbol_list[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(symbol_list) + batch_size - 1) // batch_size
            
            # Create batches by date range
            date_batches = {}
            for symbol, start_date in batch_symbols_data:
                date_key = start_date.isoformat()
                if date_key not in date_batches:
                    date_batches[date_key] = []
                date_batches[date_key].append(symbol)
            
            # Process each date batch
            for start_date_str, batch_symbols in date_batches.items():
                start_date = datetime.fromisoformat(start_date_str).date()
                
                logger.info(f"Batch {batch_num}/{total_batches}: {len(batch_symbols)} symbols from {start_date} to {end_date}")
                
                try:
                    # Fetch daily data
                    result = await ingester.ingest_ohlcv_batch(
                        symbols=batch_symbols,
                        from_date=start_date,
                        to_date=end_date,
                        timeframe="1d"
                    )
                    
                    records = result.get("total_records", 0)
                    errors = result.get("errors", [])
                    
                    total_records += records
                    
                    if errors:
                        logger.warning(f"Batch had {len(errors)} errors")
                        failed_symbols.extend(errors)
                        
                    logger.info(f"Batch complete: {records} records inserted")
                    
                    # Small delay between batches to avoid rate limits
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Batch failed: {e}")
                    failed_symbols.extend(batch_symbols)
    
    # Summary
    logger.info("=== Daily Update Complete ===")
    logger.info(f"Total records: {total_records}")
    logger.info(f"Successful symbols: {len(symbols) - len(failed_symbols)}")
    logger.info(f"Failed symbols: {len(failed_symbols)}")
    
    if failed_symbols:
        logger.warning(f"Failed symbols: {failed_symbols[:10]}...")
    
    # Try yfinance as backup for failed symbols
    if failed_symbols and len(failed_symbols) < 20:
        logger.info("Attempting backup fetch with yfinance...")
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
    
    logger.info(f"Final total records: {total_records}")
    
    # Update economic data (can run on any day including weekends)
    logger.info("\n=== Updating Economic Data ===")
    try:
        from src.pipelines.data_pipeline.steps.economic_data import EconomicDataStep
        from src.pipelines.base import PipelineContext
        
        econ_step = EconomicDataStep()
        context = PipelineContext(data={
            "mode": "daily",
            "start_date": datetime.now().date() - timedelta(days=7),
            "end_date": datetime.now().date()
        })
        
        econ_result = await econ_step.execute(context)
        econ_records = econ_result.get("total_records", 0)
        logger.info(f"Economic data update complete: {econ_records} new records")
        
    except Exception as e:
        logger.error(f"Economic data update failed: {e}")
    
    return {
        "total_records": total_records,
        "symbols_processed": len(symbols) - len(failed_symbols),
        "symbols_failed": len(failed_symbols),
        "date": end_date.isoformat(),
        "is_trading_day": calendar.is_trading_day(today)
    }


if __name__ == "__main__":
    try:
        result = asyncio.run(run_daily_update())
        logger.info(f"Update result: {result}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Daily update failed: {e}")
        sys.exit(1)