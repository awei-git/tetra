#!/usr/bin/env python
"""
Update economic data for August 2025
"""

import asyncio
import sys
from datetime import datetime, timedelta, date
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.clients.economic_data_client import EconomicDataClient
from src.data_definitions.economic_indicators import EconomicIndicators
from src.utils.logging import logger
from src.db.base import get_session
from src.db.models import EconomicDataModel
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert


async def save_economic_data(data_points, session):
    """Save economic data to database"""
    if not data_points:
        return 0
    
    # Prepare values for insert
    values = []
    for point in data_points:
        values.append({
            "symbol": point.symbol,
            "date": point.date,
            "value": point.value,
            "source": point.source,
            "created_at": datetime.utcnow(),
        })
    
    # Use PostgreSQL upsert
    stmt = insert(EconomicDataModel).values(values)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_econ_symbol_date",
        set_={
            "value": stmt.excluded.value,
            "updated_at": datetime.utcnow(),
        }
    )
    
    await session.execute(stmt)
    await session.commit()
    
    return len(values)


async def update_economic_data():
    """Update economic data for August 2025"""
    logger.info("=== Updating Economic Data for August 2025 ===")
    
    # Get all indicators
    all_indicators = EconomicIndicators.get_all_indicators()
    
    # Set date range for August
    # FRED data is sometimes delayed, so we'll fetch from July 15 to ensure we get all August data
    start_date = date(2025, 7, 15)
    end_date = date.today()
    
    logger.info(f"Fetching data from {start_date} to {end_date}")
    logger.info(f"Total indicators to update: {len(all_indicators)}")
    
    # Initialize client
    async with EconomicDataClient(provider="fred") as client:
        total_records = 0
        failed_indicators = []
        
        # Process in batches to avoid overwhelming the API
        batch_size = 10
        
        for i in range(0, len(all_indicators), batch_size):
            batch = all_indicators[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(all_indicators) + batch_size - 1) // batch_size
            
            logger.info(f"\nBatch {batch_num}/{total_batches}")
            
            for symbol, name, frequency in batch:
                try:
                    logger.info(f"Fetching {symbol} ({name})")
                    
                    # Fetch data
                    data_points = await client.get_indicator_data(
                        symbol=symbol,
                        from_date=start_date,
                        to_date=end_date
                    )
                    
                    if data_points:
                        # Filter for August data
                        august_data = [
                            point for point in data_points 
                            if point.date.month == 8 and point.date.year == 2025
                        ]
                        
                        if august_data:
                            # Save to database
                            async for session in get_session():
                                records_saved = await save_economic_data(august_data, session)
                                total_records += records_saved
                                logger.info(f"  Saved {records_saved} August data points for {symbol}")
                                break
                        else:
                            # Also save July data if available (for indicators that haven't updated yet)
                            july_data = [
                                point for point in data_points 
                                if point.date.month == 7 and point.date.year == 2025
                            ]
                            if july_data:
                                async for session in get_session():
                                    records_saved = await save_economic_data(july_data, session)
                                    total_records += records_saved
                                    logger.info(f"  Saved {records_saved} July data points for {symbol} (August not yet available)")
                                    break
                            else:
                                logger.warning(f"  No recent data found for {symbol}")
                    else:
                        logger.warning(f"  No data returned for {symbol}")
                    
                    # Small delay between requests
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    logger.error(f"  Error fetching {symbol}: {e}")
                    failed_indicators.append((symbol, name))
            
            # Delay between batches
            await asyncio.sleep(1)
    
    # Summary
    logger.info("\n=== Update Complete ===")
    logger.info(f"Total records saved: {total_records}")
    logger.info(f"Successful indicators: {len(all_indicators) - len(failed_indicators)}")
    logger.info(f"Failed indicators: {len(failed_indicators)}")
    
    if failed_indicators:
        logger.warning("Failed indicators:")
        for symbol, name in failed_indicators[:10]:
            logger.warning(f"  {symbol}: {name}")
        if len(failed_indicators) > 10:
            logger.warning(f"  ... and {len(failed_indicators) - 10} more")
    
    return {
        "total_records": total_records,
        "indicators_processed": len(all_indicators) - len(failed_indicators),
        "indicators_failed": len(failed_indicators)
    }


if __name__ == "__main__":
    try:
        result = asyncio.run(update_economic_data())
        logger.info(f"Result: {result}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Economic update failed: {e}")
        sys.exit(1)