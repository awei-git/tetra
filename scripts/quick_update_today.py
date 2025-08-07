#!/usr/bin/env python3
"""Quick update for today's data only"""

import asyncio
import sys
from pathlib import Path
from datetime import date

sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.data_ingester import DataIngester
from src.utils.logging import logger

async def main():
    # Get stock symbols only (not crypto)
    stock_symbols = [
        'SPY', 'QQQ', 'IWM', 'DIA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META',
        'TSLA', 'BRK-B', 'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'MA'
    ]
    
    # Today's date
    today = date(2025, 8, 6)
    
    logger.info(f"Updating {len(stock_symbols)} stocks for {today}")
    
    ingester = DataIngester(provider="polygon")
    stats = await ingester.ingest_ohlcv_batch(
        symbols=stock_symbols,
        from_date=today,
        to_date=today,
        timeframe="1d"
    )
    
    logger.info(f"Update complete: {stats}")
    
    # Quick verification
    from src.db.base import get_session
    from sqlalchemy import text
    
    async for session in get_session():
        result = await session.execute(
            text("""
                SELECT COUNT(*) as count, MAX(timestamp)::date as latest
                FROM market_data.ohlcv
                WHERE timestamp::date = :today
            """),
            {"today": today}
        )
        row = result.fetchone()
        logger.info(f"Verification: {row.count} records for {row.latest}")
        break

if __name__ == "__main__":
    asyncio.run(main())