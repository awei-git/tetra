#!/usr/bin/env python3
"""Update all stocks for today's date"""

import asyncio
import sys
from pathlib import Path
from datetime import date
import psycopg2

sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.data_ingester import DataIngester
from src.utils.logging import logger

async def main():
    # Get all stock symbols from database
    conn = psycopg2.connect(
        host='localhost', 
        port=5432, 
        database='tetra', 
        user='tetra_user', 
        password='tetra_password'
    )
    cur = conn.cursor()
    cur.execute('''
        SELECT DISTINCT symbol 
        FROM market_data.ohlcv 
        WHERE symbol NOT LIKE '%-USD'
        ORDER BY symbol
    ''')
    all_symbols = [row[0] for row in cur.fetchall()]
    conn.close()
    
    # Today's date
    today = date(2025, 8, 6)
    
    logger.info(f"Updating {len(all_symbols)} stocks for {today}")
    
    # Process in batches to avoid overwhelming the system
    batch_size = 50
    total_stats = {
        "total_records": 0,
        "inserted": 0,
        "updated": 0,
        "errors": 0,
        "symbols_processed": 0
    }
    
    for i in range(0, len(all_symbols), batch_size):
        batch = all_symbols[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_symbols) + batch_size - 1)//batch_size}: {len(batch)} symbols")
        
        ingester = DataIngester(provider="polygon")
        stats = await ingester.ingest_ohlcv_batch(
            symbols=batch,
            from_date=today,
            to_date=today,
            timeframe="1d"
        )
        
        # Aggregate stats
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)
        
        logger.info(f"Batch complete: {stats}")
        
        # Small delay between batches
        if i + batch_size < len(all_symbols):
            await asyncio.sleep(1)
    
    logger.info(f"All updates complete: {total_stats}")
    
    # Quick verification
    from src.db.base import get_session
    from sqlalchemy import text
    
    async for session in get_session():
        result = await session.execute(
            text("""
                SELECT 
                    COUNT(*) as count, 
                    COUNT(DISTINCT symbol) as symbols,
                    MAX(timestamp)::date as latest
                FROM market_data.ohlcv
                WHERE timestamp::date = :today
            """),
            {"today": today}
        )
        row = result.fetchone()
        logger.info(f"Verification: {row.count} records for {row.symbols} symbols on {row.latest}")
        break

if __name__ == "__main__":
    asyncio.run(main())