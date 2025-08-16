#!/usr/bin/env python
"""
Backfill historical market data using yfinance for data before 2021.
This supplements Polygon data with historical data from Yahoo Finance.
"""

import asyncio
import logging
from datetime import datetime
import os
import sys
import yfinance as yf
import pandas as pd
import asyncpg

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.definitions.market_universe import MarketUniverse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def backfill_symbol_data(symbol: str, start_date: str, end_date: str, conn: asyncpg.Connection):
    """Backfill data for a single symbol using yfinance."""
    try:
        # Download data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty:
            logger.warning(f"No data available for {symbol}")
            return 0
        
        # Prepare data for insertion
        records = []
        for date, row in df.iterrows():
            # Convert volume to millions as per Polygon format
            volume_millions = row['Volume'] / 1_000_000 if row['Volume'] else 0
            
            records.append((
                symbol,
                date.to_pydatetime(),  # timestamp
                row['Open'],           # open
                row['High'],           # high
                row['Low'],            # low
                row['Close'],          # close
                volume_millions,       # volume in millions
                0,                     # vwap (not available from yfinance)
                0                      # transactions (not available)
            ))
        
        # Insert into database
        if records:
            inserted = await conn.executemany("""
                INSERT INTO market_data.ohlcv 
                (symbol, timestamp, open, high, low, close, volume, vwap, transactions)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (symbol, timestamp) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                RETURNING symbol
            """, records)
            
            return len(records)
        return 0
        
    except Exception as e:
        logger.error(f"Failed to backfill {symbol}: {e}")
        return 0


async def backfill_historical_data():
    """Backfill historical data from 2005 to 2020 using yfinance."""
    
    # Initialize components
    universe = MarketUniverse()
    
    # Get all symbols
    symbols = universe.get_all_symbols()
    
    # Remove crypto symbols as yfinance uses different format
    symbols = [s for s in symbols if not s.endswith('-USD')]
    
    logger.info(f"Backfilling data for {len(symbols)} symbols")
    
    # Set date range for backfill (2005-2020, before Polygon data)
    start_date = '2005-01-01'
    end_date = '2020-12-31'
    
    logger.info(f"Backfilling from {start_date} to {end_date}")
    
    # Connect to database
    conn = await asyncpg.connect('postgresql://tetra_user:tetra_password@localhost:5432/tetra')
    
    try:
        # Track progress
        total_records = 0
        successful_symbols = []
        failed_symbols = []
        
        # Process symbols in batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            
            for symbol in batch:
                logger.info(f"[{i + batch.index(symbol) + 1}/{len(symbols)}] Processing {symbol}...")
                
                records = await backfill_symbol_data(symbol, start_date, end_date, conn)
                
                if records > 0:
                    total_records += records
                    successful_symbols.append(symbol)
                    logger.info(f"  ✓ Inserted {records} records for {symbol}")
                else:
                    failed_symbols.append(symbol)
                    logger.warning(f"  ⚠ No data for {symbol}")
            
            # Small delay between batches
            await asyncio.sleep(2)
        
        # Summary
        logger.info("=" * 60)
        logger.info("BACKFILL COMPLETE")
        logger.info(f"Total records inserted: {total_records:,}")
        logger.info(f"Successful symbols: {len(successful_symbols)}")
        logger.info(f"Failed symbols: {len(failed_symbols)}")
        
        if failed_symbols and len(failed_symbols) < 20:
            logger.info(f"Failed symbols: {failed_symbols}")
        
        # Get updated coverage
        result = await conn.fetchrow("""
            SELECT 
                MIN(timestamp) as earliest,
                MAX(timestamp) as latest,
                COUNT(*) as total_records
            FROM market_data.ohlcv
        """)
        
        logger.info(f"Database now contains:")
        logger.info(f"  Date range: {result['earliest']} to {result['latest']}")
        logger.info(f"  Total records: {result['total_records']:,}")
        logger.info("=" * 60)
        
    finally:
        await conn.close()


if __name__ == "__main__":
    # Install yfinance if not present
    try:
        import yfinance
    except ImportError:
        import subprocess
        logger.info("Installing yfinance...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
        import yfinance
    
    asyncio.run(backfill_historical_data())