#!/usr/bin/env python3
"""
Quick script to update crypto data using yfinance
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.data_ingester import DataIngester
from src.utils.logging import logger


async def main():
    """Update crypto symbols using yfinance"""
    # Crypto symbols to update
    crypto_symbols = [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
        'SOL-USD', 'DOGE-USD', 'DOT-USD', 'AVAX-USD', 'MATIC-USD'
    ]
    
    logger.info(f"Updating {len(crypto_symbols)} crypto symbols...")
    
    # Use yfinance for crypto
    ingester = DataIngester(provider="yfinance")
    
    # Get last 30 days to ensure we have all data
    start_date = datetime.now().date() - timedelta(days=30)
    end_date = datetime.now().date()
    
    try:
        result = await ingester.ingest_ohlcv_batch(
            symbols=crypto_symbols,
            from_date=start_date,
            to_date=end_date,
            timeframe="1d"
        )
        
        records = result.get("total_records", 0)
        logger.info(f"Updated crypto: {records} records")
        
        # Check individual results
        if "symbols_processed" in result:
            logger.info(f"Symbols processed: {result['symbols_processed']}")
        
    except Exception as e:
        logger.error(f"Failed to update crypto: {e}")
    
    # Quick check
    logger.info("\nChecking crypto data status...")
    import subprocess
    subprocess.run([
        sys.executable, "-c",
        """
import psycopg2
conn = psycopg2.connect(host='localhost', port=5432, database='tetra', user='tetra_user', password='tetra_password')
cur = conn.cursor()
cur.execute('''
    SELECT symbol, MAX(timestamp)::date as last_date
    FROM market_data.ohlcv
    WHERE symbol LIKE '%-USD' AND timeframe = '1d'
    GROUP BY symbol
    ORDER BY symbol
''')
print('\\nCrypto Symbol Status:')
for symbol, last_date in cur.fetchall():
    days_ago = (datetime.now().date() - last_date).days
    status = '✓' if days_ago <= 1 else f'⚠️  {days_ago}d'
    print(f'{symbol:12} {last_date} {status}')
"""
    ])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Update failed: {e}")