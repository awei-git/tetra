#!/usr/bin/env python3
"""
Script to query and display PLTR (Palantir) daily price data for the past quarter
"""

import asyncio
import asyncpg
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import settings

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    
    # Simple replacement for tabulate
    def tabulate(data, headers=None, tablefmt=None):
        if headers:
            print(" | ".join(str(h).ljust(12) for h in headers))
            print("-" * (len(headers) * 13))
        for row in data:
            print(" | ".join(str(cell).ljust(12) for cell in row))


async def get_pltr_data():
    """Query PLTR daily price data for the past quarter"""
    
    # Calculate date range (approximately 90 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # Create database connection
    conn = await asyncpg.connect(
        host=settings.database_host,
        port=settings.database_port,
        database=settings.database_name,
        user=settings.database_user,
        password=settings.database_password
    )
    
    try:
        # Query for PLTR daily data
        query = """
        SELECT 
            timestamp::date as date,
            open,
            high,
            low,
            close,
            volume,
            vwap,
            source
        FROM market_data.ohlcv
        WHERE 
            symbol = 'PLTR' 
            AND timeframe = '1d'
            AND timestamp >= $1
            AND timestamp <= $2
        ORDER BY timestamp DESC
        """
        
        rows = await conn.fetch(query, start_date, end_date)
        
        if not rows:
            print(f"No data found for PLTR between {start_date.date()} and {end_date.date()}")
            return
        
        # Convert to list for tabulate
        data = []
        total_volume = 0
        min_price = float('inf')
        max_price = 0
        
        for row in rows:
            data.append([
                row['date'],
                f"${row['open']:.2f}",
                f"${row['high']:.2f}",
                f"${row['low']:.2f}",
                f"${row['close']:.2f}",
                f"{row['volume']:,}",
                f"${row['vwap']:.2f}" if row['vwap'] else "N/A",
                row['source']
            ])
            
            total_volume += row['volume']
            min_price = min(min_price, float(row['low']))
            max_price = max(max_price, float(row['high']))
        
        # Display results
        print(f"\n=== PLTR (Palantir) Daily Price Data ===")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Total trading days: {len(data)}\n")
        
        # Display table
        headers = ["Date", "Open", "High", "Low", "Close", "Volume", "VWAP", "Source"]
        print(tabulate(data, headers=headers, tablefmt="grid"))
        
        # Display summary statistics
        if rows:
            first_close = float(rows[-1]['close'])
            last_close = float(rows[0]['close'])
            price_change = last_close - first_close
            price_change_pct = (price_change / first_close) * 100
            
            print(f"\n=== Summary Statistics ===")
            print(f"Period Start Price: ${first_close:.2f}")
            print(f"Period End Price: ${last_close:.2f}")
            print(f"Price Change: ${price_change:.2f} ({price_change_pct:+.2f}%)")
            print(f"Period High: ${max_price:.2f}")
            print(f"Period Low: ${min_price:.2f}")
            print(f"Average Daily Volume: {total_volume // len(data):,}")
            print(f"Total Volume: {total_volume:,}")
        
        # Check for missing dates
        print(f"\n=== Data Coverage ===")
        
        # Get count by source
        source_query = """
        SELECT 
            source,
            COUNT(*) as count,
            MIN(timestamp)::date as first_date,
            MAX(timestamp)::date as last_date
        FROM market_data.ohlcv
        WHERE 
            symbol = 'PLTR' 
            AND timeframe = '1d'
            AND timestamp >= $1
        GROUP BY source
        ORDER BY count DESC
        """
        
        source_rows = await conn.fetch(source_query, start_date)
        
        if source_rows:
            source_data = []
            for row in source_rows:
                source_data.append([
                    row['source'],
                    row['count'],
                    row['first_date'],
                    row['last_date']
                ])
            
            print(tabulate(source_data, 
                         headers=["Source", "Count", "First Date", "Last Date"], 
                         tablefmt="grid"))
        
    finally:
        await conn.close()


async def check_all_pltr_data():
    """Check all available PLTR data in the database"""
    
    conn = await asyncpg.connect(
        host=settings.database_host,
        port=settings.database_port,
        database=settings.database_name,
        user=settings.database_user,
        password=settings.database_password
    )
    
    try:
        query = """
        SELECT 
            timeframe,
            COUNT(*) as record_count,
            MIN(timestamp)::date as first_date,
            MAX(timestamp)::date as last_date,
            array_agg(DISTINCT source) as sources
        FROM market_data.ohlcv
        WHERE symbol = 'PLTR'
        GROUP BY timeframe
        ORDER BY 
            CASE timeframe
                WHEN '1m' THEN 1
                WHEN '5m' THEN 2
                WHEN '15m' THEN 3
                WHEN '30m' THEN 4
                WHEN '1h' THEN 5
                WHEN '4h' THEN 6
                WHEN '1d' THEN 7
                WHEN '1w' THEN 8
                WHEN '1M' THEN 9
                ELSE 10
            END
        """
        
        rows = await conn.fetch(query)
        
        if not rows:
            print("\nNo PLTR data found in the database.")
            return
        
        print("\n=== All Available PLTR Data by Timeframe ===")
        
        data = []
        for row in rows:
            data.append([
                row['timeframe'],
                f"{row['record_count']:,}",
                row['first_date'],
                row['last_date'],
                ", ".join(row['sources'])
            ])
        
        headers = ["Timeframe", "Records", "First Date", "Last Date", "Sources"]
        print(tabulate(data, headers=headers, tablefmt="grid"))
        
    finally:
        await conn.close()


async def main():
    """Main function"""
    print("Checking PLTR (Palantir) price data in the database...")
    
    # First check what data is available
    await check_all_pltr_data()
    
    # Then show the quarterly data
    await get_pltr_data()


if __name__ == "__main__":
    asyncio.run(main())