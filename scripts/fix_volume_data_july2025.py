#!/usr/bin/env python3
import asyncio
import asyncpg
from datetime import datetime, timedelta
import subprocess
import sys

async def fix_volume_data():
    conn = await asyncpg.connect(
        host='localhost', port=5432, user='tetra_user', 
        password='tetra_password', database='tetra'
    )
    
    try:
        # First, let's see what symbols have data after July 22, 2025
        print("Checking affected symbols...")
        check_query = '''
            SELECT DISTINCT symbol, COUNT(*) as records
            FROM market_data.ohlcv
            WHERE timestamp >= '2025-07-22'
              AND timeframe = '1d'
            GROUP BY symbol
            ORDER BY symbol
        '''
        
        affected_symbols = await conn.fetch(check_query)
        print(f"Found {len(affected_symbols)} symbols with data on/after July 22, 2025")
        
        # Show a sample
        if affected_symbols:
            print("\nSample of affected symbols:")
            for row in affected_symbols[:10]:
                print(f"  {row['symbol']}: {row['records']} records")
        
        # Delete the incorrect data
        print("\nDeleting incorrect volume data from July 22, 2025 onwards...")
        delete_query = '''
            DELETE FROM market_data.ohlcv
            WHERE timestamp >= '2025-07-22'
              AND timeframe = '1d'
        '''
        
        result = await conn.execute(delete_query)
        print(f"Deleted {result.split()[-1]} records")
        
        # Get list of symbols to refetch
        symbols_to_update = [row['symbol'] for row in affected_symbols]
        
        print(f"\nNeed to refetch data for {len(symbols_to_update)} symbols")
        print("Symbols:", ', '.join(symbols_to_update[:20]), '...' if len(symbols_to_update) > 20 else '')
        
    finally:
        await conn.close()
    
    # Now run the data update script
    print("\n" + "="*60)
    print("Running data update to fetch correct data...")
    print("="*60 + "\n")
    
    # Run the existing update script
    try:
        # First check if the update script exists
        update_script = "../scripts/update_market_data.py"
        import os
        if not os.path.exists(update_script):
            print("Looking for update script in scripts directory...")
            # Try to find the correct script
            possible_paths = [
                "scripts/update_market_data.py",
                "../scripts/run_data_updates.py",
                "scripts/run_data_updates.py",
                "../scripts/update_ohlcv.py",
                "scripts/update_ohlcv.py"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    update_script = path
                    print(f"Found update script at: {path}")
                    break
            else:
                print("ERROR: Could not find data update script!")
                print("Please run the data update manually")
                return
        
        # Run the update
        print(f"Running: python {update_script}")
        subprocess.run([sys.executable, update_script], check=True)
        
        print("\nData update completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running update script: {e}")
        print("You may need to run the data update manually")
    except Exception as e:
        print(f"Unexpected error: {e}")

async def verify_fix():
    """Verify the data has been fixed"""
    conn = await asyncpg.connect(
        host='localhost', port=5432, user='tetra_user', 
        password='tetra_password', database='tetra'
    )
    
    try:
        print("\n" + "="*60)
        print("Verifying the fix...")
        print("="*60)
        
        query = '''
            SELECT 
                symbol,
                DATE(timestamp) as date,
                volume,
                close
            FROM market_data.ohlcv
            WHERE symbol IN ('AAPL', 'META', 'AMZN')
              AND timestamp BETWEEN '2025-07-20' AND '2025-07-31'
              AND timeframe = '1d'
            ORDER BY symbol, timestamp
            LIMIT 15
        '''
        
        results = await conn.fetch(query)
        
        if not results:
            print("No data found for July 22-31, 2025. Run data update to fetch new data.")
        else:
            print("\nSample of data around July 22, 2025:")
            current_symbol = None
            for row in results:
                if current_symbol != row['symbol']:
                    if current_symbol:
                        print()
                    print(f"{row['symbol']}:")
                    current_symbol = row['symbol']
                print(f"  {row['date']}: {row['volume']:>15,} shares @ ${row['close']:.2f}")
    
    finally:
        await conn.close()

if __name__ == "__main__":
    print("Volume Data Fix Script for July 22, 2025 onwards")
    print("=" * 50)
    
    # Run the fix
    asyncio.run(fix_volume_data())
    
    # Verify the results
    asyncio.run(verify_fix())