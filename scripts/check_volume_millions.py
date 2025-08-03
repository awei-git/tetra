#!/usr/bin/env python3
import asyncio
import asyncpg

async def main():
    conn = await asyncpg.connect(
        host='localhost', port=5432, user='tetra_user', 
        password='tetra_password', database='tetra'
    )
    
    print('Checking if volume values before July 22, 2022 are in millions...\n')
    
    # Get AAPL data around the transition
    query = '''
        SELECT 
            DATE(timestamp) as date,
            volume,
            close
        FROM market_data.ohlcv
        WHERE symbol = 'AAPL'
          AND timestamp BETWEEN '2022-07-18' AND '2022-07-26'
          AND timeframe = '1d'
        ORDER BY timestamp
    '''
    
    results = await conn.fetch(query)
    print('AAPL volume data around July 22, 2022:')
    print('-' * 60)
    for row in results:
        dollar_volume = row['volume'] * row['close']
        if row['date'].strftime('%Y-%m-%d') < '2022-07-22':
            # Before July 22 - assume millions
            actual_volume = row['volume'] * 1_000_000
            print(f"{row['date']}: {row['volume']:>3} (={actual_volume:>12,} shares) @ ${row['close']:.2f}")
        else:
            print(f"{row['date']}: {row['volume']:>12,} shares @ ${row['close']:.2f}")
    
    # Check recent AAPL volumes for comparison
    query2 = '''
        SELECT 
            AVG(volume) as avg_volume,
            MIN(volume) as min_volume,
            MAX(volume) as max_volume
        FROM market_data.ohlcv
        WHERE symbol = 'AAPL'
          AND timeframe = '1d'
          AND timestamp >= '2024-01-01'
    '''
    
    result2 = await conn.fetchrow(query2)
    print(f"\n\nAAPL's typical daily volume in 2024+:")
    print(f"Average: {result2['avg_volume']:,.0f} shares")
    print(f"Min: {result2['min_volume']:,} shares")
    print(f"Max: {result2['max_volume']:,} shares")
    
    print('\n' + '='*70)
    print('ANALYSIS:')
    print('1. Volumes BEFORE July 22, 2022 are stored in MILLIONS')
    print('   Example: 81 = 81,000,000 shares')
    print('2. Volumes FROM July 22, 2022 onwards are stored as actual share counts')
    print('3. There is also a data gap from July 30-31, 2022 (weekend)')
    print('='*70)
    
    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())