#!/usr/bin/env python3
import asyncio
import asyncpg

async def check_volume_and_gaps():
    conn = await asyncpg.connect(
        host='localhost',
        port=5432,
        user='tetra_user',
        password='tetra_password',
        database='tetra'
    )
    
    # First, check volume scaling around July 22, 2022
    query1 = '''
        SELECT 
            symbol,
            DATE(timestamp) as date,
            volume
        FROM market_data.ohlcv
        WHERE symbol IN ('AAPL', 'META', 'AMZN')
          AND timestamp BETWEEN '2022-07-18' AND '2022-07-26'
          AND timeframe = '1d'
        ORDER BY symbol, timestamp
    '''
    
    print('=== Volume data around July 22, 2022 ===')
    results1 = await conn.fetch(query1)
    current_symbol = None
    for row in results1:
        if current_symbol != row['symbol']:
            print(f'\n{row["symbol"]}:')
            current_symbol = row['symbol']
        print(f'  {row["date"]}: {row["volume"]:,} shares')
    
    # Check for data gaps after July 29
    query2 = '''
        SELECT 
            symbol,
            MIN(DATE(timestamp)) as first_date,
            MAX(DATE(timestamp)) as last_date,
            COUNT(DISTINCT DATE(timestamp)) as trading_days,
            MAX(DATE(timestamp)) - MIN(DATE(timestamp)) as date_span
        FROM market_data.ohlcv
        WHERE timeframe = '1d'
          AND timestamp >= '2022-07-01'
        GROUP BY symbol
        ORDER BY symbol
        LIMIT 10
    '''
    
    print('\n\n=== Data coverage from July 2022 onwards ===')
    results2 = await conn.fetch(query2)
    for row in results2:
        print(f'{row["symbol"]}: {row["first_date"]} to {row["last_date"]} ({row["trading_days"]} days)')
    
    # Check for specific gaps around July 29
    query3 = '''
        WITH date_series AS (
            SELECT generate_series('2022-07-25'::date, '2022-08-05'::date, '1 day'::interval)::date as date
        ),
        actual_data AS (
            SELECT DISTINCT DATE(timestamp) as date
            FROM market_data.ohlcv
            WHERE symbol = 'AAPL'
              AND timeframe = '1d'
              AND timestamp BETWEEN '2022-07-25' AND '2022-08-05'
        )
        SELECT 
            ds.date,
            CASE WHEN ad.date IS NULL THEN 'MISSING' ELSE 'Present' END as status
        FROM date_series ds
        LEFT JOIN actual_data ad ON ds.date = ad.date
        ORDER BY ds.date
    '''
    
    print('\n\n=== AAPL data presence/absence around July 29, 2022 ===')
    results3 = await conn.fetch(query3)
    for row in results3:
        status_marker = ' <-- Gap here!' if row["status"] == 'MISSING' else ''
        print(f'{row["date"]}: {row["status"]}{status_marker}')
    
    # Check if volumes before July 22 are in millions
    query4 = '''
        SELECT 
            AVG(CASE WHEN timestamp < '2022-07-22' THEN volume END) as avg_before,
            AVG(CASE WHEN timestamp >= '2022-07-22' THEN volume END) as avg_after,
            AVG(CASE WHEN timestamp < '2022-07-22' THEN volume END) / 
            NULLIF(AVG(CASE WHEN timestamp >= '2022-07-22' THEN volume END), 0) as ratio
        FROM market_data.ohlcv
        WHERE symbol = 'AAPL'
          AND timeframe = '1d'
          AND timestamp BETWEEN '2022-01-01' AND '2023-01-01'
    '''
    
    print('\n\n=== AAPL volume scaling analysis ===')
    result4 = await conn.fetchrow(query4)
    print(f'Average volume before July 22: {result4["avg_before"]:,.0f}')
    print(f'Average volume after July 22: {result4["avg_after"]:,.0f}')
    print(f'Ratio (before/after): {result4["ratio"]:,.1f}x')
    
    if result4['ratio'] > 100000:
        print('\n*** WARNING: Volumes before July 22 appear to be in millions!')
        print('*** Example: A volume of 75 before July 22 likely means 75 million shares')
    
    # Check other high-volume stocks
    query5 = '''
        SELECT 
            symbol,
            AVG(CASE WHEN timestamp < '2022-07-22' THEN volume END) / 
            NULLIF(AVG(CASE WHEN timestamp >= '2022-07-22' THEN volume END), 0) as ratio
        FROM market_data.ohlcv
        WHERE timeframe = '1d'
          AND timestamp BETWEEN '2022-01-01' AND '2023-01-01'
        GROUP BY symbol
        HAVING AVG(CASE WHEN timestamp < '2022-07-22' THEN volume END) IS NOT NULL
           AND AVG(CASE WHEN timestamp >= '2022-07-22' THEN volume END) IS NOT NULL
        ORDER BY ratio DESC
        LIMIT 20
    '''
    
    print('\n\n=== Top 20 stocks by volume ratio (before/after July 22) ===')
    results5 = await conn.fetch(query5)
    for row in results5:
        if row['ratio'] and row['ratio'] > 100000:
            print(f'{row["symbol"]}: {row["ratio"]:,.0f}x ratio - LIKELY IN MILLIONS BEFORE JULY 22')
        elif row['ratio']:
            print(f'{row["symbol"]}: {row["ratio"]:.1f}x ratio')
    
    await conn.close()

if __name__ == "__main__":
    asyncio.run(check_volume_and_gaps())