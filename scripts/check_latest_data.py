#!/usr/bin/env python3
import psycopg2
from datetime import datetime

# Database connection
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="tetra",
    user="tetra_user",
    password="tetra_password"
)

cur = conn.cursor()

# Check latest data
cur.execute("""
    SELECT symbol, MAX(timestamp)::date as latest_date, COUNT(*) as records
    FROM market_data.ohlcv
    WHERE symbol IN ('SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BTC-USD')
    GROUP BY symbol
    ORDER BY symbol
""")

print("Latest market data:")
for row in cur.fetchall():
    print(f"{row[0]}: {row[1]} ({row[2]} records)")

# Check August data
cur.execute("""
    SELECT COUNT(*) as count, 
           COUNT(DISTINCT symbol) as symbols,
           MAX(timestamp)::date as latest
    FROM market_data.ohlcv
    WHERE timestamp >= '2025-08-01'
""")

result = cur.fetchone()
print(f"\nAugust 2025 data: {result[0]} records for {result[1]} symbols")
print(f"Latest date with data: {result[2]}")

# Check today's update progress
cur.execute("""
    SELECT COUNT(DISTINCT symbol) as updated_symbols
    FROM market_data.ohlcv
    WHERE timestamp = '2025-08-01'
""")

result = cur.fetchone()
print(f"\nSymbols with Aug 1 data: {result[0]}")

cur.close()
conn.close()