#!/usr/bin/env python3
"""
Check data coverage and gaps for all symbols
"""

import psycopg2
from datetime import datetime, timedelta, date
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.simulators.utils.trading_calendar import TradingCalendar

# Database connection
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="tetra",
    user="tetra_user",
    password="tetra_password"
)

cur = conn.cursor()

# Initialize trading calendar
calendar = TradingCalendar()
today = date.today()

print("=" * 80)
print("DATA COVERAGE SUMMARY")
print(f"Report Date: {today} ({today.strftime('%A')})")
print(f"Trading Day: {'Yes' if calendar.is_trading_day(today) else 'No'}")
print(f"Last Trading Day: {calendar.previous_trading_day(today)}")
print("=" * 80)

# Overall statistics
cur.execute("""
    SELECT 
        COUNT(DISTINCT symbol) as total_symbols,
        COUNT(*) as total_records,
        MIN(timestamp)::date as earliest_date,
        MAX(timestamp)::date as latest_date,
        COUNT(DISTINCT timestamp::date) as unique_days
    FROM market_data.ohlcv
    WHERE timeframe = '1d'
""")

stats = cur.fetchone()
print(f"\nOVERALL STATISTICS:")
print(f"  Total Symbols: {stats[0]:,}")
print(f"  Total Records: {stats[1]:,}")
print(f"  Date Range: {stats[2]} to {stats[3]}")
print(f"  Unique Days: {stats[4]:,}")

# Symbol coverage by latest date
cur.execute("""
    WITH symbol_latest AS (
        SELECT 
            symbol,
            MAX(timestamp)::date as latest_date,
            COUNT(*) as record_count
        FROM market_data.ohlcv
        WHERE timeframe = '1d'
        GROUP BY symbol
    )
    SELECT 
        latest_date,
        COUNT(*) as symbol_count,
        STRING_AGG(symbol, ', ' ORDER BY symbol) as symbols
    FROM symbol_latest
    GROUP BY latest_date
    ORDER BY latest_date DESC
    LIMIT 10
""")

print(f"\nSYMBOL COVERAGE BY LATEST DATE:")
print("-" * 60)
for row in cur.fetchall():
    date_val, count, symbols = row
    trading_days_ago = calendar.count_trading_days(date_val, calendar.previous_trading_day(today))
    symbols_preview = symbols[:50] + "..." if len(symbols) > 50 else symbols
    print(f"{date_val} ({trading_days_ago} trading days ago): {count} symbols")

# Check for major symbols
major_symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'BTC-USD', 'ETH-USD']
cur.execute("""
    SELECT 
        symbol,
        MIN(timestamp)::date as first_date,
        MAX(timestamp)::date as last_date,
        COUNT(*) as total_days,
        MAX(timestamp) as last_update
    FROM market_data.ohlcv
    WHERE symbol = ANY(%s) AND timeframe = '1Day'
    GROUP BY symbol
    ORDER BY symbol
""", (major_symbols,))

print(f"\nMAJOR SYMBOLS STATUS:")
print("-" * 80)
print(f"{'Symbol':<10} {'First Date':<12} {'Last Date':<12} {'Days':<8} {'Status':<20}")
print("-" * 80)

for row in cur.fetchall():
    symbol, first_date, last_date, total_days, last_update = row
    
    # Calculate trading days behind
    last_trading_day = calendar.previous_trading_day(today)
    if last_date >= last_trading_day:
        status = "✓ Current"
    else:
        days_behind = calendar.count_trading_days(last_date, last_trading_day) - 1
        status = f"⚠️  {days_behind} days behind"
    
    print(f"{symbol:<10} {str(first_date):<12} {str(last_date):<12} {total_days:<8} {status:<20}")

# Check which major symbols are missing
all_results = []
cur.execute("""
    SELECT 
        symbol,
        MIN(timestamp)::date as first_date,
        MAX(timestamp)::date as last_date,
        COUNT(*) as total_days,
        MAX(timestamp) as last_update
    FROM market_data.ohlcv
    WHERE symbol = ANY(%s) AND timeframe = '1d'
    GROUP BY symbol
    ORDER BY symbol
""", (major_symbols,))

for row in cur.fetchall():
    all_results.append(row[0])
    symbol, first_date, last_date, total_days, last_update = row
    
    # Calculate trading days behind
    last_trading_day = calendar.previous_trading_day(today)
    if last_date >= last_trading_day:
        status = "✓ Current"
    else:
        days_behind = calendar.count_trading_days(last_date, last_trading_day) - 1
        status = f"⚠️  {days_behind} days behind"
    
    print(f"{symbol:<10} {str(first_date):<12} {str(last_date):<12} {total_days:<8} {status:<20}")

# Missing major symbols
missing = [s for s in major_symbols if s not in all_results]
if missing:
    print(f"\nMissing symbols: {', '.join(missing)}")

# Check for gaps in SPY (as a benchmark)
print(f"\nDATA QUALITY CHECK (SPY):")
print("-" * 60)

# Get last 30 trading days
last_trading_day = calendar.previous_trading_day(today)
trading_days = []
current = last_trading_day
for _ in range(30):
    trading_days.append(current)
    current = calendar.previous_trading_day(current)

cur.execute("""
    SELECT timestamp::date as trading_date
    FROM market_data.ohlcv
    WHERE symbol = 'SPY' 
        AND timeframe = '1Day'
        AND timestamp::date >= %s
    ORDER BY timestamp DESC
""", (min(trading_days),))

spy_dates = {row[0] for row in cur.fetchall()}
missing_days = [d for d in trading_days if d not in spy_dates]

if missing_days:
    print(f"Missing {len(missing_days)} trading days:")
    for d in missing_days[:5]:
        print(f"  - {d}")
else:
    print("✓ No gaps in last 30 trading days")

# Economic data summary
print(f"\nECONOMIC DATA SUMMARY:")
print("-" * 60)

cur.execute("""
    SELECT 
        symbol,
        MIN(date)::date as first_date,
        MAX(date)::date as last_date,
        COUNT(*) as total_records
    FROM economic_data.economic_data
    WHERE symbol IN ('DFF', 'DGS10', 'DEXUSEU', 'CPIAUCSL', 'UNRATE', 'GDPC1')
    GROUP BY symbol
    ORDER BY symbol
""")

print(f"{'Indicator':<15} {'First Date':<12} {'Last Date':<12} {'Records':<10}")
print("-" * 60)
for row in cur.fetchall():
    symbol, first_date, last_date, count = row
    days_old = (today - last_date).days
    status = "✓" if days_old <= 31 else "⚠️ "
    print(f"{status} {symbol:<12} {str(first_date):<12} {str(last_date):<12} {count:<10}")

# News data summary
cur.execute("""
    SELECT 
        COUNT(*) as total_articles,
        MIN(published_at)::date as earliest,
        MAX(published_at)::date as latest,
        COUNT(DISTINCT source) as sources,
        SUM(CASE WHEN published_at::date >= CURRENT_DATE - 7 THEN 1 ELSE 0 END) as last_7_days
    FROM news.news_articles
""")

news = cur.fetchone()
if news[0] > 0:
    print(f"\nNEWS DATA SUMMARY:")
    print("-" * 60)
    print(f"Total Articles: {news[0]:,}")
    print(f"Date Range: {news[1]} to {news[2]}")
    print(f"Sources: {news[3]}")
    print(f"Last 7 Days: {news[4]:,} articles")

# Events summary
cur.execute("""
    SELECT 
        event_type,
        COUNT(*) as count,
        MIN(event_datetime)::date as first_event,
        MAX(event_datetime)::date as last_event
    FROM events.event_data
    WHERE event_datetime >= CURRENT_DATE - 30
    GROUP BY event_type
    ORDER BY count DESC
""")

events = cur.fetchall()
if events:
    print(f"\nRECENT EVENTS (Last 30 Days):")
    print("-" * 60)
    print(f"{'Event Type':<20} {'Count':<10} {'Date Range':<30}")
    print("-" * 60)
    for event_type, count, first_event, last_event in events:
        date_range = f"{first_event} to {last_event}"
        print(f"{event_type:<20} {count:<10} {date_range:<30}")

print("\n" + "=" * 80)

cur.close()
conn.close()