#!/usr/bin/env python3
"""Quick data coverage check"""

import psycopg2
from datetime import datetime, date
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
calendar = TradingCalendar()
today = date.today()
last_trading_day = calendar.previous_trading_day(today)

print(f"Data Coverage Report - {today}")
print(f"Last Trading Day: {last_trading_day}")
print("=" * 60)

# Check overall stats
cur.execute("""
    SELECT 
        COUNT(DISTINCT symbol) as symbols,
        COUNT(*) as records,
        MAX(timestamp)::date as latest
    FROM market_data.ohlcv
    WHERE timeframe = '1d'
""")
symbols, records, latest = cur.fetchone()
print(f"\nMarket Data: {symbols} symbols, {records:,} records, latest: {latest}")

# Check symbols by latest date
cur.execute("""
    WITH latest_dates AS (
        SELECT symbol, MAX(timestamp)::date as last_date
        FROM market_data.ohlcv
        WHERE timeframe = '1d'
        GROUP BY symbol
    )
    SELECT 
        CASE 
            WHEN last_date = %s THEN 'Current'
            WHEN last_date = %s - INTERVAL '1 day' THEN '1 day behind'
            WHEN last_date >= %s - INTERVAL '7 days' THEN 'Within 7 days'
            ELSE 'Stale'
        END as status,
        COUNT(*) as count
    FROM latest_dates
    GROUP BY status
    ORDER BY status
""", (last_trading_day, last_trading_day, last_trading_day))

print("\nSymbol Status:")
for status, count in cur.fetchall():
    print(f"  {status}: {count} symbols")

# Check specific symbols
symbols_to_check = ['SPY', 'QQQ', 'AAPL', 'BTC-USD', 'ETH-USD']
cur.execute("""
    SELECT symbol, MAX(timestamp)::date as last_date
    FROM market_data.ohlcv
    WHERE symbol = ANY(%s) AND timeframe = '1d'
    GROUP BY symbol
    ORDER BY symbol
""", (symbols_to_check,))

print(f"\nKey Symbols:")
for symbol, last_date in cur.fetchall():
    if last_date >= last_trading_day:
        status = "✓"
    else:
        days_behind = (last_trading_day - last_date).days
        status = f"⚠️  {days_behind}d"
    print(f"  {symbol:8} {last_date} {status}")

# Economic data
cur.execute("""
    WITH latest_data AS (
        SELECT symbol, MAX(date) as max_date
        FROM economic_data.economic_data
        GROUP BY symbol
    )
    SELECT 
        CASE 
            WHEN max_date >= CURRENT_DATE - 31 THEN 'Current'
            WHEN max_date >= CURRENT_DATE - 90 THEN 'Recent' 
            ELSE 'Stale'
        END as status,
        COUNT(*) as count
    FROM latest_data
    GROUP BY status
""")

print("\nEconomic Data:")
for status, count in cur.fetchall():
    print(f"  {status}: {count} indicators")

# News
cur.execute("""
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN published_at::date >= CURRENT_DATE - 7 THEN 1 END) as last_week
    FROM news.news_articles
""")
total, last_week = cur.fetchone()
print(f"\nNews: {total} articles ({last_week} in last 7 days)")

# Check for recent updates
print("\n\nDAILY UPDATE SUMMARY:")
print("=" * 60)

# Get the most recent data updates
cur.execute("""
    WITH recent_updates AS (
        SELECT 
            'Market Data' as data_type,
            symbol,
            MAX(created_at) as last_update,
            COUNT(*) FILTER (WHERE created_at::date = CURRENT_DATE) as today_count,
            COUNT(*) FILTER (WHERE created_at::date = CURRENT_DATE - 1) as yesterday_count
        FROM market_data.ohlcv
        WHERE created_at >= CURRENT_DATE - 2
        GROUP BY symbol
        
        UNION ALL
        
        SELECT 
            'Economic Data' as data_type,
            symbol,
            MAX(created_at) as last_update,
            COUNT(*) FILTER (WHERE created_at::date = CURRENT_DATE) as today_count,
            COUNT(*) FILTER (WHERE created_at::date = CURRENT_DATE - 1) as yesterday_count
        FROM economic_data.economic_data
        WHERE created_at >= CURRENT_DATE - 2
        GROUP BY symbol
    )
    SELECT 
        data_type,
        COUNT(DISTINCT symbol) as symbols_updated,
        SUM(today_count) as records_today,
        SUM(yesterday_count) as records_yesterday,
        MAX(last_update) as most_recent_update
    FROM recent_updates
    GROUP BY data_type
    ORDER BY data_type
""")

print(f"Last Update Activity:")
for data_type, symbols, today_records, yesterday_records, last_update in cur.fetchall():
    time_ago = datetime.now() - last_update.replace(tzinfo=None)
    hours_ago = time_ago.total_seconds() / 3600
    
    if hours_ago < 1:
        time_str = f"{int(time_ago.total_seconds() / 60)} minutes ago"
    elif hours_ago < 24:
        time_str = f"{int(hours_ago)} hours ago"
    else:
        time_str = f"{int(hours_ago / 24)} days ago"
    
    print(f"\n{data_type}:")
    print(f"  Last Update: {last_update.strftime('%Y-%m-%d %H:%M')} ({time_str})")
    print(f"  Symbols Updated: {symbols}")
    print(f"  Records Today: {today_records}")
    print(f"  Records Yesterday: {yesterday_records}")

# Check launchd logs if available
import os
log_path = "/Users/angwei/Repos/tetra/logs/launchd_pipeline_out.log"
if os.path.exists(log_path):
    # Get last 5 lines of the log
    with open(log_path, 'r') as f:
        lines = f.readlines()
        last_run_info = None
        for line in reversed(lines[-20:]):
            if "Daily pipeline completed" in line or "Daily update completed" in line:
                last_run_info = line.strip()
                break
        
        if last_run_info:
            print(f"\nLast Pipeline Run:")
            print(f"  {last_run_info}")

conn.close()