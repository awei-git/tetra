"""Tests for data coverage and integrity"""

import pytest
from datetime import datetime, date, timedelta
from sqlalchemy import text
from src.db.base import get_session
from src.data_definitions.market_universe import MarketUniverse
from src.data_definitions.economic_indicators import EconomicIndicators


class TestDataCoverage:
    """Test data coverage and integrity in the database"""
    
    @pytest.mark.asyncio
    async def test_market_data_coverage(self):
        """Test market data coverage"""
        async for session in get_session():
            # Check overall coverage
            result = await session.execute(text("""
                SELECT 
                    COUNT(DISTINCT symbol) as unique_symbols,
                    COUNT(*) as total_records,
                    MIN(timestamp)::date as earliest_date,
                    MAX(timestamp)::date as latest_date,
                    COUNT(DISTINCT DATE(timestamp)) as unique_days
                FROM market_data.ohlcv
            """))
            row = result.fetchone()
            
            print(f"\n=== Market Data Coverage ===")
            print(f"Total symbols: {row.unique_symbols}")
            print(f"Total records: {row.total_records:,}")
            print(f"Date range: {row.earliest_date} to {row.latest_date}")
            print(f"Unique days: {row.unique_days}")
            
            # Check we have some data
            assert row.total_records > 0, "No market data found"
            assert row.unique_symbols > 0, "No symbols found"
            
            # Check high priority symbols
            high_priority = MarketUniverse.get_high_priority_symbols()
            placeholders = ','.join([f':sym_{i}' for i in range(len(high_priority))])
            params = {f'sym_{i}': sym for i, sym in enumerate(high_priority)}
            
            result = await session.execute(text(f"""
                SELECT symbol, COUNT(*) as records, 
                       MIN(timestamp)::date as first_date,
                       MAX(timestamp)::date as last_date
                FROM market_data.ohlcv
                WHERE symbol IN ({placeholders})
                GROUP BY symbol
            """), params)
            
            covered_symbols = {}
            for row in result:
                covered_symbols[row.symbol] = {
                    'records': row.records,
                    'first_date': row.first_date,
                    'last_date': row.last_date
                }
            
            print(f"\nHigh Priority Symbol Coverage ({len(covered_symbols)}/{len(high_priority)}):")
            for symbol in high_priority[:10]:  # Show first 10
                if symbol in covered_symbols:
                    data = covered_symbols[symbol]
                    days = (data['last_date'] - data['first_date']).days
                    print(f"  {symbol}: {data['records']} records, {days} days")
                else:
                    print(f"  {symbol}: NO DATA")
            
            # Assert at least 50% of high priority symbols have data
            coverage_pct = len(covered_symbols) / len(high_priority) * 100
            assert coverage_pct >= 50, f"Only {coverage_pct:.1f}% of high priority symbols have data"
    
    @pytest.mark.asyncio
    async def test_data_freshness(self):
        """Test that we have recent data"""
        async for session in get_session():
            # Check for recent market data (excluding weekends)
            today = date.today()
            lookback = 7  # Look back 7 days
            
            # Find the most recent business day
            recent_biz_day = today
            while recent_biz_day.weekday() in [5, 6]:  # Saturday = 5, Sunday = 6
                recent_biz_day -= timedelta(days=1)
            
            result = await session.execute(text("""
                SELECT COUNT(DISTINCT symbol) as symbols_with_recent_data
                FROM market_data.ohlcv
                WHERE timestamp::date >= :cutoff_date
            """), {"cutoff_date": recent_biz_day - timedelta(days=lookback)})
            
            row = result.fetchone()
            print(f"\n=== Data Freshness ===")
            print(f"Symbols with data in last {lookback} days: {row.symbols_with_recent_data}")
            
            # We should have at least some symbols with recent data
            assert row.symbols_with_recent_data > 0, "No recent market data found"
    
    @pytest.mark.asyncio
    async def test_data_gaps(self):
        """Test for data gaps in recent history"""
        async for session in get_session():
            # Check for gaps in the last 30 days for high priority symbols
            end_date = date.today()
            start_date = end_date - timedelta(days=30)
            
            # Get a sample high priority symbol
            symbol = MarketUniverse.get_high_priority_symbols()[0]
            
            # Using raw SQL to avoid parameter binding issues with asyncpg
            query = f"""
                WITH expected_dates AS (
                    SELECT generate_series(
                        '{start_date}'::date,
                        '{end_date}'::date,
                        '1 day'::interval
                    )::date AS date
                ),
                actual_dates AS (
                    SELECT DISTINCT timestamp::date as date
                    FROM market_data.ohlcv
                    WHERE symbol = '{symbol}'
                        AND timestamp::date BETWEEN '{start_date}'::date AND '{end_date}'::date
                )
                SELECT COUNT(*) as missing_days
                FROM expected_dates e
                LEFT JOIN actual_dates a ON e.date = a.date
                WHERE a.date IS NULL
                    AND EXTRACT(dow FROM e.date) NOT IN (0, 6)  -- Exclude weekends
            """
            result = await session.execute(text(query))
            
            row = result.fetchone()
            print(f"\n=== Data Gaps for {symbol} ===")
            print(f"Missing business days in last 30 days: {row.missing_days}")
            
            # Allow some gaps but not too many (e.g., holidays)
            # Note: We're missing recent days if we haven't run backfill
            if row.missing_days > 10:
                print(f"Warning: {row.missing_days} missing days for {symbol} (need to run backfill)")
    
    @pytest.mark.asyncio
    async def test_economic_data_coverage(self):
        """Test economic data coverage"""
        async for session in get_session():
            result = await session.execute(text("""
                SELECT 
                    COUNT(DISTINCT symbol) as indicators,
                    COUNT(*) as total_records,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date
                FROM economic_data.economic_data
            """))
            row = result.fetchone()
            
            print(f"\n=== Economic Data Coverage ===")
            print(f"Total indicators: {row.indicators}")
            print(f"Total records: {row.total_records:,}")
            print(f"Date range: {row.earliest_date} to {row.latest_date}")
            
            # Check specific important indicators
            key_indicators = ["GDP", "UNRATE", "CPIAUCSL", "DFF"]
            placeholders = ','.join([f':ind_{i}' for i in range(len(key_indicators))])
            params = {f'ind_{i}': ind for i, ind in enumerate(key_indicators)}
            
            result = await session.execute(text(f"""
                SELECT symbol, COUNT(*) as records,
                       MIN(date) as first_date,
                       MAX(date) as latest_date
                FROM economic_data.economic_data
                WHERE symbol IN ({placeholders})
                GROUP BY symbol
            """), params)
            
            print("\nKey Economic Indicators:")
            for row in result:
                print(f"  {row.symbol}: {row.records} records, {row.first_date} to {row.latest_date}")
    
    @pytest.mark.asyncio
    async def test_news_data_coverage(self):
        """Test news data coverage"""
        async for session in get_session():
            # Check news articles
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as total_articles,
                    COUNT(DISTINCT source) as unique_sources,
                    MIN(published_at)::date as earliest,
                    MAX(published_at)::date as latest
                FROM news.news_articles
            """))
            row = result.fetchone()
            
            print(f"\n=== News Data Coverage ===")
            print(f"Total articles: {row.total_articles:,}")
            print(f"Unique sources: {row.unique_sources}")
            print(f"Date range: {row.earliest} to {row.latest}")
            
            # Check sentiment coverage
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as articles_with_sentiment,
                    AVG(polarity) as avg_polarity,
                    AVG(relevance_score) as avg_relevance
                FROM news.news_sentiments
            """))
            row = result.fetchone()
            
            if row.articles_with_sentiment:
                print(f"Articles with sentiment: {row.articles_with_sentiment:,}")
                print(f"Average polarity: {row.avg_polarity:.3f}")
                print(f"Average relevance: {row.avg_relevance:.3f}")
    
    @pytest.mark.asyncio
    async def test_event_data_coverage(self):
        """Test event data coverage"""
        async for session in get_session():
            result = await session.execute(text("""
                SELECT 
                    event_type,
                    COUNT(*) as count,
                    MIN(event_datetime)::date as earliest,
                    MAX(event_datetime)::date as latest
                FROM events.event_data
                GROUP BY event_type
                ORDER BY count DESC
            """))
            
            print(f"\n=== Event Data Coverage ===")
            total_events = 0
            for row in result:
                print(f"{row.event_type}: {row.count} events ({row.earliest} to {row.latest})")
                total_events += row.count
            
            print(f"Total events: {total_events:,}")
    
    @pytest.mark.asyncio
    async def test_data_integrity(self):
        """Test data integrity - no duplicates, valid values"""
        async for session in get_session():
            # Check for duplicate market data
            result = await session.execute(text("""
                SELECT symbol, timestamp, timeframe, COUNT(*) as count
                FROM market_data.ohlcv
                GROUP BY symbol, timestamp, timeframe
                HAVING COUNT(*) > 1
                LIMIT 10
            """))
            
            duplicates = result.fetchall()
            print(f"\n=== Data Integrity ===")
            print(f"Duplicate market data records: {len(duplicates)}")
            
            assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate market data records"
            
            # Check for invalid prices (negative or zero)
            result = await session.execute(text("""
                SELECT COUNT(*) as invalid_prices
                FROM market_data.ohlcv
                WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
            """))
            
            row = result.fetchone()
            print(f"Invalid price records: {row.invalid_prices}")
            
            assert row.invalid_prices == 0, f"Found {row.invalid_prices} records with invalid prices"
            
            # Check high/low consistency
            result = await session.execute(text("""
                SELECT COUNT(*) as inconsistent
                FROM market_data.ohlcv
                WHERE low > high OR low > open OR low > close
                   OR high < open OR high < close
            """))
            
            row = result.fetchone()
            print(f"Inconsistent OHLC records: {row.inconsistent}")
            
            assert row.inconsistent == 0, f"Found {row.inconsistent} records with inconsistent OHLC values"
    
    @pytest.mark.asyncio
    async def test_data_completeness_summary(self):
        """Generate a summary report of data completeness"""
        async for session in get_session():
            print("\n" + "="*60)
            print("DATA COMPLETENESS SUMMARY REPORT")
            print("="*60)
            
            # Get expected symbols
            expected_symbols = MarketUniverse.get_all_symbols()
            
            # Get actual symbols with data
            result = await session.execute(text("""
                SELECT DISTINCT symbol FROM market_data.ohlcv
            """))
            actual_symbols = {row.symbol for row in result}
            
            coverage_pct = len(actual_symbols) / len(expected_symbols) * 100
            print(f"\nSymbol Coverage: {len(actual_symbols)}/{len(expected_symbols)} ({coverage_pct:.1f}%)")
            
            # Missing symbols
            missing = set(expected_symbols) - actual_symbols
            if missing:
                print(f"\nMissing symbols ({len(missing)}):")
                for symbol in sorted(list(missing))[:20]:
                    print(f"  - {symbol}")
                if len(missing) > 20:
                    print(f"  ... and {len(missing) - 20} more")
            
            # Data recency
            result = await session.execute(text("""
                SELECT 
                    symbol,
                    MAX(timestamp)::date as last_update,
                    CURRENT_DATE - MAX(timestamp)::date as days_behind
                FROM market_data.ohlcv
                GROUP BY symbol
                HAVING CURRENT_DATE - MAX(timestamp)::date > 2
                ORDER BY days_behind DESC
                LIMIT 20
            """))
            
            stale_symbols = result.fetchall()
            if stale_symbols:
                print(f"\nStale symbols (>2 days old):")
                for row in stale_symbols:
                    print(f"  {row.symbol}: {row.days_behind} days behind (last: {row.last_update})")
            
            print("\n" + "="*60)