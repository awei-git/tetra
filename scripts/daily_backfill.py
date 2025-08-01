#!/usr/bin/env python3
"""
Daily backfill script for market and economic data
- Market data: Uses yfinance for historical data, Polygon for recent/intraday
- Economic data: Uses FRED for all economic indicators
- Handles updates intelligently to avoid duplicates
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Dict, Set, Optional

sys.path.append(str(Path(__file__).parent.parent))

from src.data_definitions.market_universe import MarketUniverse as Universe
from src.data_definitions.economic_indicators import EconomicIndicators
from src.data_definitions.event_calendar import EventCalendar
from src.ingestion.data_ingester import DataIngester
from src.clients.economic_data_client import EconomicDataClient
from src.clients.event_data_client import EventDataClient
from src.clients.news_sentiment_client import NewsSentimentClient
from src.db.base import get_session
from src.db.models import (
    OHLCVModel, EconomicDataModel, EventDataModel, EconomicEventModel, 
    EarningsEventModel, NewsArticleModel, NewsSentimentModel
)
from src.models.event_data import EventType, EventStatus
import uuid
from src.utils.logging import logger
from sqlalchemy import select, func
from config import settings


class SmartBackfiller:
    """Smart backfilling that uses multiple data sources efficiently"""
    
    def __init__(self):
        self.yfinance_ingester = DataIngester(provider="yfinance")
        self.polygon_ingester = DataIngester(provider="polygon")
        self.economic_client = None  # Lazy initialization
        self.event_client = None  # Lazy initialization
        self.news_client = None  # Lazy initialization
        
    async def get_last_update_dates(self, symbols: List[str], timeframe: str = "1d") -> Dict[str, datetime]:
        """Get the last update date for each symbol"""
        last_dates = {}
        
        async for session in get_session():
            for symbol in symbols:
                query = select(func.max(OHLCVModel.timestamp)).where(
                    OHLCVModel.symbol == symbol,
                    OHLCVModel.timeframe == timeframe
                )
                result = await session.execute(query)
                last_date = result.scalar()
                if last_date:
                    last_dates[symbol] = last_date
        
        return last_dates
    
    async def check_data_coverage(self) -> Dict[str, any]:
        """Check current data coverage in the database"""
        async for session in get_session():
            # Get summary statistics
            query = select(
                OHLCVModel.symbol,
                func.count(OHLCVModel.id).label('record_count'),
                func.min(OHLCVModel.timestamp).label('earliest_date'),
                func.max(OHLCVModel.timestamp).label('latest_date')
            ).group_by(OHLCVModel.symbol).order_by(OHLCVModel.symbol)
            
            result = await session.execute(query)
            data = result.all()
            
            # Get total records
            total_query = select(func.count(OHLCVModel.id))
            total_result = await session.execute(total_query)
            total_records = total_result.scalar()
            
            coverage = {
                'total_symbols': len(data),
                'total_records': total_records,
                'symbols': {}
            }
            
            for row in data:
                days = (row.latest_date - row.earliest_date).days if row.earliest_date and row.latest_date else 0
                coverage['symbols'][row.symbol] = {
                    'records': row.record_count,
                    'earliest': row.earliest_date,
                    'latest': row.latest_date,
                    'days': days
                }
            
            return coverage
    
    async def backfill_symbol(self, symbol: str, timeframe: str = "1d", days_back: int = 3650) -> Dict[str, int]:
        """Backfill a single symbol intelligently"""
        stats = {"yfinance": 0, "polygon": 0, "errors": 0}
        
        # Get last update date for this symbol
        last_dates = await self.get_last_update_dates([symbol], timeframe)
        last_date = last_dates.get(symbol)
        
        # Calculate date range
        end_date = date.today()
        if last_date:
            # Update from last date
            start_date = (last_date + timedelta(days=1)).date()
            if start_date > end_date:
                logger.info(f"{symbol}: Already up to date")
                return stats
        else:
            # No data, start from days_back
            start_date = end_date - timedelta(days=days_back)
        
        logger.info(f"{symbol}: Fetching {timeframe} data from {start_date} to {end_date}")
        
        # Use Polygon for recent data (last 2 years) for non-crypto
        use_polygon = (
            not Universe.is_crypto(symbol) and 
            self._has_polygon_key() and
            (end_date - start_date).days <= 730
        )
        
        if use_polygon:
            try:
                async with self.polygon_ingester as ingester:
                    result = await ingester.fetch_ohlcv(
                        symbols=[symbol],
                        from_date=start_date,
                        to_date=end_date,
                        timeframe=timeframe
                    )
                    stats["polygon"] += result.get("inserted", 0) + result.get("updated", 0)
                    logger.info(f"{symbol}: Fetched {result.get('inserted', 0)} records from Polygon")
            except Exception as e:
                logger.warning(f"{symbol}: Polygon error: {e}, falling back to yfinance")
                # Fall back to yfinance
                try:
                    async with self.yfinance_ingester as ingester:
                        result = await ingester.fetch_ohlcv(
                            symbols=[symbol],
                            from_date=start_date,
                            to_date=end_date,
                            timeframe=timeframe
                        )
                        stats["yfinance"] += result.get("inserted", 0) + result.get("updated", 0)
                except Exception as e2:
                    logger.error(f"{symbol}: yfinance fallback error: {e2}")
                    stats["errors"] += 1
        else:
            # Use yfinance for historical data or crypto
            try:
                async with self.yfinance_ingester as ingester:
                    # Check if symbol is available
                    available = await ingester.check_available_symbols([symbol])
                    if symbol not in available:
                        logger.warning(f"{symbol}: Not available on yfinance")
                        stats["errors"] += 1
                        return stats
                    
                    result = await ingester.fetch_ohlcv(
                        symbols=[symbol],
                        from_date=start_date,
                        to_date=end_date,
                        timeframe=timeframe
                    )
                    stats["yfinance"] += result.get("inserted", 0) + result.get("updated", 0)
                    logger.info(f"{symbol}: Fetched {result.get('inserted', 0)} records from yfinance")
            except Exception as e:
                logger.error(f"{symbol}: Error during yfinance fetch: {e}")
                stats["errors"] += 1
                
                # Try polygon as last resort for non-crypto
                if not Universe.is_crypto(symbol) and self._has_polygon_key():
                    try:
                        async with self.polygon_ingester as ingester:
                            result = await ingester.fetch_ohlcv(
                                symbols=[symbol],
                                from_date=max(start_date, end_date - timedelta(days=730)),
                                to_date=end_date,
                                timeframe=timeframe
                            )
                            stats["polygon"] += result.get("inserted", 0) + result.get("updated", 0)
                    except Exception as e2:
                        logger.error(f"{symbol}: Polygon fallback error: {e2}")
        
        return stats
    
    def _has_polygon_key(self) -> bool:
        """Check if Polygon API key is configured"""
        return bool(settings.polygon_api_key)
    
    def _supports_crypto(self) -> bool:
        """Check if any configured provider supports crypto"""
        return True  # yfinance supports crypto
    
    async def run_daily_backfill(self):
        """Run the daily backfill process"""
        logger.info("Starting daily backfill process")
        
        # Get all symbols from universe
        all_symbols = Universe.get_all_symbols()
        high_priority_symbols = Universe.get_high_priority_symbols()
        
        # Remove crypto symbols if provider doesn't support them
        all_symbols = [s for s in all_symbols if not (Universe.is_crypto(s) and not self._supports_crypto())]
        
        logger.info(f"Processing {len(all_symbols)} symbols total")
        logger.info(f"High priority symbols: {len(high_priority_symbols)}")
        
        # Statistics
        total_stats = {
            "symbols_processed": 0,
            "yfinance_records": 0,
            "polygon_records": 0,
            "errors": 0,
            "skipped": 0
        }
        
        # Process high priority symbols with more data
        logger.info("\n=== Processing high priority symbols ===")
        for symbol in high_priority_symbols:
            logger.info(f"\nProcessing {symbol} (high priority)...")
            
            # Get daily data
            stats = await self.backfill_symbol(symbol, "1d", days_back=3650)  # 10 years
            total_stats["yfinance_records"] += stats["yfinance"]
            total_stats["polygon_records"] += stats["polygon"]
            total_stats["errors"] += stats["errors"]
            
            # Get hourly data for last 30 days
            if not Universe.is_crypto(symbol):  # Skip hourly for crypto if not supported
                stats = await self.backfill_symbol(symbol, "1h", days_back=30)
                total_stats["yfinance_records"] += stats["yfinance"]
                total_stats["polygon_records"] += stats["polygon"]
                total_stats["errors"] += stats["errors"]
            
            total_stats["symbols_processed"] += 1
            
            # Small delay to avoid rate limits
            await asyncio.sleep(0.5)
        
        # Process remaining symbols with daily data only
        logger.info("\n=== Processing remaining symbols ===")
        remaining_symbols = [s for s in all_symbols if s not in high_priority_symbols]
        
        for i, symbol in enumerate(remaining_symbols):
            if i % 10 == 0:
                logger.info(f"\nProgress: {i}/{len(remaining_symbols)} symbols...")
            
            # Get daily data only for remaining symbols
            stats = await self.backfill_symbol(symbol, "1d", days_back=3650)  # 10 years
            total_stats["yfinance_records"] += stats["yfinance"]
            total_stats["polygon_records"] += stats["polygon"]
            total_stats["errors"] += stats["errors"]
            total_stats["symbols_processed"] += 1
            
            # Small delay to avoid rate limits
            await asyncio.sleep(0.1)
        
        # Summary
        logger.info("\n=== Backfill Summary ===")
        logger.info(f"Symbols processed: {total_stats['symbols_processed']}")
        logger.info(f"Records from yfinance: {total_stats['yfinance_records']}")
        logger.info(f"Records from Polygon: {total_stats['polygon_records']}")
        logger.info(f"Total records: {total_stats['yfinance_records'] + total_stats['polygon_records']}")
        logger.info(f"Errors: {total_stats['errors']}")
        
        return total_stats
    
    async def get_last_economic_update(self, symbol: str) -> Optional[datetime]:
        """Get the last update date for an economic indicator"""
        async for session in get_session():
            query = select(func.max(EconomicDataModel.date)).where(
                EconomicDataModel.symbol == symbol
            )
            result = await session.execute(query)
            return result.scalar()
    
    async def backfill_economic_indicator(self, symbol: str, days_back: int = 3650) -> Dict[str, int]:
        """Backfill a single economic indicator"""
        stats = {"records": 0, "errors": 0}
        
        if not self.economic_client:
            self.economic_client = EconomicDataClient(provider="fred")
        
        try:
            # Get last update date
            last_date = await self.get_last_economic_update(symbol)
            
            # Calculate date range
            end_date = date.today()
            if last_date:
                start_date = (last_date + timedelta(days=1)).date()
                if start_date > end_date:
                    logger.info(f"{symbol}: Already up to date")
                    return stats
            else:
                start_date = end_date - timedelta(days=days_back)
            
            logger.info(f"{symbol}: Fetching data from {start_date} to {end_date}")
            
            # Fetch data
            async with self.economic_client as client:
                data = await client.get_indicator_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
            
            if not data:
                logger.warning(f"{symbol}: No data returned")
                return stats
            
            # Insert into database
            async for session in get_session():
                for point in data:
                    try:
                        # Check if already exists
                        exists_query = select(EconomicDataModel).where(
                            EconomicDataModel.symbol == point.symbol,
                            EconomicDataModel.date == point.date
                        )
                        result = await session.execute(exists_query)
                        if result.scalar_one_or_none():
                            continue
                        
                        # Create new record
                        db_record = EconomicDataModel(
                            symbol=point.symbol,
                            date=point.date,
                            value=point.value,
                            revision_date=point.revision_date,
                            is_preliminary=point.is_preliminary,
                            source=point.source
                        )
                        session.add(db_record)
                        stats["records"] += 1
                        
                    except Exception as e:
                        logger.error(f"Error inserting {symbol} data: {e}")
                        stats["errors"] += 1
                
                await session.commit()
            
            logger.info(f"{symbol}: Inserted {stats['records']} records")
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            stats["errors"] += 1
        
        return stats
    
    async def run_economic_backfill(self, indicators: Optional[List[str]] = None):
        """Run backfill for economic indicators"""
        logger.info("Starting economic data backfill")
        
        # Get all indicators if not specified
        if not indicators:
            indicators = EconomicIndicators.get_all_symbols()
        
        logger.info(f"Processing {len(indicators)} economic indicators")
        
        total_stats = {"records": 0, "errors": 0, "indicators_processed": 0}
        
        for i, symbol in enumerate(indicators):
            if i % 10 == 0:
                logger.info(f"\nProgress: {i}/{len(indicators)} indicators...")
            
            stats = await self.backfill_economic_indicator(symbol)
            total_stats["records"] += stats["records"]
            total_stats["errors"] += stats["errors"]
            total_stats["indicators_processed"] += 1
            
            # Small delay to avoid rate limits
            await asyncio.sleep(0.5)
        
        logger.info("\n=== Economic Backfill Summary ===")
        logger.info(f"Indicators processed: {total_stats['indicators_processed']}")
        logger.info(f"Records inserted: {total_stats['records']}")
        logger.info(f"Errors: {total_stats['errors']}")
        
        return total_stats
    
    async def backfill_earnings_events(self, symbols: List[str], days_back: int = 365, provider: str = "yahoo") -> Dict[str, int]:
        """Backfill earnings events for given symbols"""
        stats = {"events": 0, "errors": 0}
        
        # Choose provider
        if provider.lower() in ["polygon", "finnhub"]:
            if not self.event_client or self.event_client.provider_name != provider:
                self.event_client = EventDataClient(provider=provider)
        else:
            if not self.event_client:
                self.event_client = EventDataClient(provider="yahoo")
        
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        logger.info(f"Fetching earnings from {start_date} to {end_date} using {provider}")
        
        for i, symbol in enumerate(symbols):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(symbols)} symbols...")
            
            try:
                async with self.event_client as client:
                    events = await client.get_earnings_events(
                        symbol=symbol,
                        from_date=start_date,
                        to_date=end_date
                    )
                
                # Insert into database
                async for session in get_session():
                    for event in events:
                        try:
                            # Check if already exists
                            exists_query = select(EventDataModel).where(
                                EventDataModel.source == event.source,
                                EventDataModel.source_id == event.source_id
                            )
                            result = await session.execute(exists_query)
                            if result.scalar_one_or_none():
                                continue
                            
                            # Create event record
                            db_event = EventDataModel(
                                event_type=event.event_type.value,
                                event_datetime=event.event_datetime,
                                event_name=event.event_name,
                                description=event.description,
                                impact=event.impact.value,
                                status=event.status.value,
                                symbol=event.symbol,
                                source=event.source,
                                source_id=event.source_id,
                                event_data=event.model_dump(
                                    exclude={"event_id", "created_at", "updated_at"},
                                    mode="json"
                                )
                            )
                            session.add(db_event)
                            await session.flush()  # Get the ID
                            
                            # Create earnings specific record
                            db_earnings = EarningsEventModel(
                                event_id=db_event.id,
                                symbol=event.symbol,
                                event_datetime=event.event_datetime,
                                eps_actual=event.eps_actual,
                                eps_estimate=event.eps_estimate,
                                eps_surprise=event.eps_surprise,
                                eps_surprise_pct=event.eps_surprise_pct,
                                revenue_actual=event.revenue_actual,
                                revenue_estimate=event.revenue_estimate,
                                revenue_surprise=event.revenue_surprise,
                                revenue_surprise_pct=event.revenue_surprise_pct,
                                guidance=event.guidance,
                                call_time=event.call_time,
                                fiscal_period=event.fiscal_period
                            )
                            session.add(db_earnings)
                            
                            stats["events"] += 1
                            
                        except Exception as e:
                            logger.error(f"Error inserting earnings for {symbol}: {e}")
                            stats["errors"] += 1
                    
                    await session.commit()
                    
            except Exception as e:
                logger.error(f"Error fetching earnings for {symbol}: {e}")
                stats["errors"] += 1
            
            # Small delay to avoid rate limits
            await asyncio.sleep(0.5)
        
        logger.info(f"Inserted {stats['events']} earnings events")
        return stats
    
    async def backfill_market_holidays(self, years: Optional[List[int]] = None) -> Dict[str, int]:
        """Backfill market holidays"""
        stats = {"events": 0, "errors": 0}
        
        if not self.event_client:
            self.event_client = EventDataClient(provider="yahoo")
        
        if not years:
            current_year = datetime.now().year
            years = [current_year - 1, current_year, current_year + 1]
        
        try:
            async with self.event_client as client:
                for year in years:
                    holidays = await client.get_market_holidays(year=year)
                    
                    # Insert into database
                    async for session in get_session():
                        for holiday in holidays:
                            try:
                                # Check if already exists
                                exists_query = select(EventDataModel).where(
                                    EventDataModel.source == holiday.source,
                                    EventDataModel.source_id == holiday.source_id
                                )
                                result = await session.execute(exists_query)
                                if result.scalar_one_or_none():
                                    continue
                                
                                # Create event record
                                db_event = EventDataModel(
                                    event_type=holiday.event_type.value,
                                    event_datetime=holiday.event_datetime,
                                    event_name=holiday.event_name,
                                    description=holiday.description,
                                    impact=holiday.impact.value,
                                    status=holiday.status.value,
                                    country=holiday.country,
                                    source=holiday.source,
                                    source_id=holiday.source_id,
                                    event_data=holiday.model_dump(
                                    exclude={"event_id", "created_at", "updated_at"},
                                    mode="json"
                                )
                                )
                                session.add(db_event)
                                stats["events"] += 1
                                
                            except Exception as e:
                                logger.error(f"Error inserting holiday {holiday.event_name}: {e}")
                                stats["errors"] += 1
                        
                        await session.commit()
                        
        except Exception as e:
            logger.error(f"Error during holiday backfill: {e}")
            stats["errors"] += 1
        
        logger.info(f"Inserted {stats['events']} market holidays")
        return stats
    
    async def backfill_economic_calendar(self, days_back: int = 30, currencies: List[str] = None) -> Dict[str, int]:
        """Backfill economic calendar events"""
        stats = {"events": 0, "errors": 0}
        
        # Use Finnhub for economic calendar
        if not self.event_client or self.event_client.provider_name != "finnhub":
            self.event_client = EventDataClient(provider="finnhub")
        
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days_back)
            
            if not currencies:
                currencies = ["USD", "EUR", "GBP", "JPY", "CAD"]
            
            logger.info(f"Fetching economic calendar from {start_date} to {end_date}")
            
            async with self.event_client as client:
                events = await client.get_economic_calendar(
                    from_date=start_date,
                    to_date=end_date,
                    currencies=currencies
                )
            
            if not events:
                logger.warning("No economic calendar events found")
                return stats
            
            # Insert into database
            async for session in get_session():
                for event in events:
                    try:
                        # Check if event already exists
                        exists_query = select(EventDataModel).where(
                            EventDataModel.source == event.source,
                            EventDataModel.source_id == event.source_id
                        )
                        result = await session.execute(exists_query)
                        existing = result.scalar_one_or_none()
                        
                        if existing:
                            continue
                        
                        # Create main event record
                        db_event = EventDataModel(
                            event_type=event.event_type.value,
                            event_datetime=event.event_datetime,
                            event_name=event.event_name,
                            description=event.description,
                            impact=event.impact.value,
                            status=event.status.value,
                            currency=event.currency,
                            country=event.country,
                            source=event.source,
                            source_id=event.source_id,
                            event_data=event.model_dump(
                                exclude={"event_id", "created_at", "updated_at"},
                                mode="json"
                            )
                        )
                        session.add(db_event)
                        await session.flush()
                        
                        # Create economic event specific record
                        db_econ = EconomicEventModel(
                            event_id=db_event.id,
                            event_datetime=event.event_datetime,
                            currency=event.currency,
                            actual=event.actual,
                            forecast=event.forecast,
                            previous=event.previous,
                            revised=event.revised,
                            unit=event.unit,
                            frequency=event.frequency
                        )
                        session.add(db_econ)
                        
                        stats["events"] += 1
                        
                    except Exception as e:
                        logger.error(f"Error inserting economic event {event.event_name}: {e}")
                        stats["errors"] += 1
                        continue
                
                await session.commit()
                logger.info(f"Inserted {stats['events']} economic calendar events")
                
        except Exception as e:
            logger.error(f"Error during economic calendar backfill: {e}")
            stats["errors"] += 1
        
        return stats
    
    async def run_event_backfill(self, event_types: Optional[List[str]] = None, provider: str = "yahoo"):
        """Run backfill for event data"""
        logger.info("Starting event data backfill")
        
        total_stats = {"events": 0, "errors": 0}
        
        # Backfill earnings events for high-priority symbols
        if not event_types or "earnings" in event_types:
            logger.info(f"\nBackfilling earnings events using {provider} provider...")
            symbols = EventCalendar.get_earnings_symbols()
            stats = await self.backfill_earnings_events(symbols, days_back=365, provider=provider)
            total_stats["events"] += stats["events"]
            total_stats["errors"] += stats["errors"]
        
        # Backfill market holidays
        if not event_types or "holidays" in event_types:
            logger.info("\nBackfilling market holidays...")
            stats = await self.backfill_market_holidays()
            total_stats["events"] += stats["events"]
            total_stats["errors"] += stats["errors"]
        
        # Backfill economic calendar (Finnhub only for now)
        if not event_types or "economic_calendar" in event_types:
            logger.info("\nBackfilling economic calendar events...")
            stats = await self.backfill_economic_calendar()
            total_stats["events"] += stats["events"]
            total_stats["errors"] += stats["errors"]
        
        logger.info("\n=== Event Backfill Summary ===")
        logger.info(f"Total events: {total_stats['events']}")
        logger.info(f"Errors: {total_stats['errors']}")
        
        return total_stats
    
    async def backfill_news_sentiment(
        self, 
        symbols: Optional[List[str]] = None, 
        days_back: int = 30, 
        provider: str = "alphavantage"
    ) -> Dict[str, int]:
        """Backfill news and sentiment data"""
        stats = {"articles": 0, "sentiments": 0, "errors": 0}
        
        if not self.news_client:
            self.news_client = NewsSentimentClient(provider=provider)
        
        # Use high priority symbols if not specified
        if not symbols:
            symbols = Universe.get_high_priority_symbols()[:20]  # Limit to avoid rate limits
        
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        logger.info(f"Fetching news from {start_date} to {end_date} using {provider}")
        logger.info(f"Processing {len(symbols)} symbols")
        
        async with self.news_client as client:
            for i, symbol in enumerate(symbols):
                if i % 5 == 0:
                    logger.info(f"Progress: {i}/{len(symbols)} symbols...")
                
                try:
                    # Get news with sentiment
                    if provider.lower() in ["alphavantage", "alpha_vantage", "av"]:
                        sentiments = await client.get_sentiment(
                            symbols=[symbol],
                            from_date=start_date,
                            to_date=end_date,
                            limit=50
                        )
                    else:
                        # NewsAPI doesn't provide sentiment
                        articles = await client.get_news(
                            symbols=[symbol],
                            from_date=start_date,
                            to_date=end_date,
                            limit=100
                        )
                        sentiments = []
                    
                    # Insert into database
                    async for session in get_session():
                        # Process sentiments (includes articles for AlphaVantage)
                        for sentiment in sentiments:
                            try:
                                # Check if article already exists
                                article = sentiment.article
                                exists_query = select(NewsArticleModel).where(
                                    NewsArticleModel.source == article.source,
                                    NewsArticleModel.source_id == article.source_id
                                )
                                result = await session.execute(exists_query)
                                existing_article = result.scalar_one_or_none()
                                
                                if not existing_article:
                                    # Create article record
                                    db_article = NewsArticleModel(
                                        source_id=article.source_id,
                                        source=article.source,
                                        source_category=article.source_category.value,
                                        author=article.author,
                                        title=article.title,
                                        description=article.description,
                                        content=article.content,
                                        url=str(article.url),
                                        image_url=str(article.image_url) if article.image_url else None,
                                        published_at=article.published_at,
                                        fetched_at=article.fetched_at,
                                        symbols=article.symbols,
                                        entities=article.entities,
                                        categories=[cat.value for cat in article.categories],
                                        raw_data=article.raw_data
                                    )
                                    session.add(db_article)
                                    await session.flush()
                                    stats["articles"] += 1
                                    article_id = db_article.id
                                else:
                                    article_id = existing_article.id
                                
                                # Check if sentiment already exists
                                sentiment_exists = select(NewsSentimentModel).where(
                                    NewsSentimentModel.article_id == article_id
                                )
                                result = await session.execute(sentiment_exists)
                                if not result.scalar_one_or_none():
                                    # Create sentiment record
                                    overall = sentiment.overall_sentiment
                                    db_sentiment = NewsSentimentModel(
                                        article_id=article_id,
                                        polarity=overall.polarity,
                                        subjectivity=overall.subjectivity,
                                        positive=overall.positive,
                                        negative=overall.negative,
                                        neutral=overall.neutral,
                                        bullish=overall.bullish,
                                        bearish=overall.bearish,
                                        sentiment_model=sentiment.sentiment_model,
                                        analyzed_at=sentiment.analyzed_at,
                                        symbols=article.symbols,
                                        relevance_score=sentiment.relevance_score,
                                        impact_score=sentiment.impact_score,
                                        is_breaking=sentiment.is_breaking,
                                        is_rumor=sentiment.is_rumor,
                                        requires_confirmation=sentiment.requires_confirmation,
                                        sentiment_by_type={
                                            st.value: {
                                                "polarity": score.polarity,
                                                "subjectivity": score.subjectivity,
                                                "positive": score.positive,
                                                "negative": score.negative,
                                                "neutral": score.neutral
                                            }
                                            for st, score in sentiment.sentiment_scores.items()
                                        }
                                    )
                                    session.add(db_sentiment)
                                    stats["sentiments"] += 1
                                
                            except Exception as e:
                                logger.error(f"Error inserting news/sentiment for {symbol}: {e}")
                                stats["errors"] += 1
                        
                        # Process raw articles (NewsAPI)
                        if provider.lower() == "newsapi":
                            for article in articles:
                                try:
                                    # Check if already exists
                                    exists_query = select(NewsArticleModel).where(
                                        NewsArticleModel.source == article.source,
                                        NewsArticleModel.url == str(article.url)
                                    )
                                    result = await session.execute(exists_query)
                                    if result.scalar_one_or_none():
                                        continue
                                    
                                    # Create article record
                                    db_article = NewsArticleModel(
                                        source_id=article.source_id,
                                        source=article.source,
                                        source_category=article.source_category.value,
                                        author=article.author,
                                        title=article.title,
                                        description=article.description,
                                        content=article.content,
                                        url=str(article.url),
                                        image_url=str(article.image_url) if article.image_url else None,
                                        published_at=article.published_at,
                                        fetched_at=article.fetched_at,
                                        symbols=article.symbols,
                                        entities=article.entities,
                                        categories=[cat.value for cat in article.categories],
                                        raw_data=article.raw_data
                                    )
                                    session.add(db_article)
                                    stats["articles"] += 1
                                    
                                except Exception as e:
                                    logger.error(f"Error inserting article: {e}")
                                    stats["errors"] += 1
                        
                        await session.commit()
                        
                except Exception as e:
                    logger.error(f"Error fetching news for {symbol}: {e}")
                    stats["errors"] += 1
                
                # Delay to respect rate limits
                if provider.lower() in ["alphavantage", "alpha_vantage", "av"]:
                    await asyncio.sleep(12)  # 5 requests per minute
                else:
                    await asyncio.sleep(0.5)
        
        logger.info(f"\nInserted {stats['articles']} articles and {stats['sentiments']} sentiments")
        return stats
    
    async def run_news_backfill(self, provider: str = "alphavantage", symbols: Optional[List[str]] = None):
        """Run news sentiment backfill"""
        logger.info("Starting news sentiment backfill")
        
        total_stats = {"articles": 0, "sentiments": 0, "errors": 0}
        
        # Run backfill
        stats = await self.backfill_news_sentiment(
            symbols=symbols,
            days_back=30,
            provider=provider
        )
        
        total_stats["articles"] += stats["articles"]
        total_stats["sentiments"] += stats["sentiments"]
        total_stats["errors"] += stats["errors"]
        
        logger.info("\n=== News Backfill Summary ===")
        logger.info(f"Articles inserted: {total_stats['articles']}")
        logger.info(f"Sentiments analyzed: {total_stats['sentiments']}")
        logger.info(f"Errors: {total_stats['errors']}")
        
        return total_stats


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart data backfill for Tetra Trading Platform")
    parser.add_argument("--scheduled", action="store_true", help="Run in scheduled/cron mode (no prompts)")
    parser.add_argument("--high-priority-only", action="store_true", help="Only backfill high priority symbols")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to backfill")
    parser.add_argument("--days", type=int, default=3650, help="Number of days to backfill (default: 3650/10 years)")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-error output")
    
    args = parser.parse_args()
    
    backfiller = SmartBackfiller()
    
    # Check if this is a scheduled run or manual
    if args.scheduled:
        # Scheduled run - just run without prompting
        await backfiller.run_daily_backfill()
    else:
        # Manual run - show menu
        print("\nTetra Data Backfill Utility")
        print("===========================")
        print("\nSelect data type to backfill:")
        print("1. Market data (OHLCV)")
        print("2. Economic indicators")
        print("3. Event data (earnings, holidays, economic calendar)")
        print("4. News sentiment data")
        print("5. All data types")
        print("6. Check current data coverage")
        print("0. Exit")
        
        choice = input("\nEnter choice (0-6): ").strip()
        
        if choice == "1":
            # Market data
            if args.symbols:
                # Specific symbols
                for symbol in args.symbols:
                    await backfiller.backfill_symbol(symbol, days_back=args.days)
            else:
                # All symbols
                await backfiller.run_daily_backfill()
                
        elif choice == "2":
            # Economic data
            await backfiller.run_economic_backfill()
            
        elif choice == "3":
            # Event data
            print("\nSelect event provider:")
            print("1. Yahoo Finance (default)")
            print("2. Polygon")
            print("3. Finnhub")
            provider_choice = input("Enter choice (1-3): ").strip()
            
            provider_map = {"1": "yahoo", "2": "polygon", "3": "finnhub"}
            provider = provider_map.get(provider_choice, "yahoo")
            
            await backfiller.run_event_backfill(provider=provider)
            
        elif choice == "4":
            # News sentiment
            print("\nSelect news provider:")
            print("1. Alpha Vantage (with sentiment)")
            print("2. NewsAPI (raw news only)")
            news_choice = input("Enter choice (1-2): ").strip()
            
            provider = "alphavantage" if news_choice == "1" else "newsapi"
            await backfiller.run_news_backfill(provider=provider, symbols=args.symbols)
            
        elif choice == "5":
            # All data
            print("\nRunning full backfill (this may take a while)...")
            await backfiller.run_daily_backfill()
            await backfiller.run_economic_backfill()
            await backfiller.run_event_backfill()
            await backfiller.run_news_backfill()
            
        elif choice == "6":
            # Check coverage
            coverage = await backfiller.check_data_coverage()
            print(f"\nData Coverage Summary:")
            print(f"Total symbols: {coverage['total_symbols']}")
            print(f"Total records: {coverage['total_records']:,}")
            print("\nTop 10 symbols by record count:")
            
            sorted_symbols = sorted(
                coverage['symbols'].items(), 
                key=lambda x: x[1]['records'], 
                reverse=True
            )[:10]
            
            for symbol, data in sorted_symbols:
                print(f"  {symbol}: {data['records']:,} records, "
                      f"{data['days']} days, "
                      f"{data['earliest'].strftime('%Y-%m-%d') if data['earliest'] else 'N/A'} to "
                      f"{data['latest'].strftime('%Y-%m-%d') if data['latest'] else 'N/A'}")
        
        elif choice == "0":
            print("Exiting...")
        else:
            print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())