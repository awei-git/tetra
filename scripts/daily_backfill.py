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
from src.db.base import get_session
from src.db.models import OHLCVModel, EconomicDataModel, EventDataModel, EconomicEventModel, EarningsEventModel
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
        
        # Determine date range
        end_date = date.today()
        
        # Special handling for crypto with known start dates
        crypto_start_dates = {
            "BTC-USD": date(2014, 9, 17),  # Bitcoin on Yahoo Finance
            "ETH-USD": date(2017, 11, 9),   # Ethereum on Yahoo Finance
        }
        
        if last_date:
            # Check if we have sufficient historical data
            expected_start = end_date - timedelta(days=days_back)
            
            # For crypto, use actual start date if more recent
            if symbol in crypto_start_dates:
                expected_start = max(expected_start, crypto_start_dates[symbol])
            
            # Calculate days of data we should have
            expected_days = (end_date - expected_start).days
            actual_days = (last_date.date() - expected_start).days if last_date else 0
            
            # Get earliest date in DB for this symbol
            async for session in get_session():
                query = select(func.min(OHLCVModel.timestamp)).where(
                    OHLCVModel.symbol == symbol,
                    OHLCVModel.timeframe == timeframe
                )
                result = await session.execute(query)
                earliest_date = result.scalar()
                
                if earliest_date:
                    actual_coverage = (last_date - earliest_date).days
                    expected_coverage = min(expected_days, days_back)
                    
                    # If we have less than 80% of expected data, force full backfill
                    if actual_coverage < expected_coverage * 0.8:
                        logger.info(f"{symbol}: Incomplete data detected ({actual_coverage} days vs {expected_coverage} expected)")
                        logger.info(f"{symbol}: Forcing full backfill from {expected_start}")
                        start_date = expected_start
                    else:
                        # Normal update from last date
                        start_date = (last_date + timedelta(days=1)).date()
                        logger.info(f"{symbol}: Last update was {last_date.date()}, updating from {start_date}")
                else:
                    # Have last_date but no earliest_date? Force full backfill
                    start_date = expected_start
                    logger.info(f"{symbol}: Data inconsistency detected, forcing full backfill")
        else:
            # No data, get historical
            start_date = end_date - timedelta(days=days_back)
            
            # For crypto, use actual start date if more recent
            if symbol in crypto_start_dates:
                start_date = max(start_date, crypto_start_dates[symbol])
                
            logger.info(f"{symbol}: No existing data, fetching from {start_date}")
        
        # Skip if already up to date
        if start_date >= end_date:
            logger.info(f"{symbol}: Already up to date")
            return stats
        
        # Use yfinance for historical data (more than 7 days old)
        yf_end_date = end_date - timedelta(days=7)
        if start_date < yf_end_date:
            try:
                logger.info(f"{symbol}: Using yfinance for {start_date} to {yf_end_date}")
                result = await self.yfinance_ingester.ingest_ohlcv_batch(
                    symbols=[symbol],
                    from_date=start_date,
                    to_date=yf_end_date,
                    timeframe=timeframe
                )
                stats["yfinance"] = result.get("inserted", 0) + result.get("updated", 0)
            except Exception as e:
                logger.error(f"{symbol}: yfinance error: {e}")
                stats["errors"] += 1
        
        # Use Polygon for recent data (last 7 days) if available
        if settings.polygon_api_key and start_date < end_date:
            polygon_start = max(start_date, end_date - timedelta(days=7))
            try:
                logger.info(f"{symbol}: Using Polygon for recent data {polygon_start} to {end_date}")
                result = await self.polygon_ingester.ingest_ohlcv_batch(
                    symbols=[symbol],
                    from_date=polygon_start,
                    to_date=end_date,
                    timeframe=timeframe
                )
                stats["polygon"] = result.get("inserted", 0) + result.get("updated", 0)
            except Exception as e:
                logger.error(f"{symbol}: Polygon error: {e}")
                # If Polygon fails, try yfinance for recent data too
                if stats["yfinance"] == 0:  # Only if yfinance hasn't been tried
                    try:
                        result = await self.yfinance_ingester.ingest_ohlcv_batch(
                            symbols=[symbol],
                            from_date=polygon_start,
                            to_date=end_date,
                            timeframe=timeframe
                        )
                        stats["yfinance"] += result.get("inserted", 0) + result.get("updated", 0)
                    except Exception as e2:
                        logger.error(f"{symbol}: yfinance fallback error: {e2}")
                        stats["errors"] += 1
        
        return stats
    
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
            
            # Determine date range
            end_date = date.today()
            if last_date:
                start_date = (last_date + timedelta(days=1)).date()
                logger.info(f"{symbol}: Last update was {last_date.date()}, updating from {start_date}")
            else:
                start_date = end_date - timedelta(days=days_back)
                logger.info(f"{symbol}: No existing data, fetching from {start_date}")
            
            # Skip if already up to date
            if start_date >= end_date:
                logger.info(f"{symbol}: Already up to date")
                return stats
            
            # Fetch data
            async with self.economic_client as client:
                data = await client.get_indicator_data(
                    symbol=symbol,
                    from_date=start_date,
                    to_date=end_date
                )
            
            if not data:
                logger.warning(f"{symbol}: No data returned from FRED")
                return stats
            
            # Insert into database
            async for session in get_session():
                for item in data:
                    # Check if already exists
                    exists_query = select(EconomicDataModel).where(
                        EconomicDataModel.symbol == symbol,
                        EconomicDataModel.date == item.date
                    )
                    result = await session.execute(exists_query)
                    existing = result.scalar_one_or_none()
                    
                    if not existing:
                        db_record = EconomicDataModel(
                            symbol=symbol,
                            date=item.date,
                            value=item.value,
                            source=item.source
                        )
                        session.add(db_record)
                        stats["records"] += 1
                
                await session.commit()
                logger.info(f"{symbol}: Inserted {stats['records']} new records")
                
        except Exception as e:
            logger.error(f"{symbol}: Error during backfill: {e}")
            stats["errors"] += 1
        
        return stats
    
    async def run_economic_backfill(self, indicators: Optional[List[str]] = None):
        """Run backfill for economic indicators"""
        logger.info("Starting economic data backfill")
        
        if not indicators:
            # Use high priority indicators
            indicators = [ind[0] for ind in EconomicIndicators.get_high_priority_indicators()]
        
        logger.info(f"Processing {len(indicators)} economic indicators")
        
        total_stats = {"indicators_processed": 0, "records": 0, "errors": 0}
        
        for symbol in indicators:
            logger.info(f"\nProcessing {symbol}...")
            stats = await self.backfill_economic_indicator(symbol)
            
            total_stats["records"] += stats["records"]
            total_stats["errors"] += stats["errors"]
            total_stats["indicators_processed"] += 1
            
            # Small delay to respect rate limits
            await asyncio.sleep(0.5)
        
        logger.info("\n=== Economic Backfill Summary ===")
        logger.info(f"Indicators processed: {total_stats['indicators_processed']}")
        logger.info(f"Total records: {total_stats['records']}")
        logger.info(f"Errors: {total_stats['errors']}")
        
        return total_stats
    
    def _supports_crypto(self) -> bool:
        """Check if current providers support crypto"""
        # yfinance supports crypto with -USD suffix
        # Polygon needs special subscription
        return True  # yfinance supports crypto
    
    async def get_last_event_update(self, event_type: Optional[str] = None) -> Optional[datetime]:
        """Get the last update date for events"""
        async for session in get_session():
            query = select(func.max(EventDataModel.event_datetime))
            if event_type:
                query = query.where(EventDataModel.event_type == event_type)
            result = await session.execute(query)
            return result.scalar()
    
    async def backfill_earnings_events(self, symbols: List[str], days_back: int = 365, provider: str = "finnhub") -> Dict[str, int]:
        """Backfill earnings events for specific symbols"""
        stats = {"events": 0, "errors": 0}
        
        if not self.event_client or self.event_client.provider_name != provider:
            self.event_client = EventDataClient(provider=provider)
        
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days_back)
            
            logger.info(f"Fetching earnings events for {len(symbols)} symbols from {start_date} to {end_date}")
            
            async with self.event_client as client:
                events = await client.get_earnings_calendar(
                    from_date=start_date,
                    to_date=end_date,
                    symbols=symbols
                )
            
            if not events:
                logger.warning("No earnings events found")
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
                            symbol=event.symbol,
                            source=event.source,
                            source_id=event.source_id,
                            event_data=event.model_dump(
                                exclude={"event_id", "created_at", "updated_at"},
                                mode="json"
                            )
                        )
                        session.add(db_event)
                        await session.flush()
                        
                        # Create earnings-specific record
                        db_earnings = EarningsEventModel(
                            event_id=db_event.id,
                            symbol=event.symbol,
                            event_datetime=event.event_datetime,
                            eps_actual=event.eps_actual,
                            eps_estimate=event.eps_estimate,
                            revenue_actual=event.revenue_actual,
                            revenue_estimate=event.revenue_estimate,
                            call_time=event.call_time,
                            fiscal_period=event.fiscal_period
                        )
                        session.add(db_earnings)
                        
                        stats["events"] += 1
                        
                    except Exception as e:
                        logger.error(f"Error inserting event {event.event_name}: {e}")
                        stats["errors"] += 1
                        continue
                
                await session.commit()
                logger.info(f"Inserted {stats['events']} earnings events")
                
        except Exception as e:
            logger.error(f"Error during earnings backfill: {e}")
            stats["errors"] += 1
        
        return stats
    
    async def backfill_market_holidays(self, years: List[int] = None) -> Dict[str, int]:
        """Backfill market holiday events"""
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
        logger.info("Running scheduled daily backfill")
        if args.quiet:
            logger.setLevel("ERROR")
        
        # Run market data backfill
        await backfiller.run_daily_backfill()
        
        # Run economic data backfill for high priority indicators
        await backfiller.run_economic_backfill()
        
        # Run event data backfill
        await backfiller.run_event_backfill()
    elif args.symbols:
        # Backfill specific symbols
        logger.info(f"Backfilling specific symbols: {args.symbols}")
        for symbol in args.symbols:
            stats = await backfiller.backfill_symbol(symbol, days_back=args.days)
            logger.info(f"{symbol}: {stats}")
    elif args.high_priority_only:
        # Backfill only high priority symbols
        logger.info("Backfilling high priority symbols only")
        high_priority = Universe.get_high_priority_symbols()
        for symbol in high_priority:
            stats = await backfiller.backfill_symbol(symbol, days_back=args.days)
            logger.info(f"{symbol}: {stats}")
    else:
        # Manual run - show menu
        print("🚀 Tetra Trading Platform - Smart Data Backfill")
        print("=" * 60)
        print("\nMarket Data:")
        print("- Uses yfinance for historical data (free, unlimited)")
        print("- Uses Polygon for recent/real-time data (if available)")
        print(f"- Total symbols: {len(Universe.get_all_symbols())} (ETFs: {len(Universe.get_all_etfs())}, Stocks: {len(Universe.get_all_stocks())}, Crypto: {len(Universe.CRYPTO_SYMBOLS)})")
        
        print("\nEconomic Data:")
        print("- Uses FRED for economic indicators")
        print(f"- Total indicators: {len(EconomicIndicators.get_all_indicators())}")
        
        print("\nEvent Data:")
        print("- Earnings calendar from Yahoo Finance")
        print("- Market holidays and economic events")
        print(f"- Tracking {len(EventCalendar.get_earnings_symbols())} symbols for earnings")
        
        print("\nOptions:")
        print("1. Run full daily backfill (market + economic + events)")
        print("2. Backfill specific market symbols")
        print("3. Backfill economic indicators")
        print("4. Backfill event data")
        print("5. Show universe categories")
        print("6. Check market data coverage")
        print("7. Check economic data coverage")
        print("8. Check event data coverage")
        print("9. Exit")
        
        choice = input("\nSelect option (1-9): ")
        
        if choice == "1":
            # Run full backfill
            await backfiller.run_daily_backfill()
            await backfiller.run_economic_backfill()
            await backfiller.run_event_backfill()
        
        elif choice == "2":
            symbols_input = input("Enter symbols (comma-separated): ")
            symbols = [s.strip().upper() for s in symbols_input.split(",")]
            
            for symbol in symbols:
                print(f"\nBackfilling {symbol}...")
                stats = await backfiller.backfill_symbol(symbol, days_back=3650)  # 10 years
                print(f"Results: {stats}")
        
        elif choice == "3":
            # Backfill economic indicators
            print("\nEconomic indicator options:")
            print("1. Backfill high priority indicators")
            print("2. Backfill ALL indicators (60 indicators)")
            print("3. Backfill specific indicators")
            print("4. Show all available indicators")
            
            sub_choice = input("\nSelect option (1-4): ")
            
            if sub_choice == "1":
                await backfiller.run_economic_backfill()
            elif sub_choice == "2":
                # Backfill ALL indicators
                all_indicators = EconomicIndicators.get_all_indicators()
                all_symbols = [ind[0] for ind in all_indicators]
                print(f"\nStarting backfill for ALL {len(all_symbols)} economic indicators...")
                print("This may take a while due to rate limits...")
                await backfiller.run_economic_backfill(indicators=all_symbols)
            elif sub_choice == "3":
                symbols_input = input("Enter FRED symbols (comma-separated, e.g., DFF,DGS10,CPIAUCSL): ")
                symbols = [s.strip().upper() for s in symbols_input.split(",")]
                await backfiller.run_economic_backfill(indicators=symbols)
            elif sub_choice == "4":
                categories = EconomicIndicators.get_indicators_by_category()
                for category, indicators in categories.items():
                    print(f"\n{category.upper()} ({len(indicators)} indicators):")
                    for symbol, name, freq in indicators[:5]:
                        print(f"  {symbol}: {name} ({freq.value})")
                    if len(indicators) > 5:
                        print(f"  ... and {len(indicators) - 5} more")
        
        elif choice == "4":
            # Backfill event data
            print("\nEvent data options:")
            print("1. Backfill earnings events")
            print("2. Backfill market holidays")
            print("3. Backfill economic calendar (Finnhub)")
            print("4. Backfill all event types")
            
            sub_choice = input("\nSelect option (1-4): ")
            
            # Ask for provider for earnings events
            provider = "yahoo"
            if sub_choice in ["1", "4"]:
                print("\nSelect provider for earnings events:")
                print("1. Yahoo Finance (limited)")
                print("2. Polygon")
                print("3. Finnhub")
                provider_choice = input("\nSelect provider (1-3): ")
                if provider_choice == "2":
                    provider = "polygon"
                elif provider_choice == "3":
                    provider = "finnhub"
            
            if sub_choice == "1":
                await backfiller.run_event_backfill(event_types=["earnings"], provider=provider)
            elif sub_choice == "2":
                await backfiller.run_event_backfill(event_types=["holidays"])
            elif sub_choice == "3":
                await backfiller.run_event_backfill(event_types=["economic_calendar"])
            elif sub_choice == "4":
                await backfiller.run_event_backfill(provider=provider)
        
        elif choice == "5":
            # Show universe categories
            categories = Universe.get_universe_by_category()
            for category, symbols in categories.items():
                print(f"\n{category.upper()} ({len(symbols)} symbols):")
                print(", ".join(sorted(symbols[:10])), "..." if len(symbols) > 10 else "")
        
        elif choice == "6":
            # Check market data coverage
            print("\nChecking market data coverage...")
            coverage = await backfiller.check_data_coverage()
            
            print(f"\nTotal symbols in database: {coverage['total_symbols']}")
            print(f"Total records: {coverage['total_records']:,}")
            print("-" * 80)
            print(f"{'Symbol':<10} {'Records':<10} {'Earliest':<20} {'Latest':<20} {'Days':<10}")
            print("-" * 80)
            
            for symbol, data in sorted(coverage['symbols'].items()):
                print(f"{symbol:<10} {data['records']:<10} {str(data['earliest'])[:10]:<20} {str(data['latest'])[:10]:<20} {data['days']:<10}")
            
            if not coverage['symbols']:
                print("No data found in database. Run backfill to populate data.")
        
        elif choice == "7":
            # Check economic data coverage
            print("\nChecking economic data coverage...")
            
            async for session in get_session():
                # Get summary statistics
                query = select(
                    EconomicDataModel.symbol,
                    func.count(EconomicDataModel.id).label('record_count'),
                    func.min(EconomicDataModel.date).label('earliest_date'),
                    func.max(EconomicDataModel.date).label('latest_date')
                ).group_by(EconomicDataModel.symbol).order_by(EconomicDataModel.symbol)
                
                result = await session.execute(query)
                data = result.all()
                
                if data:
                    print(f"\nTotal indicators in database: {len(data)}")
                    print("-" * 80)
                    print(f"{'Symbol':<15} {'Records':<10} {'Earliest':<20} {'Latest':<20}")
                    print("-" * 80)
                    
                    for row in data:
                        print(f"{row.symbol:<15} {row.record_count:<10} {str(row.earliest_date)[:10]:<20} {str(row.latest_date)[:10]:<20}")
                else:
                    print("No economic data found. Run economic backfill to populate data.")
        
        elif choice == "8":
            # Check event data coverage
            print("\nChecking event data coverage...")
            
            async for session in get_session():
                # Get summary statistics
                query = select(
                    EventDataModel.event_type,
                    func.count(EventDataModel.id).label('event_count'),
                    func.min(EventDataModel.event_datetime).label('earliest_date'),
                    func.max(EventDataModel.event_datetime).label('latest_date')
                ).group_by(EventDataModel.event_type).order_by(EventDataModel.event_type)
                
                result = await session.execute(query)
                data = result.all()
                
                # Get total events
                total_query = select(func.count(EventDataModel.id))
                total_result = await session.execute(total_query)
                total_events = total_result.scalar()
                
                if data:
                    print(f"\nTotal events in database: {total_events}")
                    print("-" * 80)
                    print(f"{'Event Type':<20} {'Count':<10} {'Earliest':<20} {'Latest':<20}")
                    print("-" * 80)
                    
                    for row in data:
                        print(f"{row.event_type:<20} {row.event_count:<10} {str(row.earliest_date)[:19]:<20} {str(row.latest_date)[:19]:<20}")
                else:
                    print("No event data found. Run event backfill to populate data.")
        
        else:
            print("Exiting...")


if __name__ == "__main__":
    # Install yfinance if needed
    try:
        import yfinance
    except ImportError:
        print("Installing yfinance...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "yfinance"], check=True)
    
    asyncio.run(main())