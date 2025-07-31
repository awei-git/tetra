#!/usr/bin/env python3
"""
Daily backfill script for market data
- Uses yfinance for historical data (free, no limits)
- Uses Polygon for recent/intraday data (better quality, real-time)
- Handles updates intelligently to avoid duplicates
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Dict, Set

sys.path.append(str(Path(__file__).parent.parent))

from src.universe import Universe
from src.ingestion.data_ingester import DataIngester
from src.db.base import get_session
from src.db.models import OHLCVModel
from src.utils.logging import logger
from sqlalchemy import select, func
from config import settings


class SmartBackfiller:
    """Smart backfilling that uses multiple data sources efficiently"""
    
    def __init__(self):
        self.yfinance_ingester = DataIngester(provider="yfinance")
        self.polygon_ingester = DataIngester(provider="polygon")
        
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
    
    def _supports_crypto(self) -> bool:
        """Check if current providers support crypto"""
        # yfinance supports crypto with -USD suffix
        # Polygon needs special subscription
        return True  # yfinance supports crypto


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
        await backfiller.run_daily_backfill()
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
        print("\nThis script intelligently backfills market data:")
        print("- Uses yfinance for historical data (free, unlimited)")
        print("- Uses Polygon for recent/real-time data (if available)")
        print("- Avoids duplicate data")
        print("- Updates only what's needed")
        print(f"\nTotal symbols in universe: {len(Universe.get_all_symbols())}")
        print(f"ETFs: {len(Universe.get_all_etfs())}")
        print(f"Stocks: {len(Universe.get_all_stocks())}")
        print(f"Crypto: {len(Universe.CRYPTO_SYMBOLS)}")
        
        print("\nOptions:")
        print("1. Run full daily backfill (recommended)")
        print("2. Backfill specific symbols")
        print("3. Show universe categories")
        print("4. Check current data coverage")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ")
        
        if choice == "1":
            await backfiller.run_daily_backfill()
        
        elif choice == "2":
            symbols_input = input("Enter symbols (comma-separated): ")
            symbols = [s.strip().upper() for s in symbols_input.split(",")]
            
            for symbol in symbols:
                print(f"\nBackfilling {symbol}...")
                stats = await backfiller.backfill_symbol(symbol, days_back=3650)  # 10 years
                print(f"Results: {stats}")
        
        elif choice == "3":
            categories = Universe.get_universe_by_category()
            for category, symbols in categories.items():
                print(f"\n{category.upper()} ({len(symbols)} symbols):")
                print(", ".join(sorted(symbols[:10])), "..." if len(symbols) > 10 else "")
        
        elif choice == "4":
            print("\nChecking current data coverage...")
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