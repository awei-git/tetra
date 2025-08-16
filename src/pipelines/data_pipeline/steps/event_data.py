"""Event data pipeline step"""

from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional
import asyncio
import yfinance as yf
import pandas as pd
from sqlalchemy import text

from src.pipelines.base import PipelineStep, PipelineContext
from src.clients.event_data_client import EventDataClient
from src.db.base import get_session
from src.utils.logging import logger


class EventDataStep(PipelineStep[Dict[str, Any]]):
    """
    Step for fetching event data (earnings, dividends, splits, economic calendar).
    """
    
    def __init__(self):
        super().__init__(
            name="EventDataStep",
            description="Fetch corporate and economic events"
        )
        
    async def _fetch_earnings_from_yfinance(self, symbols: List[str]) -> int:
        """Fetch earnings data from yfinance for given symbols"""
        total_inserted = 0
        
        async for db in get_session():
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    
                    # Get earnings dates
                    earnings_dates = ticker.earnings_dates
                    if earnings_dates is not None and not earnings_dates.empty:
                        for date_idx in earnings_dates.index:
                            # Parse earnings data
                            event_datetime = pd.to_datetime(date_idx)
                            eps_estimate = earnings_dates.loc[date_idx, 'EPS Estimate'] if 'EPS Estimate' in earnings_dates.columns else None
                            eps_actual = earnings_dates.loc[date_idx, 'Reported EPS'] if 'Reported EPS' in earnings_dates.columns else None
                            revenue_estimate = earnings_dates.loc[date_idx, 'Revenue Estimate'] if 'Revenue Estimate' in earnings_dates.columns else None
                            revenue_actual = earnings_dates.loc[date_idx, 'Revenue'] if 'Revenue' in earnings_dates.columns else None
                            
                            # Insert or update earnings event
                            query = text("""
                                INSERT INTO events.earnings_events 
                                (symbol, event_datetime, eps_estimate, eps_actual, revenue_estimate, revenue_actual, created_at, updated_at)
                                VALUES (:symbol, :event_datetime, :eps_estimate, :eps_actual, :revenue_estimate, :revenue_actual, NOW(), NOW())
                                ON CONFLICT (symbol, event_datetime) 
                                DO UPDATE SET 
                                    eps_estimate = EXCLUDED.eps_estimate,
                                    eps_actual = EXCLUDED.eps_actual,
                                    revenue_estimate = EXCLUDED.revenue_estimate,
                                    revenue_actual = EXCLUDED.revenue_actual,
                                    updated_at = NOW()
                            """)
                            
                            await db.execute(query, {
                                'symbol': symbol,
                                'event_datetime': event_datetime,
                                'eps_estimate': float(eps_estimate) if eps_estimate and pd.notna(eps_estimate) else None,
                                'eps_actual': float(eps_actual) if eps_actual and pd.notna(eps_actual) else None,
                                'revenue_estimate': float(revenue_estimate) if revenue_estimate and pd.notna(revenue_estimate) else None,
                                'revenue_actual': float(revenue_actual) if revenue_actual and pd.notna(revenue_actual) else None
                            })
                            total_inserted += 1
                            
                    # Get upcoming earnings
                    if hasattr(ticker, 'calendar') and ticker.calendar is not None:
                        calendar = ticker.calendar
                        if not calendar.empty and 'Earnings Date' in calendar.columns:
                            for earnings_date in calendar['Earnings Date']:
                                if pd.notna(earnings_date):
                                    event_datetime = pd.to_datetime(earnings_date)
                                    
                                    # Insert future earnings event
                                    query = text("""
                                        INSERT INTO events.earnings_events 
                                        (symbol, event_datetime, created_at, updated_at)
                                        VALUES (:symbol, :event_datetime, NOW(), NOW())
                                        ON CONFLICT (symbol, event_datetime) DO NOTHING
                                    """)
                                    
                                    await db.execute(query, {
                                        'symbol': symbol,
                                        'event_datetime': event_datetime
                                    })
                                    total_inserted += 1
                    
                    await db.commit()
                    
                except Exception as e:
                    logger.debug(f"Failed to fetch earnings for {symbol}: {e}")
                    await db.rollback()
                    
                # Rate limiting
                await asyncio.sleep(0.1)
        
        return total_inserted
    
    async def _fetch_dividends_from_yfinance(self, symbols: List[str]) -> int:
        """Fetch dividend data from yfinance for given symbols"""
        total_inserted = 0
        
        async for db in get_session():
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    
                    # Get dividend history
                    dividends = ticker.dividends
                    if dividends is not None and not dividends.empty:
                        for date_idx in dividends.index[-20:]:  # Last 20 dividends
                            event_datetime = pd.to_datetime(date_idx)
                            amount = float(dividends[date_idx])
                            
                            # Insert dividend event
                            query = text("""
                                INSERT INTO events.dividend_events 
                                (symbol, event_datetime, amount, created_at, updated_at)
                                VALUES (:symbol, :event_datetime, :amount, NOW(), NOW())
                                ON CONFLICT (symbol, event_datetime) 
                                DO UPDATE SET 
                                    amount = EXCLUDED.amount,
                                    updated_at = NOW()
                            """)
                            
                            await db.execute(query, {
                                'symbol': symbol,
                                'event_datetime': event_datetime,
                                'amount': amount
                            })
                            total_inserted += 1
                    
                    await db.commit()
                    
                except Exception as e:
                    logger.debug(f"Failed to fetch dividends for {symbol}: {e}")
                    await db.rollback()
                    
                # Rate limiting
                await asyncio.sleep(0.1)
        
        return total_inserted
    
    async def _fetch_splits_from_yfinance(self, symbols: List[str]) -> int:
        """Fetch stock split data from yfinance for given symbols"""
        total_inserted = 0
        
        async for db in get_session():
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    
                    # Get split history
                    splits = ticker.splits
                    if splits is not None and not splits.empty:
                        for date_idx in splits.index:
                            event_datetime = pd.to_datetime(date_idx)
                            ratio = float(splits[date_idx])
                            
                            # Insert split event
                            query = text("""
                                INSERT INTO events.split_events 
                                (symbol, event_datetime, ratio, created_at, updated_at)
                                VALUES (:symbol, :event_datetime, :ratio, NOW(), NOW())
                                ON CONFLICT (symbol, event_datetime) 
                                DO UPDATE SET 
                                    ratio = EXCLUDED.ratio,
                                    updated_at = NOW()
                            """)
                            
                            await db.execute(query, {
                                'symbol': symbol,
                                'event_datetime': event_datetime,
                                'ratio': ratio
                            })
                            total_inserted += 1
                    
                    await db.commit()
                    
                except Exception as e:
                    logger.debug(f"Failed to fetch splits for {symbol}: {e}")
                    await db.rollback()
                    
                # Rate limiting
                await asyncio.sleep(0.1)
        
        return total_inserted
    
    async def _update_earnings_calendar(self) -> int:
        """Update earnings calendar from existing database"""
        total_updated = 0
        
        async for db in get_session():
            try:
                # Clean up old events (older than 2 years)
                cleanup_query = text("""
                    DELETE FROM events.earnings_events 
                    WHERE event_datetime < NOW() - INTERVAL '2 years'
                """)
                await db.execute(cleanup_query)
                
                # Count current events
                count_query = text("""
                    SELECT COUNT(*) as count 
                    FROM events.earnings_events 
                    WHERE event_datetime >= NOW() - INTERVAL '1 year'
                """)
                result = await db.execute(count_query)
                row = result.fetchone()
                total_updated = row['count'] if row else 0
                
                await db.commit()
                
            except Exception as e:
                logger.error(f"Failed to update earnings calendar: {e}")
                await db.rollback()
        
        return total_updated
    
    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Execute event data fetching"""
        mode = context.data.get("mode", "daily")
        symbols = context.data.get("symbols", [])
        start_date = context.data.get("start_date")
        end_date = context.data.get("end_date")
        
        logger.debug(f"EventDataStep context keys: {list(context.data.keys())}")
        logger.debug(f"EventDataStep symbols count: {len(symbols) if symbols else 0}")
        
        # For events, we look both backward and forward
        if mode == "daily":
            # Look back 7 days for recent events, forward 30 days for upcoming
            lookback_start = start_date - timedelta(days=7)
            lookahead_end = end_date + timedelta(days=30)
        else:
            # For backfill, use the specified range
            lookback_start = start_date
            lookahead_end = end_date
            
        logger.info(f"Starting event data {mode} for {len(symbols)} symbols")
        
        results = {
            "earnings": {"success": 0, "failed": 0, "from_yfinance": 0, "from_db": 0},
            "dividends": {"success": 0, "failed": 0, "from_yfinance": 0},
            "splits": {"success": 0, "failed": 0, "from_yfinance": 0},
            "holidays": {"success": 0, "failed": 0},
            "economic_calendar": {"success": 0, "failed": 0},
            "total_records": 0
        }
        
        client = EventDataClient(provider="yahoo")
        try:
                
                # 1. Update earnings calendar from existing database
                try:
                    db_earnings_count = await self._update_earnings_calendar()
                    results["earnings"]["from_db"] = db_earnings_count
                    results["total_records"] += db_earnings_count
                    logger.info(f"Found {db_earnings_count} earnings events in database")
                except Exception as e:
                    logger.error(f"Failed to update earnings from database: {e}")
                
                # 2. Fetch new earnings from yfinance (daily mode or if requested)
                if mode == "daily" or context.data.get("fetch_earnings", False):
                    try:
                        # Fetch for high-priority symbols in daily mode
                        earnings_symbols = symbols[:50] if mode == "daily" else symbols
                        yf_earnings_count = await self._fetch_earnings_from_yfinance(earnings_symbols)
                        results["earnings"]["from_yfinance"] = yf_earnings_count
                        results["earnings"]["success"] = yf_earnings_count + results["earnings"]["from_db"]
                        results["total_records"] += yf_earnings_count
                        logger.info(f"Fetched {yf_earnings_count} new earnings events from yfinance")
                    except Exception as e:
                        logger.error(f"Failed to fetch earnings from yfinance: {e}")
                        results["earnings"]["failed"] = 1
                        context.add_error(f"Earnings fetch failed: {str(e)}")
                
                # 3. Fetch dividends and splits (only in backfill mode or if requested)
                if mode == "backfill" or context.data.get("fetch_dividends", False):
                    try:
                        # Fetch dividends for symbols
                        dividend_symbols = symbols[:100] if mode == "daily" else symbols
                        div_count = await self._fetch_dividends_from_yfinance(dividend_symbols)
                        results["dividends"]["from_yfinance"] = div_count
                        results["dividends"]["success"] = div_count
                        results["total_records"] += div_count
                        logger.info(f"Fetched {div_count} dividend events from yfinance")
                    except Exception as e:
                        results["dividends"]["failed"] = 1
                        logger.error(f"Failed to fetch dividends: {e}")
                
                if mode == "backfill" or context.data.get("fetch_splits", False):
                    try:
                        # Fetch splits for symbols
                        split_symbols = symbols[:100] if mode == "daily" else symbols
                        split_count = await self._fetch_splits_from_yfinance(split_symbols)
                        results["splits"]["from_yfinance"] = split_count
                        results["splits"]["success"] = split_count
                        results["total_records"] += split_count
                        logger.info(f"Fetched {split_count} split events from yfinance")
                    except Exception as e:
                        results["splits"]["failed"] = 1
                        logger.error(f"Failed to fetch splits: {e}")
                
                # 4. Market holidays (not symbol-specific)
                if mode == "daily" or context.data.get("fetch_holidays", True):
                    try:
                        current_year = date.today().year
                        years = [current_year - 1, current_year, current_year + 1]
                        
                        holiday_count = 0
                        for year in years:
                            holidays = await client.get_market_holidays(year=year)
                            holiday_count += len(holidays) if holidays else 0
                            
                        results["holidays"]["success"] = holiday_count
                        results["total_records"] += holiday_count
                    except Exception as e:
                        logger.error(f"Failed to fetch market holidays: {e}")
                        results["holidays"]["failed"] = 1
                        context.add_error(f"Market holidays failed: {str(e)}")
                
                # 5. Economic calendar (if using Finnhub)
                if client.provider_name == "finnhub":
                    try:
                        events = await client.get_economic_calendar(
                            from_date=lookback_start,
                            to_date=lookahead_end,
                            currencies=["USD", "EUR", "GBP", "JPY", "CAD"]
                        )
                        econ_count = len(events) if events else 0
                        results["economic_calendar"]["success"] = econ_count
                        results["total_records"] += econ_count
                    except Exception as e:
                        logger.error(f"Failed to fetch economic calendar: {e}")
                        results["economic_calendar"]["failed"] = 1
        except Exception as e:
            logger.error(f"EventDataStep failed: {e}", exc_info=True)
            raise
        
        # Update metrics
        context.set_metric("event_earnings_records", results["earnings"]["success"])
        context.set_metric("event_earnings_from_db", results["earnings"]["from_db"])
        context.set_metric("event_earnings_from_yfinance", results["earnings"]["from_yfinance"])
        context.set_metric("event_dividend_records", results["dividends"]["success"])
        context.set_metric("event_split_records", results["splits"]["success"])
        context.set_metric("event_total_records", results["total_records"])
        
        # Log summary
        logger.info(
            f"Event data complete: {results['total_records']} total events "
            f"(Earnings: {results['earnings']['success']}, "
            f"Dividends: {results['dividends']['success']}, "
            f"Splits: {results['splits']['success']})"
        )
        
        return results