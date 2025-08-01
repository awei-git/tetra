"""Event data pipeline step"""

from datetime import date, timedelta
from typing import Dict, Any
import asyncio

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
        
    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Execute event data fetching"""
        mode = context.data.get("mode", "daily")
        symbols = context.data.get("symbols", [])
        start_date = context.data.get("start_date")
        end_date = context.data.get("end_date")
        
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
            "earnings": {"success": 0, "failed": 0},
            "dividends": {"success": 0, "failed": 0},
            "splits": {"success": 0, "failed": 0},
            "holidays": {"success": 0, "failed": 0},
            "economic_calendar": {"success": 0, "failed": 0},
            "total_records": 0
        }
        
        async with EventDataClient(provider="yahoo") as client:
                
                # 1. Fetch earnings calendar
                try:
                    earnings = await client.get_earnings_calendar(
                        from_date=lookback_start,
                        to_date=lookahead_end,
                        symbols=symbols  # Will fetch for all provided symbols
                    )
                    earnings_count = len(earnings) if earnings else 0
                    results["earnings"]["success"] = earnings_count
                    results["total_records"] += earnings_count
                except Exception as e:
                    logger.error(f"Failed to fetch earnings calendar: {e}")
                    results["earnings"]["failed"] = 1
                    context.add_error(f"Earnings calendar failed: {str(e)}")
                
                # 2. Fetch dividends and splits for each symbol
                # Process in batches to avoid overwhelming the API
                batch_size = 100
                for i in range(0, len(symbols), batch_size):
                    batch = symbols[i:i + batch_size]
                    
                    for symbol in batch:
                        # Dividends
                        try:
                            # For now, dividends not implemented
                            div_count = 0
                            if div_count > 0:
                                results["dividends"]["success"] += 1
                                results["total_records"] += div_count
                        except Exception as e:
                            results["dividends"]["failed"] += 1
                            logger.debug(f"Failed to fetch dividends for {symbol}: {e}")
                        
                        # Splits
                        try:
                            # For now, splits not implemented
                            split_count = 0
                            if split_count > 0:
                                results["splits"]["success"] += 1
                                results["total_records"] += split_count
                        except Exception as e:
                            results["splits"]["failed"] += 1
                            logger.debug(f"Failed to fetch splits for {symbol}: {e}")
                        
                        await asyncio.sleep(0.1)  # Rate limiting
                
                # 3. Market holidays (not symbol-specific)
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
                
                # 4. Economic calendar (if using Finnhub)
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
        
        # Update metrics
        context.set_metric("event_earnings_records", results["earnings"]["success"])
        context.set_metric("event_dividend_records", results["dividends"]["success"])
        context.set_metric("event_split_records", results["splits"]["success"])
        context.set_metric("event_total_records", results["total_records"])
        
        logger.info(
            f"Event data complete: {results['total_records']} total events fetched"
        )
        
        return results