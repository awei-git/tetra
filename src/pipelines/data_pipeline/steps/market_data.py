"""Market data pipeline step"""

from datetime import date, timedelta
from typing import Dict, List, Any
import asyncio

from src.pipelines.base import PipelineStep, PipelineContext
from src.ingestion.data_ingester import DataIngester
from src.utils.logging import logger


class MarketDataStep(PipelineStep[Dict[str, Any]]):
    """
    Step for fetching market data (OHLCV).
    Adapts behavior based on pipeline mode (daily vs backfill).
    """
    
    def __init__(self):
        super().__init__(
            name="MarketDataStep",
            description="Fetch OHLCV data for all symbols"
        )
        
    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Execute market data fetching"""
        mode = context.data.get("mode", "daily")
        symbols = context.data.get("symbols", [])
        start_date = context.data.get("start_date")
        end_date = context.data.get("end_date")
        
        logger.info(f"Starting market data {mode} for {len(symbols)} symbols")
        
        results = {
            "success": {},
            "failed": {},
            "total_records": 0
        }
        
        async with DataIngester(provider="polygon") as ingestion:
            
            # Process symbols in batches to manage memory and API limits
            batch_size = 50
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}")
                
                for symbol in batch:
                    try:
                        # For daily mode, also look back a bit to catch any gaps
                        if mode == "daily":
                            fetch_start = start_date - timedelta(days=context.data.get("days_back", 2))
                        else:
                            fetch_start = start_date
                            
                        # Fetch daily data
                        result = await ingestion.fetch_ohlcv(
                            symbols=[symbol],
                            from_date=fetch_start,
                            to_date=end_date,
                            timeframe="1d"
                        )
                        daily_count = result.get("inserted", 0) + result.get("updated", 0)
                        
                        intraday_count = 0
                        # For daily mode, also fetch intraday if configured
                        if mode == "daily" and context.data.get("fetch_intraday", True):
                            try:
                                result = await ingestion.fetch_ohlcv(
                                    symbols=[symbol],
                                    from_date=end_date,  # Only today
                                    to_date=end_date,
                                    timeframe="5m"
                                )
                                intraday_count = result.get("inserted", 0) + result.get("updated", 0)
                            except Exception as e:
                                logger.debug(f"Intraday fetch failed for {symbol}: {e}")
                        
                        total_count = daily_count + intraday_count
                        results["success"][symbol] = total_count
                        results["total_records"] += total_count
                        
                        # Small delay to respect rate limits
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"Failed to fetch data for {symbol}: {error_msg}")
                        results["failed"][symbol] = error_msg
                        context.add_error(f"Market data failed for {symbol}: {error_msg}")
                
                # Log batch progress
                logger.info(
                    f"Batch complete: {len([s for s in batch if s in results['success']])} success, "
                    f"{len([s for s in batch if s in results['failed']])} failed"
                )
        
        # Update metrics
        context.set_metric("market_data_records", results["total_records"])
        context.set_metric("market_data_symbols_success", len(results["success"]))
        context.set_metric("market_data_symbols_failed", len(results["failed"]))
        
        logger.info(
            f"Market data complete: {len(results['success'])} symbols successful, "
            f"{len(results['failed'])} failed, {results['total_records']} total records"
        )
        
        return results