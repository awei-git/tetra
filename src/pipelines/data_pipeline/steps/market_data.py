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
        symbols = context.data.get("symbols")
        
        logger.debug(f"MarketDataStep context keys: {list(context.data.keys())}")
        logger.debug(f"Symbols in context: {symbols[:5] if symbols else 'None'}...")
        
        if symbols is None or len(symbols) == 0:
            logger.error(f"No symbols provided to MarketDataStep. Context keys: {list(context.data.keys())}")
            return {"success": {}, "failed": {}, "total_records": 0}
        
        start_date = context.data.get("start_date")
        end_date = context.data.get("end_date")
        
        logger.info(f"Starting market data {mode} for {len(symbols)} symbols")
        
        results = {
            "success": {},
            "failed": {},
            "total_records": 0
        }
        
        # DataIngester doesn't have context manager support, create instance directly
        ingester = DataIngester(provider="polygon")
        
        # Process symbols in batches to manage memory and API limits
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}")
            
            # For daily mode, also look back a bit to catch any gaps
            if mode == "daily":
                fetch_start = start_date - timedelta(days=context.data.get("days_back", 2))
            else:
                fetch_start = start_date
                
            # Fetch daily data using the correct method
            result = await ingester.ingest_ohlcv_batch(
                symbols=batch,
                from_date=fetch_start,
                to_date=end_date,
                timeframe="1d"
            )
            
            # Track results
            batch_records = result.get("total_records", 0)
            results["total_records"] += batch_records
            
            # Track success/failure per symbol
            symbols_processed = result.get("symbols_processed", 0)
            errors = result.get("errors", 0)
            
            # Assume symbols are processed successfully unless errors
            for symbol in batch[:symbols_processed]:
                results["success"][symbol] = batch_records // max(symbols_processed, 1)
            
            # For daily mode, also fetch intraday if configured
            if mode == "daily" and context.data.get("fetch_intraday", True):
                try:
                    intraday_result = await ingester.ingest_ohlcv_batch(
                        symbols=batch,
                        from_date=end_date,  # Only today
                        to_date=end_date,
                        timeframe="5m"
                    )
                    results["total_records"] += intraday_result.get("total_records", 0)
                except Exception as e:
                    logger.debug(f"Intraday fetch failed for batch: {e}")
            
            # Log batch progress
            logger.info(
                f"Batch complete: {symbols_processed} symbols processed, "
                f"{batch_records} records, {errors} errors"
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