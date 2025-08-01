"""Unified data pipeline that handles both daily updates and historical backfills"""

from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any

from src.pipelines.base import Pipeline, PipelineContext, PipelineStatus, PipelineStep, ParallelStep
from src.pipelines.data_pipeline.steps import (
    MarketDataStep,
    EconomicDataStep,
    EventDataStep,
    NewsSentimentStep,
    DataQualityCheckStep
)
from src.data_definitions.market_universe import MarketUniverse
from src.db.base import get_session
from src.utils.logging import logger
from sqlalchemy import text


class DataPipeline(Pipeline):
    """
    Unified data pipeline for ingesting all data types.
    
    Supports two modes:
    - daily: Fetch latest data (start_date = end_date = today or specified date)
    - backfill: Fetch historical data for a date range
    """
    
    def __init__(self):
        super().__init__(
            name="DataPipeline",
            description="Ingest market data, economic indicators, events, and news"
        )
        
    def _configure_steps(self, context: PipelineContext):
        """Configure pipeline steps based on mode and parameters"""
        mode = context.data.get("mode", "daily")
        skip_steps = context.data.get("skip_steps", [])
        parallel = context.data.get("parallel", True)
        
        # Create steps with appropriate configuration
        all_steps = []
        
        if "market_data" not in skip_steps:
            all_steps.append(MarketDataStep())
            
        if "economic_data" not in skip_steps:
            all_steps.append(EconomicDataStep())
            
        if "event_data" not in skip_steps:
            all_steps.append(EventDataStep())
            
        if "news_sentiment" not in skip_steps:
            all_steps.append(NewsSentimentStep())
        
        # Add steps based on execution mode
        if parallel and len(all_steps) > 1:
            # Run data ingestion steps in parallel
            self.add_step(ParallelStep(all_steps, name="ParallelDataIngestion"))
        else:
            # Run sequentially
            self.add_steps(all_steps)
            
        # Always run quality check at the end (unless skipped)
        if "data_quality" not in skip_steps:
            self.add_step(DataQualityCheckStep())
            
    async def _get_universe_symbols(self) -> List[str]:
        """Get all symbols from the market universe"""
        # First try to get from MarketUniverse data definition
        all_symbols = MarketUniverse.get_all_symbols()
        
        # Optionally, also fetch from database if we have a symbols table
        try:
            async for db in get_session():
                query = """
                SELECT DISTINCT symbol 
                FROM market_data.symbols 
                WHERE is_active = true
                ORDER BY symbol
                """
                result = await db.execute(text(query))
                rows = result.fetchall()
                db_symbols = [row['symbol'] for row in rows]
                
                # Combine both sources
                if db_symbols:
                    all_symbols = list(set(all_symbols + db_symbols))
        except Exception as e:
            logger.warning(f"Could not fetch symbols from database: {e}")
            
        logger.info(f"Total universe contains {len(all_symbols)} symbols")
        return sorted(all_symbols)
        
    async def setup(self) -> PipelineContext:
        """Setup pipeline context"""
        context = PipelineContext()
        context.data["pipeline_start_time"] = datetime.now()
        
        # Determine mode and date range
        mode = context.data.get("mode", "daily")
        
        if mode == "daily":
            # Daily mode: use today or specified date
            target_date = context.data.get("start_date", date.today())
            context.data["start_date"] = target_date
            context.data["end_date"] = target_date
            context.data["days_back"] = 2  # Look back 2 days to catch any gaps
            
            # Daily defaults
            context.data.setdefault("news_provider", "newsapi")
            context.data.setdefault("fetch_intraday", True)
            
        else:  # backfill mode
            # Backfill mode: use specified range
            start_date = context.data.get("start_date")
            end_date = context.data.get("end_date", date.today())
            
            if not start_date:
                raise ValueError("Backfill mode requires start_date")
                
            context.data["start_date"] = start_date
            context.data["end_date"] = end_date
            context.data["days_back"] = (end_date - start_date).days
            
            # Backfill defaults
            context.data.setdefault("news_provider", "alphavantage")
            context.data.setdefault("fetch_intraday", False)
        
        # Get symbols - use provided list or fetch entire universe
        if "symbols" not in context.data or context.data["symbols"] is None:
            # Fetch entire universe
            context.data["symbols"] = await self._get_universe_symbols()
        else:
            # Use provided symbols
            logger.info(f"Using provided symbols: {len(context.data['symbols'])} symbols")
        
        # For news in daily mode, we might want to limit symbols due to API constraints
        if mode == "daily" and "news_sentiment" not in context.data.get("skip_steps", []):
            # Store full symbol list
            context.data["all_symbols"] = context.data["symbols"]
            
            # For news, use high priority symbols only
            context.data["news_symbols"] = MarketUniverse.get_high_priority_symbols()
            logger.info(f"News will be fetched for {len(context.data['news_symbols'])} high-priority symbols")
        
        # Configure steps based on mode
        self._configure_steps(context)
        
        logger.info(
            f"Starting {mode} pipeline: "
            f"{context.data['start_date']} to {context.data['end_date']} "
            f"for {len(context.data['symbols'])} symbols"
        )
        
        return context
        
    async def teardown(self, context: PipelineContext):
        """Cleanup and report results"""
        mode = context.data.get("mode", "daily")
        
        logger.info(f"=== {mode.title()} Pipeline Summary ===")
        logger.info(f"Status: {context.status}")
        logger.info(f"Duration: {context.metrics.get('duration_seconds', 0):.2f} seconds")
        logger.info(f"Symbols processed: {len(context.data.get('symbols', []))}")
        
        # Report record counts
        total_records = 0
        for key, value in context.metrics.items():
            if key.endswith("_records") or key in ["news_articles", "news_sentiments"]:
                total_records += value
                
        logger.info(f"Total records processed: {total_records}")
        
        # Detailed metrics
        if context.metrics:
            logger.info("\nDetailed metrics:")
            for key, value in sorted(context.metrics.items()):
                if key.endswith("_records") or key in ["news_articles", "news_sentiments"]:
                    logger.info(f"  {key}: {value}")
                    
        # Quality check results
        quality_checks = context.data.get("quality_checks", {})
        if quality_checks:
            if quality_checks.get("stale_data"):
                logger.warning(
                    f"Found {len(quality_checks['stale_data'])} symbols with stale data"
                )
            if quality_checks.get("missing_symbols"):
                logger.warning(
                    f"Missing data for symbols: {quality_checks['missing_symbols']}"
                )
                
        # Errors summary
        if context.errors:
            logger.error(f"\nErrors encountered: {len(context.errors)}")
            for i, error in enumerate(context.errors[:5], 1):
                logger.error(f"{i}. {error}")
            if len(context.errors) > 5:
                logger.error(f"... and {len(context.errors) - 5} more errors")
                
    def should_continue_on_error(self, step: PipelineStep, error: Exception) -> bool:
        """Continue pipeline even if a step fails"""
        # Always continue - we want to get as much data as possible
        return True