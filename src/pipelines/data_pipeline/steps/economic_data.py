"""Economic data pipeline step"""

from datetime import date, timedelta, datetime, timezone
from typing import Dict, Any
import asyncio

from src.pipelines.base import PipelineStep, PipelineContext
from src.clients.economic_data_client import EconomicDataClient
from src.data_definitions.economic_indicators import EconomicIndicators
from src.utils.logging import logger
from src.db.base import get_session
from src.db.models import EconomicDataModel
from sqlalchemy.dialects.postgresql import insert


class EconomicDataStep(PipelineStep[Dict[str, Any]]):
    """
    Step for fetching economic indicators data.
    Economic indicators are not symbol-specific but rather macro indicators.
    """
    
    def __init__(self):
        super().__init__(
            name="EconomicDataStep",
            description="Fetch economic indicators from FRED"
        )
        
    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Execute economic data fetching"""
        mode = context.data.get("mode", "daily")
        start_date = context.data.get("start_date")
        end_date = context.data.get("end_date")
        
        # Get all economic indicators
        indicators = EconomicIndicators.get_all_symbols()
        
        # For daily mode, focus on frequently updated indicators
        if mode == "daily":
            indicators = EconomicIndicators.get_daily_symbols()
            # Look back more days for economic data as it may have delays
            fetch_start = start_date - timedelta(days=7)
        else:
            fetch_start = start_date
            
        logger.info(f"Starting economic data {mode} for {len(indicators)} indicators")
        
        results = {
            "success": {},
            "failed": {},
            "total_records": 0
        }
        
        async with EconomicDataClient(provider="fred") as client:
            
            for indicator in indicators:
                try:
                    data = await client.get_indicator_data(
                        symbol=indicator,
                        start_date=fetch_start,
                        end_date=end_date
                    )
                    count = len(data) if data else 0
                    
                    # Save to database if we have data
                    if data:
                        async for db in get_session():
                            await self._save_economic_data(data, db)
                            break
                    
                    results["success"][indicator] = count
                    results["total_records"] += count
                    
                    # FRED has generous rate limits but let's be respectful
                    await asyncio.sleep(0.3)
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Failed to fetch {indicator}: {error_msg}")
                    results["failed"][indicator] = error_msg
                    context.add_error(f"Economic data failed for {indicator}: {error_msg}")
        
        # Update metrics
        context.set_metric("economic_data_records", results["total_records"])
        context.set_metric("economic_indicators_success", len(results["success"]))
        context.set_metric("economic_indicators_failed", len(results["failed"]))
        
        logger.info(
            f"Economic data complete: {len(results['success'])} indicators successful, "
            f"{len(results['failed'])} failed, {results['total_records']} total records"
        )
        
        return results
    
    async def _save_economic_data(self, data, db):
        """Save economic data to database with upsert logic"""
        try:
            # Prepare values for insert
            values = []
            for item in data:
                values.append({
                    "symbol": item.symbol,
                    "date": item.date,
                    "value": item.value,
                    "source": item.source,
                })
            
            # Use PostgreSQL upsert
            stmt = insert(EconomicDataModel).values(values)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_econ_symbol_date",
                set_={
                    "value": stmt.excluded.value,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            
            await db.execute(stmt)
            await db.commit()
            
        except Exception as e:
            logger.error(f"Error saving economic data: {e}")
            await db.rollback()
            raise