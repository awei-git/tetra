"""Data quality check pipeline step"""

from datetime import date, timedelta
from typing import Dict, Any, List

from src.pipelines.base import PipelineStep, PipelineContext
from src.db.base import get_session
from sqlalchemy import text
from src.utils.logging import logger


class DataQualityCheckStep(PipelineStep[Dict[str, Any]]):
    """
    Step for checking data quality and completeness.
    Identifies gaps, stale data, and missing symbols.
    """
    
    def __init__(self):
        super().__init__(
            name="DataQualityCheckStep",
            description="Verify data quality and completeness"
        )
        
    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Execute data quality checks"""
        logger.info("Running data quality checks")
        
        symbols = context.data.get("symbols", [])
        mode = context.data.get("mode", "daily")
        today = date.today()
        
        checks = {
            "market_data_gaps": [],
            "stale_data": [],
            "missing_symbols": [],
            "low_data_symbols": [],
            "summary": {}
        }
        
        async for db in get_session():
            # 1. Check market data coverage
            query = """
            WITH symbol_stats AS (
                SELECT 
                    symbol,
                    COUNT(*) as record_count,
                    MIN(timestamp::date) as first_date,
                    MAX(timestamp::date) as last_date,
                    COUNT(DISTINCT timestamp::date) as days_with_data
                FROM market_data.aggregates
                WHERE symbol = ANY(:symbols)
                    AND timeframe = '1Day'
                GROUP BY symbol
            )
            SELECT * FROM symbol_stats
            """
            
            result = await db.execute(text(query), {"symbols": symbols})
            results = result.fetchall()
            symbol_data = {row['symbol']: row for row in results}
            
            # Check each symbol
            for symbol in symbols:
                if symbol not in symbol_data:
                    checks["missing_symbols"].append(symbol)
                else:
                    data = symbol_data[symbol]
                    last_date = data['last_date']
                    days_behind = (today - last_date).days
                    
                    # Check for stale data (more than 2 business days old)
                    if days_behind > 2:
                        checks["stale_data"].append({
                            "symbol": symbol,
                            "last_date": last_date.isoformat(),
                            "days_behind": days_behind
                        })
                    
                    # Check for low data (less than 10 days of data)
                    if data['days_with_data'] < 10:
                        checks["low_data_symbols"].append({
                            "symbol": symbol,
                            "days_with_data": data['days_with_data'],
                            "first_date": data['first_date'].isoformat(),
                            "last_date": data['last_date'].isoformat()
                        })
            
            # 2. Check for data gaps in recent history
            if mode == "daily":
                gap_query = """
                WITH date_series AS (
                    SELECT generate_series(
                        CURRENT_DATE - INTERVAL '30 days',
                        CURRENT_DATE,
                        '1 day'::interval
                    )::date AS expected_date
                ),
                symbol_dates AS (
                    SELECT DISTINCT 
                        symbol,
                        timestamp::date as data_date
                    FROM market_data.aggregates
                    WHERE symbol = ANY(:symbols1)
                        AND timeframe = '1Day'
                        AND timestamp >= CURRENT_DATE - INTERVAL '30 days'
                )
                SELECT 
                    s.symbol,
                    COUNT(*) FILTER (WHERE sd.data_date IS NULL) as missing_days
                FROM (SELECT DISTINCT symbol FROM market_data.aggregates WHERE symbol = ANY(:symbols2)) s
                CROSS JOIN date_series ds
                LEFT JOIN symbol_dates sd ON s.symbol = sd.symbol AND ds.expected_date = sd.data_date
                WHERE EXTRACT(dow FROM ds.expected_date) NOT IN (0, 6)  -- Exclude weekends
                GROUP BY s.symbol
                HAVING COUNT(*) FILTER (WHERE sd.data_date IS NULL) > 0
                """
                
                result = await db.execute(text(gap_query), {"symbols1": symbols, "symbols2": symbols})
                gap_results = result.fetchall()
                for row in gap_results:
                    if row['missing_days'] > 0:
                        checks["market_data_gaps"].append({
                            "symbol": row['symbol'],
                            "missing_days": row['missing_days']
                        })
            
            # 3. Summary statistics
            total_query = """
            SELECT 
                COUNT(DISTINCT symbol) as unique_symbols,
                COUNT(*) as total_records,
                MIN(timestamp) as earliest_data,
                MAX(timestamp) as latest_data
            FROM market_data.aggregates
            WHERE symbol = ANY(:symbols)
            """
            
            result = await db.execute(text(total_query), {"symbols": symbols})
            summary = result.fetchone()
            checks["summary"] = {
                "total_symbols_requested": len(symbols),
                "symbols_with_data": len(symbol_data),
                "symbols_missing": len(checks["missing_symbols"]),
                "symbols_stale": len(checks["stale_data"]),
                "symbols_with_gaps": len(checks["market_data_gaps"]),
                "total_records": summary['total_records'] or 0,
                "date_range": f"{summary['earliest_data']} to {summary['latest_data']}" 
                              if summary['earliest_data'] else "No data"
            }
        
        # Update context with quality results
        context.data["quality_checks"] = checks
        context.set_metric("missing_symbols_count", len(checks["missing_symbols"]))
        context.set_metric("stale_symbols_count", len(checks["stale_data"]))
        context.set_metric("symbols_with_gaps", len(checks["market_data_gaps"]))
        
        # Log warnings
        if checks["missing_symbols"]:
            logger.warning(
                f"Found {len(checks['missing_symbols'])} symbols with no data: "
                f"{', '.join(checks['missing_symbols'][:10])}"
                f"{'...' if len(checks['missing_symbols']) > 10 else ''}"
            )
            
        if checks["stale_data"]:
            logger.warning(
                f"Found {len(checks['stale_data'])} symbols with stale data"
            )
            
        if checks["market_data_gaps"]:
            logger.warning(
                f"Found {len(checks['market_data_gaps'])} symbols with data gaps"
            )
        
        logger.info(f"Data quality check complete: {checks['summary']}")
        
        return checks