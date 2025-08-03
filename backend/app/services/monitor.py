from typing import Dict, Any, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MonitorService:
    """Service for monitoring database coverage and statistics"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    async def get_data_coverage(self) -> Dict[str, Any]:
        """Get coverage statistics for all data topics"""
        # Return mock data if database is not available
        if not self.db:
            return {
                "market_data": {
                    "name": "Market Data",
                    "status": "healthy",
                    "record_count": 1425000,
                    "date_range": {"start": "2014-01-01", "end": "2024-12-01"},
                    "tables": ["ohlcv", "ticks", "quotes"],
                    "symbols": 153
                },
                "economic_data": {
                    "name": "Economic Data",
                    "status": "healthy",
                    "record_count": 25000,
                    "date_range": {"start": "2010-01-01", "end": "2024-12-01"},
                    "tables": ["economic_data", "releases", "forecasts"],
                    "symbols": 15
                },
                "news": {
                    "name": "News",
                    "status": "healthy",
                    "record_count": 150000,
                    "date_range": {"start": "2023-01-01", "end": "2024-12-01"},
                    "tables": ["news_articles", "news_sentiments", "news_clusters"],
                    "symbols": None
                },
                "events": {
                    "name": "Events",
                    "status": "healthy",
                    "record_count": 50000,
                    "date_range": {"start": "2020-01-01", "end": "2024-12-01"},
                    "tables": ["event_data", "currency_events", "economic_events", "earnings_events"],
                    "symbols": None
                }
            }
        
        coverage = {}
        
        # Market data coverage
        coverage["market_data"] = await self._get_market_data_coverage()
        
        # Economic data coverage
        coverage["economic_data"] = await self._get_economic_data_coverage()
        
        # News coverage
        coverage["news"] = await self._get_news_coverage()
        
        # Events coverage
        coverage["events"] = await self._get_events_coverage()
        
        return coverage
    
    async def _get_market_data_coverage(self) -> Dict[str, Any]:
        """Get market data coverage statistics"""
        query = """
            SELECT 
                COUNT(DISTINCT symbol) as unique_symbols,
                COUNT(*) as total_records,
                MIN(timestamp)::date as earliest_date,
                MAX(timestamp)::date as latest_date
            FROM market_data.ohlcv
        """
        
        result = await self.db.fetchrow(query)
        
        return {
            "name": "Market Data",
            "status": "healthy" if result["total_records"] > 0 else "empty",
            "record_count": result["total_records"] or 0,
            "date_range": {
                "start": str(result["earliest_date"]) if result["earliest_date"] else None,
                "end": str(result["latest_date"]) if result["latest_date"] else None
            },
            "tables": ["ohlcv", "ticks", "quotes"],
            "symbols": result["unique_symbols"] or 0
        }
    
    async def _get_economic_data_coverage(self) -> Dict[str, Any]:
        """Get economic data coverage statistics"""
        query = """
            SELECT 
                COUNT(DISTINCT symbol) as indicators,
                COUNT(*) as total_records,
                MIN(date) as earliest_date,
                MAX(date) as latest_date
            FROM economic_data.economic_data
        """
        
        result = await self.db.fetchrow(query)
        
        return {
            "name": "Economic Data",
            "status": "healthy" if result["total_records"] > 0 else "empty",
            "record_count": result["total_records"] or 0,
            "date_range": {
                "start": str(result["earliest_date"]) if result["earliest_date"] else None,
                "end": str(result["latest_date"]) if result["latest_date"] else None
            },
            "tables": ["economic_data", "releases", "forecasts"],
            "symbols": result["indicators"] or 0
        }
    
    async def _get_news_coverage(self) -> Dict[str, Any]:
        """Get news coverage statistics"""
        query = """
            SELECT 
                COUNT(*) as total_articles,
                COUNT(DISTINCT source) as unique_sources,
                MIN(published_at)::date as earliest,
                MAX(published_at)::date as latest
            FROM news.news_articles
        """
        
        result = await self.db.fetchrow(query)
        
        return {
            "name": "News",
            "status": "healthy" if result["total_articles"] > 0 else "empty",
            "record_count": result["total_articles"] or 0,
            "date_range": {
                "start": str(result["earliest"]) if result["earliest"] else None,
                "end": str(result["latest"]) if result["latest"] else None
            },
            "tables": ["news_articles", "news_sentiments", "news_clusters"],
            "symbols": None,
            "sources": result["unique_sources"] or 0
        }
    
    async def _get_events_coverage(self) -> Dict[str, Any]:
        """Get events coverage statistics"""
        query = """
            SELECT 
                COUNT(*) as total_events,
                COUNT(DISTINCT event_type) as event_types,
                MIN(event_datetime)::date as earliest,
                MAX(event_datetime)::date as latest
            FROM events.event_data
        """
        
        result = await self.db.fetchrow(query)
        
        return {
            "name": "Events",
            "status": "healthy" if result["total_events"] > 0 else "empty",
            "record_count": result["total_events"] or 0,
            "date_range": {
                "start": str(result["earliest"]) if result["earliest"] else None,
                "end": str(result["latest"]) if result["latest"] else None
            },
            "tables": ["event_data", "currency_events", "economic_events", "earnings_events"],
            "symbols": None,
            "event_types": result["event_types"] or 0
        }
    
    async def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information"""
        query = """
            SELECT 
                table_schema,
                table_name,
                column_name,
                data_type,
                is_nullable
            FROM information_schema.columns
            WHERE table_schema IN ('market_data', 'economic_data', 'news', 'events')
            ORDER BY table_schema, table_name, ordinal_position
        """
        
        rows = await self.db.fetch(query)
        
        # Group by schema and table
        schemas = {}
        for row in rows:
            schema_name = row["table_schema"]
            table_name = row["table_name"]
            
            if schema_name not in schemas:
                schemas[schema_name] = {"name": schema_name, "tables": {}}
            
            if table_name not in schemas[schema_name]["tables"]:
                schemas[schema_name]["tables"][table_name] = {
                    "name": table_name,
                    "columns": []
                }
            
            schemas[schema_name]["tables"][table_name]["columns"].append({
                "name": row["column_name"],
                "type": row["data_type"],
                "nullable": row["is_nullable"] == "YES"
            })
        
        # Convert to list format
        return {
            "schemas": [
                {
                    "name": schema_name,
                    "tables": list(schema_data["tables"].values())
                }
                for schema_name, schema_data in schemas.items()
            ]
        }
    
    async def get_schema_statistics(self, schema_name: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific schema"""
        # Table sizes
        query = f"""
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                n_live_tup as row_count
            FROM pg_stat_user_tables
            WHERE schemaname = $1
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """
        
        tables = await self.db.fetch(query, schema_name)
        
        return {
            "schema": schema_name,
            "tables": [
                {
                    "name": table["tablename"],
                    "size": table["size"],
                    "row_count": table["row_count"]
                }
                for table in tables
            ],
            "total_size": await self._get_schema_size(schema_name)
        }
    
    async def _get_schema_size(self, schema_name: str) -> str:
        """Get total size of a schema"""
        query = """
            SELECT pg_size_pretty(
                SUM(pg_total_relation_size(schemaname||'.'||tablename))
            ) as total_size
            FROM pg_stat_user_tables
            WHERE schemaname = $1
        """
        
        result = await self.db.fetchrow(query, schema_name)
        return result["total_size"] or "0 bytes"
    
    async def get_symbol_details(self, schema: str = "market_data") -> List[Dict[str, Any]]:
        """Get detailed coverage information for each symbol"""
        if schema == "news":
            query = """
                SELECT 
                    source,
                    COUNT(*) as article_count,
                    MIN(published_at)::date as start_date,
                    MAX(published_at)::date as end_date,
                    COUNT(DISTINCT DATE_TRUNC('day', published_at)) as days_with_articles
                FROM news.news_articles
                GROUP BY source
                ORDER BY article_count DESC
            """
            rows = await self.db.fetch(query)
            return [
                {
                    "source": row["source"],
                    "article_count": row["article_count"],
                    "date_range": {
                        "start": str(row["start_date"]),
                        "end": str(row["end_date"])
                    },
                    "days_with_articles": row["days_with_articles"],
                    "coverage_quality": "good" if row["article_count"] > 100 else "fair",
                    "type": "news"
                }
                for row in rows
            ]
        elif schema == "events":
            query = """
                SELECT 
                    event_type,
                    COUNT(*) as event_count,
                    MIN(event_datetime)::date as start_date,
                    MAX(event_datetime)::date as end_date,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    COUNT(DISTINCT country) as unique_countries
                FROM events.event_data
                GROUP BY event_type
                ORDER BY event_count DESC
            """
            rows = await self.db.fetch(query)
            return [
                {
                    "event_type": row["event_type"],
                    "event_count": row["event_count"],
                    "date_range": {
                        "start": str(row["start_date"]),
                        "end": str(row["end_date"])
                    },
                    "unique_symbols": row["unique_symbols"],
                    "unique_countries": row["unique_countries"],
                    "coverage_quality": "good" if row["event_count"] > 1000 else "fair",
                    "type": "events"
                }
                for row in rows
            ]
        elif schema == "market_data":
            query = """
                SELECT 
                    symbol,
                    COUNT(*) as record_count,
                    MIN(timestamp)::date as start_date,
                    MAX(timestamp)::date as end_date,
                    COUNT(DISTINCT DATE_TRUNC('day', timestamp)) as trading_days,
                    -- Calculate missing trading days (252 trading days per year)
                    ROUND(
                        GREATEST(0, 
                            ((MAX(timestamp)::date - MIN(timestamp)::date + 1) * 252.0 / 365.0) 
                            - COUNT(DISTINCT DATE_TRUNC('day', timestamp))
                        )::numeric, 0
                    ) as missing_days
                FROM market_data.ohlcv
                GROUP BY symbol
                ORDER BY symbol
            """
        elif schema == "economic_data":
            query = """
                WITH date_gaps AS (
                    SELECT 
                        symbol,
                        date,
                        LEAD(date) OVER (PARTITION BY symbol ORDER BY date) as next_date,
                        EXTRACT(EPOCH FROM (LEAD(date) OVER (PARTITION BY symbol ORDER BY date) - date)) / 86400 as days_between
                    FROM economic_data.economic_data
                ),
                symbol_stats AS (
                    SELECT 
                        symbol,
                        COUNT(*) as record_count,
                        MIN(date) as start_date,
                        MAX(date) as end_date,
                        COUNT(DISTINCT date) as data_points,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY days_between) as median_days_between
                    FROM date_gaps
                    WHERE days_between IS NOT NULL
                    GROUP BY symbol
                )
                SELECT 
                    symbol,
                    record_count,
                    start_date,
                    end_date,
                    data_points,
                    CASE
                        WHEN median_days_between <= 1.5 THEN 'Daily'
                        WHEN median_days_between <= 8 THEN 'Weekly'
                        WHEN median_days_between <= 35 THEN 'Monthly'
                        WHEN median_days_between <= 95 THEN 'Quarterly'
                        ELSE 'Annual'
                    END as frequency,
                    CASE
                        WHEN median_days_between <= 1.5 THEN 
                            GREATEST(0, (EXTRACT(EPOCH FROM (end_date - start_date)) / 86400 + 1)::int - data_points)
                        WHEN median_days_between <= 8 THEN 
                            GREATEST(0, (EXTRACT(EPOCH FROM (end_date - start_date)) / 86400 / 7)::int - data_points)
                        WHEN median_days_between <= 35 THEN 
                            GREATEST(0, (EXTRACT(EPOCH FROM (end_date - start_date)) / 86400 / 30)::int - data_points)
                        WHEN median_days_between <= 95 THEN 
                            GREATEST(0, (EXTRACT(EPOCH FROM (end_date - start_date)) / 86400 / 90)::int - data_points)
                        ELSE 
                            GREATEST(0, (EXTRACT(EPOCH FROM (end_date - start_date)) / 86400 / 365)::int - data_points)
                    END as missing_points
                FROM symbol_stats
                ORDER BY symbol
            """
        else:
            return []
        
        rows = await self.db.fetch(query)
        
        if schema == "market_data":
            return [
                {
                    "symbol": row["symbol"],
                    "record_count": row["record_count"],
                    "date_range": {
                        "start": str(row["start_date"]),
                        "end": str(row["end_date"])
                    },
                    "trading_days": row.get("trading_days"),
                    "missing_days": int(row["missing_days"]) if row.get("missing_days") is not None else None,
                    "coverage_quality": self._calculate_coverage_quality(row),
                    "type": "market"
                }
                for row in rows
            ]
        else:  # economic_data
            return [
                {
                    "symbol": row["symbol"],
                    "record_count": row["record_count"],
                    "date_range": {
                        "start": str(row["start_date"]),
                        "end": str(row["end_date"])
                    },
                    "data_points": row.get("data_points"),
                    "frequency": row.get("frequency"),
                    "missing_points": row.get("missing_points", 0),
                    "coverage_quality": self._calculate_economic_coverage_quality(row),
                    "type": "economic"
                }
                for row in rows
            ]
    
    def _calculate_coverage_quality(self, row: Dict[str, Any]) -> str:
        """Calculate coverage quality based on missing days"""
        if "missing_days" not in row or row["missing_days"] is None:
            return "unknown"
        
        missing_days = row["missing_days"]
        if missing_days == 0:
            return "excellent"
        elif missing_days < 10:
            return "good"
        elif missing_days < 50:
            return "fair"
        else:
            return "poor"
    
    def _calculate_economic_coverage_quality(self, row: Dict[str, Any]) -> str:
        """Calculate coverage quality for economic data based on data points"""
        if "data_points" not in row or row["data_points"] is None:
            return "unknown"
        
        # For economic data, quality is based on data consistency
        data_points = row["data_points"]
        if data_points > 100:
            return "excellent"
        elif data_points > 50:
            return "good"
        elif data_points > 20:
            return "fair"
        else:
            return "poor"