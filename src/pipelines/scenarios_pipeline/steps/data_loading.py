"""
Data loading step for Scenarios Pipeline.
"""

import logging
from typing import Any
from src.pipelines.base import PipelineStep, PipelineContext

logger = logging.getLogger(__name__)


class DataLoadingStep(PipelineStep):
    """Load historical market data for scenario generation."""
    
    def __init__(self):
        super().__init__("Data Loading", "Load historical market data")
    
    async def execute(self, context: PipelineContext) -> Any:
        """Load data from database or API."""
        logger.info("Loading historical market data...")
        
        from src.db import async_session_maker
        from sqlalchemy import text
        import pandas as pd
        
        # Get date range from context
        start_date = context.data.get('start_date')
        end_date = context.data.get('end_date')
        
        # Get symbols to load - use what's available in database
        query = text("""
            SELECT DISTINCT symbol 
            FROM market_data.ohlcv 
            WHERE timestamp >= :start_date 
                AND timestamp <= :end_date
            ORDER BY symbol
        """)
        
        async with async_session_maker() as session:
            result = await session.execute(
                query,
                {"start_date": start_date, "end_date": end_date}
            )
            symbols = [row[0] for row in result.fetchall()]
        
        logger.info(f"Found {len(symbols)} symbols with data in date range")
        
        # Load OHLCV data for all symbols
        market_data = {}
        total_records = 0
        
        if symbols:
            # Load data in batches to avoid memory issues
            batch_size = 50
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                
                query = text("""
                    SELECT 
                        symbol,
                        timestamp as date,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        vwap
                    FROM market_data.ohlcv
                    WHERE symbol = ANY(:symbols)
                        AND timestamp >= :start_date
                        AND timestamp <= :end_date
                    ORDER BY symbol, timestamp
                """
                )
                
                async with async_session_maker() as session:
                    result = await session.execute(
                        query,
                        {
                            "symbols": batch_symbols,
                            "start_date": start_date,
                            "end_date": end_date
                        }
                    )
                    
                    rows = result.fetchall()
                    
                    # Group by symbol and convert to DataFrames
                    symbol_data = {}
                    for row in rows:
                        symbol = row.symbol
                        if symbol not in symbol_data:
                            symbol_data[symbol] = []
                        
                        symbol_data[symbol].append({
                            'date': row.date,
                            'open': float(row.open) if row.open else None,
                            'high': float(row.high) if row.high else None,
                            'low': float(row.low) if row.low else None,
                            'close': float(row.close) if row.close else None,
                            'volume': float(row.volume) if row.volume else None,
                            'vwap': float(row.vwap) if row.vwap else None
                        })
                        total_records += 1
                    
                    # Convert to DataFrames
                    for symbol, data in symbol_data.items():
                        if data:
                            df = pd.DataFrame(data)
                            # Calculate returns for convenience
                            if 'close' in df.columns:
                                df['returns'] = df['close'].pct_change()
                            if symbol not in market_data:
                                market_data[symbol] = df
                            else:
                                market_data[symbol] = pd.concat([market_data[symbol], df], ignore_index=True)
        
        context.data['market_data'] = market_data
        context.data['market_data_loaded'] = True
        context.data['available_symbols'] = symbols
        context.set_metric('data_records_loaded', total_records)
        
        logger.info(f"Loaded {total_records} records for {len(symbols)} symbols")
        
        return context