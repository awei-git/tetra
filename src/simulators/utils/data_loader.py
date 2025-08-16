"""Efficient data loading utilities."""

from datetime import date, datetime
from typing import Dict, List, Optional, Any
import pandas as pd
from sqlalchemy import text
import asyncio

from ...db.base import async_session_maker


class DataLoader:
    """Efficient data loading for simulations."""
    
    def __init__(self, batch_size: int = 1000):
        """
        Initialize data loader.
        
        Args:
            batch_size: Number of rows to fetch at once
        """
        self.batch_size = batch_size
        
    async def load_ohlcv_batch(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data for multiple symbols.
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Dict mapping symbols to DataFrames
        """
        # Split into batches to avoid query size limits
        symbol_batches = [
            symbols[i:i + 50] 
            for i in range(0, len(symbols), 50)
        ]
        
        all_data = {}
        
        for batch in symbol_batches:
            batch_data = await self._load_batch(batch, start_date, end_date)
            all_data.update(batch_data)
            
        return all_data
    
    async def _load_batch(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Dict[str, pd.DataFrame]:
        """Load a batch of symbols."""
        query = text("""
            SELECT 
                symbol,
                timestamp,
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
        """)
        
        async with async_session_maker() as session:
            result = await session.execute(
                query,
                {
                    "symbols": symbols,
                    "start_date": start_date,
                    "end_date": end_date
                }
            )
            
            rows = result.fetchall()
        
        # Group by symbol
        data_by_symbol = {}
        for row in rows:
            symbol = row.symbol
            if symbol not in data_by_symbol:
                data_by_symbol[symbol] = []
            data_by_symbol[symbol].append(row)
        
        # Convert to DataFrames
        result = {}
        for symbol, symbol_rows in data_by_symbol.items():
            df = pd.DataFrame(symbol_rows)
            df.set_index('timestamp', inplace=True)
            df.drop('symbol', axis=1, inplace=True)
            df.index = pd.to_datetime(df.index)
            
            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            result[symbol] = df
            
        return result
    
    async def load_economic_data(
        self,
        indicators: List[str],
        start_date: date,
        end_date: date
    ) -> Dict[str, pd.DataFrame]:
        """Load economic indicator data."""
        query = text("""
            SELECT 
                symbol,
                date,
                value
            FROM economic_data.economic_data
            WHERE symbol = ANY(:indicators)
                AND date >= :start_date
                AND date <= :end_date
            ORDER BY symbol, date
        """)
        
        async with async_session_maker() as session:
            result = await session.execute(
                query,
                {
                    "indicators": indicators,
                    "start_date": start_date,
                    "end_date": end_date
                }
            )
            
            rows = result.fetchall()
        
        # Group by indicator
        data_by_indicator = {}
        for row in rows:
            indicator = row.symbol
            if indicator not in data_by_indicator:
                data_by_indicator[indicator] = []
            data_by_indicator[indicator].append(row)
        
        # Convert to DataFrames
        result = {}
        for indicator, indicator_rows in data_by_indicator.items():
            df = pd.DataFrame(indicator_rows)
            df.set_index('date', inplace=True)
            df.drop('symbol', axis=1, inplace=True)
            df.index = pd.to_datetime(df.index)
            df['value'] = pd.to_numeric(df['value'])
            result[indicator] = df
            
        return result
    
    async def load_latest_prices(
        self,
        symbols: List[str],
        as_of_date: Optional[date] = None
    ) -> Dict[str, float]:
        """
        Load latest prices for symbols.
        
        Args:
            symbols: List of symbols
            as_of_date: Date to get prices for (latest if None)
            
        Returns:
            Dict mapping symbols to prices
        """
        if as_of_date:
            query = text("""
                SELECT DISTINCT ON (symbol)
                    symbol,
                    close
                FROM market_data.ohlcv
                WHERE symbol = ANY(:symbols)
                    AND DATE(timestamp) <= :as_of_date
                ORDER BY symbol, timestamp DESC
            """)
            params = {"symbols": symbols, "as_of_date": as_of_date}
        else:
            query = text("""
                SELECT DISTINCT ON (symbol)
                    symbol,
                    close
                FROM market_data.ohlcv
                WHERE symbol = ANY(:symbols)
                ORDER BY symbol, timestamp DESC
            """)
            params = {"symbols": symbols}
        
        async with async_session_maker() as session:
            result = await session.execute(query, params)
            rows = result.fetchall()
            
        return {row.symbol: float(row.close) for row in rows}
    
    async def check_data_availability(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Dict[str, Dict[str, Any]]:
        """
        Check data availability for symbols.
        
        Returns dict with coverage info for each symbol.
        """
        query = text("""
            SELECT 
                symbol,
                MIN(DATE(timestamp)) as first_date,
                MAX(DATE(timestamp)) as last_date,
                COUNT(*) as record_count,
                COUNT(DISTINCT DATE(timestamp)) as days_count
            FROM market_data.ohlcv
            WHERE symbol = ANY(:symbols)
                AND timestamp >= :start_date
                AND timestamp <= :end_date
            GROUP BY symbol
        """)
        
        async with async_session_maker() as session:
            result = await session.execute(
                query,
                {
                    "symbols": symbols,
                    "start_date": start_date,
                    "end_date": end_date
                }
            )
            
            rows = result.fetchall()
            
        availability = {}
        for row in rows:
            availability[row.symbol] = {
                'first_date': row.first_date,
                'last_date': row.last_date,
                'record_count': row.record_count,
                'days_count': row.days_count,
                'has_data': True
            }
            
        # Mark missing symbols
        for symbol in symbols:
            if symbol not in availability:
                availability[symbol] = {
                    'has_data': False,
                    'record_count': 0,
                    'days_count': 0
                }
                
        return availability