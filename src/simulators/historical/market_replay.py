"""Market data replay engine for historical simulation."""

import asyncio
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Set
import pandas as pd
from sqlalchemy import text

from ...db.base import async_session_maker
from ..utils.trading_calendar import TradingCalendar


class MarketReplay:
    """Efficiently replay market data for simulation."""
    
    def __init__(self, cache_days: int = 30):
        """
        Initialize market replay engine.
        
        Args:
            cache_days: Number of days to cache in memory
        """
        self.cache_days = cache_days
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_dates: Dict[str, Set[date]] = {}
        self.calendar = TradingCalendar()
        
    async def load_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        preload: bool = True
    ) -> None:
        """
        Load market data for simulation.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date
            end_date: End date  
            preload: Whether to preload all data into cache
        """
        if preload:
            await self._preload_data(symbols, start_date, end_date)
        else:
            # Just initialize cache structure
            for symbol in symbols:
                self._cache[symbol] = pd.DataFrame()
                self._cache_dates[symbol] = set()
    
    async def get_market_data(
        self,
        symbols: List[str],
        query_date: date
    ) -> Dict[str, Dict]:
        """
        Get market data for specific date.
        
        Args:
            symbols: List of symbols
            query_date: Date to get data for
            
        Returns:
            Dict mapping symbols to their OHLCV data
        """
        result = {}
        missing_symbols = []
        
        # Check cache first
        for symbol in symbols:
            if symbol in self._cache_dates and query_date in self._cache_dates[symbol]:
                df = self._cache[symbol]
                day_data = df[df.index.date == query_date]
                if not day_data.empty:
                    result[symbol] = day_data.iloc[-1].to_dict()
                    continue
            missing_symbols.append(symbol)
        
        # Load missing data
        if missing_symbols:
            missing_data = await self._load_day_data(missing_symbols, query_date)
            result.update(missing_data)
            
        return result
    
    async def get_price_series(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        price_type: str = "close"
    ) -> pd.Series:
        """
        Get price series for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            price_type: Type of price (open, high, low, close, vwap)
            
        Returns:
            Price series
        """
        # Check if we have data in cache
        if symbol in self._cache and not self._cache[symbol].empty:
            df = self._cache[symbol]
            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            return df.loc[mask, price_type]
        
        # Load from database
        df = await self._load_symbol_data(symbol, start_date, end_date)
        if not df.empty:
            return df[price_type]
        
        return pd.Series()
    
    async def _preload_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> None:
        """Preload data into cache."""
        # Extend date range for cache
        cache_start = start_date - timedelta(days=self.cache_days)
        cache_end = end_date + timedelta(days=self.cache_days)
        
        # Load data for each symbol
        tasks = [
            self._load_symbol_data(symbol, cache_start, cache_end)
            for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Store in cache
        for symbol, df in zip(symbols, results):
            if not df.empty:
                self._cache[symbol] = df
                self._cache_dates[symbol] = set(df.index.date)
    
    async def _load_symbol_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Load data for a single symbol from database."""
        query = text("""
            SELECT 
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                vwap
            FROM market_data.ohlcv
            WHERE symbol = :symbol
                AND timestamp >= :start_date
                AND timestamp <= CAST(:end_date AS date)
            ORDER BY timestamp
        """)
        
        async with async_session_maker() as session:
            result = await session.execute(
                query,
                {
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date
                }
            )
            
            rows = result.fetchall()
            
        if not rows:
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        df.set_index('timestamp', inplace=True)
        df.index = pd.to_datetime(df.index)
        
        # Ensure numeric types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
        return df
    
    async def _load_day_data(
        self,
        symbols: List[str],
        query_date: date
    ) -> Dict[str, Dict]:
        """Load single day data for multiple symbols."""
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
                AND DATE(timestamp) = :query_date
            ORDER BY symbol, timestamp DESC
        """)
        
        async with async_session_maker() as session:
            result = await session.execute(
                query,
                {
                    "symbols": symbols,
                    "query_date": query_date
                }
            )
            
            rows = result.fetchall()
        
        # Group by symbol and take latest entry for each
        data = {}
        for row in rows:
            symbol = row.symbol
            if symbol not in data:
                data[symbol] = {
                    'timestamp': row.timestamp,
                    'open': float(row.open),
                    'high': float(row.high),
                    'low': float(row.low),
                    'close': float(row.close),
                    'volume': int(row.volume),
                    'vwap': float(row.vwap) if row.vwap else None
                }
        
        return data
    
    async def get_dividends(
        self,
        symbols: List[str],
        query_date: date
    ) -> List[Dict]:
        """Get dividends for given date."""
        # TODO: Implement dividends table in database
        # For now, return empty list as dividends are not yet implemented
        return []
        
        # Future implementation:
        # query = text("""
        #     SELECT 
        #         symbol,
        #         ex_date,
        #         amount,
        #         payment_date
        #     FROM market_data.dividends
        #     WHERE symbol = ANY(:symbols)
        #         AND ex_date = :query_date
        # """)
        # 
        # async with async_session_maker() as session:
        #     result = await session.execute(
        #         query,
        #         {"symbols": symbols, "query_date": query_date}
        #     )
        #     
        #     rows = result.fetchall()
        #     
        # return [
        #     {
        #         'symbol': row.symbol,
        #         'ex_date': row.ex_date,
        #         'amount': float(row.amount),
        #         'payment_date': row.payment_date
        #     }
        #     for row in rows
        # ]
    
    async def get_splits(
        self,
        symbols: List[str],
        query_date: date
    ) -> List[Dict]:
        """Get stock splits for given date."""
        # TODO: Implement splits table in database
        # For now, return empty list as splits are not yet implemented
        return []
        
        # Future implementation:
        # query = text("""
        #     SELECT 
        #         symbol,
        #         split_date,
        #         split_ratio
        #     FROM market_data.splits
        #     WHERE symbol = ANY(:symbols)
        #         AND split_date = :query_date
        # """)
        # 
        # async with async_session_maker() as session:
        #     result = await session.execute(
        #         query,
        #         {"symbols": symbols, "query_date": query_date}
        #     )
        #     
        #     rows = result.fetchall()
        #     
        # return [
        #     {
        #         'symbol': row.symbol,
        #         'split_date': row.split_date,
        #         'ratio': float(row.split_ratio)
        #     }
        #     for row in rows
        # ]
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._cache_dates.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'symbols_cached': len(self._cache),
            'total_days_cached': sum(len(dates) for dates in self._cache_dates.values()),
            'memory_usage_mb': sum(
                df.memory_usage(deep=True).sum() / 1024 / 1024 
                for df in self._cache.values()
            )
        }