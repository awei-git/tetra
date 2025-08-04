"""Data handler for backtesting engine that uses the simulator."""

import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import asyncio

from ..simulators.historical.market_replay import MarketReplay

logger = logging.getLogger(__name__)


class DataHandler:
    """Handle data loading using the simulator's MarketReplay."""
    
    def __init__(self, cache_days: int = 30):
        """Initialize data handler.
        
        Args:
            cache_days: Number of days to cache in memory
        """
        self.market_replay = MarketReplay(cache_days=cache_days)
        self._loop = None
        
    def _get_event_loop(self):
        """Get or create event loop."""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop
        
    def load_market_data(self,
                        symbols: List[str],
                        start_date: datetime,
                        end_date: datetime,
                        frequency: str = "1d") -> pd.DataFrame:
        """Load market data for multiple symbols.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date for data
            end_date: End date for data  
            frequency: Data frequency (1m, 5m, 1h, 1d)
            
        Returns:
            DataFrame with MultiIndex (timestamp, symbol)
        """
        logger.info(f"Loading market data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Convert datetime to date for simulator
        start_dt = start_date.date() if isinstance(start_date, datetime) else start_date
        end_dt = end_date.date() if isinstance(end_date, datetime) else end_date
        
        # Run async method synchronously
        loop = self._get_event_loop()
        loop.run_until_complete(
            self.market_replay.load_data(symbols, start_dt, end_dt, preload=True)
        )
        
        # Get all data for the date range
        all_data = []
        current_date = start_dt
        
        while current_date <= end_dt:
            # Get data for this date
            day_data = loop.run_until_complete(
                self.market_replay.get_market_data(symbols, current_date)
            )
            
            # Convert to rows
            for symbol, data in day_data.items():
                if data:
                    all_data.append({
                        'timestamp': datetime.combine(current_date, datetime.min.time()),
                        'symbol': symbol,
                        'open': data['open'],
                        'high': data['high'],
                        'low': data['low'],
                        'close': data['close'],
                        'volume': data['volume'],
                        'adj_close': data['close']  # Using close as adj_close
                    })
            
            # Move to next day
            current_date += timedelta(days=1)
        
        if not all_data:
            logger.warning(f"No market data found for symbols {symbols}")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        df.set_index(['timestamp', 'symbol'], inplace=True)
        df.sort_index(inplace=True)
        
        logger.info(f"Loaded {len(df)} rows of market data")
        return df
    
    def load_event_data(self,
                       symbols: List[str],
                       start_date: datetime,
                       end_date: datetime) -> pd.DataFrame:
        """Load event data for multiple symbols.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with event data
        """
        logger.info(f"Loading event data for {len(symbols)} symbols")
        
        # For now, return empty DataFrame as simulator doesn't have event data yet
        # This can be enhanced later
        return pd.DataFrame()
    
    def get_historical_data(self,
                           symbol: str,
                           end_date: datetime,
                           periods: int,
                           frequency: str = "1d") -> pd.DataFrame:
        """Get historical data for a single symbol.
        
        Args:
            symbol: Symbol to get data for
            end_date: End date (inclusive)
            periods: Number of periods to look back
            frequency: Data frequency
            
        Returns:
            DataFrame with OHLCV data
        """
        # Calculate start date based on frequency
        freq_map = {
            "1m": timedelta(minutes=periods),
            "5m": timedelta(minutes=periods * 5),
            "1h": timedelta(hours=periods),
            "1d": timedelta(days=periods)
        }
        
        lookback = freq_map.get(frequency, timedelta(days=periods))
        start_date = end_date - lookback
        
        # Convert to dates
        start_dt = start_date.date() if isinstance(start_date, datetime) else start_date
        end_dt = end_date.date() if isinstance(end_date, datetime) else end_date
        
        # Get price series from simulator
        loop = self._get_event_loop()
        series = loop.run_until_complete(
            self.market_replay.get_price_series(
                symbol, start_dt, end_dt, price_type="close"
            )
        )
        
        if series.empty:
            return pd.DataFrame()
        
        # Convert to DataFrame format expected by backtester
        df = pd.DataFrame({
            'close': series,
            'open': series,  # Simplified - using close for all prices
            'high': series,
            'low': series,
            'volume': 0
        })
        
        return df.tail(periods)
    
    def get_latest_price(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """Get latest price for a symbol at given timestamp.
        
        Args:
            symbol: Symbol to get price for
            timestamp: Timestamp to get price at
            
        Returns:
            Latest price or None if not found
        """
        # Convert to date
        query_date = timestamp.date() if isinstance(timestamp, datetime) else timestamp
        
        # Get market data for that date
        loop = self._get_event_loop()
        day_data = loop.run_until_complete(
            self.market_replay.get_market_data([symbol], query_date)
        )
        
        if symbol in day_data and day_data[symbol]:
            return day_data[symbol]['close']
        
        return None
    
    def get_trading_days(self,
                        start_date: datetime,
                        end_date: datetime,
                        exchange: str = "NYSE") -> pd.DatetimeIndex:
        """Get trading days between dates.
        
        Args:
            start_date: Start date
            end_date: End date
            exchange: Exchange calendar to use
            
        Returns:
            DatetimeIndex of trading days
        """
        # For now, use a simple approach - exclude weekends
        # The simulator has a TradingCalendar that could be used here
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        return date_range
    
    def clear_cache(self):
        """Clear all cached data."""
        self.market_replay.clear_cache()
        logger.info("Cleared data cache")
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get information about cached data.
        
        Returns:
            Dictionary with cache statistics
        """
        return self.market_replay.get_cache_stats()