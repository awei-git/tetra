"""Yahoo Finance data provider implementation."""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import yfinance as yf
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .base import BaseProvider

logger = logging.getLogger(__name__)


class YFinanceProvider(BaseProvider):
    """
    Yahoo Finance data provider for free market data.
    
    Supports:
    - OHLCV data for stocks, ETFs, indices
    - Company events (earnings, dividends, splits)
    - Basic company information
    
    Note: YFinance is free but has rate limits and occasional reliability issues.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize YFinance provider.
        
        Args:
            api_key: Not used for YFinance (free service)
        """
        super().__init__(api_key=None)
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    def _initialize(self):
        """Initialize provider."""
        # YFinance doesn't require initialization
        pass
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        from_date: Union[date, datetime],
        to_date: Union[date, datetime],
        timeframe: str = "1d"
    ) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV data from Yahoo Finance.
        
        Args:
            symbol: Ticker symbol
            from_date: Start date
            to_date: End date
            timeframe: Time interval (1m, 5m, 1h, 1d, etc.)
            
        Returns:
            List of OHLCV records
        """
        from_date, to_date = self._validate_date_range(from_date, to_date)
        
        # Map timeframe to yfinance format
        interval_map = {
            "1m": "1m",
            "2m": "2m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "60m": "60m",
            "1h": "1h",
            "1d": "1d",
            "5d": "5d",
            "1w": "1wk",
            "1M": "1mo",
            "3M": "3mo"
        }
        
        interval = interval_map.get(timeframe, "1d")
        
        # Run synchronous yfinance call in executor
        loop = asyncio.get_event_loop()
        
        try:
            df = await loop.run_in_executor(
                self.executor,
                self._fetch_ohlcv_sync,
                symbol,
                from_date,
                to_date,
                interval
            )
            
            if df is None or df.empty:
                logger.warning(f"No data returned for {symbol} from {from_date} to {to_date}")
                return []
            
            # Convert DataFrame to list of dicts
            results = []
            for timestamp, row in df.iterrows():
                # Skip rows with NaN values
                if pd.isna(row['Close']):
                    continue
                    
                results.append({
                    "timestamp": timestamp.to_pydatetime(),
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": int(row['Volume']),
                    "vwap": None,  # YFinance doesn't provide VWAP
                    "trade_count": None
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return []
    
    def _fetch_ohlcv_sync(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """
        Synchronous method to fetch OHLCV data.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            interval: YFinance interval string
            
        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Add one day to end_date because yfinance is exclusive on end date
            end_date_adjusted = end_date + timedelta(days=1)
            
            df = ticker.history(
                start=start_date,
                end=end_date_adjusted,
                interval=interval,
                auto_adjust=True,
                prepost=False,
                actions=False
            )
            
            return df
            
        except Exception as e:
            logger.error(f"YFinance sync fetch failed for {symbol}: {e}")
            return None
    
    async def fetch_economic_indicator(
        self,
        indicator: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        YFinance doesn't support economic indicators directly.
        
        Some market indices can be used as proxies (e.g., ^VIX, ^TNX).
        """
        logger.warning("YFinance doesn't support economic indicators. Use FRED provider instead.")
        
        # Some common market indices that can serve as economic proxies
        economic_proxies = {
            "VIX": "^VIX",      # Volatility Index
            "DXY": "DX-Y.NYB",  # US Dollar Index
            "TNX": "^TNX",      # 10-Year Treasury Yield
            "TYX": "^TYX",      # 30-Year Treasury Yield
            "GOLD": "GC=F",     # Gold Futures
            "OIL": "CL=F"       # Crude Oil Futures
        }
        
        if indicator in economic_proxies:
            symbol = economic_proxies[indicator]
            ohlcv_data = await self.fetch_ohlcv(symbol, from_date, to_date, "1d")
            
            # Convert to indicator format
            results = []
            for bar in ohlcv_data:
                results.append({
                    "date": bar["timestamp"].date(),
                    "value": bar["close"],
                    "previous_value": None,
                    "period": "daily",
                    "unit": "points"
                })
            
            return results
        
        return []
    
    async def fetch_news(
        self,
        symbols: Optional[List[str]] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles.
        
        Note: YFinance has very limited news functionality.
        """
        if not symbols:
            logger.warning("YFinance requires symbols for news fetching")
            return []
        
        results = []
        loop = asyncio.get_event_loop()
        
        for symbol in symbols[:5]:  # Limit to avoid rate limiting
            try:
                news_items = await loop.run_in_executor(
                    self.executor,
                    self._fetch_news_sync,
                    symbol
                )
                
                for item in news_items:
                    # Filter by date if provided
                    pub_date = datetime.fromtimestamp(item.get("providerPublishTime", 0))
                    
                    if from_date and pub_date.date() < from_date:
                        continue
                    if to_date and pub_date.date() > to_date:
                        continue
                    
                    results.append({
                        "source": item.get("publisher", "Unknown"),
                        "author": None,
                        "title": item.get("title", ""),
                        "description": None,
                        "url": item.get("link", ""),
                        "published_at": pub_date,
                        "content": None,
                        "symbols": [symbol]
                    })
                    
            except Exception as e:
                logger.error(f"Failed to fetch news for {symbol}: {e}")
        
        return results
    
    def _fetch_news_sync(self, symbol: str) -> List[Dict]:
        """
        Synchronous method to fetch news.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            List of news items
        """
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            return news if news else []
        except Exception as e:
            logger.error(f"YFinance news fetch failed for {symbol}: {e}")
            return []
    
    async def fetch_events(
        self,
        event_type: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch event data from Yahoo Finance.
        
        Args:
            event_type: Type of event (earnings, dividends, splits)
            from_date: Start date
            to_date: End date
            
        Returns:
            List of events
        """
        # YFinance doesn't support bulk event fetching
        # Would need to fetch per symbol which is inefficient
        logger.warning(f"YFinance doesn't support bulk {event_type} fetching. Use Polygon provider.")
        return []
    
    async def fetch_ticker_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed information about a ticker.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Ticker information or None
        """
        loop = asyncio.get_event_loop()
        
        try:
            info = await loop.run_in_executor(
                self.executor,
                self._fetch_ticker_info_sync,
                symbol
            )
            return info
            
        except Exception as e:
            logger.error(f"Failed to fetch ticker info for {symbol}: {e}")
            return None
    
    def _fetch_ticker_info_sync(self, symbol: str) -> Optional[Dict]:
        """
        Synchronous method to fetch ticker info.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Ticker info dictionary or None
        """
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logger.error(f"YFinance info fetch failed for {symbol}: {e}")
            return None
    
    async def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)