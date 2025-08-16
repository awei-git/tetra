"""Alpha Vantage data provider implementation."""

import os
import yaml
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import aiohttp
import asyncio
from urllib.parse import urlencode

from .base import BaseProvider

logger = logging.getLogger(__name__)


class AlphaVantageProvider(BaseProvider):
    """
    Alpha Vantage data provider for financial and economic data.
    
    Supports:
    - OHLCV data for stocks, forex, crypto
    - Economic indicators
    - Technical indicators (pre-calculated)
    - Fundamental data
    
    Note: Free tier has 5 API calls per minute limit.
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    RATE_LIMIT_DELAY = 12.5  # 5 calls per minute = 12 seconds between calls
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage provider.
        
        Args:
            api_key: Alpha Vantage API key (or from secrets.yml)
        """
        if not api_key:
            # Read from secrets.yml
            try:
                with open('config/secrets.yml', 'r') as f:
                    secrets = yaml.safe_load(f)
                    api_key = secrets.get('api_keys', {}).get('alphavantage')
            except:
                api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        
        if not api_key:
            logger.warning("No Alpha Vantage API key provided, functionality will be limited")
        
        super().__init__(api_key)
        self.last_request_time = None
    
    def _initialize(self):
        """Initialize provider."""
        pass
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def _rate_limit(self):
        """Enforce rate limiting."""
        if self.last_request_time:
            elapsed = datetime.now().timestamp() - self.last_request_time
            if elapsed < self.RATE_LIMIT_DELAY:
                await asyncio.sleep(self.RATE_LIMIT_DELAY - elapsed)
        
        self.last_request_time = datetime.now().timestamp()
    
    async def _make_request(self, params: Dict[str, Any]) -> Dict:
        """
        Make HTTP request to Alpha Vantage API.
        
        Args:
            params: Query parameters including function
            
        Returns:
            JSON response data
        """
        await self._rate_limit()
        
        session = await self._get_session()
        
        params['apikey'] = self.api_key
        params['datatype'] = 'json'
        
        try:
            async with session.get(self.BASE_URL, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Check for API errors
                if "Error Message" in data:
                    raise ValueError(f"API Error: {data['Error Message']}")
                if "Note" in data:
                    logger.warning(f"API Note: {data['Note']}")
                    
                return data
                
        except aiohttp.ClientError as e:
            logger.error(f"Alpha Vantage API request failed: {e}")
            raise
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        from_date: Union[date, datetime],
        to_date: Union[date, datetime],
        timeframe: str = "1d"
    ) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV data from Alpha Vantage.
        
        Args:
            symbol: Ticker symbol
            from_date: Start date
            to_date: End date
            timeframe: Time interval (1m, 5m, 15m, 30m, 60m, 1d, 1w, 1M)
            
        Returns:
            List of OHLCV records
        """
        from_date, to_date = self._validate_date_range(from_date, to_date)
        
        # Map timeframe to Alpha Vantage function
        function_map = {
            "1m": ("TIME_SERIES_INTRADAY", "1min"),
            "5m": ("TIME_SERIES_INTRADAY", "5min"),
            "15m": ("TIME_SERIES_INTRADAY", "15min"),
            "30m": ("TIME_SERIES_INTRADAY", "30min"),
            "60m": ("TIME_SERIES_INTRADAY", "60min"),
            "1h": ("TIME_SERIES_INTRADAY", "60min"),
            "1d": ("TIME_SERIES_DAILY", None),
            "1w": ("TIME_SERIES_WEEKLY", None),
            "1M": ("TIME_SERIES_MONTHLY", None)
        }
        
        function, interval = function_map.get(timeframe, ("TIME_SERIES_DAILY", None))
        
        params = {
            "function": function,
            "symbol": symbol,
            "outputsize": "full"  # Get full data, not just last 100 points
        }
        
        if interval:
            params["interval"] = interval
        
        try:
            response = await self._make_request(params)
            
            # Find the time series key in response
            time_series_key = None
            for key in response.keys():
                if "Time Series" in key:
                    time_series_key = key
                    break
            
            if not time_series_key:
                logger.warning(f"No time series data found for {symbol}")
                return []
            
            time_series = response[time_series_key]
            
            # Convert to standard format
            results = []
            for timestamp_str, values in time_series.items():
                # Parse timestamp
                if timeframe in ["1m", "5m", "15m", "30m", "60m", "1h"]:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                else:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d")
                
                # Filter by date range
                if timestamp.date() < from_date or timestamp.date() > to_date:
                    continue
                
                results.append({
                    "timestamp": timestamp,
                    "open": float(values.get("1. open", values.get("1a. open (USD)", 0))),
                    "high": float(values.get("2. high", values.get("2a. high (USD)", 0))),
                    "low": float(values.get("3. low", values.get("3a. low (USD)", 0))),
                    "close": float(values.get("4. close", values.get("4a. close (USD)", 0))),
                    "volume": int(values.get("5. volume", values.get("6. volume", 0))),
                    "vwap": None,
                    "trade_count": None
                })
            
            # Sort by timestamp
            results.sort(key=lambda x: x["timestamp"])
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return []
    
    async def fetch_economic_indicator(
        self,
        indicator: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch economic indicator data from Alpha Vantage.
        
        Args:
            indicator: Indicator symbol (GDP, CPI, INFLATION, etc.)
            from_date: Start date (optional)
            to_date: End date (optional)
            
        Returns:
            List of indicator records
        """
        # Map common indicators to Alpha Vantage functions
        indicator_map = {
            "GDP": "REAL_GDP",
            "CPI": "CPI",
            "INFLATION": "INFLATION",
            "UNEMPLOYMENT": "UNEMPLOYMENT",
            "INTEREST_RATE": "FEDERAL_FUNDS_RATE",
            "CONSUMER_SENTIMENT": "CONSUMER_SENTIMENT"
        }
        
        function = indicator_map.get(indicator.upper(), indicator)
        
        params = {
            "function": function,
            "interval": "monthly"  # or "quarterly" for some indicators
        }
        
        try:
            response = await self._make_request(params)
            
            # Find data key
            data_key = "data"
            if data_key not in response:
                # Try to find any key containing data
                for key in response.keys():
                    if "data" in key.lower() or indicator.lower() in key.lower():
                        data_key = key
                        break
            
            if data_key not in response:
                logger.warning(f"No data found for indicator {indicator}")
                return []
            
            # Convert to standard format
            results = []
            for item in response[data_key]:
                item_date = date.fromisoformat(item.get("date", ""))
                
                # Filter by date range if provided
                if from_date and item_date < from_date:
                    continue
                if to_date and item_date > to_date:
                    continue
                
                results.append({
                    "date": item_date,
                    "value": float(item.get("value", 0)),
                    "previous_value": None,
                    "period": params.get("interval", "monthly"),
                    "unit": response.get("unit", "")
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch economic indicator {indicator}: {e}")
            return []
    
    async def fetch_news(
        self,
        symbols: Optional[List[str]] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch news and sentiment data.
        
        Args:
            symbols: List of symbols to get news for
            from_date: Start date
            to_date: End date
            categories: News categories (optional)
            
        Returns:
            List of news articles
        """
        if not symbols:
            logger.warning("Alpha Vantage requires symbols for news/sentiment")
            return []
        
        results = []
        
        for symbol in symbols[:3]:  # Limit due to rate limiting
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "limit": 50
            }
            
            if from_date:
                params["time_from"] = from_date.strftime("%Y%m%dT0000")
            if to_date:
                params["time_to"] = to_date.strftime("%Y%m%dT2359")
            
            try:
                response = await self._make_request(params)
                
                feed = response.get("feed", [])
                
                for article in feed:
                    # Parse timestamp
                    time_str = article.get("time_published", "")
                    if time_str:
                        published_at = datetime.strptime(time_str, "%Y%m%dT%H%M%S")
                    else:
                        published_at = datetime.now()
                    
                    results.append({
                        "source": article.get("source", "Unknown"),
                        "author": None,
                        "title": article.get("title", ""),
                        "description": article.get("summary", ""),
                        "url": article.get("url", ""),
                        "published_at": published_at,
                        "content": article.get("summary"),
                        "symbols": [t["ticker"] for t in article.get("ticker_sentiment", [])]
                    })
                    
            except Exception as e:
                logger.error(f"Failed to fetch news for {symbol}: {e}")
        
        return results
    
    async def fetch_events(
        self,
        event_type: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        Alpha Vantage doesn't directly support event fetching.
        
        Some data can be extracted from company overview or earnings.
        """
        logger.warning(f"Alpha Vantage has limited support for {event_type} events")
        
        if event_type == "earnings":
            # Can fetch earnings data
            params = {
                "function": "EARNINGS_CALENDAR",
                "horizon": "3month"
            }
            
            try:
                response = await self._make_request(params)
                # Process CSV response (Alpha Vantage returns CSV for this endpoint)
                # This would need CSV parsing
                return []
                
            except Exception as e:
                logger.error(f"Failed to fetch earnings calendar: {e}")
                return []
        
        return []
    
    async def fetch_company_overview(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch company fundamental data.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Company overview data or None
        """
        params = {
            "function": "OVERVIEW",
            "symbol": symbol
        }
        
        try:
            response = await self._make_request(params)
            
            if response and "Symbol" in response:
                return response
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch company overview for {symbol}: {e}")
            return None