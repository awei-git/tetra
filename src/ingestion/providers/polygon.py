"""Polygon.io data provider implementation."""

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


class PolygonProvider(BaseProvider):
    """
    Polygon.io data provider for comprehensive market data.
    
    Supports:
    - OHLCV data (stocks, options, forex, crypto)
    - News articles
    - Company events (earnings, dividends, splits)
    - Economic indicators (via partnerships)
    """
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Polygon provider.
        
        Args:
            api_key: Polygon API key (or from secrets.yml)
        """
        if not api_key:
            # Read from secrets.yml
            try:
                with open('config/secrets.yml', 'r') as f:
                    secrets = yaml.safe_load(f)
                    api_key = secrets.get('api_keys', {}).get('polygon')
            except:
                api_key = os.getenv("POLYGON_API_KEY")
        
        if not api_key:
            logger.warning("No Polygon API key provided, functionality will be limited")
        
        super().__init__(api_key)
    
    def _initialize(self):
        """Initialize HTTP session."""
        # Session will be created when needed for async context
        pass
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """
        Make HTTP request to Polygon API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response data
        """
        session = await self._get_session()
        
        params = params or {}
        params['apiKey'] = self.api_key
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Polygon API request failed: {e}")
            raise
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        from_date: Union[date, datetime],
        to_date: Union[date, datetime],
        timeframe: str = "1d"
    ) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV data from Polygon.
        
        Args:
            symbol: Ticker symbol
            from_date: Start date
            to_date: End date
            timeframe: Time interval (1d, 1h, 5m, etc.)
            
        Returns:
            List of OHLCV records
        """
        from_date, to_date = self._validate_date_range(from_date, to_date)
        
        # Map timeframe to Polygon format
        timeframe_map = {
            "1m": (1, "minute"),
            "5m": (5, "minute"),
            "15m": (15, "minute"),
            "30m": (30, "minute"),
            "1h": (1, "hour"),
            "4h": (4, "hour"),
            "1d": (1, "day"),
            "1w": (1, "week"),
            "1M": (1, "month")
        }
        
        multiplier, timespan = timeframe_map.get(timeframe, (1, "day"))
        
        # Polygon aggregates endpoint
        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000
        }
        
        try:
            response = await self._make_request(endpoint, params)
            
            if response.get("status") != "OK" or not response.get("results"):
                logger.warning(f"No data returned for {symbol} from {from_date} to {to_date}")
                return []
            
            # Convert Polygon format to standard format
            results = []
            for bar in response["results"]:
                results.append({
                    "timestamp": datetime.fromtimestamp(bar["t"] / 1000),  # Convert ms to seconds
                    "open": bar["o"],
                    "high": bar["h"],
                    "low": bar["l"],
                    "close": bar["c"],
                    "volume": bar["v"],
                    "vwap": bar.get("vw"),
                    "trade_count": bar.get("n")
                })
            
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
        Fetch economic indicator data.
        
        Note: Polygon has limited economic data. Consider using FRED provider.
        """
        logger.warning("Polygon has limited economic data support. Consider using FRED provider.")
        
        # Polygon doesn't have direct economic indicators
        # Would need to use partner APIs or alternative endpoints
        return []
    
    async def fetch_news(
        self,
        symbols: Optional[List[str]] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles from Polygon.
        
        Args:
            symbols: List of symbols to get news for
            from_date: Start date
            to_date: End date
            categories: News categories (not used by Polygon)
            
        Returns:
            List of news articles
        """
        endpoint = "/v2/reference/news"
        
        params = {}
        
        if symbols:
            params["ticker"] = ",".join(symbols)
        
        if from_date:
            params["published_utc.gte"] = from_date.isoformat()
        
        if to_date:
            params["published_utc.lte"] = to_date.isoformat()
        
        params["limit"] = 100
        params["sort"] = "published_utc"
        
        try:
            response = await self._make_request(endpoint, params)
            
            if response.get("status") != "OK" or not response.get("results"):
                logger.warning(f"No news returned for symbols: {symbols}")
                return []
            
            # Convert to standard format
            results = []
            for article in response["results"]:
                results.append({
                    "source": article.get("publisher", {}).get("name", "Unknown"),
                    "author": article.get("author"),
                    "title": article.get("title", ""),
                    "description": article.get("description"),
                    "url": article.get("article_url", ""),
                    "published_at": datetime.fromisoformat(article.get("published_utc", "")),
                    "content": None,  # Polygon doesn't provide full content
                    "symbols": article.get("tickers", [])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch news: {e}")
            return []
    
    async def fetch_events(
        self,
        event_type: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch event data from Polygon.
        
        Args:
            event_type: Type of event (earnings, dividends, splits)
            from_date: Start date
            to_date: End date
            
        Returns:
            List of events
        """
        # Map event types to Polygon endpoints
        event_endpoints = {
            "earnings": "/vX/reference/financials",
            "dividends": "/v3/reference/dividends",
            "splits": "/v3/reference/splits"
        }
        
        endpoint = event_endpoints.get(event_type)
        if not endpoint:
            logger.warning(f"Unsupported event type: {event_type}")
            return []
        
        params = {}
        
        if from_date:
            if event_type == "dividends":
                params["ex_dividend_date.gte"] = from_date.isoformat()
            elif event_type == "splits":
                params["execution_date.gte"] = from_date.isoformat()
        
        if to_date:
            if event_type == "dividends":
                params["ex_dividend_date.lte"] = to_date.isoformat()
            elif event_type == "splits":
                params["execution_date.lte"] = to_date.isoformat()
        
        params["limit"] = 1000
        
        try:
            response = await self._make_request(endpoint, params)
            
            if response.get("status") != "OK" or not response.get("results"):
                logger.warning(f"No {event_type} events found")
                return []
            
            # Convert to standard format
            results = []
            
            if event_type == "dividends":
                for event in response["results"]:
                    results.append({
                        "symbol": event.get("ticker", ""),
                        "date": date.fromisoformat(event.get("ex_dividend_date", "")),
                        "time": None,
                        "data": {
                            "amount": event.get("cash_amount", 0),
                            "currency": event.get("currency", "USD"),
                            "frequency": event.get("frequency"),
                            "pay_date": event.get("pay_date"),
                            "record_date": event.get("record_date")
                        },
                        "importance": "medium"
                    })
            
            elif event_type == "splits":
                for event in response["results"]:
                    results.append({
                        "symbol": event.get("ticker", ""),
                        "date": date.fromisoformat(event.get("execution_date", "")),
                        "time": None,
                        "data": {
                            "split_from": event.get("split_from", 1),
                            "split_to": event.get("split_to", 1),
                            "ratio": f"{event.get('split_to', 1)}:{event.get('split_from', 1)}"
                        },
                        "importance": "high"
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch {event_type} events: {e}")
            return []
    
    async def fetch_ticker_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed information about a ticker.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Ticker details or None
        """
        endpoint = f"/v3/reference/tickers/{symbol}"
        
        try:
            response = await self._make_request(endpoint)
            
            if response.get("status") == "OK" and response.get("results"):
                return response["results"]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch ticker details for {symbol}: {e}")
            return None
    
    async def fetch_market_status(self) -> Dict[str, Any]:
        """
        Get current market status.
        
        Returns:
            Market status information
        """
        endpoint = "/v1/marketstatus/now"
        
        try:
            response = await self._make_request(endpoint)
            return response
            
        except Exception as e:
            logger.error(f"Failed to fetch market status: {e}")
            return {}