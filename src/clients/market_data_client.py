from datetime import datetime, date
from typing import List, Optional, Dict, Any, Protocol
from decimal import Decimal
from abc import ABC, abstractmethod
import yfinance as yf
import pandas as pd

from config import settings
from src.clients.base_client import BaseAPIClient, RateLimiter
from src.models import OHLCVData, TickData, Quote
from src.utils.logging import logger


class MarketDataProvider(Protocol):
    """Protocol for market data providers"""
    
    async def get_aggregates(
        self,
        symbol: str,
        multiplier: int,
        timespan: str,
        from_date: date,
        to_date: date,
        **kwargs
    ) -> List[OHLCVData]:
        ...
    
    async def get_last_trade(self, symbol: str) -> Optional[TickData]:
        ...
    
    async def get_last_quote(self, symbol: str) -> Optional[Quote]:
        ...


class PolygonProvider(BaseAPIClient):
    """Polygon.io data provider implementation"""
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or settings.polygon_api_key
        if not api_key:
            raise ValueError("Polygon API key not provided")
        
        rate_limiter = RateLimiter(
            calls=settings.polygon_rate_limit,
            period=60
        )
        
        super().__init__(
            base_url=settings.polygon_base_url,
            api_key=None,  # Polygon uses API key in URL params
            rate_limiter=rate_limiter,
            timeout=30
        )
        
        self.api_key = api_key
    
    def _add_api_key(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add API key to parameters"""
        if params is None:
            params = {}
        params["apiKey"] = self.api_key
        return params
    
    async def get_aggregates(
        self,
        symbol: str,
        multiplier: int,
        timespan: str,
        from_date: date,
        to_date: date,
        adjusted: bool = True,
        sort: str = "asc",
        limit: int = 5000,
    ) -> List[OHLCVData]:
        """Get aggregate bars from Polygon"""
        endpoint = f"/v2/aggs/ticker/{symbol.upper()}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        
        params = self._add_api_key({
            "adjusted": str(adjusted).lower(),
            "sort": sort,
            "limit": limit,
        })
        
        try:
            data = await self.get(endpoint, params=params)
            
            # Polygon returns "DELAYED" for free tier, which is fine
            if data.get("status") not in ["OK", "DELAYED"]:
                logger.error(f"Polygon API error: {data}")
                return []
            
            results = data.get("results", [])
            
            # Convert to our OHLCV model
            ohlcv_list = []
            for bar in results:
                timeframe_map = {
                    "minute": f"{multiplier}m",
                    "hour": f"{multiplier}h",
                    "day": f"{multiplier}d",
                    "week": f"{multiplier}w",
                    "month": f"{multiplier}M",
                }
                
                ohlcv = OHLCVData(
                    symbol=symbol.upper(),
                    timestamp=datetime.fromtimestamp(bar["t"] / 1000),
                    open=Decimal(str(bar["o"])),
                    high=Decimal(str(bar["h"])),
                    low=Decimal(str(bar["l"])),
                    close=Decimal(str(bar["c"])),
                    volume=int(bar["v"] / 1_000_000),  # Convert to millions to match yfinance
                    vwap=Decimal(str(bar["vw"])) if "vw" in bar else None,
                    trades_count=bar.get("n"),
                    timeframe=timeframe_map.get(timespan, "1d"),
                    source="polygon",
                )
                ohlcv_list.append(ohlcv)
            
            logger.info(f"Fetched {len(ohlcv_list)} bars for {symbol} from Polygon")
            return ohlcv_list
            
        except Exception as e:
            logger.error(f"Error fetching Polygon aggregates: {e}")
            raise
    
    async def get_last_trade(self, symbol: str) -> Optional[TickData]:
        """Get last trade from Polygon"""
        endpoint = f"/v2/last/trade/{symbol.upper()}"
        params = self._add_api_key()
        
        try:
            data = await self.get(endpoint, params=params)
            
            if data.get("status") not in ["OK", "DELAYED"]:
                logger.error(f"Polygon API error: {data}")
                return None
            
            results = data.get("results")
            if not results:
                return None
            
            return TickData(
                symbol=symbol.upper(),
                timestamp=datetime.fromtimestamp(results["t"] / 1000000000),
                price=Decimal(str(results["p"])),
                size=results["s"],
                conditions=results.get("c", []),
                exchange=results.get("x"),
                source="polygon",
            )
            
        except Exception as e:
            logger.error(f"Error fetching last trade from Polygon: {e}")
            raise
    
    async def get_last_quote(self, symbol: str) -> Optional[Quote]:
        """Get last quote from Polygon"""
        endpoint = f"/v2/last/nbbo/{symbol.upper()}"
        params = self._add_api_key()
        
        try:
            data = await self.get(endpoint, params=params)
            
            if data.get("status") not in ["OK", "DELAYED"]:
                logger.error(f"Polygon API error: {data}")
                return None
            
            results = data.get("results")
            if not results:
                return None
            
            return Quote(
                symbol=symbol.upper(),
                timestamp=datetime.fromtimestamp(results["t"] / 1000000000),
                bid_price=Decimal(str(results["p"])),
                bid_size=results["s"],
                ask_price=Decimal(str(results["P"])),
                ask_size=results["S"],
                bid_exchange=results.get("x"),
                ask_exchange=results.get("X"),
                source="polygon",
            )
            
        except Exception as e:
            logger.error(f"Error fetching last quote from Polygon: {e}")
            raise


class YFinanceProvider:
    """Yahoo Finance data provider implementation"""
    
    def __init__(self):
        """Initialize yfinance provider"""
        # yfinance doesn't need API keys
        pass
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass
    
    async def get_aggregates(
        self,
        symbol: str,
        multiplier: int,
        timespan: str,
        from_date: date,
        to_date: date,
        **kwargs
    ) -> List[OHLCVData]:
        """Get aggregate bars from Yahoo Finance"""
        try:
            # Map our timespan to yfinance interval
            interval_map = {
                ("minute", 1): "1m",
                ("minute", 5): "5m",
                ("minute", 15): "15m",
                ("minute", 30): "30m",
                ("minute", 60): "60m",
                ("hour", 1): "1h",
                ("day", 1): "1d",
                ("week", 1): "1wk",
                ("month", 1): "1mo",
            }
            
            yf_interval = interval_map.get((timespan, multiplier), "1d")
            
            # Download data
            ticker = yf.Ticker(symbol.upper())
            df = ticker.history(
                start=from_date,
                end=to_date,
                interval=yf_interval,
                auto_adjust=True,  # Adjust for splits
                prepost=False,
                actions=False
            )
            
            if df.empty:
                logger.warning(f"No data found for {symbol} from yfinance")
                return []
            
            # Convert to our OHLCV model
            ohlcv_list = []
            timeframe_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "60m": "1h", "1h": "1h", "1d": "1d", "1wk": "1w", "1mo": "1M"
            }
            
            for timestamp, row in df.iterrows():
                # Skip rows with NaN values
                if pd.isna(row['Open']) or pd.isna(row['Volume']):
                    continue
                    
                ohlcv = OHLCVData(
                    symbol=symbol.upper(),
                    timestamp=timestamp.to_pydatetime(),
                    open=Decimal(str(round(row['Open'], 4))),
                    high=Decimal(str(round(row['High'], 4))),
                    low=Decimal(str(round(row['Low'], 4))),
                    close=Decimal(str(round(row['Close'], 4))),
                    volume=int(row['Volume'] / 1_000_000),  # Convert to millions to avoid overflow
                    vwap=None,  # yfinance doesn't provide VWAP
                    trades_count=None,
                    timeframe=timeframe_map.get(yf_interval, "1d"),
                    source="yfinance",
                )
                ohlcv_list.append(ohlcv)
            
            logger.info(f"Fetched {len(ohlcv_list)} bars for {symbol} from yfinance")
            return ohlcv_list
            
        except Exception as e:
            logger.error(f"Error fetching data from yfinance: {e}")
            return []
    
    async def get_last_trade(self, symbol: str) -> Optional[TickData]:
        """Get last trade - yfinance doesn't provide tick data"""
        logger.warning("yfinance doesn't provide tick-level trade data")
        return None
    
    async def get_last_quote(self, symbol: str) -> Optional[Quote]:
        """Get current quote from yfinance"""
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            
            # Get bid/ask from info if available
            bid = info.get('bid', 0)
            ask = info.get('ask', 0)
            bid_size = info.get('bidSize', 0)
            ask_size = info.get('askSize', 0)
            
            if bid and ask:
                return Quote(
                    symbol=symbol.upper(),
                    timestamp=datetime.now(),
                    bid_price=Decimal(str(bid)),
                    bid_size=bid_size,
                    ask_price=Decimal(str(ask)),
                    ask_size=ask_size,
                    source="yfinance",
                )
            return None
            
        except Exception as e:
            logger.error(f"Error fetching quote from yfinance: {e}")
            return None


class MarketDataClient:
    """Unified market data client that can use multiple providers"""
    
    def __init__(self, provider: str = "polygon"):
        """
        Initialize market data client
        
        Args:
            provider: Data provider to use (polygon, yfinance, finnhub, etc.)
        """
        self.provider_name = provider
        self.provider = self._get_provider(provider)
    
    def _get_provider(self, provider: str) -> MarketDataProvider:
        """Get provider instance based on name"""
        providers = {
            "polygon": PolygonProvider,
            "yfinance": YFinanceProvider,
            # Add more providers here as needed:
            # "finnhub": FinnhubProvider,
            # "alpaca": AlpacaProvider,
        }
        
        if provider not in providers:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")
        
        return providers[provider]()
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.provider.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.provider.__aexit__(exc_type, exc_val, exc_tb)
    
    async def get_daily_bars(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
        **kwargs
    ) -> List[OHLCVData]:
        """Get daily bars for a symbol"""
        return await self.provider.get_aggregates(
            symbol=symbol,
            multiplier=1,
            timespan="day",
            from_date=from_date,
            to_date=to_date,
            **kwargs
        )
    
    async def get_intraday_bars(
        self,
        symbol: str,
        interval: str,
        from_date: date,
        to_date: date,
        **kwargs
    ) -> List[OHLCVData]:
        """
        Get intraday bars for a symbol
        
        Args:
            symbol: Ticker symbol
            interval: Time interval (1m, 5m, 15m, 30m, 1h, etc.)
            from_date: Start date
            to_date: End date
        """
        # Parse interval
        import re
        match = re.match(r'(\d+)([mhd])', interval)
        if not match:
            raise ValueError(f"Invalid interval format: {interval}")
        
        multiplier = int(match.group(1))
        unit_map = {"m": "minute", "h": "hour", "d": "day"}
        timespan = unit_map.get(match.group(2))
        
        return await self.provider.get_aggregates(
            symbol=symbol,
            multiplier=multiplier,
            timespan=timespan,
            from_date=from_date,
            to_date=to_date,
            **kwargs
        )
    
    async def get_last_trade(self, symbol: str) -> Optional[TickData]:
        """Get the last trade for a symbol"""
        return await self.provider.get_last_trade(symbol)
    
    async def get_last_quote(self, symbol: str) -> Optional[Quote]:
        """Get the last quote for a symbol"""
        return await self.provider.get_last_quote(symbol)
    
    def switch_provider(self, provider: str):
        """Switch to a different data provider"""
        self.provider_name = provider
        self.provider = self._get_provider(provider)
        logger.info(f"Switched to {provider} data provider")