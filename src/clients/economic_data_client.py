"""Economic data client with support for multiple providers"""

from datetime import datetime, date
from typing import List, Optional, Dict, Any, Protocol
from decimal import Decimal
from abc import ABC, abstractmethod
import pandas as pd
import httpx

from config import settings
from src.clients.base_client import BaseAPIClient, RateLimiter
from src.models.economic_data import EconomicData, EconomicRelease, EconomicForecast
from src.utils.logging import logger


class EconomicDataProvider(Protocol):
    """Protocol for economic data providers"""
    
    async def get_indicator_data(
        self,
        symbol: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        **kwargs
    ) -> List[EconomicData]:
        """Get historical data for an economic indicator"""
        ...
    
    async def get_releases(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        **kwargs
    ) -> List[EconomicRelease]:
        """Get economic data releases"""
        ...
    
    async def get_forecasts(
        self,
        symbol: str,
        **kwargs
    ) -> List[EconomicForecast]:
        """Get economic forecasts"""
        ...


class FREDProvider(BaseAPIClient):
    """FRED (Federal Reserve Economic Data) provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or settings.fred_api_key
        if not api_key:
            raise ValueError("FRED API key not provided")
        
        rate_limiter = RateLimiter(
            calls=settings.fred_rate_limit,
            period=60
        )
        
        super().__init__(
            base_url=settings.fred_base_url,
            api_key=api_key,
            rate_limiter=rate_limiter
        )
        
        self.api_key = api_key
    
    async def __aenter__(self):
        """Async context manager entry"""
        await super().__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await super().__aexit__(exc_type, exc_val, exc_tb)
    
    async def _make_fred_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a request to FRED API"""
        if params is None:
            params = {}
        
        # Add API key and format to params
        params["api_key"] = self.api_key
        params["file_type"] = "json"
        
        # Use the base class's get method
        return await self.get(endpoint, params=params)
    
    async def get_indicator_data(
        self,
        symbol: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        **kwargs
    ) -> List[EconomicData]:
        """Get historical data for a FRED series"""
        endpoint = "series/observations"
        params = {"series_id": symbol}
        
        if from_date:
            params["observation_start"] = from_date.isoformat()
        if to_date:
            params["observation_end"] = to_date.isoformat()
        
        data = await self._make_fred_request(endpoint, params)
        
        observations = []
        for obs in data["observations"]:
            # Skip missing values (shown as ".")
            if obs["value"] == ".":
                continue
                
            observations.append(
                EconomicData(
                    symbol=symbol,
                    date=datetime.fromisoformat(obs["date"]),
                    value=Decimal(obs["value"]),
                    source="FRED"
                )
            )
        
        logger.info(f"Fetched {len(observations)} observations for {symbol} from FRED")
        return observations
    
    async def get_releases(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        **kwargs
    ) -> List[EconomicRelease]:
        """FRED doesn't provide release event data in the same format"""
        # This would need to be implemented differently or sourced elsewhere
        logger.warning("FRED provider doesn't support release events")
        return []
    
    async def get_forecasts(
        self,
        symbol: str,
        **kwargs
    ) -> List[EconomicForecast]:
        """FRED doesn't provide forecast data"""
        logger.warning("FRED provider doesn't support forecast data")
        return []


class EconomicDataClient:
    """Main client for economic data with support for multiple providers"""
    
    def __init__(self, provider: str = "fred"):
        """Initialize with specified provider"""
        self.provider_name = provider.lower()
        self.provider = self._create_provider(provider)
    
    def _create_provider(self, provider: str) -> EconomicDataProvider:
        """Create provider instance based on name"""
        provider = provider.lower()
        
        if provider == "fred":
            return FREDProvider()
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.provider.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.provider.__aexit__(exc_type, exc_val, exc_tb)
    
    async def get_indicator_data(
        self,
        symbol: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        **kwargs
    ) -> List[EconomicData]:
        """Get historical data for an economic indicator"""
        return await self.provider.get_indicator_data(
            symbol, from_date, to_date, **kwargs
        )
    
    async def get_releases(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        **kwargs
    ) -> List[EconomicRelease]:
        """Get economic data releases"""
        return await self.provider.get_releases(
            from_date, to_date, **kwargs
        )
    
    async def get_forecasts(
        self,
        symbol: str,
        **kwargs
    ) -> List[EconomicForecast]:
        """Get economic forecasts"""
        return await self.provider.get_forecasts(symbol, **kwargs)
    
    async def get_multiple_indicators(
        self,
        symbols: List[str],
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        **kwargs
    ) -> Dict[str, List[EconomicData]]:
        """Get data for multiple indicators"""
        import asyncio
        
        tasks = []
        for symbol in symbols:
            task = self.get_indicator_data(symbol, from_date, to_date, **kwargs)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {symbol}: {result}")
                data[symbol] = []
            else:
                data[symbol] = result
        
        return data