"""FRED (Federal Reserve Economic Data) provider implementation."""

import os
import yaml
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import aiohttp
import asyncio

from .base import BaseProvider

logger = logging.getLogger(__name__)


class FREDProvider(BaseProvider):
    """
    FRED data provider for comprehensive economic indicators.
    
    Supports:
    - GDP, CPI, unemployment, interest rates
    - Housing data, consumer sentiment
    - Manufacturing indices, trade data
    - Thousands of other economic time series
    
    Note: Requires FRED API key (free from St. Louis Fed website).
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    # Common economic indicators
    COMMON_SERIES = {
        "GDP": "GDP",                           # Gross Domestic Product
        "GDPC1": "GDPC1",                      # Real GDP
        "CPI": "CPIAUCSL",                     # Consumer Price Index
        "INFLATION": "T10YIE",                 # 10-Year Breakeven Inflation Rate
        "UNEMPLOYMENT": "UNRATE",              # Unemployment Rate
        "FED_FUNDS": "DFF",                    # Federal Funds Rate
        "TREASURY_10Y": "DGS10",               # 10-Year Treasury Rate
        "TREASURY_2Y": "DGS2",                 # 2-Year Treasury Rate
        "VIX": "VIXCLS",                       # VIX Index
        "CONSUMER_SENTIMENT": "UMCSENT",       # U of Michigan Consumer Sentiment
        "HOUSING_STARTS": "HOUST",             # Housing Starts
        "RETAIL_SALES": "RSXFS",               # Retail Sales
        "INDUSTRIAL_PRODUCTION": "INDPRO",     # Industrial Production Index
        "PMI": "MANEMP",                       # Manufacturing Employment
        "M2": "M2SL",                          # M2 Money Supply
        "DXY": "DTWEXBGS",                     # Trade Weighted US Dollar Index
        "JOBLESS_CLAIMS": "ICSA",              # Initial Jobless Claims
        "PERSONAL_INCOME": "PI",               # Personal Income
        "PCE": "PCE",                          # Personal Consumption Expenditures
        "SAVINGS_RATE": "PSAVERT"              # Personal Savings Rate
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED provider.
        
        Args:
            api_key: FRED API key (or from secrets.yml)
        """
        if not api_key:
            # Read from secrets.yml
            try:
                with open('config/secrets.yml', 'r') as f:
                    secrets = yaml.safe_load(f)
                    api_key = secrets.get('api_keys', {}).get('fred')
            except:
                api_key = os.getenv("FRED_API_KEY")
        
        if not api_key:
            logger.warning("No FRED API key provided, functionality will be limited")
        
        super().__init__(api_key)
    
    def _initialize(self):
        """Initialize provider."""
        pass
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict:
        """
        Make HTTP request to FRED API.
        
        Args:
            endpoint: API endpoint (e.g., /series/observations)
            params: Query parameters
            
        Returns:
            JSON response data
        """
        session = await self._get_session()
        
        params['api_key'] = self.api_key
        params['file_type'] = 'json'
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Check for errors
                if "error_code" in data:
                    raise ValueError(f"FRED API Error: {data.get('error_message', 'Unknown error')}")
                
                return data
                
        except aiohttp.ClientError as e:
            logger.error(f"FRED API request failed: {e}")
            raise
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        from_date: Union[date, datetime],
        to_date: Union[date, datetime],
        timeframe: str = "1d"
    ) -> List[Dict[str, Any]]:
        """
        FRED doesn't provide OHLCV data.
        
        Use Polygon or YFinance providers for market data.
        """
        logger.warning("FRED doesn't provide OHLCV data. Use market data providers instead.")
        return []
    
    async def fetch_economic_indicator(
        self,
        indicator: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch economic indicator data from FRED.
        
        Args:
            indicator: Indicator symbol or FRED series ID
            from_date: Start date (optional)
            to_date: End date (optional)
            
        Returns:
            List of indicator records
        """
        # Map common names to FRED series IDs
        series_id = self.COMMON_SERIES.get(indicator.upper(), indicator)
        
        params = {
            "series_id": series_id,
            "sort_order": "asc"
        }
        
        if from_date:
            params["observation_start"] = from_date.isoformat()
        
        if to_date:
            params["observation_end"] = to_date.isoformat()
        
        try:
            # Fetch observations
            response = await self._make_request("/series/observations", params)
            
            observations = response.get("observations", [])
            
            if not observations:
                logger.warning(f"No data found for indicator {indicator} (series: {series_id})")
                return []
            
            # Also fetch series info for metadata
            info_params = {"series_id": series_id}
            info_response = await self._make_request("/series", info_params)
            series_info = info_response.get("seriess", [{}])[0]
            
            # Convert to standard format
            results = []
            previous_value = None
            
            for obs in observations:
                # Skip missing values
                if obs.get("value") == ".":
                    continue
                
                try:
                    value = float(obs["value"])
                except (ValueError, TypeError):
                    continue
                
                results.append({
                    "date": date.fromisoformat(obs["date"]),
                    "value": value,
                    "previous_value": previous_value,
                    "period": series_info.get("frequency_short", ""),
                    "unit": series_info.get("units_short", "")
                })
                
                previous_value = value
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch economic indicator {indicator}: {e}")
            return []
    
    async def fetch_multiple_indicators(
        self,
        indicators: List[str],
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch multiple economic indicators efficiently.
        
        Args:
            indicators: List of indicator symbols
            from_date: Start date (optional)
            to_date: End date (optional)
            
        Returns:
            Dictionary mapping indicator to data
        """
        results = {}
        
        # Create tasks for parallel fetching
        tasks = []
        for indicator in indicators:
            task = self.fetch_economic_indicator(indicator, from_date, to_date)
            tasks.append(task)
        
        # Execute in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Map results
        for indicator, response in zip(indicators, responses):
            if isinstance(response, Exception):
                logger.error(f"Failed to fetch {indicator}: {response}")
                results[indicator] = []
            else:
                results[indicator] = response
        
        return results
    
    async def fetch_news(
        self,
        symbols: Optional[List[str]] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        FRED doesn't provide news data.
        
        Use NewsAPI or Polygon providers for news.
        """
        logger.warning("FRED doesn't provide news data. Use news providers instead.")
        return []
    
    async def fetch_events(
        self,
        event_type: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch economic release events.
        
        FRED can provide release dates for economic data.
        
        Args:
            event_type: Type of event (economic_releases)
            from_date: Start date
            to_date: End date
            
        Returns:
            List of events
        """
        if event_type != "economic_releases":
            logger.warning(f"FRED only supports economic_releases events, not {event_type}")
            return []
        
        # Fetch release calendar
        params = {}
        
        if from_date:
            params["realtime_start"] = from_date.isoformat()
        
        if to_date:
            params["realtime_end"] = to_date.isoformat()
        
        try:
            response = await self._make_request("/releases/dates", params)
            
            releases = response.get("release_dates", [])
            
            results = []
            for release in releases:
                results.append({
                    "symbol": release.get("release_id", ""),
                    "date": date.fromisoformat(release.get("date", "")),
                    "time": None,
                    "data": {
                        "release_name": release.get("release_name", ""),
                        "release_id": release.get("release_id", "")
                    },
                    "importance": "medium"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch economic releases: {e}")
            return []
    
    async def search_series(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for FRED series by text.
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results
            
        Returns:
            List of matching series
        """
        params = {
            "search_text": search_text,
            "limit": limit,
            "sort_order": "desc",
            "order_by": "popularity"
        }
        
        try:
            response = await self._make_request("/series/search", params)
            
            series_list = response.get("seriess", [])
            
            results = []
            for series in series_list:
                results.append({
                    "id": series.get("id"),
                    "title": series.get("title"),
                    "units": series.get("units"),
                    "frequency": series.get("frequency"),
                    "popularity": series.get("popularity"),
                    "observation_start": series.get("observation_start"),
                    "observation_end": series.get("observation_end")
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search FRED series: {e}")
            return []
    
    async def get_series_info(self, series_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a FRED series.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Series information or None
        """
        params = {"series_id": series_id}
        
        try:
            response = await self._make_request("/series", params)
            
            series_list = response.get("seriess", [])
            if series_list:
                return series_list[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get series info for {series_id}: {e}")
            return None