"""Base provider interface for data ingestion."""

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """
    Abstract base class for all data providers.
    
    All data providers must implement these methods to ensure
    consistent interface for the DataIngester.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize provider with optional API key.
        
        Args:
            api_key: API key for the provider (if required)
        """
        self.api_key = api_key
        self.session = None
        self._initialize()
    
    def _initialize(self):
        """Initialize provider-specific resources."""
        pass
    
    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        from_date: Union[date, datetime],
        to_date: Union[date, datetime],
        timeframe: str = "1d"
    ) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV data for a symbol.
        
        Args:
            symbol: Ticker symbol
            from_date: Start date
            to_date: End date
            timeframe: Time interval (1m, 5m, 1h, 1d, etc.)
            
        Returns:
            List of OHLCV records with keys:
            - timestamp: datetime
            - open: float
            - high: float
            - low: float
            - close: float
            - volume: int
            - vwap: float (optional)
            - trade_count: int (optional)
        """
        pass
    
    @abstractmethod
    async def fetch_economic_indicator(
        self,
        indicator: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch economic indicator data.
        
        Args:
            indicator: Indicator symbol (GDP, CPI, etc.)
            from_date: Start date (optional)
            to_date: End date (optional)
            
        Returns:
            List of indicator records with keys:
            - date: date
            - value: float
            - previous_value: float (optional)
            - period: str (optional)
            - unit: str (optional)
        """
        pass
    
    @abstractmethod
    async def fetch_news(
        self,
        symbols: Optional[List[str]] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles.
        
        Args:
            symbols: List of symbols to get news for
            from_date: Start date
            to_date: End date
            categories: News categories to filter
            
        Returns:
            List of news articles with keys:
            - source: str
            - author: str (optional)
            - title: str
            - description: str (optional)
            - url: str
            - published_at: datetime
            - content: str (optional)
            - symbols: List[str] (optional)
        """
        pass
    
    @abstractmethod
    async def fetch_events(
        self,
        event_type: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch event data (earnings, dividends, splits, etc.).
        
        Args:
            event_type: Type of event
            from_date: Start date
            to_date: End date
            
        Returns:
            List of events with keys:
            - symbol: str
            - date: date
            - time: str (optional)
            - data: Dict (event-specific data)
            - importance: str (low, medium, high)
        """
        pass
    
    async def close(self):
        """Clean up provider resources."""
        if self.session:
            await self.session.close()
    
    def _validate_date_range(
        self,
        from_date: Union[date, datetime],
        to_date: Union[date, datetime]
    ) -> tuple:
        """
        Validate and normalize date range.
        
        Args:
            from_date: Start date
            to_date: End date
            
        Returns:
            Tuple of (from_date, to_date) as date objects
        """
        # Convert datetime to date if needed
        if isinstance(from_date, datetime):
            from_date = from_date.date()
        if isinstance(to_date, datetime):
            to_date = to_date.date()
        
        # Validate range
        if from_date > to_date:
            raise ValueError(f"from_date ({from_date}) cannot be after to_date ({to_date})")
        
        # Ensure not future dates for market data
        today = date.today()
        if to_date > today:
            logger.warning(f"to_date ({to_date}) is in the future, adjusting to today ({today})")
            to_date = today
        
        return from_date, to_date