"""Event data client for fetching economic calendar and earnings data"""

from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Protocol
from decimal import Decimal
import json
import pandas as pd
import yfinance as yf
import httpx

from config import settings
from src.clients.base_client import BaseAPIClient, RateLimiter
from src.models.event_data import (
    EventData, EconomicEvent, EarningsEvent, EventType, 
    EventImpact, EventStatus, create_event
)
from src.utils.logging import logger


class EventDataProvider(Protocol):
    """Protocol for event data providers"""
    
    async def get_economic_calendar(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        currencies: Optional[List[str]] = None,
        **kwargs
    ) -> List[EconomicEvent]:
        """Get economic calendar events"""
        ...
    
    async def get_earnings_calendar(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        **kwargs
    ) -> List[EarningsEvent]:
        """Get earnings calendar events"""
        ...
    
    async def get_market_holidays(
        self,
        year: Optional[int] = None,
        **kwargs
    ) -> List[EventData]:
        """Get market holidays"""
        ...


class YahooFinanceProvider:
    """Yahoo Finance provider for earnings and basic events"""
    
    def __init__(self):
        """Initialize Yahoo Finance provider"""
        # yfinance doesn't need API keys
        pass
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass
    
    async def get_earnings_calendar(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        **kwargs
    ) -> List[EarningsEvent]:
        """Get earnings calendar from Yahoo Finance"""
        if not from_date:
            from_date = date.today()
        if not to_date:
            to_date = from_date + timedelta(days=30)
        
        # For now, return empty list - Yahoo Finance earnings calendar API is unstable
        # In production, you would use a more reliable source like:
        # - Polygon.io earnings calendar
        # - Alpha Vantage earnings calendar
        # - Web scraping from financial sites
        
        logger.warning("Yahoo Finance earnings calendar is currently not implemented properly")
        return []
    
    async def get_economic_calendar(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        currencies: Optional[List[str]] = None,
        **kwargs
    ) -> List[EconomicEvent]:
        """Yahoo Finance doesn't provide economic calendar"""
        logger.warning("Yahoo Finance doesn't provide economic calendar data")
        return []
    
    async def get_market_holidays(
        self,
        year: Optional[int] = None,
        **kwargs
    ) -> List[EventData]:
        """Get market holidays (simplified)"""
        if not year:
            year = datetime.now().year
        
        # Basic US market holidays (simplified list)
        holidays = []
        holiday_dates = {
            f"{year}-01-01": "New Year's Day",
            f"{year}-07-04": "Independence Day",
            f"{year}-12-25": "Christmas Day",
            # Add more holidays as needed
        }
        
        for date_str, name in holiday_dates.items():
            event = EventData(
                event_type=EventType.MARKET_HOLIDAY,
                event_datetime=datetime.strptime(date_str, "%Y-%m-%d"),
                event_name=name,
                description=f"US Markets Closed - {name}",
                impact=EventImpact.HIGH,
                country="US",
                source="yahoo_finance",
                source_id=f"holiday_{date_str}"
            )
            holidays.append(event)
        
        return holidays


class PolygonEventProvider(BaseAPIClient):
    """Polygon.io provider for market events and earnings"""
    
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
    
    async def get_earnings_calendar(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        **kwargs
    ) -> List[EarningsEvent]:
        """Get earnings calendar from Polygon"""
        if not from_date:
            from_date = date.today()
        if not to_date:
            to_date = from_date + timedelta(days=30)
        
        earnings_events = []
        
        # Polygon doesn't have a single earnings calendar endpoint
        # We need to check each symbol's financials
        if symbols:
            for symbol in symbols:
                try:
                    # Get financial data for the symbol
                    # Polygon uses stock financials endpoint
                    endpoint = f"/v3/reference/financials"
                    params = self._add_api_key({
                        "ticker": symbol.upper(),
                        "limit": 10,
                        "order": "desc"
                    })
                    
                    data = await self.get(endpoint, params=params)
                    
                    if data.get("status") == "OK" and data.get("results"):
                        for result in data["results"]:
                            # Check if this is an earnings report
                            report_date = result.get("filed_date")
                            if report_date:
                                event_date = datetime.strptime(report_date, "%Y-%m-%d")
                                
                                # Skip if outside date range
                                if event_date.date() < from_date or event_date.date() > to_date:
                                    continue
                                
                                # Create earnings event
                                event = EarningsEvent(
                                    event_type=EventType.EARNINGS,
                                    event_datetime=event_date,
                                    event_name=f"{symbol} Earnings Report",
                                    symbol=symbol.upper(),
                                    impact=EventImpact.HIGH if symbol in ["AAPL", "MSFT", "GOOGL", "AMZN"] else EventImpact.MEDIUM,
                                    source="polygon",
                                    source_id=f"polygon_{symbol}_{report_date}",
                                    fiscal_period=result.get("fiscal_period", ""),
                                    eps_actual=Decimal(str(result.get("financials", {}).get("income_statement", {}).get("diluted_earnings_per_share", {}).get("value", 0))) if result.get("financials") else None
                                )
                                
                                earnings_events.append(event)
                                
                except Exception as e:
                    logger.error(f"Error fetching Polygon earnings for {symbol}: {e}")
                    continue
        
        logger.info(f"Fetched {len(earnings_events)} earnings events from Polygon")
        return earnings_events
    
    async def get_market_holidays(
        self,
        year: Optional[int] = None,
        **kwargs
    ) -> List[EventData]:
        """Get market holidays from Polygon"""
        try:
            endpoint = "/v1/marketstatus/upcoming"
            params = self._add_api_key()
            
            data = await self.get(endpoint, params=params)
            
            holidays = []
            for holiday in data:
                if holiday.get("status") == "closed":
                    event = EventData(
                        event_type=EventType.MARKET_HOLIDAY,
                        event_datetime=datetime.strptime(holiday["date"], "%Y-%m-%d"),
                        event_name=holiday.get("name", "Market Holiday"),
                        description=f"Markets closed: {', '.join(holiday.get('exchange', []))}",
                        impact=EventImpact.HIGH,
                        country="US",
                        source="polygon",
                        source_id=f"polygon_holiday_{holiday['date']}"
                    )
                    holidays.append(event)
            
            return holidays
            
        except Exception as e:
            logger.error(f"Error fetching Polygon holidays: {e}")
            return []
    
    async def get_economic_calendar(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        currencies: Optional[List[str]] = None,
        **kwargs
    ) -> List[EconomicEvent]:
        """Polygon doesn't provide economic calendar"""
        return []


class FinnhubEventProvider(BaseAPIClient):
    """Finnhub provider for earnings and economic calendar"""
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or settings.finnhub_api_key
        if not api_key:
            raise ValueError("Finnhub API key not provided")
        
        rate_limiter = RateLimiter(
            calls=settings.finnhub_rate_limit,
            period=60
        )
        
        super().__init__(
            base_url=settings.finnhub_base_url,
            api_key=api_key,
            rate_limiter=rate_limiter,
            timeout=30
        )
        
        self.api_key = api_key
    
    async def get_earnings_calendar(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        **kwargs
    ) -> List[EarningsEvent]:
        """Get earnings calendar from Finnhub"""
        if not from_date:
            from_date = date.today()
        if not to_date:
            to_date = from_date + timedelta(days=30)
        
        earnings_events = []
        
        try:
            endpoint = "/calendar/earnings"
            params = {
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
                "token": self.api_key
            }
            
            data = await self.get(endpoint, params=params)
            
            if data.get("earningsCalendar"):
                for earning in data["earningsCalendar"]:
                    symbol = earning.get("symbol", "").upper()
                    
                    # Filter by symbols if provided
                    if symbols and symbol not in [s.upper() for s in symbols]:
                        continue
                    
                    # Create earnings event
                    event_date = datetime.strptime(earning["date"], "%Y-%m-%d")
                    
                    event = EarningsEvent(
                        event_type=EventType.EARNINGS,
                        event_datetime=event_date,
                        event_name=f"{symbol} Earnings",
                        symbol=symbol,
                        impact=EventImpact.HIGH if symbol in ["AAPL", "MSFT", "GOOGL", "AMZN"] else EventImpact.MEDIUM,
                        source="finnhub",
                        source_id=f"finnhub_{symbol}_{earning['date']}",
                        eps_actual=Decimal(str(earning["epsActual"])) if earning.get("epsActual") else None,
                        eps_estimate=Decimal(str(earning["epsEstimate"])) if earning.get("epsEstimate") else None,
                        revenue_actual=Decimal(str(earning["revenueActual"])) if earning.get("revenueActual") else None,
                        revenue_estimate=Decimal(str(earning["revenueEstimate"])) if earning.get("revenueEstimate") else None,
                        call_time=earning.get("hour", ""),
                        fiscal_period=f"{earning.get('quarter', '')} {earning.get('year', '')}"
                    )
                    
                    earnings_events.append(event)
            
            logger.info(f"Fetched {len(earnings_events)} earnings events from Finnhub")
            return earnings_events
            
        except Exception as e:
            logger.error(f"Error fetching Finnhub earnings calendar: {e}")
            return []
    
    async def get_economic_calendar(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        currencies: Optional[List[str]] = None,
        **kwargs
    ) -> List[EconomicEvent]:
        """Get economic calendar from Finnhub"""
        if not from_date:
            from_date = date.today()
        if not to_date:
            to_date = from_date + timedelta(days=30)
        
        economic_events = []
        
        try:
            endpoint = "/calendar/economic"
            params = {
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
                "token": self.api_key
            }
            
            data = await self.get(endpoint, params=params)
            
            if data.get("economicCalendar"):
                for event in data["economicCalendar"]:
                    # Map impact levels
                    impact_map = {
                        "low": EventImpact.LOW,
                        "medium": EventImpact.MEDIUM,
                        "high": EventImpact.HIGH
                    }
                    
                    # Create economic event
                    event_datetime = datetime.strptime(f"{event['time']} {event.get('hour', '00:00')}", "%Y-%m-%d %H:%M")
                    
                    econ_event = EconomicEvent(
                        event_type=EventType.ECONOMIC_RELEASE,
                        event_datetime=event_datetime,
                        event_name=event.get("event", ""),
                        description=event.get("event", ""),
                        impact=impact_map.get(event.get("impact", "medium").lower(), EventImpact.MEDIUM),
                        currency=event.get("country", ""),
                        country=event.get("country", ""),
                        source="finnhub",
                        source_id=f"finnhub_econ_{event['time']}_{event.get('event', '').replace(' ', '_')}",
                        actual=Decimal(str(event["actual"])) if event.get("actual") else None,
                        forecast=Decimal(str(event["estimate"])) if event.get("estimate") else None,
                        previous=Decimal(str(event["prev"])) if event.get("prev") else None,
                        unit=event.get("unit", "")
                    )
                    
                    # Filter by currencies if provided
                    if not currencies or econ_event.currency in currencies:
                        economic_events.append(econ_event)
            
            logger.info(f"Fetched {len(economic_events)} economic events from Finnhub")
            return economic_events
            
        except Exception as e:
            logger.error(f"Error fetching Finnhub economic calendar: {e}")
            return []
    
    async def get_market_holidays(
        self,
        year: Optional[int] = None,
        **kwargs
    ) -> List[EventData]:
        """Finnhub doesn't provide market holidays"""
        return []


class EconomicCalendarProvider(BaseAPIClient):
    """Provider for economic calendar data (placeholder for future implementation)"""
    
    def __init__(self):
        """Initialize economic calendar provider"""
        # This is a placeholder - in production you'd use a real API
        # Options: ForexFactory scraping, TradingEconomics API, etc.
        super().__init__(
            base_url="https://api.example.com",  # Placeholder
            api_key=None,
            rate_limiter=RateLimiter(calls=60, period=60),
            timeout=30
        )
    
    async def get_economic_calendar(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        currencies: Optional[List[str]] = None,
        **kwargs
    ) -> List[EconomicEvent]:
        """Get economic calendar events"""
        # Placeholder implementation
        logger.warning("Economic calendar provider not yet implemented")
        return []
    
    async def get_earnings_calendar(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        **kwargs
    ) -> List[EarningsEvent]:
        """Economic calendar provider doesn't provide earnings"""
        return []
    
    async def get_market_holidays(
        self,
        year: Optional[int] = None,
        **kwargs
    ) -> List[EventData]:
        """Get market holidays"""
        return []


class EventDataClient:
    """Main client for event data with support for multiple providers"""
    
    def __init__(self, provider: str = "yahoo"):
        """Initialize with specified provider"""
        self.provider_name = provider.lower()
        self.provider = self._create_provider(provider)
    
    def _create_provider(self, provider: str) -> EventDataProvider:
        """Create provider instance based on name"""
        provider = provider.lower()
        
        if provider == "yahoo":
            return YahooFinanceProvider()
        elif provider == "polygon":
            return PolygonEventProvider()
        elif provider == "finnhub":
            return FinnhubEventProvider()
        elif provider == "economic_calendar":
            return EconomicCalendarProvider()
        else:
            raise ValueError(f"Unknown provider: {provider}. Available: yahoo, polygon, finnhub")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.provider.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.provider.__aexit__(exc_type, exc_val, exc_tb)
    
    async def get_economic_calendar(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        currencies: Optional[List[str]] = None,
        impact_filter: Optional[int] = None,
        **kwargs
    ) -> List[EconomicEvent]:
        """Get economic calendar events"""
        events = await self.provider.get_economic_calendar(
            from_date, to_date, currencies, **kwargs
        )
        
        # Filter by impact if specified
        if impact_filter:
            events = [e for e in events if e.impact >= impact_filter]
        
        return events
    
    async def get_earnings_calendar(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        **kwargs
    ) -> List[EarningsEvent]:
        """Get earnings calendar events"""
        return await self.provider.get_earnings_calendar(
            from_date, to_date, symbols, **kwargs
        )
    
    async def get_market_holidays(
        self,
        year: Optional[int] = None,
        **kwargs
    ) -> List[EventData]:
        """Get market holidays"""
        return await self.provider.get_market_holidays(year, **kwargs)
    
    async def get_all_events(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        event_types: Optional[List[EventType]] = None,
        **kwargs
    ) -> List[EventData]:
        """Get all types of events"""
        all_events = []
        
        # Get economic calendar
        if not event_types or EventType.ECONOMIC_RELEASE in event_types:
            econ_events = await self.get_economic_calendar(from_date, to_date)
            all_events.extend(econ_events)
        
        # Get earnings calendar
        if not event_types or EventType.EARNINGS in event_types:
            earnings_events = await self.get_earnings_calendar(from_date, to_date)
            all_events.extend(earnings_events)
        
        # Get market holidays
        if not event_types or EventType.MARKET_HOLIDAY in event_types:
            year = from_date.year if from_date else datetime.now().year
            holidays = await self.get_market_holidays(year)
            # Filter holidays by date range
            if from_date and to_date:
                holidays = [h for h in holidays 
                           if from_date <= h.event_datetime.date() <= to_date]
            all_events.extend(holidays)
        
        # Sort by datetime
        all_events.sort(key=lambda x: x.event_datetime)
        
        return all_events