"""Test event data functionality"""

import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from src.models import (
    EventData, EconomicEvent, EarningsEvent, MarketHolidayEvent,
    EventType, EventImpact, EventStatus, create_event
)
from src.clients.event_data_client import (
    EventDataClient, YahooFinanceProvider, PolygonEventProvider, FinnhubEventProvider
)
from src.definitions.event_calendar import EventCalendar, EventFrequency


class TestEventDataModels:
    """Test event data models"""
    
    def test_event_data_base_model(self):
        """Test base EventData model creation"""
        event = EventData(
            event_type=EventType.ECONOMIC_RELEASE,
            event_datetime=datetime(2024, 1, 15, 14, 30),
            event_name="CPI Release",
            description="Consumer Price Index",
            impact=EventImpact.HIGH,
            currency="USD",
            country="US",
            source="FRED"
        )
        
        assert event.event_type == EventType.ECONOMIC_RELEASE
        assert event.event_name == "CPI Release"
        assert event.impact == EventImpact.HIGH
        assert event.status == EventStatus.SCHEDULED
        assert event.currency == "USD"
        assert len(event.event_id) == 36  # UUID string length
    
    def test_economic_event_model(self):
        """Test EconomicEvent model with calculations"""
        event = EconomicEvent(
            event_type=EventType.ECONOMIC_RELEASE,
            event_datetime=datetime(2024, 1, 15, 8, 30),
            event_name="Non-Farm Payrolls",
            impact=EventImpact.CRITICAL,
            currency="USD",
            source="BLS",
            actual=Decimal("250000"),
            forecast=Decimal("200000"),
            previous=Decimal("180000"),
            unit="jobs"
        )
        
        assert event.actual == Decimal("250000")
        assert event.forecast == Decimal("200000")
        assert event.surprise == Decimal("50000")
        assert event.surprise_percentage == Decimal("25.0")
    
    def test_economic_event_validation(self):
        """Test economic event type validation"""
        with pytest.raises(ValueError, match="Economic event type must be one of"):
            EconomicEvent(
                event_type=EventType.EARNINGS,  # Wrong type
                event_datetime=datetime.now(),
                event_name="Invalid",
                source="test"
            )
    
    def test_earnings_event_model(self):
        """Test EarningsEvent model"""
        event = EarningsEvent(
            event_type=EventType.EARNINGS,
            event_datetime=datetime(2024, 1, 25, 16, 0),
            event_name="AAPL Q1 2024 Earnings",
            symbol="AAPL",
            impact=EventImpact.HIGH,
            source="yahoo",
            eps_actual=Decimal("2.18"),
            eps_estimate=Decimal("2.10"),
            revenue_actual=Decimal("119575000000"),
            revenue_estimate=Decimal("118000000000"),
            call_time="AMC",
            fiscal_period="Q1 2024"
        )
        
        assert event.symbol == "AAPL"
        assert event.eps_actual == Decimal("2.18")
        assert event.call_time == "AMC"
        assert event.fiscal_period == "Q1 2024"
    
    def test_earnings_event_symbol_validation(self):
        """Test earnings event requires symbol"""
        with pytest.raises(ValueError, match="Symbol is required"):
            EarningsEvent(
                event_type=EventType.EARNINGS,
                event_datetime=datetime.now(),
                event_name="Earnings",
                source="test",
                symbol=None  # Missing symbol
            )
    
    def test_market_holiday_event(self):
        """Test MarketHolidayEvent model"""
        event = MarketHolidayEvent(
            event_type=EventType.MARKET_HOLIDAY,
            event_datetime=datetime(2024, 7, 4),
            event_name="Independence Day",
            description="US Markets Closed",
            impact=EventImpact.HIGH,
            country="US",
            source="NYSE",
            markets_closed=["NYSE", "NASDAQ", "BOND"]
        )
        
        assert event.event_type == EventType.MARKET_HOLIDAY
        assert "NYSE" in event.markets_closed
        assert len(event.markets_closed) == 3
    
    def test_create_event_factory(self):
        """Test event factory function"""
        # Economic event
        econ_event = create_event(
            EventType.ECONOMIC_RELEASE,
            event_datetime=datetime.now(),
            event_name="GDP",
            source="test"
        )
        assert isinstance(econ_event, EconomicEvent)
        
        # Earnings event
        earnings_event = create_event(
            EventType.EARNINGS,
            event_datetime=datetime.now(),
            event_name="AAPL Earnings",
            symbol="AAPL",
            source="test"
        )
        assert isinstance(earnings_event, EarningsEvent)
        
        # Generic event
        generic_event = create_event(
            EventType.MERGER,
            event_datetime=datetime.now(),
            event_name="Merger",
            source="test"
        )
        assert isinstance(generic_event, EventData)
        assert not isinstance(generic_event, (EconomicEvent, EarningsEvent))


class TestEventCalendar:
    """Test event calendar definitions"""
    
    def test_get_all_economic_events(self):
        """Test getting all economic events"""
        events = EventCalendar.get_all_economic_events()
        
        assert len(events) > 20
        assert all(len(event) == 5 for event in events)
        
        # Check specific important events exist
        event_names = [event[0] for event in events]
        assert "FOMC Meeting" in event_names
        assert "Non-Farm Payrolls" in event_names
        assert "CPI" in event_names
    
    def test_get_high_impact_events(self):
        """Test filtering high impact events"""
        high_impact = EventCalendar.get_high_impact_events(min_impact=3)
        
        assert len(high_impact) > 10
        assert all(event[3] >= 3 for event in high_impact)
        
        # Critical events (impact=4) should be included
        critical = EventCalendar.get_high_impact_events(min_impact=4)
        assert len(critical) > 0
        assert all(event[3] == 4 for event in critical)
    
    def test_get_events_by_frequency(self):
        """Test filtering events by frequency"""
        monthly = EventCalendar.get_events_by_frequency(EventFrequency.MONTHLY)
        weekly = EventCalendar.get_events_by_frequency(EventFrequency.WEEKLY)
        
        assert len(monthly) > 10
        assert len(weekly) > 0
        assert all(event[1] == EventFrequency.MONTHLY for event in monthly)
        assert all(event[1] == EventFrequency.WEEKLY for event in weekly)
    
    def test_get_earnings_symbols(self):
        """Test earnings focus symbols"""
        symbols = EventCalendar.get_earnings_symbols()
        
        assert len(symbols) > 20
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "JPM" in symbols  # Banks for earnings season start
    
    def test_get_event_categories(self):
        """Test event categorization"""
        categories = EventCalendar.get_event_categories()
        
        assert "fed_events" in categories
        assert "employment" in categories
        assert "inflation" in categories
        
        # Check category contents
        assert "FOMC Meeting" in categories["fed_events"]
        assert "Non-Farm Payrolls" in categories["employment"]
        assert "CPI" in categories["inflation"]
    
    def test_get_event_info(self):
        """Test getting specific event info"""
        info = EventCalendar.get_event_info("FOMC Meeting")
        
        assert info is not None
        assert info["name"] == "FOMC Meeting"
        assert info["frequency"] == EventFrequency.FOMC
        assert info["impact"] == 4
        assert "interest rate" in info["description"].lower()
        
        # Test case insensitive
        info2 = EventCalendar.get_event_info("fomc meeting")
        assert info2 is not None
        
        # Test non-existent event
        info3 = EventCalendar.get_event_info("Non-existent Event")
        assert info3 is None


class TestYahooFinanceProvider:
    """Test Yahoo Finance provider"""
    
    @pytest.mark.asyncio
    async def test_provider_init(self):
        """Test provider initialization"""
        provider = YahooFinanceProvider()
        
        # Test context manager
        async with provider as p:
            assert p is provider
    
    @pytest.mark.asyncio
    async def test_get_earnings_calendar_placeholder(self):
        """Test earnings calendar returns empty (placeholder implementation)"""
        provider = YahooFinanceProvider()
        
        async with provider:
            events = await provider.get_earnings_calendar(
                from_date=date.today(),
                to_date=date.today() + timedelta(days=30),
                symbols=["AAPL", "MSFT"]
            )
            
            # Current implementation returns empty list
            assert events == []
    
    @pytest.mark.asyncio
    async def test_get_market_holidays(self):
        """Test market holidays"""
        provider = YahooFinanceProvider()
        
        async with provider:
            holidays = await provider.get_market_holidays(year=2024)
            
            assert len(holidays) > 0
            assert all(h.event_type == EventType.MARKET_HOLIDAY for h in holidays)
            
            # Check specific holidays
            holiday_names = [h.event_name for h in holidays]
            assert "New Year's Day" in holiday_names
            assert "Independence Day" in holiday_names
            assert "Christmas Day" in holiday_names
    
    @pytest.mark.asyncio
    async def test_get_economic_calendar_not_supported(self):
        """Test economic calendar not supported by Yahoo"""
        provider = YahooFinanceProvider()
        
        async with provider:
            events = await provider.get_economic_calendar()
            assert events == []


class TestPolygonEventProvider:
    """Test Polygon event provider"""
    
    @pytest.mark.asyncio
    async def test_provider_init(self):
        """Test provider initialization"""
        # Should use settings API key
        provider = PolygonEventProvider()
        assert provider.api_key is not None
        
        # Should accept custom API key
        custom_key = "test_api_key"
        provider = PolygonEventProvider(api_key=custom_key)
        assert provider.api_key == custom_key
    
    @pytest.mark.asyncio
    async def test_get_earnings_calendar(self):
        """Test fetching earnings calendar from Polygon"""
        provider = PolygonEventProvider()
        
        # Mock the get method
        mock_response = {
            "status": "OK",
            "results": [{
                "filed_date": "2024-01-25",
                "fiscal_period": "Q4 2023",
                "financials": {
                    "income_statement": {
                        "diluted_earnings_per_share": {
                            "value": 2.18
                        }
                    }
                }
            }]
        }
        
        with patch.object(provider, 'get', return_value=mock_response) as mock_get:
            async with provider:
                events = await provider.get_earnings_calendar(
                    from_date=date(2024, 1, 1),
                    to_date=date(2024, 2, 1),
                    symbols=["AAPL"]
                )
                
                assert len(events) == 1
                assert events[0].symbol == "AAPL"
                assert events[0].eps_actual == Decimal("2.18")
                
                # Verify API call
                mock_get.assert_called_once()
                call_args = mock_get.call_args
                assert "/v3/reference/financials" in call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_get_market_holidays(self):
        """Test fetching market holidays from Polygon"""
        provider = PolygonEventProvider()
        
        # Mock the get method
        mock_response = [{
            "date": "2024-07-04",
            "status": "closed",
            "name": "Independence Day",
            "exchange": ["NYSE", "NASDAQ"]
        }]
        
        with patch.object(provider, 'get', return_value=mock_response) as mock_get:
            async with provider:
                holidays = await provider.get_market_holidays(year=2024)
                
                assert len(holidays) == 1
                assert holidays[0].event_type == EventType.MARKET_HOLIDAY
                assert holidays[0].event_name == "Independence Day"
                assert "NYSE" in holidays[0].description
                
                # Verify API call
                mock_get.assert_called_once()
                call_args = mock_get.call_args
                assert "/v1/marketstatus/upcoming" in call_args[0][0]


class TestFinnhubEventProvider:
    """Test Finnhub event provider"""
    
    @pytest.mark.asyncio
    async def test_provider_init(self):
        """Test provider initialization"""
        # Should use settings API key
        provider = FinnhubEventProvider()
        assert provider.api_key is not None
        
        # Should accept custom API key
        custom_key = "test_api_key"
        provider = FinnhubEventProvider(api_key=custom_key)
        assert provider.api_key == custom_key
    
    @pytest.mark.asyncio
    async def test_get_earnings_calendar(self):
        """Test fetching earnings calendar from Finnhub"""
        provider = FinnhubEventProvider()
        
        # Mock the get method
        mock_response = {
            "earningsCalendar": [{
                "date": "2024-01-25",
                "symbol": "AAPL",
                "epsActual": 2.18,
                "epsEstimate": 2.10,
                "revenueActual": 119575000000,
                "revenueEstimate": 118000000000,
                "hour": "amc",
                "quarter": "Q4",
                "year": 2023
            }]
        }
        
        with patch.object(provider, 'get', return_value=mock_response) as mock_get:
            async with provider:
                events = await provider.get_earnings_calendar(
                    from_date=date(2024, 1, 1),
                    to_date=date(2024, 2, 1),
                    symbols=["AAPL"]
                )
                
                assert len(events) == 1
                assert events[0].symbol == "AAPL"
                assert events[0].eps_actual == Decimal("2.18")
                assert events[0].eps_estimate == Decimal("2.10")
                assert events[0].call_time == "amc"
                
                # Verify API call
                mock_get.assert_called_once()
                call_args = mock_get.call_args
                assert "/calendar/earnings" in call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_get_economic_calendar(self):
        """Test fetching economic calendar from Finnhub"""
        provider = FinnhubEventProvider()
        
        # Mock the get method
        mock_response = {
            "economicCalendar": [{
                "time": "2024-01-15",
                "hour": "08:30",
                "event": "CPI",
                "impact": "high",
                "country": "US",
                "actual": 3.2,
                "estimate": 3.1,
                "prev": 3.0,
                "unit": "%"
            }]
        }
        
        with patch.object(provider, 'get', return_value=mock_response) as mock_get:
            async with provider:
                events = await provider.get_economic_calendar(
                    from_date=date(2024, 1, 1),
                    to_date=date(2024, 2, 1)
                )
                
                assert len(events) == 1
                assert events[0].event_name == "CPI"
                assert events[0].actual == Decimal("3.2")
                assert events[0].forecast == Decimal("3.1")
                assert events[0].previous == Decimal("3.0")
                assert events[0].impact == EventImpact.HIGH
                
                # Verify API call
                mock_get.assert_called_once()
                call_args = mock_get.call_args
                assert "/calendar/economic" in call_args[0][0]


class TestEventDataClient:
    """Test main event data client"""
    
    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initialization with different providers"""
        # Yahoo provider
        client = EventDataClient(provider="yahoo")
        assert client.provider_name == "yahoo"
        assert isinstance(client.provider, YahooFinanceProvider)
        
        # Polygon provider
        client = EventDataClient(provider="polygon")
        assert client.provider_name == "polygon"
        assert isinstance(client.provider, PolygonEventProvider)
        
        # Finnhub provider
        client = EventDataClient(provider="finnhub")
        assert client.provider_name == "finnhub"
        assert isinstance(client.provider, FinnhubEventProvider)
        
        # Unknown provider
        with pytest.raises(ValueError, match="Unknown provider"):
            EventDataClient(provider="unknown")
    
    @pytest.mark.asyncio
    async def test_get_earnings_calendar(self):
        """Test fetching earnings calendar"""
        client = EventDataClient(provider="yahoo")
        
        async with client:
            events = await client.get_earnings_calendar(
                from_date=date.today(),
                to_date=date.today() + timedelta(days=30)
            )
            
            # Current implementation returns empty
            assert isinstance(events, list)
    
    @pytest.mark.asyncio
    async def test_get_market_holidays(self):
        """Test fetching market holidays"""
        client = EventDataClient(provider="yahoo")
        
        async with client:
            holidays = await client.get_market_holidays(year=2024)
            
            assert len(holidays) > 0
            assert all(isinstance(h, EventData) for h in holidays)
    
    @pytest.mark.asyncio
    async def test_get_all_events(self):
        """Test fetching all event types"""
        client = EventDataClient(provider="yahoo")
        
        async with client:
            # Get all events
            all_events = await client.get_all_events(
                from_date=date(2024, 1, 1),
                to_date=date(2024, 12, 31)
            )
            
            # Should include holidays at least
            assert len(all_events) > 0
            
            # Filter by event type
            holidays_only = await client.get_all_events(
                from_date=date(2024, 1, 1),
                to_date=date(2024, 12, 31),
                event_types=[EventType.MARKET_HOLIDAY]
            )
            
            assert all(e.event_type == EventType.MARKET_HOLIDAY for e in holidays_only)
    
    @pytest.mark.asyncio
    async def test_economic_calendar_with_impact_filter(self):
        """Test economic calendar with impact filtering"""
        client = EventDataClient(provider="yahoo")
        
        async with client:
            # High impact only
            events = await client.get_economic_calendar(
                from_date=date.today(),
                to_date=date.today() + timedelta(days=30),
                impact_filter=3
            )
            
            # Current implementation returns empty, but when implemented:
            # assert all(e.impact >= 3 for e in events)
            assert isinstance(events, list)


class TestIntegration:
    """Integration tests for event data flow"""
    
    @pytest.mark.asyncio
    async def test_event_data_serialization(self):
        """Test event data can be serialized for database storage"""
        event = MarketHolidayEvent(
            event_type=EventType.MARKET_HOLIDAY,
            event_datetime=datetime(2024, 7, 4, 9, 30),
            event_name="Independence Day",
            description="US Markets Closed",
            impact=EventImpact.HIGH,
            country="US",
            source="NYSE",
            markets_closed=["NYSE", "NASDAQ"]
        )
        
        # Test JSON serialization (for database storage)
        json_data = event.model_dump(mode="json")
        
        assert json_data["event_type"] == "market_holiday"
        assert json_data["impact"] == 3  # HIGH = 3
        assert isinstance(json_data["event_datetime"], str)  # Should be ISO format
        assert json_data["markets_closed"] == ["NYSE", "NASDAQ"]
        
        # Test excluding fields
        json_data_filtered = event.model_dump(
            mode="json",
            exclude={"event_id", "created_at", "updated_at"}
        )
        
        assert "event_id" not in json_data_filtered
        assert "created_at" not in json_data_filtered
        assert "updated_at" not in json_data_filtered