"""Tests for event periods functionality."""

import pytest
from datetime import date
from src.simulators.historical.event_periods import (
    EventPeriod, 
    EVENT_PERIODS,
    get_event_by_date,
    get_overlapping_events
)


class TestEventPeriod:
    """Test EventPeriod class."""
    
    def test_event_period_creation(self):
        """Test creating an event period."""
        event = EventPeriod(
            name="Test Event",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 31),
            description="Test description",
            volatility_multiplier=2.0
        )
        
        assert event.name == "Test Event"
        assert event.start_date == date(2020, 1, 1)
        assert event.end_date == date(2020, 1, 31)
        assert event.volatility_multiplier == 2.0
    
    def test_contains_date(self):
        """Test date containment check."""
        event = EventPeriod(
            name="Test",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 31),
            description="Test"
        )
        
        assert event.contains_date(date(2020, 1, 15))
        assert event.contains_date(date(2020, 1, 1))
        assert event.contains_date(date(2020, 1, 31))
        assert not event.contains_date(date(2019, 12, 31))
        assert not event.contains_date(date(2020, 2, 1))
    
    def test_days_duration(self):
        """Test duration calculation."""
        event = EventPeriod(
            name="Test",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 31),
            description="Test"
        )
        
        assert event.days_duration() == 30


class TestEventPeriods:
    """Test pre-defined event periods."""
    
    def test_all_events_have_required_fields(self):
        """Test that all events have required fields."""
        for event_name, event in EVENT_PERIODS.items():
            assert event.name
            assert event.start_date
            assert event.end_date
            assert event.description
            assert event.volatility_multiplier >= 1.0
            assert event.start_date <= event.end_date
    
    def test_covid_crash_event(self):
        """Test COVID crash event details."""
        covid = EVENT_PERIODS["covid_crash"]
        
        assert covid.name == "COVID-19 Market Crash"
        assert covid.start_date == date(2020, 2, 20)
        assert covid.end_date == date(2020, 4, 30)
        assert covid.volatility_multiplier == 3.5
        assert len(covid.key_dates) > 0
        assert date(2020, 3, 23) in covid.key_dates  # Market bottom
    
    def test_gme_squeeze_event(self):
        """Test GameStop squeeze event."""
        gme = EVENT_PERIODS["gme_squeeze"]
        
        assert gme.name == "GameStop Short Squeeze"
        assert gme.affected_symbols is not None
        assert "GME" in gme.affected_symbols
        assert "AMC" in gme.affected_symbols
        assert gme.volatility_multiplier == 5.0  # Extreme volatility
    
    def test_no_overlapping_events(self):
        """Test that major events don't overlap significantly."""
        # Some events might have minor overlaps, but major ones shouldn't
        major_events = ["covid_crash", "svb_collapse", "financial_crisis"]
        
        for i, event1_name in enumerate(major_events):
            event1 = EVENT_PERIODS[event1_name]
            for event2_name in major_events[i+1:]:
                event2 = EVENT_PERIODS[event2_name]
                
                # Check no overlap
                assert (event1.end_date < event2.start_date or 
                       event2.end_date < event1.start_date)


class TestEventLookup:
    """Test event lookup functions."""
    
    def test_get_event_by_date(self):
        """Test finding event by date."""
        # Test COVID crash
        event = get_event_by_date(date(2020, 3, 15))
        assert event is not None
        assert event.name == "COVID-19 Market Crash"
        
        # Test no event
        event = get_event_by_date(date(2019, 6, 15))
        assert event is None
    
    def test_get_overlapping_events(self):
        """Test finding overlapping events."""
        # Test period covering COVID
        events = get_overlapping_events(
            date(2020, 1, 1),
            date(2020, 12, 31)
        )
        
        event_names = [e.name for e in events]
        assert "COVID-19 Market Crash" in event_names
        
        # Test period with no events
        events = get_overlapping_events(
            date(2015, 1, 1),
            date(2015, 6, 30)
        )
        
        assert len(events) == 0
    
    def test_multiple_overlapping_events(self):
        """Test finding multiple overlapping events."""
        # 2021 had both GME squeeze and Archegos
        events = get_overlapping_events(
            date(2021, 1, 1),
            date(2021, 4, 30)
        )
        
        event_names = [e.name for e in events]
        assert "GameStop Short Squeeze" in event_names
        assert "Archegos Capital Collapse" in event_names