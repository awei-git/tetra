"""Event calendar definitions for economic and market events to track"""

from enum import Enum
from typing import List, Tuple, Dict, Optional
from datetime import time


class EventFrequency(str, Enum):
    """Frequency of recurring events"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    FOMC = "fomc"  # ~8 times per year
    OPEX = "opex"  # Monthly options expiration
    EARNINGS = "earnings"  # Quarterly earnings seasons


class EventCalendar:
    """Central definition of events to track"""
    
    # Economic events to track
    # Format: (event_name, frequency, typical_time, impact_level, description)
    ECONOMIC_EVENTS = [
        # Federal Reserve Events
        ("FOMC Meeting", EventFrequency.FOMC, "14:00", 4, "Federal Open Market Committee interest rate decision"),
        ("FOMC Minutes", EventFrequency.FOMC, "14:00", 3, "Minutes from previous FOMC meeting"),
        ("Fed Chair Speech", EventFrequency.MONTHLY, None, 3, "Federal Reserve Chair public appearance"),
        
        # Employment
        ("Non-Farm Payrolls", EventFrequency.MONTHLY, "08:30", 4, "Monthly employment report"),
        ("Unemployment Rate", EventFrequency.MONTHLY, "08:30", 3, "Monthly unemployment rate"),
        ("Initial Jobless Claims", EventFrequency.WEEKLY, "08:30", 2, "Weekly unemployment claims"),
        ("ADP Employment", EventFrequency.MONTHLY, "08:15", 3, "ADP private payrolls report"),
        
        # Inflation
        ("CPI", EventFrequency.MONTHLY, "08:30", 4, "Consumer Price Index"),
        ("Core CPI", EventFrequency.MONTHLY, "08:30", 4, "Core CPI excluding food and energy"),
        ("PPI", EventFrequency.MONTHLY, "08:30", 3, "Producer Price Index"),
        ("PCE Price Index", EventFrequency.MONTHLY, "08:30", 3, "Personal Consumption Expenditure Price Index"),
        
        # GDP and Growth
        ("GDP", EventFrequency.QUARTERLY, "08:30", 4, "Gross Domestic Product"),
        ("Retail Sales", EventFrequency.MONTHLY, "08:30", 3, "Monthly retail sales"),
        ("Industrial Production", EventFrequency.MONTHLY, "09:15", 2, "Industrial production index"),
        ("Durable Goods Orders", EventFrequency.MONTHLY, "08:30", 3, "Durable goods orders"),
        
        # Housing
        ("Housing Starts", EventFrequency.MONTHLY, "08:30", 2, "New residential construction"),
        ("Building Permits", EventFrequency.MONTHLY, "08:30", 2, "Building permits issued"),
        ("Existing Home Sales", EventFrequency.MONTHLY, "10:00", 2, "Existing home sales"),
        ("New Home Sales", EventFrequency.MONTHLY, "10:00", 2, "New home sales"),
        
        # Consumer
        ("Consumer Confidence", EventFrequency.MONTHLY, "10:00", 3, "Conference Board Consumer Confidence"),
        ("Michigan Consumer Sentiment", EventFrequency.MONTHLY, "10:00", 2, "University of Michigan Consumer Sentiment"),
        
        # Manufacturing
        ("ISM Manufacturing PMI", EventFrequency.MONTHLY, "10:00", 3, "ISM Manufacturing Index"),
        ("ISM Services PMI", EventFrequency.MONTHLY, "10:00", 3, "ISM Non-Manufacturing Index"),
        
        # International Central Banks
        ("ECB Rate Decision", EventFrequency.MONTHLY, "07:45", 3, "European Central Bank rate decision"),
        ("BOE Rate Decision", EventFrequency.MONTHLY, "07:00", 3, "Bank of England rate decision"),
        ("BOJ Rate Decision", EventFrequency.QUARTERLY, "23:00", 3, "Bank of Japan rate decision"),
    ]
    
    # Market structure events
    # Format: (event_name, frequency, typical_time, impact_level, description)
    MARKET_EVENTS = [
        # Options expiration
        ("Monthly OpEx", EventFrequency.OPEX, "16:00", 3, "Monthly options expiration"),
        ("Quarterly OpEx", EventFrequency.QUARTERLY, "16:00", 4, "Quarterly options expiration (Triple Witching)"),
        
        # Index rebalancing
        ("S&P 500 Rebalance", EventFrequency.QUARTERLY, "16:00", 3, "S&P 500 index rebalancing"),
        ("Russell Rebalance", EventFrequency.ANNUAL, "16:00", 4, "Russell index reconstitution"),
        
        # Futures
        ("VIX Futures Expiration", EventFrequency.MONTHLY, "09:30", 2, "VIX futures expiration"),
        ("Futures Rollover", EventFrequency.QUARTERLY, None, 2, "Quarterly futures rollover"),
    ]
    
    # Major US market holidays
    MARKET_HOLIDAYS = [
        "New Year's Day",
        "Martin Luther King Jr. Day",
        "Presidents Day",
        "Good Friday",
        "Memorial Day",
        "Independence Day",
        "Labor Day",
        "Thanksgiving Day",
        "Christmas Day",
    ]
    
    # Earnings seasons (approximate windows)
    EARNINGS_SEASONS = [
        ("Q4 Earnings", "January", "February"),  # Q4 previous year
        ("Q1 Earnings", "April", "May"),
        ("Q2 Earnings", "July", "August"),
        ("Q3 Earnings", "October", "November"),
    ]
    
    # High-impact stock symbols for earnings tracking
    EARNINGS_FOCUS_SYMBOLS = [
        # Mega-caps (market movers)
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
        
        # Major banks (start of earnings season)
        "JPM", "BAC", "WFC", "GS", "MS", "C",
        
        # Key sectors
        "XOM", "CVX",  # Energy
        "JNJ", "PFE", "UNH",  # Healthcare
        "WMT", "HD", "COST",  # Consumer
        "BA", "CAT",  # Industrial
        "NFLX", "DIS",  # Media
        "AMD", "INTC", "CRM",  # Tech
    ]
    
    @classmethod
    def get_all_economic_events(cls) -> List[Tuple[str, EventFrequency, Optional[str], int, str]]:
        """Get all economic events"""
        return cls.ECONOMIC_EVENTS
    
    @classmethod
    def get_all_market_events(cls) -> List[Tuple[str, EventFrequency, Optional[str], int, str]]:
        """Get all market structure events"""
        return cls.MARKET_EVENTS
    
    @classmethod
    def get_high_impact_events(cls, min_impact: int = 3) -> List[Tuple[str, EventFrequency, Optional[str], int, str]]:
        """Get high impact events (impact >= min_impact)"""
        all_events = cls.ECONOMIC_EVENTS + cls.MARKET_EVENTS
        return [event for event in all_events if event[3] >= min_impact]
    
    @classmethod
    def get_events_by_frequency(cls, frequency: EventFrequency) -> List[Tuple[str, EventFrequency, Optional[str], int, str]]:
        """Get events by frequency"""
        all_events = cls.ECONOMIC_EVENTS + cls.MARKET_EVENTS
        return [event for event in all_events if event[1] == frequency]
    
    @classmethod
    def get_earnings_symbols(cls) -> List[str]:
        """Get symbols to track for earnings"""
        return cls.EARNINGS_FOCUS_SYMBOLS
    
    @classmethod
    def get_event_categories(cls) -> Dict[str, List[str]]:
        """Get events organized by category"""
        categories = {
            "fed_events": ["FOMC Meeting", "FOMC Minutes", "Fed Chair Speech"],
            "employment": ["Non-Farm Payrolls", "Unemployment Rate", "Initial Jobless Claims", "ADP Employment"],
            "inflation": ["CPI", "Core CPI", "PPI", "PCE Price Index"],
            "growth": ["GDP", "Retail Sales", "Industrial Production", "Durable Goods Orders"],
            "housing": ["Housing Starts", "Building Permits", "Existing Home Sales", "New Home Sales"],
            "sentiment": ["Consumer Confidence", "Michigan Consumer Sentiment"],
            "manufacturing": ["ISM Manufacturing PMI", "ISM Services PMI"],
            "central_banks": ["ECB Rate Decision", "BOE Rate Decision", "BOJ Rate Decision"],
            "options": ["Monthly OpEx", "Quarterly OpEx"],
            "index": ["S&P 500 Rebalance", "Russell Rebalance"],
        }
        return categories
    
    @classmethod
    def get_event_info(cls, event_name: str) -> Optional[Dict[str, any]]:
        """Get information about a specific event"""
        all_events = cls.ECONOMIC_EVENTS + cls.MARKET_EVENTS
        
        for event in all_events:
            if event[0].lower() == event_name.lower():
                return {
                    "name": event[0],
                    "frequency": event[1],
                    "typical_time": event[2],
                    "impact": event[3],
                    "description": event[4]
                }
        
        return None