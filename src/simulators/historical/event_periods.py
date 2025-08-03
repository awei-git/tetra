"""Pre-defined market event periods for simulation."""

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional


@dataclass
class EventPeriod:
    """Definition of a market event period."""
    
    name: str
    start_date: date
    end_date: date
    description: str
    key_dates: Optional[Dict[date, str]] = None
    affected_symbols: Optional[list[str]] = None
    volatility_multiplier: float = 1.0
    
    def contains_date(self, check_date: date) -> bool:
        """Check if date falls within event period."""
        return self.start_date <= check_date <= self.end_date
    
    def days_duration(self) -> int:
        """Get event duration in days."""
        return (self.end_date - self.start_date).days


# Pre-defined market events
EVENT_PERIODS = {
    "covid_crash": EventPeriod(
        name="COVID-19 Market Crash",
        start_date=date(2020, 2, 20),
        end_date=date(2020, 4, 30),
        description="Initial COVID panic and recovery",
        key_dates={
            date(2020, 3, 9): "First circuit breaker since 1997",
            date(2020, 3, 12): "Worst day since 1987 crash",
            date(2020, 3, 16): "Second circuit breaker", 
            date(2020, 3, 18): "Third circuit breaker",
            date(2020, 3, 23): "Market bottom, Fed announces unlimited QE",
            date(2020, 3, 24): "Start of recovery rally"
        },
        volatility_multiplier=3.5
    ),
    
    "svb_collapse": EventPeriod(
        name="Silicon Valley Bank Collapse",
        start_date=date(2023, 3, 8),
        end_date=date(2023, 3, 31),
        description="Regional banking crisis triggered by SVB failure",
        key_dates={
            date(2023, 3, 10): "SVB seized by regulators",
            date(2023, 3, 12): "Signature Bank fails",
            date(2023, 3, 13): "Fed backstop announced",
            date(2023, 3, 16): "Credit Suisse rescue",
            date(2023, 3, 19): "UBS acquires Credit Suisse"
        },
        affected_symbols=["KRE", "SIVB", "SBNY", "FRC", "PACW"],
        volatility_multiplier=2.0
    ),
    
    "gme_squeeze": EventPeriod(
        name="GameStop Short Squeeze",
        start_date=date(2021, 1, 11),
        end_date=date(2021, 2, 10),
        description="Retail-driven short squeeze via r/wallstreetbets",
        key_dates={
            date(2021, 1, 22): "GME hits $65, shorts under pressure",
            date(2021, 1, 27): "GME peaks near $350",
            date(2021, 1, 28): "Robinhood restricts trading",
            date(2021, 2, 2): "Second squeeze attempt"
        },
        affected_symbols=["GME", "AMC", "BB", "NOK", "BBBY"],
        volatility_multiplier=5.0
    ),
    
    "fed_taper_2022": EventPeriod(
        name="Fed Rate Hike Cycle 2022",
        start_date=date(2022, 3, 1),
        end_date=date(2022, 12, 31),
        description="Aggressive rate hikes to combat inflation",
        key_dates={
            date(2022, 3, 16): "First rate hike (25bp)",
            date(2022, 5, 4): "50bp hike",
            date(2022, 6, 15): "75bp hike (largest since 1994)",
            date(2022, 7, 27): "Second 75bp hike",
            date(2022, 9, 21): "Third 75bp hike",
            date(2022, 11, 2): "Fourth 75bp hike",
            date(2022, 12, 14): "50bp hike, peak hawkishness"
        },
        volatility_multiplier=1.8
    ),
    
    "trump_election": EventPeriod(
        name="Trump Election 2016",
        start_date=date(2016, 11, 1),
        end_date=date(2016, 12, 31),
        description="Election surprise and subsequent rally",
        key_dates={
            date(2016, 11, 8): "Election day",
            date(2016, 11, 9): "Trump victory confirmed, futures limit down then rally",
            date(2016, 11, 30): "DOW breaks 19,000"
        },
        volatility_multiplier=1.5
    ),
    
    "financial_crisis": EventPeriod(
        name="2008 Financial Crisis",
        start_date=date(2008, 9, 1),
        end_date=date(2009, 3, 31),
        description="Global financial system meltdown",
        key_dates={
            date(2008, 9, 15): "Lehman Brothers bankruptcy",
            date(2008, 9, 16): "AIG bailout",
            date(2008, 9, 29): "TARP rejected, DOW drops 777 points",
            date(2008, 10, 3): "TARP passed",
            date(2008, 11, 20): "S&P hits crisis low",
            date(2009, 3, 9): "Market bottom"
        },
        volatility_multiplier=4.0
    ),
    
    "dotcom_crash": EventPeriod(
        name="Dot-Com Bubble Burst",
        start_date=date(2000, 3, 10),
        end_date=date(2002, 10, 9),
        description="Tech bubble burst and bear market",
        key_dates={
            date(2000, 3, 10): "NASDAQ peaks at 5,048",
            date(2000, 4, 14): "Black Friday tech selloff",
            date(2001, 9, 11): "9/11 attacks",
            date(2002, 7, 24): "WorldCom bankruptcy",
            date(2002, 10, 9): "NASDAQ bottoms at 1,114"
        },
        affected_symbols=["QQQ", "CSCO", "INTC", "MSFT", "ORCL"],
        volatility_multiplier=2.5
    ),
    
    "flash_crash": EventPeriod(
        name="2010 Flash Crash",
        start_date=date(2010, 5, 6),
        end_date=date(2010, 5, 6),
        description="1000-point intraday crash and recovery",
        key_dates={
            date(2010, 5, 6): "DOW drops 1000 points in minutes"
        },
        volatility_multiplier=10.0  # Extreme intraday volatility
    ),
    
    "brexit": EventPeriod(
        name="Brexit Vote",
        start_date=date(2016, 6, 20),
        end_date=date(2016, 6, 30),
        description="UK votes to leave European Union",
        key_dates={
            date(2016, 6, 23): "Brexit referendum",
            date(2016, 6, 24): "Leave vote wins, markets crash",
            date(2016, 6, 27): "Continued volatility"
        },
        affected_symbols=["EWU", "FXB", "EWG", "EU"],
        volatility_multiplier=2.0
    ),
    
    "volmageddon": EventPeriod(
        name="Volmageddon",
        start_date=date(2018, 2, 2),
        end_date=date(2018, 2, 9),
        description="VIX spike causes inverse volatility product collapse",
        key_dates={
            date(2018, 2, 5): "VIX spikes 116%, XIV loses 90%",
            date(2018, 2, 6): "XIV termination announced"
        },
        affected_symbols=["VXX", "UVXY", "SVXY", "VIX"],
        volatility_multiplier=3.0
    ),
    
    "china_trade_war": EventPeriod(
        name="US-China Trade War Escalation",
        start_date=date(2019, 5, 1),
        end_date=date(2019, 8, 31),
        description="Trade war escalation and tariff threats",
        key_dates={
            date(2019, 5, 5): "Trump threatens new tariffs",
            date(2019, 5, 13): "China retaliates",
            date(2019, 8, 1): "New 10% tariffs announced",
            date(2019, 8, 5): "Yuan breaks 7.0",
            date(2019, 8, 23): "Powell Jackson Hole speech"
        },
        volatility_multiplier=1.7
    ),
    
    "archegos_blowup": EventPeriod(
        name="Archegos Capital Collapse",
        start_date=date(2021, 3, 22),
        end_date=date(2021, 3, 31),
        description="Family office margin call triggers bank losses",
        key_dates={
            date(2021, 3, 26): "Block trades begin",
            date(2021, 3, 29): "Credit Suisse, Nomura warn of losses"
        },
        affected_symbols=["VIAC", "DISCA", "BIDU", "TME", "CS", "NMR"],
        volatility_multiplier=2.0
    )
}


def get_event_by_date(check_date: date) -> Optional[EventPeriod]:
    """Find event period containing given date."""
    for event in EVENT_PERIODS.values():
        if event.contains_date(check_date):
            return event
    return None


def get_overlapping_events(start_date: date, end_date: date) -> list[EventPeriod]:
    """Find all events overlapping with date range."""
    overlapping = []
    for event in EVENT_PERIODS.values():
        if (event.start_date <= end_date and event.end_date >= start_date):
            overlapping.append(event)
    return overlapping