"""Trading calendar utilities."""

from datetime import date, datetime, timedelta
from typing import List, Set, Dict
import pandas as pd
import pandas_market_calendars as mcal


class TradingCalendar:
    """Handle trading days and market hours."""
    
    def __init__(self, exchange: str = "NYSE"):
        """
        Initialize trading calendar.
        
        Args:
            exchange: Exchange calendar to use
        """
        self.exchange = exchange
        self.calendar = mcal.get_calendar(exchange)
        self._cache: Dict[tuple, List[date]] = {}
        
    def get_trading_days(self, start_date: date, end_date: date) -> List[date]:
        """
        Get list of trading days between dates.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of trading days
        """
        cache_key = (start_date, end_date)
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        # Get valid trading days
        schedule = self.calendar.valid_days(
            start_date=pd.Timestamp(start_date),
            end_date=pd.Timestamp(end_date)
        )
        
        trading_days = [ts.date() for ts in schedule]
        self._cache[cache_key] = trading_days
        
        return trading_days
    
    def is_trading_day(self, check_date: date) -> bool:
        """Check if given date is a trading day."""
        trading_days = self.get_trading_days(check_date, check_date)
        return len(trading_days) > 0
    
    def next_trading_day(self, from_date: date) -> date:
        """Get next trading day after given date."""
        # Look ahead up to 10 days (covers long weekends)
        for i in range(1, 11):
            next_date = from_date + timedelta(days=i)
            if self.is_trading_day(next_date):
                return next_date
        raise ValueError(f"No trading day found within 10 days of {from_date}")
    
    def previous_trading_day(self, from_date: date) -> date:
        """Get previous trading day before given date."""
        # Look back up to 10 days
        for i in range(1, 11):
            prev_date = from_date - timedelta(days=i)
            if self.is_trading_day(prev_date):
                return prev_date
        raise ValueError(f"No trading day found within 10 days before {from_date}")
    
    def count_trading_days(self, start_date: date, end_date: date) -> int:
        """Count trading days between dates."""
        trading_days = self.get_trading_days(start_date, end_date)
        return len(trading_days)
    
    def get_market_hours(self, trading_date: date) -> tuple[datetime, datetime]:
        """
        Get market open and close times for a trading day.
        
        Args:
            trading_date: Trading date
            
        Returns:
            Tuple of (market_open, market_close) datetimes
        """
        schedule = self.calendar.schedule(
            start_date=pd.Timestamp(trading_date),
            end_date=pd.Timestamp(trading_date)
        )
        
        if schedule.empty:
            raise ValueError(f"{trading_date} is not a trading day")
            
        market_open = schedule.iloc[0]['market_open'].to_pydatetime()
        market_close = schedule.iloc[0]['market_close'].to_pydatetime()
        
        return market_open, market_close
    
    def is_market_open(self, check_time: datetime) -> bool:
        """Check if market is open at given time."""
        check_date = check_time.date()
        
        if not self.is_trading_day(check_date):
            return False
            
        market_open, market_close = self.get_market_hours(check_date)
        return market_open <= check_time <= market_close
    
    def get_holidays(self, year: int) -> List[date]:
        """Get list of market holidays for a year."""
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        
        # Get all calendar days
        all_days = pd.date_range(start, end, freq='B')  # Business days
        
        # Get trading days
        trading_days = set(self.get_trading_days(start, end))
        
        # Holidays are business days that aren't trading days
        holidays = [d.date() for d in all_days if d.date() not in trading_days]
        
        return holidays
    
    def clear_cache(self) -> None:
        """Clear cached trading days."""
        self._cache.clear()