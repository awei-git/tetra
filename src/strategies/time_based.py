"""Time-based trading strategies."""

from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import logging
from zoneinfo import ZoneInfo

from .base import BaseStrategy, PositionSide, Trade, Position

logger = logging.getLogger(__name__)


class TimeFrame(Enum):
    """Trading time frames."""
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class SessionType(Enum):
    """Trading session types."""
    PREMARKET = "premarket"
    REGULAR = "regular"
    AFTERHOURS = "afterhours"
    OVERNIGHT = "overnight"
    ASIAN = "asian"
    EUROPEAN = "european"
    US = "us"


@dataclass
class TradingWindow:
    """Defines a time window for trading."""
    start_time: time
    end_time: time
    days_of_week: List[int] = field(default_factory=lambda: list(range(5)))  # Mon-Fri
    timezone: str = "America/New_York"
    session_type: SessionType = SessionType.REGULAR
    
    def is_active(self, timestamp: datetime) -> bool:
        """Check if timestamp is within trading window."""
        # Convert to market timezone
        tz = ZoneInfo(self.timezone)
        market_time = timestamp.astimezone(tz)
        
        # Check day of week (0=Monday, 6=Sunday)
        if market_time.weekday() not in self.days_of_week:
            return False
        
        # Check time of day
        current_time = market_time.time()
        
        # Handle overnight sessions
        if self.start_time > self.end_time:
            return current_time >= self.start_time or current_time <= self.end_time
        else:
            return self.start_time <= current_time <= self.end_time


@dataclass
class TradingSchedule:
    """Complete trading schedule with multiple windows."""
    windows: List[TradingWindow]
    blackout_dates: Set[datetime.date] = field(default_factory=set)
    max_trades_per_day: int = 10
    max_trades_per_window: int = 3
    force_close_end_of_day: bool = True
    force_close_time: time = time(15, 45)  # 15 minutes before market close
    
    def get_active_window(self, timestamp: datetime) -> Optional[TradingWindow]:
        """Get the currently active trading window."""
        # Check blackout dates
        if timestamp.date() in self.blackout_dates:
            return None
        
        # Find active window
        for window in self.windows:
            if window.is_active(timestamp):
                return window
        
        return None
    
    def should_force_close(self, timestamp: datetime) -> bool:
        """Check if positions should be force closed."""
        if not self.force_close_end_of_day:
            return False
        
        tz = ZoneInfo("America/New_York")
        market_time = timestamp.astimezone(tz)
        
        return (
            market_time.time() >= self.force_close_time and
            market_time.weekday() < 5  # Weekday
        )


class TimeBasedStrategy(BaseStrategy):
    """Strategy that trades based on time constraints."""
    
    def __init__(self,
                 name: str,
                 trading_schedule: TradingSchedule,
                 entry_strategy: Optional[Callable] = None,
                 exit_strategy: Optional[Callable] = None,
                 **kwargs):
        super().__init__(name, **kwargs)
        self.trading_schedule = trading_schedule
        self.entry_strategy = entry_strategy
        self.exit_strategy = exit_strategy
        self.daily_trades: Dict[datetime.date, int] = {}
        self.window_trades: Dict[Tuple[datetime.date, TradingWindow], int] = {}
    
    def generate_signals(self, 
                        data: pd.DataFrame,
                        signals: Optional[pd.DataFrame] = None,
                        events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate trading signals based on time."""
        output_signals = pd.DataFrame(index=data.index)
        output_signals['signal'] = 0
        output_signals['position_size'] = 0
        output_signals['window'] = None
        
        for timestamp in data.index:
            # Check if we're in a trading window
            window = self.trading_schedule.get_active_window(timestamp)
            
            if window:
                output_signals.loc[timestamp, 'window'] = window.session_type.value
                
                # Check trading limits
                date = timestamp.date()
                if self._check_trading_limits(date, window):
                    # Apply entry strategy if provided
                    if self.entry_strategy:
                        entry_signal = self.entry_strategy(
                            data.loc[timestamp],
                            signals.loc[timestamp] if signals is not None else None,
                            window
                        )
                        if entry_signal:
                            output_signals.loc[timestamp, 'signal'] = entry_signal['signal']
                            output_signals.loc[timestamp, 'position_size'] = entry_signal['size']
                            self._record_trade(date, window)
            
            # Check force close
            if self.trading_schedule.should_force_close(timestamp):
                output_signals.loc[timestamp, 'signal'] = 0  # Close all positions
        
        return output_signals
    
    def should_enter(self, 
                    symbol: str,
                    timestamp: datetime,
                    data: pd.Series,
                    signals: Optional[pd.Series] = None,
                    events: Optional[pd.Series] = None) -> Tuple[bool, PositionSide, float]:
        """Determine if should enter position based on time."""
        # Check if we're in a trading window
        window = self.trading_schedule.get_active_window(timestamp)
        if not window:
            return False, PositionSide.FLAT, 0.0
        
        # Check trading limits
        date = timestamp.date()
        if not self._check_trading_limits(date, window):
            return False, PositionSide.FLAT, 0.0
        
        # Apply custom entry logic
        if self.entry_strategy:
            result = self.entry_strategy(data, signals, window)
            if result:
                self._record_trade(date, window)
                return result['should_enter'], result['side'], result['size']
        
        # Default time-based entry logic
        if window.session_type == SessionType.PREMARKET:
            # Trade momentum in pre-market
            if signals and 'gap' in signals and abs(signals['gap']) > 0.02:
                side = PositionSide.LONG if signals['gap'] > 0 else PositionSide.SHORT
                return True, side, self.position_size
        
        elif window.session_type == SessionType.REGULAR:
            # Trade during regular hours based on signals
            if signals and 'trend' in signals:
                if signals['trend'] > 0.5:
                    return True, PositionSide.LONG, self.position_size
                elif signals['trend'] < -0.5:
                    return True, PositionSide.SHORT, self.position_size
        
        return False, PositionSide.FLAT, 0.0
    
    def should_exit(self,
                   position: Position,
                   timestamp: datetime,
                   data: pd.Series,
                   signals: Optional[pd.Series] = None,
                   events: Optional[pd.Series] = None) -> bool:
        """Determine if should exit position."""
        # Check standard risk limits
        if self.check_risk_limits(position, data['close']):
            return True
        
        # Force close at end of day
        if self.trading_schedule.should_force_close(timestamp):
            logger.info(f"Force closing {position.symbol} at end of day")
            return True
        
        # Check if we've left the trading window
        window = self.trading_schedule.get_active_window(timestamp)
        if not window and 'entry_window' in position.metadata:
            logger.info(f"Exiting {position.symbol} - outside trading window")
            return True
        
        # Apply custom exit logic
        if self.exit_strategy:
            return self.exit_strategy(position, data, signals, window)
        
        return False
    
    def _check_trading_limits(self, date: datetime.date, window: TradingWindow) -> bool:
        """Check if trading limits allow new trades."""
        # Check daily limit
        daily_count = self.daily_trades.get(date, 0)
        if daily_count >= self.trading_schedule.max_trades_per_day:
            return False
        
        # Check window limit
        window_key = (date, id(window))  # Use id for hashable key
        window_count = self.window_trades.get(window_key, 0)
        if window_count >= self.trading_schedule.max_trades_per_window:
            return False
        
        return True
    
    def _record_trade(self, date: datetime.date, window: TradingWindow):
        """Record a trade for limit tracking."""
        self.daily_trades[date] = self.daily_trades.get(date, 0) + 1
        window_key = (date, id(window))  # Use id for hashable key
        self.window_trades[window_key] = self.window_trades.get(window_key, 0) + 1


# Predefined time-based strategies
class IntradayStrategy(TimeBasedStrategy):
    """Intraday trading strategy with no overnight positions."""
    
    def __init__(self, name: str = "Intraday Strategy", **kwargs):
        # Define intraday trading windows
        windows = [
            TradingWindow(
                start_time=time(9, 30),
                end_time=time(10, 30),
                session_type=SessionType.REGULAR
            ),
            TradingWindow(
                start_time=time(14, 0),
                end_time=time(15, 45),
                session_type=SessionType.REGULAR
            )
        ]
        
        schedule = TradingSchedule(
            windows=windows,
            force_close_end_of_day=True,
            max_trades_per_day=5,
            max_trades_per_window=2
        )
        
        super().__init__(
            name=name,
            trading_schedule=schedule,
            entry_strategy=self._intraday_entry,
            exit_strategy=self._intraday_exit,
            **kwargs
        )
    
    def _intraday_entry(self, data, signals, window):
        """Intraday entry logic."""
        # Morning session - trade breakouts
        if window.start_time == time(9, 30):
            if signals and 'opening_range_break' in signals:
                if signals['opening_range_break'] > 0:
                    return {'should_enter': True, 'side': PositionSide.LONG, 'size': self.position_size}
                elif signals['opening_range_break'] < 0:
                    return {'should_enter': True, 'side': PositionSide.SHORT, 'size': self.position_size}
        
        # Afternoon session - trade momentum
        elif window.start_time == time(14, 0):
            if signals and 'momentum' in signals:
                if signals['momentum'] > 0.7:
                    return {'should_enter': True, 'side': PositionSide.LONG, 'size': self.position_size * 0.8}
        
        return None
    
    def _intraday_exit(self, position, data, signals, window):
        """Intraday exit logic."""
        # Exit if momentum reverses
        if signals and 'momentum' in signals:
            if position.side == PositionSide.LONG and signals['momentum'] < -0.3:
                return True
            elif position.side == PositionSide.SHORT and signals['momentum'] > 0.3:
                return True
        
        # Time-based exit for morning positions
        if 'entry_time' in position.metadata:
            entry_time = position.metadata['entry_time']
            if (datetime.now() - entry_time).total_seconds() > 3600:  # 1 hour
                return True
        
        return False


class OvernightStrategy(TimeBasedStrategy):
    """Strategy for overnight/gap trading."""
    
    def __init__(self, name: str = "Overnight Strategy", **kwargs):
        windows = [
            TradingWindow(
                start_time=time(15, 30),
                end_time=time(16, 0),
                session_type=SessionType.REGULAR
            ),
            TradingWindow(
                start_time=time(9, 0),
                end_time=time(9, 45),
                session_type=SessionType.PREMARKET
            )
        ]
        
        schedule = TradingSchedule(
            windows=windows,
            force_close_end_of_day=False,  # Allow overnight positions
            max_trades_per_day=2
        )
        
        super().__init__(
            name=name,
            trading_schedule=schedule,
            **kwargs
        )


class AsianSessionStrategy(TimeBasedStrategy):
    """Strategy for Asian market hours."""
    
    def __init__(self, name: str = "Asian Session Strategy", **kwargs):
        windows = [
            TradingWindow(
                start_time=time(19, 0),  # 7 PM ET
                end_time=time(3, 0),     # 3 AM ET next day
                timezone="America/New_York",
                session_type=SessionType.ASIAN
            )
        ]
        
        schedule = TradingSchedule(
            windows=windows,
            force_close_end_of_day=False,
            max_trades_per_day=3
        )
        
        super().__init__(
            name=name,
            trading_schedule=schedule,
            **kwargs
        )