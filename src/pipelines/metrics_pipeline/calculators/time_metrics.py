"""Time-based metrics calculator for intraday and session-based analysis."""

import pandas as pd
import numpy as np
from datetime import time, datetime
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from src.utils.logging import logger


class TimeMetricsCalculator:
    """Calculate time-based metrics for trading strategies."""
    
    def __init__(self, timezone: str = "America/New_York"):
        """
        Initialize time metrics calculator.
        
        Args:
            timezone: Market timezone (default: US Eastern)
        """
        self.timezone = ZoneInfo(timezone)
        
        # Define trading sessions (ET)
        self.sessions = {
            'premarket': (time(4, 0), time(9, 30)),
            'regular': (time(9, 30), time(16, 0)),
            'afterhours': (time(16, 0), time(20, 0)),
            'overnight': (time(20, 0), time(4, 0))  # Next day
        }
        
        # Define intraday periods
        self.intraday_periods = {
            'opening_range': (time(9, 30), time(10, 0)),
            'morning': (time(10, 0), time(12, 0)),
            'lunch': (time(12, 0), time(13, 0)),
            'afternoon': (time(13, 0), time(15, 0)),
            'power_hour': (time(15, 0), time(16, 0)),
            'closing': (time(15, 45), time(16, 0))
        }
    
    def calculate_time_metrics(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Calculate all time-based metrics.
        
        Args:
            data: DataFrame with OHLCV data and datetime index
            symbol: Symbol being analyzed
            
        Returns:
            DataFrame with time-based metrics added
        """
        logger.debug(f"Calculating time metrics for {symbol}")
        
        # Ensure we have a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Add time components
        data['hour'] = data.index.hour
        data['minute'] = data.index.minute
        data['day_of_week'] = data.index.dayofweek  # 0=Monday, 6=Sunday
        data['is_weekday'] = data['day_of_week'] < 5
        
        # Calculate gap
        data['prev_close'] = data['close'].shift(1)
        data['gap'] = (data['open'] - data['prev_close']) / data['prev_close']
        data['gap_size'] = np.abs(data['gap'])
        data['gap_up'] = data['gap'] > 0
        data['gap_down'] = data['gap'] < 0
        
        # Identify sessions (simplified for daily data)
        data['session'] = 'regular'  # Default for daily data
        
        # If we have intraday data (check if we have multiple data points per day)
        if len(data) > 0 and data.index[0].date() == data.index[-1].date():
            # This is intraday data
            data = self._add_intraday_metrics(data)
        
        # Opening range calculations (for first 30 min of day)
        data['opening_high'] = np.nan
        data['opening_low'] = np.nan
        data['opening_range'] = np.nan
        data['opening_range_break'] = 0
        
        # Group by date for daily calculations
        for date in data.index.normalize().unique():
            day_mask = data.index.normalize() == date
            day_data = data[day_mask]
            
            if len(day_data) > 0:
                # Calculate opening range (first value of the day)
                opening_idx = day_data.index[0]
                data.loc[opening_idx, 'opening_high'] = day_data.iloc[0]['high']
                data.loc[opening_idx, 'opening_low'] = day_data.iloc[0]['low']
                data.loc[opening_idx, 'opening_range'] = day_data.iloc[0]['high'] - day_data.iloc[0]['low']
                
                # Check for opening range breakout
                if len(day_data) > 1:
                    opening_high = day_data.iloc[0]['high']
                    opening_low = day_data.iloc[0]['low']
                    
                    for idx in day_data.index[1:]:
                        if data.loc[idx, 'high'] > opening_high:
                            data.loc[idx, 'opening_range_break'] = 1  # Upside break
                        elif data.loc[idx, 'low'] < opening_low:
                            data.loc[idx, 'opening_range_break'] = -1  # Downside break
        
        # Time-based volume analysis
        data['volume_percentile'] = data['volume'].rolling(20).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
        )
        
        # Day of week effects
        data['monday'] = data['day_of_week'] == 0
        data['friday'] = data['day_of_week'] == 4
        
        # Month-end effects
        data['month'] = data.index.month
        data['is_month_end'] = data.index.is_month_end
        data['is_quarter_end'] = data.index.is_quarter_end
        
        # Options expiration (simplified - third Friday)
        data['is_opex'] = False
        for idx in data.index:
            if idx.weekday() == 4:  # Friday
                # Check if it's the third Friday
                if 15 <= idx.day <= 21:
                    data.loc[idx, 'is_opex'] = True
        
        # Momentum by time of day (simplified for daily data)
        data['morning_momentum'] = np.nan
        data['afternoon_momentum'] = np.nan
        
        # For daily data, use open-to-high and low-to-close as proxies
        data['morning_momentum'] = (data['high'] - data['open']) / data['open']
        data['afternoon_momentum'] = (data['close'] - data['low']) / data['low']
        
        # Time decay factor (for options-like strategies)
        data['days_to_friday'] = (4 - data['day_of_week']) % 7
        data['week_of_month'] = (data.index.day - 1) // 7 + 1
        
        # Overnight vs intraday returns
        data['overnight_return'] = data['gap']  # Gap is overnight return
        data['intraday_return'] = (data['close'] - data['open']) / data['open']
        
        # Session volatility (simplified)
        data['session_volatility'] = (data['high'] - data['low']) / data['open']
        
        # Time-weighted average price (already calculated as vwap if available)
        if 'vwap' not in data.columns and 'volume' in data.columns:
            # Simple VWAP calculation
            data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        
        # VWAP deviation
        if 'vwap' in data.columns:
            data['vwap_deviation'] = (data['close'] - data['vwap']) / data['vwap']
        
        return data
    
    def _add_intraday_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add intraday-specific metrics."""
        
        # Convert to market timezone
        data.index = data.index.tz_localize(self.timezone) if data.index.tz is None else data.index.tz_convert(self.timezone)
        
        # Identify trading session
        for idx in data.index:
            current_time = idx.time()
            
            # Check each session
            for session_name, (start, end) in self.sessions.items():
                if session_name == 'overnight':
                    # Overnight spans midnight
                    if current_time >= start or current_time < end:
                        data.loc[idx, 'session'] = session_name
                        break
                else:
                    if start <= current_time < end:
                        data.loc[idx, 'session'] = session_name
                        break
            
            # Identify intraday period
            for period_name, (start, end) in self.intraday_periods.items():
                if start <= current_time < end:
                    data.loc[idx, 'intraday_period'] = period_name
                    break
        
        # Calculate session-specific metrics
        for session in ['premarket', 'regular', 'afterhours']:
            session_mask = data['session'] == session
            if session_mask.any():
                # Session volume
                data.loc[session_mask, f'{session}_volume'] = data.loc[session_mask, 'volume']
                
                # Session volatility
                session_data = data.loc[session_mask]
                if len(session_data) > 1:
                    session_vol = session_data['close'].pct_change().std()
                    data.loc[session_mask, f'{session}_volatility'] = session_vol
        
        return data
    
    def get_session_type(self, timestamp: datetime) -> str:
        """
        Get the trading session type for a given timestamp.
        
        Args:
            timestamp: Datetime to check
            
        Returns:
            Session type string
        """
        # Convert to market timezone
        market_time = timestamp.astimezone(self.timezone) if timestamp.tzinfo else timestamp.replace(tzinfo=self.timezone)
        current_time = market_time.time()
        
        # Check each session
        for session_name, (start, end) in self.sessions.items():
            if session_name == 'overnight':
                if current_time >= start or current_time < end:
                    return session_name
            else:
                if start <= current_time < end:
                    return session_name
        
        return 'closed'
    
    def is_market_open(self, timestamp: datetime) -> bool:
        """
        Check if market is open at given timestamp.
        
        Args:
            timestamp: Datetime to check
            
        Returns:
            True if market is open
        """
        session = self.get_session_type(timestamp)
        return session == 'regular'