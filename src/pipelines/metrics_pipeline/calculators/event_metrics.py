"""Event metrics calculator for the Metrics Pipeline."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


class EventMetricsCalculator:
    """Calculate event-based metrics for trading strategies."""
    
    def __init__(self, db_connection_string: Optional[str] = None):
        """Initialize the event metrics calculator."""
        self.db_connection = db_connection_string or 'postgresql://tetra_user:tetra_password@localhost:5432/tetra'
        self.engine = None
        self.earnings_cache = {}
        self.dividend_cache = {}
        
    def connect_db(self):
        """Connect to database."""
        if not self.engine:
            self.engine = create_engine(self.db_connection)
    
    def load_earnings_events(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load earnings events from database."""
        self.connect_db()
        
        # Convert symbols to SQL list
        symbol_list = "','".join(symbols)
        
        query = f"""
        SELECT 
            symbol,
            event_datetime,
            eps_actual,
            eps_estimate,
            eps_surprise_pct,
            revenue_actual,
            revenue_estimate,
            revenue_surprise_pct,
            call_time
        FROM events.earnings_events
        WHERE symbol IN ('{symbol_list}')
        AND event_datetime BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY symbol, event_datetime
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"Loaded {len(df)} earnings events for {len(symbols)} symbols")
            return df
        except Exception as e:
            logger.warning(f"Failed to load earnings events: {e}")
            return pd.DataFrame()
    
    def calculate_event_metrics(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate event-based metrics for a symbol."""
        
        # Initialize event metrics columns
        data['days_to_earnings'] = np.nan
        data['earnings_surprise'] = 0.0
        data['is_earnings_week'] = False
        data['post_earnings_drift'] = 0.0
        
        # For now, generate synthetic event signals for testing
        # In production, this would use real earnings dates
        
        # Simulate quarterly earnings (every ~63 trading days)
        earnings_days = list(range(60, len(data), 63))
        
        for earnings_day in earnings_days:
            if earnings_day >= len(data):
                break
                
            # Mark earnings week (5 days before to 2 days after)
            start_idx = max(0, earnings_day - 5)
            end_idx = min(len(data), earnings_day + 3)
            
            # Calculate days to earnings
            for i in range(start_idx, earnings_day):
                data.iloc[i, data.columns.get_loc('days_to_earnings')] = earnings_day - i
            
            # Mark earnings week
            data.iloc[start_idx:end_idx, data.columns.get_loc('is_earnings_week')] = True
            
            # Simulate earnings surprise (random for now)
            surprise = np.random.normal(0, 0.1)  # ±10% surprise on average
            data.iloc[earnings_day, data.columns.get_loc('earnings_surprise')] = surprise
            
            # Post-earnings drift (price momentum after surprise)
            if earnings_day < len(data) - 10:
                drift_days = min(10, len(data) - earnings_day - 1)
                drift_magnitude = surprise * 0.5  # 50% of surprise translates to drift
                for i in range(1, drift_days + 1):
                    data.iloc[earnings_day + i, data.columns.get_loc('post_earnings_drift')] = \
                        drift_magnitude * (1 - i / drift_days)  # Decay over time
        
        # Add dividend metrics (simulated for now)
        data['days_to_dividend'] = np.nan
        data['dividend_yield'] = 0.0
        
        # Simulate quarterly dividends for dividend-paying stocks
        if symbol in ['AAPL', 'JNJ', 'JPM', 'XOM', 'KO']:
            # Quarterly dividends
            dividend_days = list(range(45, len(data), 63))
            
            for div_day in dividend_days:
                if div_day >= len(data):
                    break
                    
                # Days to dividend (10 days before)
                start_idx = max(0, div_day - 10)
                for i in range(start_idx, div_day):
                    data.iloc[i, data.columns.get_loc('days_to_dividend')] = div_day - i
                
                # Dividend yield (varies by symbol)
                yields = {
                    'AAPL': 0.005,  # 0.5% quarterly
                    'JNJ': 0.007,   # 0.7% quarterly  
                    'JPM': 0.008,   # 0.8% quarterly
                    'XOM': 0.009,   # 0.9% quarterly
                    'KO': 0.0075    # 0.75% quarterly
                }
                data.iloc[div_day, data.columns.get_loc('dividend_yield')] = yields.get(symbol, 0.005)
        
        # Add event trading signals
        data['event_signal'] = 0
        
        # Earnings momentum signal
        earnings_momentum = (data['days_to_earnings'] <= 3) & (data['days_to_earnings'] > 0)
        data.loc[earnings_momentum, 'event_signal'] = 1
        
        # Dividend capture signal  
        dividend_capture = (data['days_to_dividend'] <= 5) & (data['days_to_dividend'] > 0)
        data.loc[dividend_capture, 'event_signal'] = 1
        
        # Post-earnings drift signal
        drift_signal = data['post_earnings_drift'] > 0.03  # 3% drift threshold
        data.loc[drift_signal, 'event_signal'] = 1
        
        return data
    
    def add_event_metrics_to_scenario(self, scenario_data: pd.DataFrame) -> pd.DataFrame:
        """Add event metrics to scenario data."""
        
        # Group by symbol and calculate metrics
        symbols = scenario_data['symbol'].unique() if 'symbol' in scenario_data.columns else []
        
        if len(symbols) == 0:
            logger.warning("No symbols found in scenario data")
            return scenario_data
        
        results = []
        for symbol in symbols:
            symbol_data = scenario_data[scenario_data['symbol'] == symbol].copy()
            symbol_data = self.calculate_event_metrics(symbol_data, symbol)
            results.append(symbol_data)
        
        # Combine results
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return scenario_data
    
    def get_next_earnings_date(self, symbol: str, current_date: datetime) -> Optional[datetime]:
        """Get next earnings date for a symbol."""
        if symbol not in self.earnings_cache:
            # Load from database
            self.connect_db()
            query = f"""
            SELECT event_datetime 
            FROM events.earnings_events
            WHERE symbol = '{symbol}'
            AND event_datetime > '{current_date}'
            ORDER BY event_datetime
            LIMIT 1
            """
            try:
                result = pd.read_sql(query, self.engine)
                if len(result) > 0:
                    self.earnings_cache[symbol] = result.iloc[0]['event_datetime']
                else:
                    return None
            except:
                return None
        
        return self.earnings_cache.get(symbol)
    
    def get_earnings_history(self, symbol: str, lookback_quarters: int = 4) -> pd.DataFrame:
        """Get historical earnings surprises for a symbol."""
        self.connect_db()
        
        query = f"""
        SELECT 
            event_datetime,
            eps_actual,
            eps_estimate,
            eps_surprise_pct,
            revenue_surprise_pct
        FROM events.earnings_events
        WHERE symbol = '{symbol}'
        ORDER BY event_datetime DESC
        LIMIT {lookback_quarters}
        """
        
        try:
            return pd.read_sql(query, self.engine)
        except:
            return pd.DataFrame()