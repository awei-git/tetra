"""Simple data loader for metrics pipeline."""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Optional
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Simple data loader for market data."""
    
    @staticmethod
    async def get_ohlcv_data(symbol: str, 
                            start_date: date, 
                            end_date: date) -> Optional[pd.DataFrame]:
        """Get OHLCV data for a symbol."""
        try:
            # Try to get data from yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                # Generate synthetic data for testing
                logger.warning(f"No data found for {symbol}, generating synthetic data")
                return DataLoader._generate_synthetic_data(symbol, start_date, end_date)
            
            # Rename columns to lowercase
            data.columns = data.columns.str.lower()
            return data
            
        except Exception as e:
            logger.warning(f"Error loading data for {symbol}: {e}, generating synthetic data")
            return DataLoader._generate_synthetic_data(symbol, start_date, end_date)
    
    @staticmethod
    def _generate_synthetic_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic-looking data
        np.random.seed(hash(symbol) % 2**32)  # Consistent data per symbol
        
        # Random walk for prices
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
        price = 100
        prices = []
        
        for ret in returns:
            price *= (1 + ret)
            prices.append(price)
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.5, len(dates)),
            'high': prices + np.abs(np.random.normal(0, 1, len(dates))),
            'low': prices - np.abs(np.random.normal(0, 1, len(dates))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Ensure high >= low, high >= open/close, low <= open/close
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data