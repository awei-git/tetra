"""Vectorized implementations of common signals for performance."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import talib


class VectorizedSignals:
    """Vectorized signal computations for multiple symbols at once."""
    
    @staticmethod
    def compute_rsi_vectorized(close_prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute RSI for multiple symbols at once.
        
        Args:
            close_prices: DataFrame with symbols as columns
            period: RSI period
            
        Returns:
            DataFrame with RSI values for each symbol
        """
        # Calculate price changes
        delta = close_prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period, min_periods=period).mean()
        avg_losses = losses.rolling(window=period, min_periods=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def compute_sma_vectorized(prices: pd.DataFrame, period: int) -> pd.DataFrame:
        """Compute SMA for multiple symbols at once."""
        return prices.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def compute_ema_vectorized(prices: pd.DataFrame, period: int) -> pd.DataFrame:
        """Compute EMA for multiple symbols at once."""
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def compute_bollinger_bands_vectorized(
        close_prices: pd.DataFrame, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Dict[str, pd.DataFrame]:
        """Compute Bollinger Bands for multiple symbols at once."""
        # Middle band (SMA)
        middle = close_prices.rolling(window=period, min_periods=1).mean()
        
        # Standard deviation
        std = close_prices.rolling(window=period, min_periods=1).std()
        
        # Upper and lower bands
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'middle': middle,
            'upper': upper,
            'lower': lower,
            'bandwidth': (upper - lower) / middle,
            'percent_b': (close_prices - lower) / (upper - lower)
        }
    
    @staticmethod
    def compute_macd_vectorized(
        close_prices: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, pd.DataFrame]:
        """Compute MACD for multiple symbols at once."""
        # Calculate EMAs
        ema_fast = close_prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close_prices.ewm(span=slow_period, adjust=False).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # MACD histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def compute_atr_vectorized(
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        period: int = 14
    ) -> pd.DataFrame:
        """Compute ATR for multiple symbols at once."""
        # Calculate true range components
        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()
        
        # True range is the maximum of the three
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Reshape back to original format
        if isinstance(close.columns, pd.Index):
            true_range = true_range.unstack(level=-1)
        
        # ATR is the EMA of true range
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr
    
    @staticmethod
    def compute_stochastic_vectorized(
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3
    ) -> Dict[str, pd.DataFrame]:
        """Compute Stochastic Oscillator for multiple symbols at once."""
        # Calculate rolling high and low
        highest_high = high.rolling(window=period, min_periods=1).max()
        lowest_low = low.rolling(window=period, min_periods=1).min()
        
        # %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Smooth %K
        k_smooth = k_percent.rolling(window=smooth_k, min_periods=1).mean()
        
        # %D is SMA of %K
        d_percent = k_smooth.rolling(window=smooth_d, min_periods=1).mean()
        
        return {
            'k': k_smooth,
            'd': d_percent
        }
    
    @staticmethod
    def compute_returns_vectorized(
        prices: pd.DataFrame,
        periods: list = [1, 5, 20, 60]
    ) -> Dict[str, pd.DataFrame]:
        """Compute returns for multiple periods and symbols."""
        returns = {}
        
        for period in periods:
            returns[f'return_{period}'] = prices.pct_change(period)
        
        # Log returns
        returns['log_return'] = np.log(prices / prices.shift(1))
        
        return returns
    
    @staticmethod
    def compute_volatility_vectorized(
        returns: pd.DataFrame,
        window: int = 20,
        annualize: bool = True
    ) -> pd.DataFrame:
        """Compute rolling volatility for multiple symbols."""
        volatility = returns.rolling(window=window, min_periods=1).std()
        
        if annualize:
            # Assume 252 trading days
            volatility = volatility * np.sqrt(252)
        
        return volatility
    
    @staticmethod
    def compute_volume_indicators_vectorized(
        close: pd.DataFrame,
        volume: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Compute volume-based indicators for multiple symbols."""
        indicators = {}
        
        # On-Balance Volume (OBV)
        price_change = close.diff()
        obv_direction = price_change.apply(np.sign).fillna(0)
        obv = (volume * obv_direction).cumsum()
        indicators['obv'] = obv
        
        # Volume-Weighted Average Price (VWAP) - daily reset
        # For intraday, would need to group by date
        cumulative_volume = volume.cumsum()
        cumulative_pv = (close * volume).cumsum()
        vwap = cumulative_pv / cumulative_volume
        indicators['vwap'] = vwap
        
        # Money Flow Index components
        typical_price = close  # Simplified, normally (H+L+C)/3
        money_flow = typical_price * volume
        
        # Positive and negative money flow
        positive_flow = money_flow.where(price_change > 0, 0)
        negative_flow = money_flow.where(price_change < 0, 0)
        
        # 14-period sums
        positive_sum = positive_flow.rolling(window=14, min_periods=1).sum()
        negative_sum = negative_flow.rolling(window=14, min_periods=1).sum()
        
        # Money Flow Ratio and MFI
        money_ratio = positive_sum / negative_sum
        mfi = 100 - (100 / (1 + money_ratio))
        indicators['mfi'] = mfi
        
        return indicators
    
    @staticmethod
    def compute_correlation_matrix_vectorized(
        returns: pd.DataFrame,
        window: int = 60
    ) -> pd.DataFrame:
        """Compute rolling correlation matrix for all symbols."""
        # This returns a 3D structure: date x symbol x symbol
        rolling_corr = returns.rolling(window=window, min_periods=20).corr()
        
        return rolling_corr