"""Numba-accelerated signal computations for performance-critical operations."""

import numpy as np
import pandas as pd
from numba import jit, njit, prange
from typing import Tuple, Optional


@njit(cache=True)
def calculate_rsi_numba(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI using Numba for acceleration.
    
    Args:
        prices: Array of closing prices
        period: RSI period
        
    Returns:
        Array of RSI values
    """
    n = len(prices)
    rsi = np.full(n, np.nan, dtype=np.float64)
    
    if n < period + 1:
        return rsi
    
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # Calculate first RSI value
    if avg_loss != 0:
        rs = avg_gain / avg_loss
        rsi[period] = 100 - (100 / (1 + rs))
    else:
        rsi[period] = 100
    
    # Calculate remaining RSI values using EMA
    for i in range(period + 1, n):
        gain = gains[i - 1]
        loss = losses[i - 1]
        
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    return rsi


@njit(cache=True)
def calculate_ema_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate EMA using Numba.
    
    Args:
        prices: Array of prices
        period: EMA period
        
    Returns:
        Array of EMA values
    """
    n = len(prices)
    ema = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return ema
    
    # Calculate multiplier
    multiplier = 2.0 / (period + 1)
    
    # Start with SMA
    ema[period - 1] = np.mean(prices[:period])
    
    # Calculate EMA
    for i in range(period, n):
        ema[i] = prices[i] * multiplier + ema[i - 1] * (1 - multiplier)
    
    return ema


@njit(cache=True)
def calculate_bollinger_bands_numba(
    prices: np.ndarray, 
    period: int = 20, 
    std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Bollinger Bands using Numba.
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    n = len(prices)
    upper = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        window = prices[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        
        middle[i] = mean
        upper[i] = mean + std_dev * std
        lower[i] = mean - std_dev * std
    
    return upper, middle, lower


@njit(cache=True)
def calculate_atr_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """Calculate ATR using Numba.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
        
    Returns:
        Array of ATR values
    """
    n = len(high)
    atr = np.full(n, np.nan, dtype=np.float64)
    
    if n < 2:
        return atr
    
    # Calculate true range
    true_range = np.zeros(n - 1)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        true_range[i - 1] = max(hl, hc, lc)
    
    # Calculate initial ATR
    if n >= period + 1:
        atr[period] = np.mean(true_range[:period])
        
        # Calculate remaining ATR values using EMA
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + true_range[i - 1]) / period
    
    return atr


@njit(cache=True)
def calculate_stochastic_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
    smooth_k: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Stochastic Oscillator using Numba.
    
    Returns:
        Tuple of (k_values, d_values)
    """
    n = len(high)
    k_raw = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate raw %K
    for i in range(period - 1, n):
        window_high = high[i - period + 1:i + 1]
        window_low = low[i - period + 1:i + 1]
        
        highest = np.max(window_high)
        lowest = np.min(window_low)
        
        if highest != lowest:
            k_raw[i] = ((close[i] - lowest) / (highest - lowest)) * 100
        else:
            k_raw[i] = 50  # Default when range is 0
    
    # Smooth %K
    k_smooth = np.full(n, np.nan, dtype=np.float64)
    for i in range(period + smooth_k - 2, n):
        k_smooth[i] = np.mean(k_raw[i - smooth_k + 1:i + 1])
    
    # Calculate %D (3-period SMA of %K)
    d_values = np.full(n, np.nan, dtype=np.float64)
    for i in range(period + smooth_k + 1, n):
        d_values[i] = np.mean(k_smooth[i - 2:i + 1])
    
    return k_smooth, d_values


@njit(cache=True, parallel=True)
def calculate_correlation_matrix_numba(returns: np.ndarray, window: int = 60) -> np.ndarray:
    """Calculate rolling correlation matrix using Numba with parallelization.
    
    Args:
        returns: 2D array of returns (time x assets)
        window: Rolling window size
        
    Returns:
        3D array of correlation matrices (time x assets x assets)
    """
    n_periods, n_assets = returns.shape
    n_corr = n_periods - window + 1
    
    # Initialize output
    corr_matrices = np.full((n_corr, n_assets, n_assets), np.nan, dtype=np.float64)
    
    # Calculate correlations for each window
    for t in prange(n_corr):
        window_data = returns[t:t + window, :]
        
        # Calculate correlation matrix for this window
        for i in range(n_assets):
            for j in range(i, n_assets):
                # Calculate correlation
                x = window_data[:, i]
                y = window_data[:, j]
                
                # Remove mean
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                x_centered = x - x_mean
                y_centered = y - y_mean
                
                # Calculate correlation
                numerator = np.sum(x_centered * y_centered)
                denominator = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
                
                if denominator != 0:
                    corr = numerator / denominator
                else:
                    corr = 0
                
                corr_matrices[t, i, j] = corr
                corr_matrices[t, j, i] = corr  # Symmetric
    
    return corr_matrices


class NumbaAcceleratedSignals:
    """Wrapper class for Numba-accelerated signal functions."""
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI with Numba acceleration."""
        values = calculate_rsi_numba(prices.values, period)
        return pd.Series(values, index=prices.index, name=f'rsi_{period}')
    
    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """Calculate EMA with Numba acceleration."""
        values = calculate_ema_numba(prices.values, period)
        return pd.Series(values, index=prices.index, name=f'ema_{period}')
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Calculate Bollinger Bands with Numba acceleration."""
        upper, middle, lower = calculate_bollinger_bands_numba(prices.values, period, std_dev)
        
        return pd.DataFrame({
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower
        }, index=prices.index)
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ATR with Numba acceleration."""
        values = calculate_atr_numba(high.values, low.values, close.values, period)
        return pd.Series(values, index=high.index, name=f'atr_{period}')
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   period: int = 14, smooth_k: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator with Numba acceleration."""
        k_values, d_values = calculate_stochastic_numba(
            high.values, low.values, close.values, period, smooth_k
        )
        
        return pd.DataFrame({
            'stoch_k': k_values,
            'stoch_d': d_values
        }, index=high.index)
    
    @staticmethod
    def correlation_matrix(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """Calculate rolling correlation matrix with Numba acceleration.
        
        Returns:
            MultiIndex DataFrame with dates and asset pairs
        """
        # Convert to numpy array
        returns_array = returns.values
        
        # Calculate correlations
        corr_matrices = calculate_correlation_matrix_numba(returns_array, window)
        
        # Convert back to DataFrame with proper indexing
        dates = returns.index[window - 1:]
        assets = returns.columns
        
        # Create MultiIndex DataFrame
        results = []
        for i, date in enumerate(dates):
            corr_matrix = pd.DataFrame(
                corr_matrices[i],
                index=assets,
                columns=assets
            )
            corr_matrix['date'] = date
            results.append(corr_matrix)
        
        return pd.concat(results, keys=dates)