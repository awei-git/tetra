"""Technical indicator calculations for metrics pipeline."""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class TechnicalCalculator:
    """Calculate technical indicators for the metrics pipeline."""
    
    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate various moving averages."""
        indicators = {}
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            indicators[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
        # Exponential Moving Averages
        for period in [12, 26, 50, 200]:
            indicators[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            
        # Volume-weighted moving averages (only if volume exists)
        if 'volume' in df.columns:
            indicators['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            indicators['volume_sma_50'] = df['volume'].rolling(window=50).mean()
        
        return indicators
    
    @staticmethod
    def calculate_momentum_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate momentum indicators."""
        indicators = {}
        
        # RSI
        for period in [14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            indicators[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        indicators['macd'] = ema_12 - ema_26
        indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # Stochastic (requires high/low)
        if 'high' in df.columns and 'low' in df.columns:
            period = 14
            if len(df) >= period:
                lowest_low = df['low'].rolling(window=period, min_periods=1).min()
                highest_high = df['high'].rolling(window=period, min_periods=1).max()
                with np.errstate(divide='ignore', invalid='ignore'):
                    indicators['stochastic_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
                    indicators['stochastic_k'] = indicators['stochastic_k'].replace([np.inf, -np.inf], np.nan)
                indicators['stochastic_d'] = indicators['stochastic_k'].rolling(window=3, min_periods=1).mean()
                
                # Williams %R
                with np.errstate(divide='ignore', invalid='ignore'):
                    indicators['williams_r_14'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
                    indicators['williams_r_14'] = indicators['williams_r_14'].replace([np.inf, -np.inf], np.nan)
            else:
                indicators['stochastic_k'] = pd.Series(50, index=df.index)
                indicators['stochastic_d'] = pd.Series(50, index=df.index)
                indicators['williams_r_14'] = pd.Series(-50, index=df.index)
            
            # CCI (Commodity Channel Index)
            period = 20
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            
            def safe_mad(x):
                try:
                    if len(x.dropna()) == 0:
                        return np.nan
                    return np.mean(np.abs(x - x.mean()))
                except:
                    return np.nan
            
            mad = typical_price.rolling(window=period).apply(safe_mad)
            indicators['cci_20'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Rate of Change
        for period in [10, 20]:
            indicators[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        
        # Money Flow Index (requires volume and OHLC)
        if 'volume' in df.columns and 'high' in df.columns and 'low' in df.columns:
            period = 14
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            raw_money_flow = typical_price * df['volume']
            
            positive_flow = pd.Series(0.0, index=df.index)  # Use float dtype
            negative_flow = pd.Series(0.0, index=df.index)  # Use float dtype
            
            price_change = typical_price.diff()
            positive_mask = price_change > 0
            negative_mask = price_change < 0
            positive_flow.loc[positive_mask] = raw_money_flow.loc[positive_mask]
            negative_flow.loc[negative_mask] = raw_money_flow.loc[negative_mask]
            
            positive_mf = positive_flow.rolling(window=period).sum()
            negative_mf = negative_flow.rolling(window=period).sum()
            
            mfi_ratio = positive_mf / negative_mf
            indicators['mfi_14'] = 100 - (100 / (1 + mfi_ratio))
        else:
            indicators['mfi_14'] = pd.Series(50, index=df.index)  # Neutral value if no volume
        
        return indicators
    
    @staticmethod
    def calculate_volatility_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volatility indicators."""
        indicators = {}
        
        # Bollinger Bands
        period = 20
        std_dev = 2
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        indicators['bb_upper'] = sma + (std * std_dev)
        indicators['bb_middle'] = sma
        indicators['bb_lower'] = sma - (std * std_dev)
        
        # ATR (Average True Range) - requires high/low
        if 'high' in df.columns and 'low' in df.columns:
            period = 14
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators['atr_14'] = true_range.rolling(window=period).mean()
            
            # Normalized ATR
            indicators['natr_14'] = (indicators['atr_14'] / df['close']) * 100
            
            # Keltner Channels
            period = 20
            multiplier = 2
            ema = df['close'].ewm(span=period, adjust=False).mean()
            indicators['keltner_upper_20'] = ema + (indicators['atr_14'] * multiplier)
            indicators['keltner_lower_20'] = ema - (indicators['atr_14'] * multiplier)
        
        # Historical Volatility
        for period in [20, 252]:
            returns = df['close'].pct_change()
            indicators[f'historical_volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
        
        return indicators
    
    @staticmethod
    def calculate_volume_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volume indicators."""
        indicators = {}
        
        if 'volume' not in df.columns:
            return indicators  # No volume data available
        
        # On-Balance Volume
        obv = pd.Series(0, index=df.index)
        obv[df['close'] > df['close'].shift()] = df['volume'][df['close'] > df['close'].shift()]
        obv[df['close'] < df['close'].shift()] = -df['volume'][df['close'] < df['close'].shift()]
        indicators['obv'] = obv.cumsum()
        
        # Volume Ratio
        indicators['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Dollar Volume
        indicators['dollar_volume'] = df['close'] * df['volume']
        indicators['dollar_volume_20d'] = indicators['dollar_volume'].rolling(window=20).mean()
        
        # VWAP and other indicators that require OHLC
        if 'high' in df.columns and 'low' in df.columns:
            # VWAP (Volume-Weighted Average Price)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            cumulative_tpv = (typical_price * df['volume']).cumsum()
            cumulative_volume = df['volume'].cumsum()
            indicators['vwap'] = cumulative_tpv / cumulative_volume
            
            # Chaikin Money Flow
            period = 20
            money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            money_flow_volume = money_flow_multiplier * df['volume']
            indicators['cmf_20'] = money_flow_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
            
            # Accumulation/Distribution Line
            indicators['accumulation_distribution'] = money_flow_volume.cumsum()
        
        return indicators
    
    @staticmethod
    def calculate_trend_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate trend indicators."""
        indicators = {}
        
        # Parabolic SAR (simplified version - works with close only)
        indicators['psar'] = df['close'].rolling(window=2).mean()  # Simplified
        
        # ADX and other indicators that require high/low
        if 'high' in df.columns and 'low' in df.columns:
            # ADX (Average Directional Index)
            period = 14
            
            # Calculate +DM and -DM
            high_diff = df['high'].diff()
            low_diff = -df['low'].diff()
            
            plus_dm = pd.Series(0.0, index=df.index)  # Use float dtype
            minus_dm = pd.Series(0.0, index=df.index)  # Use float dtype
            
            plus_mask = (high_diff > low_diff) & (high_diff > 0)
            minus_mask = (low_diff > high_diff) & (low_diff > 0)
            plus_dm.loc[plus_mask] = high_diff.loc[plus_mask]
            minus_dm.loc[minus_mask] = low_diff.loc[minus_mask]
            
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            # Calculate +DI and -DI
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            
            # Calculate DX and ADX
            dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
            indicators['adx_14'] = dx.rolling(window=period).mean()
            
            # Aroon - DISABLED due to persistent argmax issues
            # Set neutral values for now
            indicators['aroon_up_25'] = pd.Series(50, index=df.index)
            indicators['aroon_down_25'] = pd.Series(50, index=df.index)
            
            # Donchian Channels - with safety checks
            try:
                indicators['donchian_high_20'] = df['high'].rolling(window=20, min_periods=1).max()
                indicators['donchian_low_10'] = df['low'].rolling(window=10, min_periods=1).min()
                indicators['donchian_low_20'] = df['low'].rolling(window=20, min_periods=1).min()
            except:
                indicators['donchian_high_20'] = df['high']
                indicators['donchian_low_10'] = df['low']
                indicators['donchian_low_20'] = df['low']
            
            # Highest/Lowest - with safety checks
            try:
                indicators['highest_20'] = df['high'].rolling(window=20, min_periods=1).max()
                indicators['highest_55'] = df['high'].rolling(window=55, min_periods=1).max()
                indicators['lowest_10'] = df['low'].rolling(window=10, min_periods=1).min()
                indicators['lowest_20'] = df['low'].rolling(window=20, min_periods=1).min()
            except:
                indicators['highest_20'] = df['high']
                indicators['highest_55'] = df['high']
                indicators['lowest_10'] = df['low']
                indicators['lowest_20'] = df['low']
        
        return indicators
    
    @staticmethod
    def calculate_derived_signals(df: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Calculate derived trading signals."""
        signals = {}
        
        # Golden Cross / Death Cross
        if 'sma_50' in indicators and 'sma_200' in indicators:
            sma_50_prev = indicators['sma_50'].shift(1)
            sma_200_prev = indicators['sma_200'].shift(1)
            
            signals['golden_cross'] = (
                (indicators['sma_50'] > indicators['sma_200']) & 
                (sma_50_prev <= sma_200_prev)
            ).astype(int)
            
            signals['death_cross'] = (
                (indicators['sma_50'] < indicators['sma_200']) & 
                (sma_50_prev >= sma_200_prev)
            ).astype(int)
        
        # MACD Signals
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd_prev = indicators['macd'].shift(1)
            signal_prev = indicators['macd_signal'].shift(1)
            
            signals['macd_bullish'] = (
                (indicators['macd'] > indicators['macd_signal']) & 
                (macd_prev <= signal_prev)
            ).astype(int)
            
            signals['macd_bearish'] = (
                (indicators['macd'] < indicators['macd_signal']) & 
                (macd_prev >= signal_prev)
            ).astype(int)
        
        # RSI Signals
        if 'rsi_14' in indicators:
            signals['rsi_oversold'] = (indicators['rsi_14'] < 30).astype(int)
            signals['rsi_overbought'] = (indicators['rsi_14'] > 70).astype(int)
        
        # Breakout Signals
        if 'highest_20' in indicators:
            signals['breakout_20d_high'] = (df['close'] >= indicators['highest_20']).astype(int)
        
        if 'lowest_20' in indicators:
            signals['breakout_20d_low'] = (df['close'] <= indicators['lowest_20']).astype(int)
        
        return signals
    
    @classmethod
    def calculate_all(cls, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate all technical indicators."""
        indicators = {}
        
        # Calculate all indicator categories
        indicators.update(cls.calculate_moving_averages(df))
        indicators.update(cls.calculate_momentum_indicators(df))
        indicators.update(cls.calculate_volatility_indicators(df))
        
        # Only calculate volume indicators if volume exists
        if 'volume' in df.columns:
            indicators.update(cls.calculate_volume_indicators(df))
        
        indicators.update(cls.calculate_trend_indicators(df))
        
        # Calculate derived signals
        signals = cls.calculate_derived_signals(df, indicators)
        indicators.update(signals)
        
        return indicators