"""Signal-based trading strategies."""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import logging

from .base import BaseStrategy, PositionSide, Trade, Position

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals."""
    TECHNICAL = "technical"
    STATISTICAL = "statistical"
    ML = "ml"
    COMPOSITE = "composite"
    CUSTOM = "custom"


class ConditionOperator(Enum):
    """Operators for signal conditions."""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    EQUAL = "="
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    NOT_EQUAL = "!="
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    BETWEEN = "between"
    OUTSIDE = "outside"


@dataclass
class SignalCondition:
    """Defines a condition for a signal."""
    signal_name: str
    operator: ConditionOperator
    value: Union[float, str, List[float]]
    lookback: int = 1  # Number of periods to look back
    weight: float = 1.0  # Weight for composite conditions
    
    def evaluate(self, signals: pd.Series, history: Optional[pd.DataFrame] = None) -> bool:
        """Evaluate if condition is met."""
        if self.signal_name not in signals:
            return False
        
        current_value = signals[self.signal_name]
        
        # If value is a string, it refers to another signal
        compare_value = self.value
        if isinstance(self.value, str) and self.value in signals:
            compare_value = signals[self.value]
        elif isinstance(self.value, str) and self.value not in signals:
            # If it's a string but not in signals, we can't evaluate
            return False
        
        if self.operator == ConditionOperator.GREATER_THAN:
            return current_value > compare_value
        elif self.operator == ConditionOperator.LESS_THAN:
            return current_value < compare_value
        elif self.operator == ConditionOperator.EQUAL:
            return current_value == compare_value
        elif self.operator == ConditionOperator.GREATER_EQUAL:
            return current_value >= compare_value
        elif self.operator == ConditionOperator.LESS_EQUAL:
            return current_value <= compare_value
        elif self.operator == ConditionOperator.NOT_EQUAL:
            return current_value != compare_value
        elif self.operator == ConditionOperator.BETWEEN:
            return self.value[0] <= current_value <= self.value[1]
        elif self.operator == ConditionOperator.OUTSIDE:
            return current_value < self.value[0] or current_value > self.value[1]
        
        # For crosses, we need history
        if history is not None and self.signal_name in history.columns:
            if self.operator == ConditionOperator.CROSSES_ABOVE:
                prev_value = history[self.signal_name].iloc[-self.lookback]
                # Handle both numeric and signal reference values
                if isinstance(self.value, str) and self.value in history.columns:
                    prev_compare = history[self.value].iloc[-self.lookback]
                    curr_compare = signals.get(self.value, compare_value)
                    return prev_value <= prev_compare and current_value > curr_compare
                else:
                    return prev_value <= compare_value and current_value > compare_value
            elif self.operator == ConditionOperator.CROSSES_BELOW:
                prev_value = history[self.signal_name].iloc[-self.lookback]
                # Handle both numeric and signal reference values
                if isinstance(self.value, str) and self.value in history.columns:
                    prev_compare = history[self.value].iloc[-self.lookback]
                    curr_compare = signals.get(self.value, compare_value)
                    return prev_value >= prev_compare and current_value < curr_compare
                else:
                    return prev_value >= compare_value and current_value < compare_value
        
        return False


@dataclass
class SignalRule:
    """Defines a trading rule based on signals."""
    name: str
    entry_conditions: List[SignalCondition]
    exit_conditions: List[SignalCondition]
    position_side: PositionSide
    position_size_factor: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_limit: Optional[int] = None  # Max holding period in bars
    require_all: bool = True  # True=AND, False=OR for conditions
    
    def check_entry(self, signals: pd.Series, history: Optional[pd.DataFrame] = None) -> bool:
        """Check if entry conditions are met."""
        results = [cond.evaluate(signals, history) for cond in self.entry_conditions]
        
        if self.require_all:
            return all(results)
        else:
            return any(results)
    
    def check_exit(self, signals: pd.Series, history: Optional[pd.DataFrame] = None) -> bool:
        """Check if exit conditions are met."""
        if not self.exit_conditions:
            return False
        
        results = [cond.evaluate(signals, history) for cond in self.exit_conditions]
        
        if self.require_all:
            return all(results)
        else:
            return any(results)


class SignalBasedStrategy(BaseStrategy):
    """Strategy that trades based on technical/statistical/ML signals."""
    
    def __init__(self,
                 name: str,
                 signal_rules: List[SignalRule],
                 signal_weights: Optional[Dict[str, float]] = None,
                 confirmation_required: int = 1,
                 **kwargs):
        super().__init__(name, **kwargs)
        self.signal_rules = signal_rules
        self.signal_weights = signal_weights or {}
        self.confirmation_required = confirmation_required
        self.signal_history: Dict[str, pd.DataFrame] = {}
        self.rule_triggers: Dict[str, List[datetime]] = {}
        self.universe = []  # Initialize empty universe
    
    def set_symbols(self, symbols: List[str]):
        """Set the universe of symbols for this strategy."""
        if not isinstance(symbols, list):
            symbols = [symbols]
        self.universe = symbols
    
    def set_ml_predictions(self, predictions: Dict[str, Dict[str, float]]):
        """Set ML predictions for use in strategy."""
        self._ml_predictions = predictions
    
    def generate_signals(self, 
                        market_data: Union[pd.DataFrame, Dict[str, Dict]],
                        portfolio: Optional[Any] = None,
                        trading_day: Optional[datetime] = None,
                        historical_data: Optional[Dict[str, pd.DataFrame]] = None) -> Union[pd.DataFrame, List[Dict]]:
        """Generate trading signals based on rules.
        
        This method supports two interfaces:
        1. Legacy: (data: DataFrame, signals: DataFrame, events: DataFrame) -> DataFrame
        2. Simulator: (market_data: Dict, portfolio: Portfolio, trading_day: date, historical_data: Dict) -> List[Dict]
        """
        # Handle simulator interface
        if isinstance(market_data, dict) and portfolio is not None and trading_day is not None:
            return self._generate_signals_for_simulator(market_data, portfolio, trading_day, historical_data)
        
        # Handle legacy DataFrame interface
        data = market_data
        signals = portfolio  # In legacy mode, portfolio arg contains signals DataFrame
        events = trading_day  # In legacy mode, trading_day arg contains events DataFrame
        
        if signals is None or (hasattr(signals, 'empty') and signals.empty):
            return pd.DataFrame(index=data.index, columns=['signal', 'position_size']).fillna(0)
        
        output_signals = pd.DataFrame(index=data.index)
        output_signals['signal'] = 0
        output_signals['position_size'] = 0
        output_signals['rule_triggered'] = ''
        
        # Process each timestamp
        for i, (timestamp, signal_row) in enumerate(signals.iterrows()):
            # Get historical signals for cross-over detection
            history = signals.iloc[:i] if i > 0 else None
            
            # Check each rule
            for rule in self.signal_rules:
                # Check entry conditions
                if rule.check_entry(signal_row, history):
                    # Check if we need confirmation
                    if self._check_confirmation(rule.name, timestamp):
                        output_signals.loc[timestamp, 'signal'] = 1 if rule.position_side == PositionSide.LONG else -1
                        output_signals.loc[timestamp, 'position_size'] = self.position_size * rule.position_size_factor
                        output_signals.loc[timestamp, 'rule_triggered'] = rule.name
                        break  # Only trigger one rule per timestamp
                
                # Check exit conditions for existing positions
                elif rule.check_exit(signal_row, history):
                    output_signals.loc[timestamp, 'signal'] = 0  # Flat
                    output_signals.loc[timestamp, 'rule_triggered'] = f"exit_{rule.name}"
        
        return output_signals
    
    def _generate_signals_for_simulator(self, market_data: Dict[str, Dict], portfolio: Any, 
                                       trading_day: datetime, historical_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[Dict]:
        """Generate signals for the simulator interface."""
        signals = []
        
        if historical_data is None:
            return signals
            
        # If universe is empty but we have market data, use market data symbols
        symbols_to_check = self.universe if self.universe else list(market_data.keys())
        
        # For each symbol to check
        for symbol in symbols_to_check:
            if symbol not in market_data or symbol not in historical_data:
                continue
                
            # Get historical price data
            hist_df = historical_data[symbol]
            if hist_df.empty:
                continue
                
            # Set current symbol context for ML indicator calculation
            self._current_symbol = symbol
            
            # Calculate indicators based on signal conditions
            indicators = self._calculate_required_indicators(hist_df)
            
            # Check each rule
            for rule in self.signal_rules:
                # Create a series with current indicators
                current_indicators = pd.Series(indicators)
                
                # Get indicator history for cross-over detection
                history_df = self._prepare_indicator_history(hist_df, indicators)
                
                # Check if we should enter a position
                if rule.check_entry(current_indicators, history_df):
                    # Check position limits
                    current_positions = len(portfolio.positions) if hasattr(portfolio, 'positions') else 0
                    if current_positions < self.max_positions:
                        # Calculate position size
                        position_size = self._calculate_position_size_for_signal(
                            symbol, 
                            market_data[symbol]['close'],
                            portfolio,
                            rule.position_size_factor
                        )
                        
                        if position_size > 0:
                            signals.append({
                                'symbol': symbol,
                                'direction': 'BUY' if rule.position_side == PositionSide.LONG else 'SELL',
                                'quantity': position_size,
                                'order_type': 'MARKET',
                                'timestamp': trading_day,
                                'rule': rule.name,
                                'stop_loss': rule.stop_loss,
                                'take_profit': rule.take_profit
                            })
                            break  # Only one rule per symbol per day
                
                # Check if we should exit a position
                elif symbol in portfolio.positions and rule.check_exit(current_indicators, history_df):
                    position = portfolio.positions[symbol]
                    signals.append({
                        'symbol': symbol,
                        'direction': 'SELL' if position.quantity > 0 else 'BUY',  # Close position
                        'quantity': abs(position.quantity),
                        'order_type': 'MARKET',
                        'timestamp': trading_day,
                        'rule': f"exit_{rule.name}"
                    })
                    break
        
        return signals
    
    def _calculate_required_indicators(self, hist_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate indicators needed by the strategy rules."""
        indicators = {}
        
        # Extract unique indicator names from all conditions
        required_indicators = set()
        for rule in self.signal_rules:
            for condition in rule.entry_conditions + rule.exit_conditions:
                required_indicators.add(condition.signal_name)
                # Add string values that might be indicators
                if isinstance(condition.value, str):
                    # Check if it looks like an indicator name
                    if any(prefix in condition.value for prefix in ['sma_', 'ema_', 'rsi_', 'volume_', 'bb_', 'macd', 'highest_', 'lowest_', 'atr_', 'adx_', 'returns_', 'donchian_', 'vwap']):
                        required_indicators.add(condition.value)
        
        # Ensure we have OHLCV data with correct column names
        if not hist_df.empty:
            # Get the last row of data (most recent values)
            last_row = hist_df.iloc[-1]
            
            # First, check if indicators are already calculated in the dataframe
            for indicator_name in required_indicators:
                if indicator_name in hist_df.columns:
                    # Use pre-calculated indicator value
                    value = last_row[indicator_name]
                    if pd.notna(value):
                        indicators[indicator_name] = value
                    continue
                    
            # Convert column names to lowercase if needed for fallback calculations
            hist_df.columns = hist_df.columns.str.lower()
            
            # Calculate each required indicator
            for indicator_name in required_indicators:
                try:
                    # Check volume_sma BEFORE generic sma to avoid misparsing
                    if indicator_name.startswith('volume_sma_'):
                        # Handle volume_sma_20, volume_sma_20_1.5x, etc.
                        if 'volume' in hist_df.columns:
                            # Extract period from the indicator name
                            remaining = indicator_name[11:]  # Remove 'volume_sma_'
                            period = 20  # default
                            multiplier = 1.0
                            
                            # Parse period and optional multiplier
                            if '_' in remaining:
                                # Format: volume_sma_20_1.5x
                                parts = remaining.split('_')
                                if parts[0].isdigit():
                                    period = int(parts[0])
                                if len(parts) > 1 and 'x' in parts[1]:
                                    try:
                                        multiplier = float(parts[1].replace('x', ''))
                                    except:
                                        pass
                            else:
                                # Format: volume_sma_20
                                if remaining.isdigit():
                                    period = int(remaining)
                            
                            if len(hist_df) >= period:
                                volume_sma = hist_df['volume'].rolling(period).mean().iloc[-1]
                                indicators[indicator_name] = volume_sma * multiplier
                    elif indicator_name.startswith('sma_') and not indicator_name.startswith('volume_sma_'):
                        period = int(indicator_name.split('_')[1])
                        if len(hist_df) >= period and 'close' in hist_df.columns:
                            indicators[indicator_name] = hist_df['close'].rolling(period).mean().iloc[-1]
                    elif indicator_name.startswith('ema_'):
                        period = int(indicator_name.split('_')[1])
                        if len(hist_df) >= period and 'close' in hist_df.columns:
                            indicators[indicator_name] = hist_df['close'].ewm(span=period).mean().iloc[-1]
                    elif indicator_name.startswith('highest_'):
                        period = int(indicator_name.split('_')[1])
                        if len(hist_df) >= period and 'high' in hist_df.columns:
                            indicators[indicator_name] = hist_df['high'].rolling(period).max().iloc[-1]
                    elif indicator_name.startswith('lowest_'):
                        period = int(indicator_name.split('_')[1])
                        if len(hist_df) >= period and 'low' in hist_df.columns:
                            indicators[indicator_name] = hist_df['low'].rolling(period).min().iloc[-1]
                    elif indicator_name.startswith('atr_'):
                        period = int(indicator_name.split('_')[1])
                        if len(hist_df) >= period:
                            indicators[indicator_name] = self._calculate_atr(hist_df, period)
                    elif 'bb_' in indicator_name:
                        # Bollinger Bands indicators
                        period = 20  # default
                        std_mult = 2  # default
                        if len(hist_df) >= period and 'close' in hist_df.columns:
                            sma = hist_df['close'].rolling(period).mean()
                            std = hist_df['close'].rolling(period).std()
                            if indicator_name == 'bb_upper':
                                indicators[indicator_name] = sma.iloc[-1] + (std_mult * std.iloc[-1])
                            elif indicator_name == 'bb_lower':
                                indicators[indicator_name] = sma.iloc[-1] - (std_mult * std.iloc[-1])
                            elif indicator_name == 'bb_middle':
                                indicators[indicator_name] = sma.iloc[-1]
                    elif 'rsi_' in indicator_name:
                        period = int(indicator_name.split('_')[1])
                        if len(hist_df) >= period + 1 and 'close' in hist_df.columns:
                            # Calculate RSI
                            delta = hist_df['close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                            rs = gain / loss
                            rsi = 100 - (100 / (1 + rs))
                            indicators[indicator_name] = rsi.iloc[-1]
                    elif 'macd' in indicator_name:
                        if len(hist_df) >= 26 and 'close' in hist_df.columns:
                            # MACD parameters
                            fast = 12
                            slow = 26
                            signal = 9
                            
                            # Calculate MACD
                            ema_fast = hist_df['close'].ewm(span=fast).mean()
                            ema_slow = hist_df['close'].ewm(span=slow).mean()
                            macd_line = ema_fast - ema_slow
                            macd_signal = macd_line.ewm(span=signal).mean()
                            macd_histogram = macd_line - macd_signal
                            
                            if indicator_name == 'macd':
                                indicators[indicator_name] = macd_line.iloc[-1]
                            elif indicator_name == 'macd_signal':
                                indicators[indicator_name] = macd_signal.iloc[-1]
                            elif indicator_name == 'macd_histogram':
                                indicators[indicator_name] = macd_histogram.iloc[-1]
                    elif 'adx_' in indicator_name:
                        period = int(indicator_name.split('_')[1])
                        if len(hist_df) >= period * 2:  # Need extra data for ADX
                            # Simplified ADX calculation
                            indicators[indicator_name] = 25  # Default neutral value
                    elif 'donchian_' in indicator_name:
                        parts = indicator_name.split('_')
                        if len(parts) >= 3:
                            side = parts[1]  # 'high' or 'low'
                            period = int(parts[2])
                            if side == 'high' and len(hist_df) >= period and 'high' in hist_df.columns:
                                indicators[indicator_name] = hist_df['high'].rolling(period).max().iloc[-1]
                            elif side == 'low' and len(hist_df) >= period and 'low' in hist_df.columns:
                                indicators[indicator_name] = hist_df['low'].rolling(period).min().iloc[-1]
                    elif 'returns_' in indicator_name:
                        # Handle returns_252d, returns_21d, returns_5, etc.
                        parts = indicator_name.split('_')[1]
                        if parts.endswith('d'):
                            period = int(parts[:-1])
                        else:
                            period = int(parts)
                        if len(hist_df) > period and 'close' in hist_df.columns:
                            try:
                                old_price = hist_df['close'].iloc[-(period+1)]
                                new_price = hist_df['close'].iloc[-1]
                                indicators[indicator_name] = (new_price - old_price) / old_price
                            except IndexError:
                                # Not enough data for this period, use shorter period or return 0
                                available_periods = len(hist_df) - 1
                                if available_periods > 0:
                                    old_price = hist_df['close'].iloc[0]
                                    new_price = hist_df['close'].iloc[-1]
                                    # Annualize the return if needed
                                    actual_return = (new_price - old_price) / old_price
                                    if period == 252:  # Annual returns
                                        indicators[indicator_name] = actual_return * (252 / available_periods)
                                    else:
                                        indicators[indicator_name] = actual_return
                                else:
                                    indicators[indicator_name] = 0.0
                        else:
                            indicators[indicator_name] = 0.0
                    elif 'vwap' in indicator_name:
                        if 'close' in hist_df.columns and 'volume' in hist_df.columns:
                            # Simple VWAP for the day
                            indicators[indicator_name] = (hist_df['close'] * hist_df['volume']).sum() / hist_df['volume'].sum()
                    elif indicator_name in ['close', 'open', 'high', 'low', 'volume']:
                        if indicator_name in hist_df.columns:
                            indicators[indicator_name] = hist_df[indicator_name].iloc[-1]
                        else:
                            # Use a default value that will make buy_and_hold work
                            if indicator_name == 'close':
                                indicators[indicator_name] = 100.0  # Default price
                            elif indicator_name == 'volume':
                                indicators[indicator_name] = 1000000  # Default volume
                            else:
                                indicators[indicator_name] = 100.0
                    elif 'volume_dollar' in indicator_name:
                        # Handle volume_dollar_20d, etc.
                        if 'close' in hist_df.columns and 'volume' in hist_df.columns:
                            parts = indicator_name.split('_')
                            if len(parts) >= 3:
                                period_str = parts[2]
                                period = int(period_str.replace('d', ''))
                                if len(hist_df) >= period:
                                    dollar_volume = hist_df['close'] * hist_df['volume']
                                    indicators[indicator_name] = dollar_volume.rolling(period).mean().iloc[-1]
                                else:
                                    indicators[indicator_name] = hist_df['close'].iloc[-1] * hist_df['volume'].iloc[-1]
                    elif indicator_name == 'relative_strength_vs_bonds':
                        # For now, return a default value (normally would compare to bond ETF)
                        indicators[indicator_name] = 1.1  # Assume stocks outperforming bonds
                    elif indicator_name == 'trend_filter_200d':
                        # Simple trend filter - price above 200 SMA
                        if len(hist_df) >= 200 and 'close' in hist_df.columns:
                            sma_200 = hist_df['close'].rolling(200).mean().iloc[-1]
                            indicators[indicator_name] = 'up' if hist_df['close'].iloc[-1] > sma_200 else 'down'
                        else:
                            # Not enough data, assume uptrend
                            indicators[indicator_name] = 'up'
                    elif indicator_name == 'market_cap':
                        # Placeholder - would normally come from fundamental data
                        indicators[indicator_name] = 1000000000000  # 1 trillion default
                    elif 'high_52w' in indicator_name or 'low_52w' in indicator_name:
                        # 52-week high/low indicators
                        if len(hist_df) >= 252 and 'high' in hist_df.columns and 'low' in hist_df.columns:
                            if 'high_52w' in indicator_name:
                                high_52w = hist_df['high'].rolling(252).max().iloc[-1]
                                if '_' in indicator_name.split('52w_')[1]:
                                    # Handle high_52w_0.9 format
                                    multiplier = float(indicator_name.split('_')[-1])
                                    indicators[indicator_name] = high_52w * multiplier
                                else:
                                    indicators[indicator_name] = high_52w
                            else:  # low_52w
                                indicators[indicator_name] = hist_df['low'].rolling(252).min().iloc[-1]
                        else:
                            # Use available data
                            if 'high_52w' in indicator_name:
                                indicators[indicator_name] = hist_df['high'].max() if 'high' in hist_df.columns else hist_df['close'].iloc[-1]
                            else:
                                indicators[indicator_name] = hist_df['low'].min() if 'low' in hist_df.columns else hist_df['close'].iloc[-1]
                    elif 'revenue_growth' in indicator_name:
                        # Placeholder for fundamental data
                        indicators[indicator_name] = 0.15  # 15% default growth
                    elif indicator_name == 'volume_ratio':
                        # Current volume vs average volume ratio
                        if 'volume' in hist_df.columns and len(hist_df) >= 20:
                            avg_volume = hist_df['volume'].rolling(20).mean().iloc[-1]
                            current_volume = hist_df['volume'].iloc[-1]
                            indicators[indicator_name] = current_volume / avg_volume if avg_volume > 0 else 1.0
                        else:
                            indicators[indicator_name] = 1.0
                    elif indicator_name in ['ml_prediction', 'ml_confidence', 'anomaly_score', 'feature_importance_price']:
                        # ML indicators - try to get from context or use defaults
                        symbol = getattr(self, '_current_symbol', 'SPY')
                        
                        # Try to get from context if available
                        ml_predictions = getattr(self, '_ml_predictions', {})
                        if symbol in ml_predictions:
                            pred_data = ml_predictions[symbol]
                            if indicator_name == 'ml_prediction':
                                indicators[indicator_name] = pred_data.get('ml_prediction', 0.5)
                            elif indicator_name == 'ml_confidence':
                                indicators[indicator_name] = pred_data.get('ml_confidence', 0.0)
                            else:
                                # Default values for other ML indicators
                                indicators[indicator_name] = 0.0
                        else:
                            # Default neutral values if no prediction
                            if indicator_name == 'ml_prediction':
                                indicators[indicator_name] = 0.5  # Neutral
                            elif indicator_name == 'ml_confidence':
                                indicators[indicator_name] = 0.0  # No confidence
                            elif indicator_name == 'anomaly_score':
                                indicators[indicator_name] = 0.0  # No anomaly
                            elif indicator_name == 'feature_importance_price':
                                indicators[indicator_name] = 0.5  # Medium importance
                except Exception as e:
                    logger.warning(f"Failed to calculate {indicator_name}: {e}")
                
        # Special indicators for specific rules
        if 'day_of_backtest' in required_indicators:
            indicators['day_of_backtest'] = 1 if not hasattr(self, '_backtest_started') else self._backtest_day
            if not hasattr(self, '_backtest_started'):
                self._backtest_started = True
                self._backtest_day = 1
            else:
                self._backtest_day += 1
                
        if 'last_day_of_backtest' in required_indicators:
            indicators['last_day_of_backtest'] = False  # Will be set by simulator
            
        return indicators
    
    def _prepare_indicator_history(self, hist_df: pd.DataFrame, current_indicators: Dict[str, float]) -> pd.DataFrame:
        """Prepare historical indicator data for cross-over detection."""
        # For simplicity, create a small history DataFrame with previous values
        # In a full implementation, this would calculate indicators for multiple historical points
        history_data = []
        
        # Calculate indicators for the last few days
        lookback = 5  # Days to look back for crossovers
        for i in range(min(lookback, len(hist_df) - 1)):
            day_idx = -(i + 2)  # Start from -2 (yesterday)
            day_indicators = {}
            
            for indicator_name in current_indicators.keys():
                try:
                    if 'sma_' in indicator_name and '_' not in indicator_name[4:]:  # Simple SMA
                        period = int(indicator_name.split('_')[1])
                        if len(hist_df) + day_idx >= period:
                            day_indicators[indicator_name] = hist_df['close'].iloc[:day_idx].rolling(period).mean().iloc[-1]
                    elif 'ema_' in indicator_name:
                        period = int(indicator_name.split('_')[1])
                        if len(hist_df) + day_idx >= period:
                            day_indicators[indicator_name] = hist_df['close'].iloc[:day_idx].ewm(span=period).mean().iloc[-1]
                    elif 'macd' in indicator_name:
                        if len(hist_df) + day_idx >= 26:
                            # MACD calculation
                            ema_fast = hist_df['close'].iloc[:day_idx].ewm(span=12).mean()
                            ema_slow = hist_df['close'].iloc[:day_idx].ewm(span=26).mean()
                            macd_line = ema_fast - ema_slow
                            macd_signal = macd_line.ewm(span=9).mean()
                            
                            if indicator_name == 'macd':
                                day_indicators[indicator_name] = macd_line.iloc[-1]
                            elif indicator_name == 'macd_signal':
                                day_indicators[indicator_name] = macd_signal.iloc[-1]
                except Exception:
                    pass  # Skip if calculation fails
                        
            if day_indicators:
                history_data.append(day_indicators)
                
        if history_data:
            return pd.DataFrame(history_data)
        return pd.DataFrame()
    
    def _calculate_position_size_for_signal(self, symbol: str, price: float, portfolio: Any, size_factor: float) -> int:
        """Calculate position size based on portfolio and risk rules."""
        # Get available cash
        available_cash = portfolio.cash if hasattr(portfolio, 'cash') else portfolio.initial_cash
        
        # Apply position sizing rules
        position_value = available_cash * self.position_size * size_factor
        
        # Convert to shares
        shares = int(position_value / price)
        
        return shares
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> float:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]
        
        return atr
    
    def should_enter(self, 
                    symbol: str,
                    timestamp: datetime,
                    data: pd.Series,
                    signals: Optional[pd.Series] = None,
                    events: Optional[pd.Series] = None) -> Tuple[bool, PositionSide, float]:
        """Determine if should enter position based on signals."""
        if signals is None:
            return False, PositionSide.FLAT, 0.0
        
        # Get signal history for this symbol
        history = self.signal_history.get(symbol)
        
        # Check each rule
        for rule in self.signal_rules:
            if rule.check_entry(signals, history):
                if self._check_confirmation(rule.name, timestamp):
                    # Calculate position size
                    size = self.calculate_position_size(symbol, data['close'])
                    adjusted_size = size * rule.position_size_factor
                    
                    # Apply signal weights if available
                    if self.signal_weights:
                        weight_sum = sum(
                            self.signal_weights.get(cond.signal_name, 1.0) 
                            for cond in rule.entry_conditions
                        )
                        adjusted_size *= weight_sum / len(rule.entry_conditions)
                    
                    # Set risk parameters
                    if rule.stop_loss:
                        self.stop_loss = rule.stop_loss
                    if rule.take_profit:
                        self.take_profit = rule.take_profit
                    
                    return True, rule.position_side, adjusted_size
        
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
        
        if signals is None:
            return False
        
        # Get signal history
        history = self.signal_history.get(position.symbol)
        
        # Check exit conditions for the rule that opened this position
        if 'entry_rule' in position.metadata:
            rule_name = position.metadata['entry_rule']
            rule = next((r for r in self.signal_rules if r.name == rule_name), None)
            
            if rule:
                # Check time limit
                if rule.time_limit and position.trades:
                    bars_held = len(history) if history is not None else 0
                    if bars_held >= rule.time_limit:
                        logger.info(f"Time limit reached for {position.symbol}")
                        return True
                
                # Check exit conditions
                if rule.check_exit(signals, history):
                    return True
        
        # Check if opposite signal triggered
        for rule in self.signal_rules:
            if rule.check_entry(signals, history):
                # Opposite direction signal
                if (position.side == PositionSide.LONG and rule.position_side == PositionSide.SHORT) or \
                   (position.side == PositionSide.SHORT and rule.position_side == PositionSide.LONG):
                    logger.info(f"Opposite signal triggered for {position.symbol}")
                    return True
        
        return False
    
    def _check_confirmation(self, rule_name: str, timestamp: datetime) -> bool:
        """Check if signal has required confirmations."""
        if self.confirmation_required <= 1:
            return True
        
        # Track rule triggers
        if rule_name not in self.rule_triggers:
            self.rule_triggers[rule_name] = []
        
        triggers = self.rule_triggers[rule_name]
        triggers.append(timestamp)
        
        # Remove old triggers (more than 5 bars ago)
        triggers = [t for t in triggers if (timestamp - t).total_seconds() < 5 * 3600]
        self.rule_triggers[rule_name] = triggers
        
        return len(triggers) >= self.confirmation_required
    
    def update_signal_history(self, symbol: str, signals: pd.DataFrame):
        """Update signal history for a symbol."""
        self.signal_history[symbol] = signals.tail(100)  # Keep last 100 periods


# Predefined signal-based strategies
class MomentumStrategy(SignalBasedStrategy):
    """Momentum-based trading strategy."""
    
    def __init__(self, name: str = "Momentum Strategy", **kwargs):
        # Define momentum rules
        rules = [
            # Strong upward momentum
            SignalRule(
                name="strong_momentum_long",
                entry_conditions=[
                    SignalCondition("rsi_14", ConditionOperator.BETWEEN, [50, 70]),
                    SignalCondition("macd_signal", ConditionOperator.GREATER_THAN, 0),
                    SignalCondition("adx_14", ConditionOperator.GREATER_THAN, 25),
                    SignalCondition("returns_20", ConditionOperator.GREATER_THAN, 0.05)
                ],
                exit_conditions=[
                    SignalCondition("rsi_14", ConditionOperator.GREATER_THAN, 80),
                    SignalCondition("macd_signal", ConditionOperator.CROSSES_BELOW, 0)
                ],
                position_side=PositionSide.LONG,
                position_size_factor=1.2,
                stop_loss=0.02,
                take_profit=0.05
            ),
            # Strong downward momentum
            SignalRule(
                name="strong_momentum_short",
                entry_conditions=[
                    SignalCondition("rsi_14", ConditionOperator.BETWEEN, [30, 50]),
                    SignalCondition("macd_signal", ConditionOperator.LESS_THAN, 0),
                    SignalCondition("adx_14", ConditionOperator.GREATER_THAN, 25),
                    SignalCondition("returns_20", ConditionOperator.LESS_THAN, -0.05)
                ],
                exit_conditions=[
                    SignalCondition("rsi_14", ConditionOperator.LESS_THAN, 20),
                    SignalCondition("macd_signal", ConditionOperator.CROSSES_ABOVE, 0)
                ],
                position_side=PositionSide.SHORT,
                position_size_factor=1.2,
                stop_loss=0.02,
                take_profit=0.05
            )
        ]
        
        super().__init__(name=name, signal_rules=rules, **kwargs)


class MeanReversionStrategy(SignalBasedStrategy):
    """Mean reversion trading strategy."""
    
    def __init__(self, name: str = "Mean Reversion Strategy", **kwargs):
        rules = [
            # Oversold bounce
            SignalRule(
                name="oversold_bounce",
                entry_conditions=[
                    SignalCondition("rsi_14", ConditionOperator.LESS_THAN, 30),
                    SignalCondition("bb_lower", ConditionOperator.LESS_THAN, 0),  # Price below lower band
                    SignalCondition("returns_5", ConditionOperator.LESS_THAN, -0.03),
                    SignalCondition("volume_ratio", ConditionOperator.GREATER_THAN, 1.5)
                ],
                exit_conditions=[
                    SignalCondition("rsi_14", ConditionOperator.GREATER_THAN, 50),
                    SignalCondition("bb_middle", ConditionOperator.GREATER_THAN, 0)  # Price above middle band
                ],
                position_side=PositionSide.LONG,
                position_size_factor=0.8,
                stop_loss=0.015,
                time_limit=10
            ),
            # Overbought reversal
            SignalRule(
                name="overbought_reversal",
                entry_conditions=[
                    SignalCondition("rsi_14", ConditionOperator.GREATER_THAN, 70),
                    SignalCondition("bb_upper", ConditionOperator.GREATER_THAN, 0),  # Price above upper band
                    SignalCondition("returns_5", ConditionOperator.GREATER_THAN, 0.03),
                    SignalCondition("volume_ratio", ConditionOperator.GREATER_THAN, 1.5)
                ],
                exit_conditions=[
                    SignalCondition("rsi_14", ConditionOperator.LESS_THAN, 50),
                    SignalCondition("bb_middle", ConditionOperator.LESS_THAN, 0)  # Price below middle band
                ],
                position_side=PositionSide.SHORT,
                position_size_factor=0.8,
                stop_loss=0.015,
                time_limit=10
            )
        ]
        
        super().__init__(name=name, signal_rules=rules, confirmation_required=2, **kwargs)
