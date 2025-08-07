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
    value: Union[float, List[float]]
    lookback: int = 1  # Number of periods to look back
    weight: float = 1.0  # Weight for composite conditions
    
    def evaluate(self, signals: pd.Series, history: Optional[pd.DataFrame] = None) -> bool:
        """Evaluate if condition is met."""
        if self.signal_name not in signals:
            return False
        
        current_value = signals[self.signal_name]
        
        if self.operator == ConditionOperator.GREATER_THAN:
            return current_value > self.value
        elif self.operator == ConditionOperator.LESS_THAN:
            return current_value < self.value
        elif self.operator == ConditionOperator.EQUAL:
            return current_value == self.value
        elif self.operator == ConditionOperator.GREATER_EQUAL:
            return current_value >= self.value
        elif self.operator == ConditionOperator.LESS_EQUAL:
            return current_value <= self.value
        elif self.operator == ConditionOperator.NOT_EQUAL:
            return current_value != self.value
        elif self.operator == ConditionOperator.BETWEEN:
            return self.value[0] <= current_value <= self.value[1]
        elif self.operator == ConditionOperator.OUTSIDE:
            return current_value < self.value[0] or current_value > self.value[1]
        
        # For crosses, we need history
        if history is not None and self.signal_name in history.columns:
            if self.operator == ConditionOperator.CROSSES_ABOVE:
                prev_value = history[self.signal_name].iloc[-self.lookback]
                return prev_value <= self.value and current_value > self.value
            elif self.operator == ConditionOperator.CROSSES_BELOW:
                prev_value = history[self.signal_name].iloc[-self.lookback]
                return prev_value >= self.value and current_value < self.value
        
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
        self.universe = symbols
    
    def generate_signals(self, 
                        data: pd.DataFrame,
                        signals: Optional[pd.DataFrame] = None,
                        events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate trading signals based on rules."""
        if signals is None or signals.empty:
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


class MLStrategy(SignalBasedStrategy):
    """Machine learning based strategy."""
    
    def __init__(self, name: str = "ML Strategy", **kwargs):
        rules = [
            # ML model predicts up
            SignalRule(
                name="ml_long",
                entry_conditions=[
                    SignalCondition("ml_prediction", ConditionOperator.GREATER_THAN, 0.6),
                    SignalCondition("ml_confidence", ConditionOperator.GREATER_THAN, 0.7),
                    SignalCondition("feature_importance_price", ConditionOperator.GREATER_THAN, 0.3)
                ],
                exit_conditions=[
                    SignalCondition("ml_prediction", ConditionOperator.LESS_THAN, 0.4),
                    SignalCondition("ml_confidence", ConditionOperator.LESS_THAN, 0.5)
                ],
                position_side=PositionSide.LONG,
                position_size_factor=1.0,
                stop_loss=0.02
            ),
            # ML model predicts down
            SignalRule(
                name="ml_short",
                entry_conditions=[
                    SignalCondition("ml_prediction", ConditionOperator.LESS_THAN, 0.4),
                    SignalCondition("ml_confidence", ConditionOperator.GREATER_THAN, 0.7),
                    SignalCondition("feature_importance_price", ConditionOperator.GREATER_THAN, 0.3)
                ],
                exit_conditions=[
                    SignalCondition("ml_prediction", ConditionOperator.GREATER_THAN, 0.6),
                    SignalCondition("ml_confidence", ConditionOperator.LESS_THAN, 0.5)
                ],
                position_side=PositionSide.SHORT,
                position_size_factor=1.0,
                stop_loss=0.02
            ),
            # Anomaly detection
            SignalRule(
                name="anomaly_exit",
                entry_conditions=[],  # This is exit only
                exit_conditions=[
                    SignalCondition("anomaly_score", ConditionOperator.GREATER_THAN, 0.8)
                ],
                position_side=PositionSide.FLAT,
                position_size_factor=0
            )
        ]
        
        # Higher weight for ML signals
        signal_weights = {
            "ml_prediction": 2.0,
            "ml_confidence": 1.5,
            "feature_importance_price": 1.0
        }
        
        super().__init__(
            name=name, 
            signal_rules=rules, 
            signal_weights=signal_weights,
            **kwargs
        )