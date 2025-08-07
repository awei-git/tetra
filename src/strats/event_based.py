"""Event-based trading strategies."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import logging

from .base import BaseStrategy, PositionSide, Trade, Position

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of market events."""
    EARNINGS = "earnings"
    ECONOMIC = "economic"
    DIVIDEND = "dividend"
    SPLIT = "split"
    FOMC = "fomc"
    ECB = "ecb"
    NEWS = "news"
    CUSTOM = "custom"


class EventImpact(Enum):
    """Expected event impact levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class EventTrigger:
    """Defines an event trigger for trading."""
    event_type: EventType
    symbol: Optional[str] = None  # None for market-wide events
    impact: EventImpact = EventImpact.UNKNOWN
    pre_event_days: int = 5  # Days before event to start monitoring
    post_event_days: int = 3  # Days after event to hold/monitor
    entry_conditions: Dict[str, Any] = field(default_factory=dict)
    exit_conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketEvent:
    """Represents a market event."""
    timestamp: datetime
    event_type: EventType
    symbol: Optional[str]
    description: str
    impact: EventImpact
    actual_value: Optional[float] = None
    expected_value: Optional[float] = None
    previous_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def surprise(self) -> Optional[float]:
        """Calculate surprise factor if applicable."""
        if self.actual_value and self.expected_value:
            return (self.actual_value - self.expected_value) / abs(self.expected_value)
        return None
    
    @property
    def is_positive_surprise(self) -> bool:
        """Check if event was a positive surprise."""
        surprise = self.surprise
        return surprise is not None and surprise > 0


class EventBasedStrategy(BaseStrategy):
    """Strategy that trades based on market events."""
    
    def __init__(self,
                 name: str,
                 event_triggers: List[EventTrigger],
                 pre_event_strategy: Optional[Callable] = None,
                 post_event_strategy: Optional[Callable] = None,
                 **kwargs):
        super().__init__(name, **kwargs)
        self.event_triggers = {trigger.event_type: trigger for trigger in event_triggers}
        self.pre_event_strategy = pre_event_strategy
        self.post_event_strategy = post_event_strategy
        self.active_events: Dict[str, MarketEvent] = {}
        self.event_positions: Dict[str, List[Trade]] = {}
    
    def generate_signals(self, 
                        data: pd.DataFrame,
                        signals: Optional[pd.DataFrame] = None,
                        events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate trading signals based on events."""
        if events is None or events.empty:
            return pd.DataFrame(index=data.index, columns=['signal', 'position_size']).fillna(0)
        
        output_signals = pd.DataFrame(index=data.index, columns=['signal', 'position_size'])
        output_signals['signal'] = 0
        output_signals['position_size'] = 0
        
        # Process each event
        for idx, event_row in events.iterrows():
            event = self._create_event_from_row(event_row)
            if event.event_type not in self.event_triggers:
                continue
            
            trigger = self.event_triggers[event.event_type]
            
            # Determine trading window
            start_date = event.timestamp - timedelta(days=trigger.pre_event_days)
            end_date = event.timestamp + timedelta(days=trigger.post_event_days)
            
            # Get data for trading window
            mask = (data.index >= start_date) & (data.index <= end_date)
            window_data = data.loc[mask]
            
            if window_data.empty:
                continue
            
            # Apply pre-event strategy
            if self.pre_event_strategy and data.index.max() < event.timestamp:
                pre_signals = self.pre_event_strategy(
                    window_data, event, trigger, signals
                )
                output_signals.loc[mask, 'signal'] = pre_signals['signal']
                output_signals.loc[mask, 'position_size'] = pre_signals['position_size']
            
            # Apply post-event strategy
            elif self.post_event_strategy and data.index.min() > event.timestamp:
                post_signals = self.post_event_strategy(
                    window_data, event, trigger, signals
                )
                output_signals.loc[mask, 'signal'] = post_signals['signal']
                output_signals.loc[mask, 'position_size'] = post_signals['position_size']
        
        return output_signals
    
    def should_enter(self, 
                    symbol: str,
                    timestamp: datetime,
                    data: pd.Series,
                    signals: Optional[pd.Series] = None,
                    events: Optional[pd.Series] = None) -> Tuple[bool, PositionSide, float]:
        """Determine if should enter position based on events."""
        if not events:
            return False, PositionSide.FLAT, 0.0
        
        # Check for relevant events
        for event in events:
            if event['event_type'] not in self.event_triggers:
                continue
            
            trigger = self.event_triggers[event['event_type']]
            
            # Check if we're in the trading window
            event_time = pd.to_datetime(event['timestamp'])
            days_to_event = (event_time - timestamp).days
            
            if -trigger.pre_event_days <= days_to_event <= trigger.post_event_days:
                # Evaluate entry conditions
                should_enter, side = self._evaluate_entry_conditions(
                    trigger, event, data, signals, days_to_event
                )
                
                if should_enter:
                    # Calculate position size based on event impact
                    size_multiplier = self._get_size_multiplier(event['impact'])
                    position_size = self.position_size * size_multiplier
                    
                    return True, side, position_size
        
        return False, PositionSide.FLAT, 0.0
    
    def should_exit(self,
                   position: Position,
                   timestamp: datetime,
                   data: pd.Series,
                   signals: Optional[pd.Series] = None,
                   events: Optional[pd.Series] = None) -> bool:
        """Determine if should exit position."""
        # Check standard risk limits first
        if self.check_risk_limits(position, data['close']):
            return True
        
        # Check if position is event-related
        if position.symbol in self.event_positions:
            trades = self.event_positions[position.symbol]
            
            for trade in trades:
                if 'event' in trade.metadata:
                    event = trade.metadata['event']
                    trigger = self.event_triggers.get(event.event_type)
                    
                    if trigger:
                        # Check if we're past the event window
                        days_since_event = (timestamp - event.timestamp).days
                        if days_since_event > trigger.post_event_days:
                            logger.info(f"Exiting {position.symbol} - past event window")
                            return True
                        
                        # Evaluate exit conditions
                        if self._evaluate_exit_conditions(trigger, event, data, signals):
                            return True
        
        return False
    
    def _create_event_from_row(self, row: pd.Series) -> MarketEvent:
        """Create MarketEvent from dataframe row."""
        return MarketEvent(
            timestamp=pd.to_datetime(row.get('timestamp', row.name)),
            event_type=EventType(row.get('event_type', 'custom')),
            symbol=row.get('symbol'),
            description=row.get('description', ''),
            impact=EventImpact(row.get('impact', 'unknown')),
            actual_value=row.get('actual'),
            expected_value=row.get('expected'),
            previous_value=row.get('previous'),
            metadata=row.get('metadata', {})
        )
    
    def _evaluate_entry_conditions(self,
                                  trigger: EventTrigger,
                                  event: Dict[str, Any],
                                  data: pd.Series,
                                  signals: Optional[pd.Series],
                                  days_to_event: int) -> Tuple[bool, PositionSide]:
        """Evaluate entry conditions for an event."""
        conditions = trigger.entry_conditions
        
        # Default conditions based on event type
        if trigger.event_type == EventType.EARNINGS:
            # Enter long if IV is elevated before earnings
            if 'implied_volatility' in signals and signals['implied_volatility'] > conditions.get('iv_threshold', 0.3):
                return True, PositionSide.LONG
            
            # Or if technical signals are bullish
            if 'rsi' in signals and signals['rsi'] < conditions.get('rsi_oversold', 30):
                return True, PositionSide.LONG
        
        elif trigger.event_type == EventType.FOMC:
            # Trade volatility expansion before FOMC
            if days_to_event > -2:  # Before announcement
                if 'vix' in data and data['vix'] < conditions.get('vix_threshold', 15):
                    return True, PositionSide.LONG  # Expect volatility to increase
        
        elif trigger.event_type == EventType.ECONOMIC:
            # Trade based on momentum and expectations
            if 'trend' in signals and signals['trend'] > conditions.get('trend_strength', 0.5):
                return True, PositionSide.LONG
        
        # Custom condition evaluation
        if 'custom_condition' in conditions:
            return conditions['custom_condition'](data, signals, event, days_to_event)
        
        return False, PositionSide.FLAT
    
    def _evaluate_exit_conditions(self,
                                 trigger: EventTrigger,
                                 event: MarketEvent,
                                 data: pd.Series,
                                 signals: Optional[pd.Series]) -> bool:
        """Evaluate exit conditions for an event."""
        conditions = trigger.exit_conditions
        
        # Default exit conditions
        if trigger.event_type == EventType.EARNINGS:
            # Exit if post-earnings move exceeded expectations
            if 'price_change' in data and abs(data['price_change']) > conditions.get('move_threshold', 0.05):
                return True
        
        elif trigger.event_type == EventType.FOMC:
            # Exit after volatility spike
            if 'vix_change' in data and data['vix_change'] > conditions.get('vix_spike', 0.2):
                return True
        
        # Custom condition evaluation
        if 'custom_condition' in conditions:
            return conditions['custom_condition'](data, signals, event)
        
        return False
    
    def _get_size_multiplier(self, impact: str) -> float:
        """Get position size multiplier based on event impact."""
        impact_multipliers = {
            'high': 1.5,
            'medium': 1.0,
            'low': 0.5,
            'unknown': 0.75
        }
        return impact_multipliers.get(impact, 1.0)


# Predefined event strategies
class EarningsStrategy(EventBasedStrategy):
    """Strategy specifically for earnings announcements."""
    
    def __init__(self, name: str = "Earnings Strategy", **kwargs):
        # Define earnings-specific triggers
        earnings_trigger = EventTrigger(
            event_type=EventType.EARNINGS,
            impact=EventImpact.HIGH,
            pre_event_days=5,
            post_event_days=2,
            entry_conditions={
                'iv_threshold': 0.3,
                'rsi_oversold': 30,
                'volume_spike': 1.5
            },
            exit_conditions={
                'move_threshold': 0.05,
                'iv_crush': 0.2
            }
        )
        
        super().__init__(
            name=name,
            event_triggers=[earnings_trigger],
            pre_event_strategy=self._pre_earnings_strategy,
            post_event_strategy=self._post_earnings_strategy,
            **kwargs
        )
    
    def _pre_earnings_strategy(self, data, event, trigger, signals):
        """Strategy for trading before earnings."""
        signals_df = pd.DataFrame(index=data.index)
        signals_df['signal'] = 0
        signals_df['position_size'] = 0
        
        # Look for IV expansion opportunities
        if 'implied_volatility' in data.columns:
            iv_percentile = data['implied_volatility'].rolling(20).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min())
            )
            
            # Enter when IV is rising but not yet extreme
            signals_df.loc[iv_percentile.between(0.3, 0.7), 'signal'] = 1
            signals_df.loc[iv_percentile.between(0.3, 0.7), 'position_size'] = 0.5
        
        return signals_df
    
    def _post_earnings_strategy(self, data, event, trigger, signals):
        """Strategy for trading after earnings."""
        signals_df = pd.DataFrame(index=data.index)
        signals_df['signal'] = 0
        signals_df['position_size'] = 0
        
        # Trade the post-earnings drift
        if event.is_positive_surprise:
            # Positive surprise - look for continuation
            signals_df.iloc[0] = [1, 1.0]  # Enter long immediately
            
            # Exit if momentum fades
            if 'rsi' in data.columns:
                signals_df.loc[data['rsi'] > 70, 'signal'] = -1
        else:
            # Negative surprise or miss
            signals_df.iloc[0] = [-1, 1.0]  # Enter short
            
            # Exit if oversold
            if 'rsi' in data.columns:
                signals_df.loc[data['rsi'] < 30, 'signal'] = 1
        
        return signals_df


class FOMCStrategy(EventBasedStrategy):
    """Strategy for FOMC announcements."""
    
    def __init__(self, name: str = "FOMC Strategy", **kwargs):
        fomc_trigger = EventTrigger(
            event_type=EventType.FOMC,
            impact=EventImpact.HIGH,
            pre_event_days=3,
            post_event_days=5,
            entry_conditions={
                'vix_threshold': 15,
                'rate_expectations': 0.25
            },
            exit_conditions={
                'vix_spike': 0.2,
                'trend_reversal': True
            }
        )
        
        super().__init__(
            name=name,
            event_triggers=[fomc_trigger],
            pre_event_strategy=self._pre_fomc_strategy,
            post_event_strategy=self._post_fomc_strategy,
            **kwargs
        )
    
    def _pre_fomc_strategy(self, data, event, trigger, signals):
        """Trade volatility before FOMC."""
        signals_df = pd.DataFrame(index=data.index)
        signals_df['signal'] = 0
        signals_df['position_size'] = 0
        
        # Volatility typically compresses before FOMC
        # Trade for expansion
        days_to_event = (event.timestamp - data.index).days
        
        # Enter volatility longs 2-3 days before
        mask = days_to_event.between(-3, -1)
        signals_df.loc[mask, 'signal'] = 1
        signals_df.loc[mask, 'position_size'] = 0.75
        
        return signals_df
    
    def _post_fomc_strategy(self, data, event, trigger, signals):
        """Trade the FOMC decision."""
        signals_df = pd.DataFrame(index=data.index)
        signals_df['signal'] = 0
        signals_df['position_size'] = 0
        
        # Trade based on surprise
        if event.surprise:
            if event.surprise > 0:  # Hawkish surprise
                signals_df.iloc[0:5] = [-1, 1.0]  # Short equities
            else:  # Dovish surprise
                signals_df.iloc[0:5] = [1, 1.0]  # Long equities
        
        return signals_df