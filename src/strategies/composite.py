"""Composite trading strategies that combine multiple approaches."""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import logging

from .base import BaseStrategy, PositionSide, Trade, StrategyState, Position
from .event_based import EventBasedStrategy
from .signal_based import SignalBasedStrategy
from .time_based import TimeBasedStrategy

logger = logging.getLogger(__name__)


class CombinationMode(Enum):
    """How to combine multiple strategy signals."""
    UNANIMOUS = "unanimous"  # All strategies must agree
    MAJORITY = "majority"    # Majority vote
    WEIGHTED = "weighted"    # Weighted combination
    SEQUENTIAL = "sequential" # Apply in sequence
    ADAPTIVE = "adaptive"    # Adapt based on performance


@dataclass
class StrategyWeight:
    """Weight configuration for a strategy."""
    strategy: BaseStrategy
    weight: float = 1.0
    min_confidence: float = 0.0
    enabled: bool = True
    performance_weight: float = 0.0  # Weight based on recent performance
    
    @property
    def effective_weight(self) -> float:
        """Get effective weight combining static and performance weights."""
        if not self.enabled:
            return 0.0
        return self.weight + self.performance_weight


class CompositeStrategy(BaseStrategy):
    """Strategy that combines multiple sub-strategies."""
    
    def __init__(self,
                 name: str,
                 strategies: List[StrategyWeight],
                 combination_mode: CombinationMode = CombinationMode.WEIGHTED,
                 min_agreement: float = 0.6,
                 adapt_weights: bool = False,
                 performance_window: int = 100,
                 **kwargs):
        super().__init__(name, **kwargs)
        self.strategies = strategies
        self.combination_mode = combination_mode
        self.min_agreement = min_agreement
        self.adapt_weights = adapt_weights
        self.performance_window = performance_window
        
        # Track individual strategy performance
        self.strategy_performance: Dict[str, List[float]] = {}
        self.strategy_signals: Dict[str, pd.DataFrame] = {}
    
    def generate_signals(self, 
                        data: pd.DataFrame,
                        signals: Optional[pd.DataFrame] = None,
                        events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate combined signals from all strategies."""
        # Collect signals from all strategies
        all_signals = {}
        
        for sw in self.strategies:
            if not sw.enabled:
                continue
            
            strategy = sw.strategy
            strategy_signals = strategy.generate_signals(data, signals, events)
            all_signals[strategy.name] = strategy_signals
            self.strategy_signals[strategy.name] = strategy_signals
        
        # Combine signals based on mode
        if self.combination_mode == CombinationMode.UNANIMOUS:
            return self._combine_unanimous(all_signals, data.index)
        elif self.combination_mode == CombinationMode.MAJORITY:
            return self._combine_majority(all_signals, data.index)
        elif self.combination_mode == CombinationMode.WEIGHTED:
            return self._combine_weighted(all_signals, data.index)
        elif self.combination_mode == CombinationMode.SEQUENTIAL:
            return self._combine_sequential(all_signals, data.index)
        elif self.combination_mode == CombinationMode.ADAPTIVE:
            return self._combine_adaptive(all_signals, data.index)
        
        return pd.DataFrame(index=data.index, columns=['signal', 'position_size']).fillna(0)
    
    def should_enter(self, 
                    symbol: str,
                    timestamp: datetime,
                    data: pd.Series,
                    signals: Optional[pd.Series] = None,
                    events: Optional[pd.Series] = None) -> Tuple[bool, PositionSide, float]:
        """Determine if should enter based on combined strategies."""
        votes = []
        weights = []
        sizes = []
        
        for sw in self.strategies:
            if not sw.enabled:
                continue
            
            should_enter, side, size = sw.strategy.should_enter(
                symbol, timestamp, data, signals, events
            )
            
            if should_enter and side != PositionSide.FLAT:
                votes.append((side, sw.effective_weight))
                weights.append(sw.effective_weight)
                sizes.append(size)
        
        if not votes:
            return False, PositionSide.FLAT, 0.0
        
        # Determine consensus
        if self.combination_mode == CombinationMode.UNANIMOUS:
            # All must agree on direction
            if len(set(v[0] for v in votes)) == 1:
                return True, votes[0][0], np.mean(sizes)
        
        elif self.combination_mode == CombinationMode.MAJORITY:
            # Majority vote
            long_weight = sum(w for (s, w) in votes if s == PositionSide.LONG)
            short_weight = sum(w for (s, w) in votes if s == PositionSide.SHORT)
            total_weight = long_weight + short_weight
            
            if long_weight / total_weight >= self.min_agreement:
                return True, PositionSide.LONG, np.average(sizes, weights=weights)
            elif short_weight / total_weight >= self.min_agreement:
                return True, PositionSide.SHORT, np.average(sizes, weights=weights)
        
        elif self.combination_mode == CombinationMode.WEIGHTED:
            # Weighted decision
            long_weight = sum(w for (s, w) in votes if s == PositionSide.LONG)
            short_weight = sum(w for (s, w) in votes if s == PositionSide.SHORT)
            
            if long_weight > short_weight * (1 + self.min_agreement):
                return True, PositionSide.LONG, np.average(sizes, weights=weights)
            elif short_weight > long_weight * (1 + self.min_agreement):
                return True, PositionSide.SHORT, np.average(sizes, weights=weights)
        
        return False, PositionSide.FLAT, 0.0
    
    def should_exit(self,
                   position: Position,
                   timestamp: datetime,
                   data: pd.Series,
                   signals: Optional[pd.Series] = None,
                   events: Optional[pd.Series] = None) -> bool:
        """Determine if should exit based on combined strategies."""
        # Check standard risk limits
        if self.check_risk_limits(position, data['close']):
            return True
        
        exit_votes = 0
        total_weight = 0
        
        for sw in self.strategies:
            if not sw.enabled:
                continue
            
            if sw.strategy.should_exit(position, timestamp, data, signals, events):
                exit_votes += sw.effective_weight
            
            total_weight += sw.effective_weight
        
        # Exit if enough strategies agree
        if total_weight > 0:
            exit_ratio = exit_votes / total_weight
            return exit_ratio >= (1 - self.min_agreement)  # Lower threshold for exits
        
        return False
    
    def _combine_unanimous(self, all_signals: Dict[str, pd.DataFrame], 
                          index: pd.Index) -> pd.DataFrame:
        """Combine signals requiring unanimous agreement."""
        combined = pd.DataFrame(index=index)
        combined['signal'] = 0
        combined['position_size'] = 0
        
        for timestamp in index:
            signals_at_time = []
            sizes_at_time = []
            
            for strategy_name, strategy_signals in all_signals.items():
                if timestamp in strategy_signals.index:
                    sig = strategy_signals.loc[timestamp, 'signal']
                    if sig != 0:
                        signals_at_time.append(sig)
                        sizes_at_time.append(strategy_signals.loc[timestamp, 'position_size'])
            
            # Check if all agree
            if signals_at_time and len(set(signals_at_time)) == 1:
                combined.loc[timestamp, 'signal'] = signals_at_time[0]
                combined.loc[timestamp, 'position_size'] = np.mean(sizes_at_time)
        
        return combined
    
    def _combine_majority(self, all_signals: Dict[str, pd.DataFrame], 
                         index: pd.Index) -> pd.DataFrame:
        """Combine signals using majority vote."""
        combined = pd.DataFrame(index=index)
        combined['signal'] = 0
        combined['position_size'] = 0
        
        for timestamp in index:
            long_votes = 0
            short_votes = 0
            flat_votes = 0
            sizes = []
            
            for strategy_name, strategy_signals in all_signals.items():
                if timestamp in strategy_signals.index:
                    sig = strategy_signals.loc[timestamp, 'signal']
                    size = strategy_signals.loc[timestamp, 'position_size']
                    
                    if sig > 0:
                        long_votes += 1
                        sizes.append(size)
                    elif sig < 0:
                        short_votes += 1
                        sizes.append(size)
                    else:
                        flat_votes += 1
            
            total_votes = long_votes + short_votes + flat_votes
            if total_votes > 0:
                if long_votes / total_votes >= self.min_agreement:
                    combined.loc[timestamp, 'signal'] = 1
                    combined.loc[timestamp, 'position_size'] = np.mean(sizes) if sizes else 0
                elif short_votes / total_votes >= self.min_agreement:
                    combined.loc[timestamp, 'signal'] = -1
                    combined.loc[timestamp, 'position_size'] = np.mean(sizes) if sizes else 0
        
        return combined
    
    def _combine_weighted(self, all_signals: Dict[str, pd.DataFrame], 
                         index: pd.Index) -> pd.DataFrame:
        """Combine signals using weighted average."""
        combined = pd.DataFrame(index=index)
        combined['signal'] = 0
        combined['position_size'] = 0
        
        for timestamp in index:
            weighted_signal = 0
            weighted_size = 0
            total_weight = 0
            
            for sw in self.strategies:
                if not sw.enabled or sw.strategy.name not in all_signals:
                    continue
                
                strategy_signals = all_signals[sw.strategy.name]
                if timestamp in strategy_signals.index:
                    sig = strategy_signals.loc[timestamp, 'signal']
                    size = strategy_signals.loc[timestamp, 'position_size']
                    weight = sw.effective_weight
                    
                    weighted_signal += sig * weight
                    weighted_size += size * weight
                    total_weight += weight
            
            if total_weight > 0:
                avg_signal = weighted_signal / total_weight
                if abs(avg_signal) >= self.min_agreement:
                    combined.loc[timestamp, 'signal'] = np.sign(avg_signal)
                    combined.loc[timestamp, 'position_size'] = weighted_size / total_weight
        
        return combined
    
    def _combine_sequential(self, all_signals: Dict[str, pd.DataFrame], 
                           index: pd.Index) -> pd.DataFrame:
        """Apply strategies sequentially with filtering."""
        combined = pd.DataFrame(index=index)
        combined['signal'] = 0
        combined['position_size'] = 0
        
        # Start with first strategy
        for sw in self.strategies:
            if not sw.enabled:
                continue
            
            if sw.strategy.name in all_signals:
                base_signals = all_signals[sw.strategy.name].copy()
                break
        else:
            return combined
        
        # Apply subsequent strategies as filters
        for sw in self.strategies[1:]:
            if not sw.enabled or sw.strategy.name not in all_signals:
                continue
            
            filter_signals = all_signals[sw.strategy.name]
            
            # Only keep signals that are confirmed by filter
            for timestamp in base_signals.index:
                if timestamp in filter_signals.index:
                    if base_signals.loc[timestamp, 'signal'] != 0:
                        # Check if filter agrees on direction
                        if np.sign(filter_signals.loc[timestamp, 'signal']) != \
                           np.sign(base_signals.loc[timestamp, 'signal']):
                            base_signals.loc[timestamp, 'signal'] = 0
                            base_signals.loc[timestamp, 'position_size'] = 0
        
        return base_signals
    
    def _combine_adaptive(self, all_signals: Dict[str, pd.DataFrame], 
                         index: pd.Index) -> pd.DataFrame:
        """Adaptively weight strategies based on recent performance."""
        # Update performance weights if needed
        if self.adapt_weights:
            self._update_performance_weights()
        
        # Use weighted combination with updated weights
        return self._combine_weighted(all_signals, index)
    
    def _update_performance_weights(self):
        """Update strategy weights based on recent performance."""
        for sw in self.strategies:
            strategy_name = sw.strategy.name
            
            if strategy_name in self.strategy_performance:
                recent_performance = self.strategy_performance[strategy_name][-self.performance_window:]
                
                if recent_performance:
                    # Calculate performance metrics
                    win_rate = sum(1 for p in recent_performance if p > 0) / len(recent_performance)
                    avg_return = np.mean(recent_performance)
                    
                    # Adjust weight based on performance
                    performance_score = win_rate * 0.5 + (avg_return + 1) * 0.5
                    sw.performance_weight = (performance_score - 0.5) * sw.weight
    
    def record_trade_result(self, strategy_name: str, pnl: float):
        """Record trade result for performance tracking."""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = []
        
        self.strategy_performance[strategy_name].append(pnl)
        
        # Keep only recent history
        if len(self.strategy_performance[strategy_name]) > self.performance_window * 2:
            self.strategy_performance[strategy_name] = \
                self.strategy_performance[strategy_name][-self.performance_window:]


# Predefined composite strategies
class ConservativeComposite(CompositeStrategy):
    """Conservative strategy requiring multiple confirmations."""
    
    def __init__(self, name: str = "Conservative Composite", **kwargs):
        from .signal_based import MeanReversionStrategy, MomentumStrategy
        from .time_based import IntradayStrategy
        
        strategies = [
            StrategyWeight(
                MeanReversionStrategy(),
                weight=1.0,
                min_confidence=0.7
            ),
            StrategyWeight(
                MomentumStrategy(),
                weight=0.8,
                min_confidence=0.6
            ),
            StrategyWeight(
                IntradayStrategy(),
                weight=0.5,
                min_confidence=0.5
            )
        ]
        
        super().__init__(
            name=name,
            strategies=strategies,
            combination_mode=CombinationMode.UNANIMOUS,
            min_agreement=0.8,
            **kwargs
        )


class AggressiveComposite(CompositeStrategy):
    """Aggressive strategy with adaptive weighting."""
    
    def __init__(self, name: str = "Aggressive Composite", **kwargs):
        from .signal_based import MomentumStrategy, MLStrategy
        from .event_based import EarningsStrategy
        
        strategies = [
            StrategyWeight(
                MomentumStrategy(),
                weight=1.5,
                min_confidence=0.5
            ),
            StrategyWeight(
                MLStrategy(),
                weight=2.0,
                min_confidence=0.6
            ),
            StrategyWeight(
                EarningsStrategy(),
                weight=1.0,
                min_confidence=0.7
            )
        ]
        
        super().__init__(
            name=name,
            strategies=strategies,
            combination_mode=CombinationMode.ADAPTIVE,
            min_agreement=0.4,
            adapt_weights=True,
            **kwargs
        )


class BalancedComposite(CompositeStrategy):
    """Balanced strategy using majority voting."""
    
    def __init__(self, name: str = "Balanced Composite", **kwargs):
        from .signal_based import MomentumStrategy, MeanReversionStrategy
        from .event_based import FOMCStrategy
        from .time_based import IntradayStrategy
        
        strategies = [
            StrategyWeight(MomentumStrategy(), weight=1.0),
            StrategyWeight(MeanReversionStrategy(), weight=1.0),
            StrategyWeight(FOMCStrategy(), weight=0.8),
            StrategyWeight(IntradayStrategy(), weight=0.7)
        ]
        
        super().__init__(
            name=name,
            strategies=strategies,
            combination_mode=CombinationMode.MAJORITY,
            min_agreement=0.6,
            **kwargs
        )