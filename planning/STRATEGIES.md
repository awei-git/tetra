# Strategies Module Documentation

## Overview
The Strategies module defines how to make trading decisions based on signals. It provides a framework for creating, testing, and combining trading strategies with clear entry/exit rules and position sizing logic.

## Architecture

### Module Structure
```
src/strategies/
├── __init__.py
├── base.py                      # Base strategy classes
│   ├── Strategy                # Abstract base strategy
│   ├── Order                   # Order representation
│   └── StrategyState           # Strategy state tracking
│
├── rules/
│   ├── __init__.py
│   ├── entry.py                # Entry rule definitions
│   ├── exit.py                 # Exit rule definitions
│   ├── filters.py              # Trade filters (time, volatility)
│   └── combinators.py          # Rule combination logic
│
├── sizing/
│   ├── __init__.py
│   ├── fixed.py                # Fixed position sizing
│   ├── dynamic.py              # Dynamic sizing (Kelly, risk-based)
│   ├── portfolio.py            # Portfolio-level sizing
│   └── optimizer.py            # Optimal position sizing
│
├── risk/
│   ├── __init__.py
│   ├── stops.py                # Stop loss management
│   ├── limits.py               # Position and exposure limits
│   ├── correlation.py          # Correlation risk management
│   └── drawdown.py             # Drawdown control
│
├── implementations/
│   ├── __init__.py
│   ├── momentum.py             # Momentum strategies
│   ├── mean_reversion.py       # Mean reversion strategies
│   ├── pairs_trading.py        # Statistical arbitrage
│   ├── breakout.py             # Breakout strategies
│   └── ml_driven.py            # ML-based strategies
│
└── portfolio/
    ├── __init__.py
    ├── multi_strategy.py       # Combine multiple strategies
    ├── allocation.py           # Strategy allocation
    └── rebalancing.py          # Portfolio rebalancing
```

## Core Components

### 1. Base Strategy Framework

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from datetime import datetime
import pandas as pd

@dataclass
class Order:
    """Represents a trading order"""
    symbol: str
    action: Literal['BUY', 'SELL', 'SHORT', 'COVER']
    quantity: int
    order_type: Literal['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'] = 'MARKET'
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: Literal['DAY', 'GTC', 'IOC', 'FOK'] = 'DAY'
    metadata: Dict = field(default_factory=dict)

class Strategy(ABC):
    """Abstract base class for all strategies"""
    
    def __init__(self, 
                 universe: List[str],
                 entry_rules: EntryRules,
                 exit_rules: ExitRules,
                 position_sizer: PositionSizer,
                 risk_manager: Optional[RiskManager] = None):
        self.universe = universe
        self.entry_rules = entry_rules
        self.exit_rules = exit_rules
        self.position_sizer = position_sizer
        self.risk_manager = risk_manager or DefaultRiskManager()
        self.state = StrategyState()
        
    @abstractmethod
    def generate_orders(self,
                       signals: pd.DataFrame,
                       portfolio: Portfolio,
                       timestamp: datetime) -> List[Order]:
        """Generate orders based on signals and portfolio state"""
        pass
    
    def filter_orders(self, 
                     orders: List[Order], 
                     portfolio: Portfolio) -> List[Order]:
        """Apply risk management and filters to orders"""
        # Apply risk limits
        orders = self.risk_manager.filter_orders(orders, portfolio)
        
        # Apply portfolio constraints
        orders = self._apply_portfolio_constraints(orders, portfolio)
        
        # Log filtered orders
        self._log_orders(orders)
        
        return orders

class StrategyState:
    """Track strategy state and history"""
    
    def __init__(self):
        self.positions = {}
        self.pending_orders = []
        self.trade_history = []
        self.performance_metrics = {}
        
    def update(self, timestamp: datetime, event: Dict):
        """Update strategy state with new event"""
        # Track position changes, orders, fills, etc.
        pass
```

### 2. Entry and Exit Rules

```python
class EntryRule(ABC):
    """Base class for entry rules"""
    
    @abstractmethod
    def check(self, 
             symbol: str, 
             signals: pd.Series, 
             portfolio: Portfolio) -> bool:
        """Check if entry conditions are met"""
        pass
    
    def get_confidence(self, signals: pd.Series) -> float:
        """Return confidence level (0-1) for the entry"""
        return 1.0

class CompositeEntryRule(EntryRule):
    """Combine multiple entry rules"""
    
    def __init__(self, 
                 rules: List[EntryRule], 
                 combination: Literal['AND', 'OR', 'WEIGHTED'] = 'AND'):
        self.rules = rules
        self.combination = combination
        
    def check(self, symbol: str, signals: pd.Series, portfolio: Portfolio) -> bool:
        if self.combination == 'AND':
            return all(rule.check(symbol, signals, portfolio) for rule in self.rules)
        elif self.combination == 'OR':
            return any(rule.check(symbol, signals, portfolio) for rule in self.rules)
        else:  # WEIGHTED
            total_weight = sum(rule.get_confidence(signals) for rule in self.rules)
            return total_weight > 0.5 * len(self.rules)

# Concrete entry rules
class RSIEntryRule(EntryRule):
    """Enter on RSI conditions"""
    
    def __init__(self, 
                 oversold_threshold: float = 30,
                 overbought_threshold: float = 70,
                 direction: Literal['LONG', 'SHORT'] = 'LONG'):
        self.oversold = oversold_threshold
        self.overbought = overbought_threshold
        self.direction = direction
        
    def check(self, symbol: str, signals: pd.Series, portfolio: Portfolio) -> bool:
        rsi = signals.get('rsi_14', 50)
        
        if self.direction == 'LONG':
            return rsi < self.oversold
        else:
            return rsi > self.overbought

class BreakoutEntryRule(EntryRule):
    """Enter on price breakouts"""
    
    def __init__(self, lookback: int = 20, breakout_factor: float = 1.0):
        self.lookback = lookback
        self.breakout_factor = breakout_factor
        
    def check(self, symbol: str, signals: pd.Series, portfolio: Portfolio) -> bool:
        current_price = signals['close']
        resistance = signals.get(f'resistance_{self.lookback}', current_price)
        
        return current_price > resistance * self.breakout_factor

class MLEntryRule(EntryRule):
    """Enter based on ML predictions"""
    
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        
    def check(self, symbol: str, signals: pd.Series, portfolio: Portfolio) -> bool:
        ml_signal = signals.get('ml_direction_buy_prob', 0.5)
        return ml_signal > self.threshold
    
    def get_confidence(self, signals: pd.Series) -> float:
        return signals.get('ml_direction_buy_prob', 0.5)
```

### 3. Position Sizing

```python
class PositionSizer(ABC):
    """Base class for position sizing"""
    
    @abstractmethod
    def calculate_size(self,
                      symbol: str,
                      signals: pd.Series,
                      portfolio: Portfolio,
                      entry_price: float) -> int:
        """Calculate position size in shares"""
        pass

class FixedDollarSizer(PositionSizer):
    """Fixed dollar amount per position"""
    
    def __init__(self, position_size: float = 10000):
        self.position_size = position_size
        
    def calculate_size(self, symbol, signals, portfolio, entry_price):
        return int(self.position_size / entry_price)

class PercentagePortfolioSizer(PositionSizer):
    """Size as percentage of portfolio"""
    
    def __init__(self, percent: float = 0.1):
        self.percent = percent
        
    def calculate_size(self, symbol, signals, portfolio, entry_price):
        portfolio_value = portfolio.get_total_value()
        position_value = portfolio_value * self.percent
        return int(position_value / entry_price)

class KellySizer(PositionSizer):
    """Kelly Criterion position sizing"""
    
    def __init__(self, 
                 lookback: int = 100,
                 kelly_fraction: float = 0.25):
        self.lookback = lookback
        self.kelly_fraction = kelly_fraction
        
    def calculate_size(self, symbol, signals, portfolio, entry_price):
        # Calculate win rate and win/loss ratio from history
        trades = portfolio.get_trade_history(symbol)[-self.lookback:]
        
        if len(trades) < 10:
            # Not enough history, use default
            return PercentagePortfolioSizer(0.02).calculate_size(
                symbol, signals, portfolio, entry_price
            )
        
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        
        win_rate = len(wins) / len(trades)
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 1
        
        # Kelly formula: f = p - q/b
        # where p = win probability, q = loss probability, b = win/loss ratio
        kelly_percent = win_rate - (1 - win_rate) / (avg_win / avg_loss)
        kelly_percent = max(0, min(kelly_percent, 0.25))  # Cap at 25%
        
        # Use fraction of Kelly
        position_percent = kelly_percent * self.kelly_fraction
        position_value = portfolio.get_total_value() * position_percent
        
        return int(position_value / entry_price)

class VolatilityAdjustedSizer(PositionSizer):
    """Size inversely proportional to volatility"""
    
    def __init__(self, 
                 target_volatility: float = 0.02,
                 lookback: int = 20):
        self.target_vol = target_volatility
        self.lookback = lookback
        
    def calculate_size(self, symbol, signals, portfolio, entry_price):
        # Get asset volatility
        returns = signals.get(f'returns_{self.lookback}d', pd.Series())
        asset_vol = returns.std() * np.sqrt(252)
        
        if asset_vol == 0:
            asset_vol = 0.2  # Default 20% volatility
        
        # Scale position by volatility
        vol_scalar = self.target_vol / asset_vol
        base_size = portfolio.get_total_value() * 0.1  # 10% base
        position_value = base_size * vol_scalar
        
        return int(position_value / entry_price)
```

### 4. Risk Management

```python
class RiskManager:
    """Manage strategy risks"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.stop_manager = StopLossManager(config)
        self.position_limiter = PositionLimiter(config)
        self.correlation_monitor = CorrelationMonitor(config)
        
    def filter_orders(self, 
                     orders: List[Order], 
                     portfolio: Portfolio) -> List[Order]:
        """Apply risk filters to orders"""
        
        # Check position limits
        orders = self.position_limiter.filter(orders, portfolio)
        
        # Check correlation limits
        orders = self.correlation_monitor.filter(orders, portfolio)
        
        # Add stop losses
        orders = self._add_stop_losses(orders)
        
        # Check daily loss limit
        if self._daily_loss_exceeded(portfolio):
            # Only allow closing orders
            orders = [o for o in orders if o.action in ['SELL', 'COVER']]
        
        return orders
    
    def _add_stop_losses(self, orders: List[Order]) -> List[Order]:
        """Add stop loss orders"""
        stop_orders = []
        
        for order in orders:
            if order.action == 'BUY':
                stop_price = order.limit_price * (1 - self.config.stop_loss_percent)
                stop_order = Order(
                    symbol=order.symbol,
                    action='SELL',
                    quantity=order.quantity,
                    order_type='STOP',
                    stop_price=stop_price,
                    metadata={'parent_order': order, 'type': 'stop_loss'}
                )
                stop_orders.append(stop_order)
        
        return orders + stop_orders

class StopLossManager:
    """Manage stop losses"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.stops = {}
        
    def update_stops(self, positions: Dict[str, Position], market_data: pd.DataFrame):
        """Update trailing stops"""
        for symbol, position in positions.items():
            current_price = market_data.loc[symbol, 'close']
            
            if symbol not in self.stops:
                # Initial stop
                self.stops[symbol] = position.entry_price * (1 - self.config.stop_loss_percent)
            else:
                # Trailing stop
                new_stop = current_price * (1 - self.config.trailing_stop_percent)
                self.stops[symbol] = max(self.stops[symbol], new_stop)
    
    def check_stops(self, positions: Dict[str, Position], market_data: pd.DataFrame) -> List[Order]:
        """Check if any stops are hit"""
        stop_orders = []
        
        for symbol, position in positions.items():
            if symbol in self.stops:
                current_price = market_data.loc[symbol, 'close']
                if current_price <= self.stops[symbol]:
                    stop_orders.append(Order(
                        symbol=symbol,
                        action='SELL',
                        quantity=position.quantity,
                        order_type='MARKET',
                        metadata={'reason': 'stop_loss_hit'}
                    ))
        
        return stop_orders
```

### 5. Strategy Implementations

```python
class MomentumStrategy(Strategy):
    """Classic momentum strategy"""
    
    def __init__(self, 
                 universe: List[str],
                 lookback: int = 20,
                 holding_period: int = 5,
                 top_n: int = 10):
        
        # Define entry rules
        entry_rules = CompositeEntryRule([
            MomentumEntryRule(lookback=lookback),
            VolumeEntryRule(min_volume_ratio=1.5),
            TrendEntryRule(min_trend_strength=0.7)
        ])
        
        # Define exit rules  
        exit_rules = CompositeExitRule([
            HoldingPeriodExitRule(days=holding_period),
            StopLossExitRule(percent=0.05),
            ProfitTargetExitRule(percent=0.15)
        ])
        
        # Position sizing
        position_sizer = EqualWeightSizer(n_positions=top_n)
        
        super().__init__(universe, entry_rules, exit_rules, position_sizer)
        
        self.lookback = lookback
        self.top_n = top_n
        self.rebalance_day = 0
        
    def generate_orders(self, signals, portfolio, timestamp):
        orders = []
        
        # Check if it's rebalance day
        self.rebalance_day += 1
        if self.rebalance_day % self.holding_period != 0:
            # Only check exits
            return self._generate_exit_orders(signals, portfolio)
        
        # Rank symbols by momentum
        momentum_scores = {}
        for symbol in self.universe:
            if symbol in signals.index:
                momentum_scores[symbol] = signals.loc[symbol, f'returns_{self.lookback}d']
        
        # Select top N
        ranked_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [s[0] for s in ranked_symbols[:self.top_n]]
        
        # Generate orders
        current_positions = portfolio.get_positions()
        
        # Exit positions not in top N
        for symbol in current_positions:
            if symbol not in top_symbols:
                orders.append(Order(
                    symbol=symbol,
                    action='SELL',
                    quantity=current_positions[symbol].quantity,
                    metadata={'reason': 'rebalance_exit'}
                ))
        
        # Enter new positions
        for symbol in top_symbols:
            if symbol not in current_positions:
                if self.entry_rules.check(symbol, signals.loc[symbol], portfolio):
                    quantity = self.position_sizer.calculate_size(
                        symbol, 
                        signals.loc[symbol], 
                        portfolio,
                        signals.loc[symbol, 'close']
                    )
                    
                    if quantity > 0:
                        orders.append(Order(
                            symbol=symbol,
                            action='BUY',
                            quantity=quantity,
                            metadata={
                                'reason': 'momentum_entry',
                                'momentum_score': momentum_scores[symbol]
                            }
                        ))
        
        return self.filter_orders(orders, portfolio)

class MeanReversionStrategy(Strategy):
    """Bollinger Band mean reversion"""
    
    def __init__(self,
                 universe: List[str],
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 holding_period: int = 5):
        
        # Entry when price touches lower band
        entry_rules = CompositeEntryRule([
            BollingerBandEntryRule(position='lower', touch_distance=0.02),
            RSIEntryRule(oversold_threshold=30),
            VolumeEntryRule(min_volume_ratio=0.8)
        ], combination='AND')
        
        # Exit at middle band or stop
        exit_rules = CompositeExitRule([
            BollingerBandExitRule(target='middle'),
            StopLossExitRule(percent=0.03),
            TimeExitRule(max_days=holding_period)
        ], combination='OR')
        
        # Size based on distance from band
        position_sizer = BandDistanceSizer(
            base_size=0.1,
            max_size=0.2
        )
        
        super().__init__(universe, entry_rules, exit_rules, position_sizer)

class PairsTradingStrategy(Strategy):
    """Statistical arbitrage between pairs"""
    
    def __init__(self,
                 pair: Tuple[str, str],
                 lookback: int = 60,
                 entry_zscore: float = 2.0,
                 exit_zscore: float = 0.0):
        
        self.pair = pair
        self.lookback = lookback
        
        # Entry on z-score extremes
        entry_rules = ZScoreEntryRule(
            threshold=entry_zscore,
            lookback=lookback
        )
        
        # Exit on z-score mean reversion
        exit_rules = ZScoreExitRule(
            threshold=exit_zscore
        )
        
        # Size based on cointegration confidence
        position_sizer = CointegrationSizer()
        
        super().__init__(list(pair), entry_rules, exit_rules, position_sizer)
        
    def generate_orders(self, signals, portfolio, timestamp):
        # Calculate spread and z-score
        price_a = signals.loc[self.pair[0], 'close']
        price_b = signals.loc[self.pair[1], 'close']
        
        spread_info = self._calculate_spread(
            signals.loc[self.pair[0]], 
            signals.loc[self.pair[1]]
        )
        
        zscore = spread_info['zscore']
        hedge_ratio = spread_info['hedge_ratio']
        
        orders = []
        positions = portfolio.get_positions()
        
        # Check entry
        if abs(zscore) > self.entry_zscore:
            if self.pair[0] not in positions:  # Not already in position
                if zscore > self.entry_zscore:
                    # Spread too high: short A, long B
                    size_a = self.position_sizer.calculate_size(
                        self.pair[0], signals.loc[self.pair[0]], portfolio, price_a
                    )
                    size_b = int(size_a * hedge_ratio)
                    
                    orders.extend([
                        Order(self.pair[0], 'SHORT', size_a),
                        Order(self.pair[1], 'BUY', size_b)
                    ])
                else:
                    # Spread too low: long A, short B
                    size_a = self.position_sizer.calculate_size(
                        self.pair[0], signals.loc[self.pair[0]], portfolio, price_a
                    )
                    size_b = int(size_a * hedge_ratio)
                    
                    orders.extend([
                        Order(self.pair[0], 'BUY', size_a),
                        Order(self.pair[1], 'SHORT', size_b)
                    ])
        
        # Check exit
        elif abs(zscore) < self.exit_zscore and self.pair[0] in positions:
            # Close both legs
            orders.extend([
                Order(
                    self.pair[0], 
                    'COVER' if positions[self.pair[0]].quantity < 0 else 'SELL',
                    abs(positions[self.pair[0]].quantity)
                ),
                Order(
                    self.pair[1],
                    'COVER' if positions[self.pair[1]].quantity < 0 else 'SELL',
                    abs(positions[self.pair[1]].quantity)
                )
            ])
        
        return self.filter_orders(orders, portfolio)
```

### 6. Multi-Strategy Portfolio

```python
class MultiStrategyPortfolio:
    """Combine multiple strategies with allocation"""
    
    def __init__(self, strategy_allocations: List[Tuple[Strategy, float]]):
        """
        Args:
            strategy_allocations: List of (strategy, allocation_weight) tuples
        """
        self.strategies = strategy_allocations
        self._normalize_weights()
        
    def _normalize_weights(self):
        """Ensure weights sum to 1"""
        total = sum(weight for _, weight in self.strategies)
        self.strategies = [(s, w/total) for s, w in self.strategies]
        
    def generate_orders(self, signals, portfolio, timestamp):
        """Generate orders from all strategies"""
        all_orders = []
        
        for strategy, weight in self.strategies:
            # Create sub-portfolio view for this strategy
            sub_portfolio = portfolio.create_sub_portfolio(weight)
            
            # Get strategy orders
            strategy_orders = strategy.generate_orders(
                signals, 
                sub_portfolio, 
                timestamp
            )
            
            # Scale orders by allocation
            for order in strategy_orders:
                order.quantity = int(order.quantity * weight)
                order.metadata['strategy'] = strategy.__class__.__name__
                order.metadata['allocation'] = weight
                
            all_orders.extend(strategy_orders)
        
        # Combine orders for same symbol/direction
        combined_orders = self._combine_orders(all_orders)
        
        return combined_orders
    
    def _combine_orders(self, orders: List[Order]) -> List[Order]:
        """Combine orders for the same symbol and direction"""
        order_map = {}
        
        for order in orders:
            key = (order.symbol, order.action)
            if key in order_map:
                # Combine quantities
                order_map[key].quantity += order.quantity
                # Merge metadata
                order_map[key].metadata['combined_from'].append(order.metadata)
            else:
                order.metadata['combined_from'] = [order.metadata]
                order_map[key] = order
        
        return list(order_map.values())
```

## Strategy Configuration

```python
@dataclass
class StrategyConfig:
    """Configuration for strategies"""
    
    # Universe selection
    universe: List[str]
    universe_selection: Literal['static', 'dynamic'] = 'static'
    
    # Risk parameters
    max_positions: int = 10
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_percent: float = 0.05
    trailing_stop_percent: float = 0.03
    
    # Execution
    allow_shorts: bool = False
    allow_margin: bool = False
    rebalance_frequency: Literal['daily', 'weekly', 'monthly'] = 'daily'
    
    # Filters
    min_price: float = 5.0
    min_volume: float = 1000000
    max_spread_percent: float = 0.01
    
    # Optimization
    lookback_period: int = 252
    walk_forward_periods: int = 63
```

## Usage Examples

### Single Strategy
```python
# Create momentum strategy
strategy = MomentumStrategy(
    universe=['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    lookback=20,
    holding_period=5,
    top_n=2
)

# Get signals
signals = signal_computer.compute_all(market_data)

# Generate orders
orders = strategy.generate_orders(signals, portfolio, datetime.now())

# Execute orders
for order in orders:
    portfolio.execute_order(order, market_data)
```

### Multi-Strategy
```python
# Create multiple strategies
momentum = MomentumStrategy(universe, lookback=20)
mean_rev = MeanReversionStrategy(universe, bb_period=20)
pairs = PairsTradingStrategy(('GLD', 'SLV'), lookback=60)

# Combine with allocation
multi_strategy = MultiStrategyPortfolio([
    (momentum, 0.4),    # 40% to momentum
    (mean_rev, 0.4),    # 40% to mean reversion
    (pairs, 0.2)        # 20% to pairs trading
])

# Generate combined orders
orders = multi_strategy.generate_orders(signals, portfolio, timestamp)
```

### Custom Strategy
```python
class MyCustomStrategy(Strategy):
    """Example custom strategy"""
    
    def __init__(self, universe):
        # Custom entry: RSI < 30 AND increasing volume
        entry = CompositeEntryRule([
            RSIEntryRule(oversold_threshold=30),
            VolumeIncreaseRule(min_increase=1.5),
            lambda s: s['close'] > s['sma_200']  # Above 200 SMA
        ])
        
        # Custom exit: 10% profit or 5% stop
        exit = CompositeExitRule([
            ProfitTargetExitRule(percent=0.10),
            StopLossExitRule(percent=0.05)
        ], combination='OR')
        
        # Size based on confidence
        sizer = ConfidenceBasedSizer(
            base_size=0.05,
            max_size=0.15
        )
        
        super().__init__(universe, entry, exit, sizer)
```

## Integration Points

### 1. With Signals Module
```python
# Signals provide input for strategies
signals = signal_computer.compute_all(market_data)
orders = strategy.generate_orders(signals, portfolio, timestamp)
```

### 2. With Performance Module
```python
# Backtest strategy
result = backtester.run(
    strategy=strategy,
    signal_computer=signal_computer,
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

### 3. With Risk Management
```python
# Risk manager filters orders
risk_manager = RiskManager(risk_config)
safe_orders = risk_manager.filter_orders(orders, portfolio)
```

## Testing Strategy

### 1. Unit Tests
- Test entry/exit rule logic
- Verify position sizing calculations
- Test order generation

### 2. Integration Tests
- Test strategy with mock signals
- Verify risk management integration
- Test multi-strategy combinations

### 3. Backtesting
- Historical performance validation
- Parameter sensitivity analysis
- Walk-forward optimization

## Best Practices

### 1. Strategy Development
- Start simple, add complexity gradually
- Always include risk management
- Test on out-of-sample data
- Consider transaction costs

### 2. Parameter Selection
- Avoid overfitting
- Use walk-forward optimization
- Test parameter stability
- Consider market regime changes

### 3. Risk Management
- Always use stop losses
- Limit position sizes
- Monitor correlations
- Set portfolio-level limits

### 4. Performance Monitoring
- Track strategy metrics daily
- Compare to benchmarks
- Monitor for strategy decay
- Regular reoptimization

## Future Enhancements

1. **Advanced Order Types**
   - Iceberg orders
   - TWAP/VWAP execution
   - Smart order routing

2. **ML Integration**
   - Reinforcement learning strategies
   - Online strategy adaptation
   - Feature importance analysis

3. **Market Microstructure**
   - Order book strategies
   - High-frequency components
   - Liquidity provision

4. **Options Strategies**
   - Covered calls/puts
   - Spreads and combinations
   - Volatility strategies

5. **Portfolio Optimization**
   - Mean-variance optimization
   - Black-Litterman allocation
   - Risk parity strategies