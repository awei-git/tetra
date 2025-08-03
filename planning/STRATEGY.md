# Trading Strategy Documentation

## Overview
The Tetra Strategy Engine provides a framework for defining, backtesting, and executing trading strategies. It supports various strategy types including pairs trading, momentum, mean reversion, and custom strategies with integrated risk management.

## Architecture

### Module Structure
```
src/strategies/
├── __init__.py
├── base.py                      # Abstract base classes
│   ├── BaseStrategy            # Strategy interface
│   ├── Signal                  # Trading signals
│   └── StrategyResult          # Backtest results
│
├── implementations/
│   ├── __init__.py
│   ├── pairs_trading.py        # Pairs/spread trading
│   ├── momentum.py             # Trend following
│   ├── mean_reversion.py       # Contrarian strategies
│   ├── arbitrage.py            # Stat arb strategies
│   └── ml_strategies.py        # ML-based strategies
│
├── indicators/
│   ├── __init__.py
│   ├── technical.py            # Technical indicators
│   ├── statistical.py          # Statistical measures
│   └── custom.py               # Custom indicators
│
├── risk/
│   ├── __init__.py
│   ├── position_sizing.py      # Kelly, fixed fraction
│   ├── stop_loss.py            # Stop management
│   └── portfolio_limits.py     # Risk constraints
│
└── optimization/
    ├── __init__.py
    ├── parameter_search.py     # Grid/random search
    ├── walk_forward.py         # Walk-forward analysis
    └── genetic_algo.py         # Genetic optimization
```

## Core Components

### 1. Base Strategy

**Strategy Interface**
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, 
                 universe: List[str],
                 params: Dict[str, Any],
                 risk_manager: Optional[RiskManager] = None):
        self.universe = universe
        self.params = params
        self.risk_manager = risk_manager or DefaultRiskManager()
        self.indicators = {}
        
    @abstractmethod
    async def generate_signals(self,
                              market_data: MarketData,
                              portfolio: Portfolio,
                              timestamp: datetime) -> List[Signal]:
        """Generate trading signals based on current market state"""
        pass
    
    @abstractmethod
    def calculate_indicators(self, 
                           historical_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate strategy-specific indicators"""
        pass
    
    def validate_signals(self, 
                        signals: List[Signal], 
                        portfolio: Portfolio) -> List[Signal]:
        """Apply risk management rules to signals"""
        return self.risk_manager.validate_signals(signals, portfolio)
```

**Signal Class**
```python
@dataclass
class Signal:
    """Trading signal representation"""
    symbol: str
    direction: Literal["BUY", "SELL", "HOLD"]
    signal_strength: float  # -1 to 1
    quantity: Optional[int] = None
    order_type: Literal["MARKET", "LIMIT", "STOP"] = "MARKET"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: Literal["DAY", "GTC", "IOC", "FOK"] = "DAY"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_order(self) -> Order:
        """Convert signal to executable order"""
        return Order(
            symbol=self.symbol,
            side="BUY" if self.direction == "BUY" else "SELL",
            quantity=self.quantity,
            order_type=self.order_type,
            limit_price=self.limit_price,
            time_in_force=self.time_in_force
        )
```

### 2. Strategy Implementations

**Pairs Trading Strategy**
```python
class PairsTradingStrategy(BaseStrategy):
    """Statistical arbitrage between correlated pairs"""
    
    def __init__(self, 
                 pair: Tuple[str, str],
                 lookback_period: int = 60,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 **kwargs):
        super().__init__(universe=list(pair), **kwargs)
        self.pair = pair
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate spread and z-score"""
        # Get prices for both symbols
        price_a = data[self.pair[0]]['close']
        price_b = data[self.pair[1]]['close']
        
        # Calculate hedge ratio using OLS
        hedge_ratio = self._calculate_hedge_ratio(price_a, price_b)
        
        # Calculate spread
        spread = price_a - hedge_ratio * price_b
        
        # Calculate z-score
        spread_mean = spread.rolling(self.lookback_period).mean()
        spread_std = spread.rolling(self.lookback_period).std()
        z_score = (spread - spread_mean) / spread_std
        
        return {
            'spread': spread,
            'z_score': z_score,
            'hedge_ratio': pd.Series(hedge_ratio, index=spread.index)
        }
    
    async def generate_signals(self, 
                              market_data: MarketData,
                              portfolio: Portfolio,
                              timestamp: datetime) -> List[Signal]:
        """Generate pairs trading signals"""
        signals = []
        
        # Get current z-score
        z_score = self.indicators['z_score'].iloc[-1]
        hedge_ratio = self.indicators['hedge_ratio'].iloc[-1]
        
        # Current positions
        pos_a = portfolio.get_position(self.pair[0])
        pos_b = portfolio.get_position(self.pair[1])
        
        # Entry signals
        if abs(z_score) > self.entry_threshold:
            if z_score > self.entry_threshold and not pos_a:
                # Spread too high: sell A, buy B
                signals.append(Signal(
                    symbol=self.pair[0],
                    direction="SELL",
                    signal_strength=-z_score/3,
                    metadata={'strategy': 'pairs', 'z_score': z_score}
                ))
                signals.append(Signal(
                    symbol=self.pair[1],
                    direction="BUY",
                    signal_strength=z_score/3,
                    quantity=int(hedge_ratio * 100),  # Example sizing
                    metadata={'strategy': 'pairs', 'z_score': z_score}
                ))
            
            elif z_score < -self.entry_threshold and not pos_a:
                # Spread too low: buy A, sell B
                signals.append(Signal(
                    symbol=self.pair[0],
                    direction="BUY",
                    signal_strength=abs(z_score)/3,
                    metadata={'strategy': 'pairs', 'z_score': z_score}
                ))
                signals.append(Signal(
                    symbol=self.pair[1],
                    direction="SELL",
                    signal_strength=abs(z_score)/3,
                    quantity=int(hedge_ratio * 100),
                    metadata={'strategy': 'pairs', 'z_score': z_score}
                ))
        
        # Exit signals
        elif abs(z_score) < self.exit_threshold and pos_a:
            # Close positions
            signals.append(Signal(
                symbol=self.pair[0],
                direction="SELL" if pos_a.quantity > 0 else "BUY",
                quantity=abs(pos_a.quantity),
                metadata={'strategy': 'pairs', 'action': 'exit'}
            ))
            signals.append(Signal(
                symbol=self.pair[1],
                direction="SELL" if pos_b.quantity > 0 else "BUY",
                quantity=abs(pos_b.quantity),
                metadata={'strategy': 'pairs', 'action': 'exit'}
            ))
        
        return self.validate_signals(signals, portfolio)
```

**Momentum Strategy**
```python
class MomentumStrategy(BaseStrategy):
    """Trend-following momentum strategy"""
    
    def __init__(self,
                 universe: List[str],
                 lookback_period: int = 20,
                 holding_period: int = 5,
                 num_positions: int = 10,
                 **kwargs):
        super().__init__(universe=universe, **kwargs)
        self.lookback_period = lookback_period
        self.holding_period = holding_period
        self.num_positions = num_positions
        self.rebalance_counter = 0
        
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate momentum scores"""
        momentum_scores = {}
        
        for symbol in self.universe:
            if symbol in data:
                prices = data[symbol]['close']
                returns = prices.pct_change(self.lookback_period)
                momentum_scores[symbol] = returns.iloc[-1]
        
        # Rank symbols by momentum
        ranked = pd.Series(momentum_scores).rank(ascending=False)
        
        return {
            'momentum_scores': pd.Series(momentum_scores),
            'momentum_ranks': ranked
        }
    
    async def generate_signals(self,
                              market_data: MarketData,
                              portfolio: Portfolio,
                              timestamp: datetime) -> List[Signal]:
        """Generate momentum signals"""
        signals = []
        
        # Check if it's time to rebalance
        self.rebalance_counter += 1
        if self.rebalance_counter % self.holding_period != 0:
            return signals
        
        # Get top momentum stocks
        ranks = self.indicators['momentum_ranks']
        top_symbols = ranks[ranks <= self.num_positions].index.tolist()
        
        # Exit positions not in top N
        for symbol, position in portfolio.positions.items():
            if symbol not in top_symbols and position.quantity > 0:
                signals.append(Signal(
                    symbol=symbol,
                    direction="SELL",
                    quantity=position.quantity,
                    metadata={'strategy': 'momentum', 'action': 'exit'}
                ))
        
        # Enter new positions
        available_slots = self.num_positions - len([
            s for s in top_symbols 
            if s in portfolio.positions and portfolio.positions[s].quantity > 0
        ])
        
        if available_slots > 0:
            # Equal weight allocation
            position_size = portfolio.cash / available_slots
            
            for symbol in top_symbols:
                if symbol not in portfolio.positions or portfolio.positions[symbol].quantity == 0:
                    price = market_data.get_price(symbol)
                    quantity = int(position_size / price)
                    
                    if quantity > 0:
                        signals.append(Signal(
                            symbol=symbol,
                            direction="BUY",
                            quantity=quantity,
                            signal_strength=self.indicators['momentum_scores'][symbol],
                            metadata={
                                'strategy': 'momentum',
                                'rank': int(ranks[symbol])
                            }
                        ))
        
        return self.validate_signals(signals, portfolio)
```

**Mean Reversion Strategy**
```python
class MeanReversionStrategy(BaseStrategy):
    """Bollinger Bands mean reversion"""
    
    def __init__(self,
                 universe: List[str],
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 rsi_period: int = 14,
                 **kwargs):
        super().__init__(universe=universe, **kwargs)
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate Bollinger Bands and RSI"""
        indicators = {}
        
        for symbol in self.universe:
            if symbol in data:
                df = data[symbol]
                
                # Bollinger Bands
                sma = df['close'].rolling(self.bb_period).mean()
                std = df['close'].rolling(self.bb_period).std()
                upper_band = sma + (self.bb_std * std)
                lower_band = sma - (self.bb_std * std)
                
                # RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(self.rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                indicators[symbol] = pd.DataFrame({
                    'close': df['close'],
                    'sma': sma,
                    'upper_band': upper_band,
                    'lower_band': lower_band,
                    'rsi': rsi
                })
        
        return indicators
```

### 3. Technical Indicators

**Indicator Library**
```python
class TechnicalIndicators:
    """Common technical indicators"""
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(series: pd.Series, 
             fast: int = 12, 
             slow: int = 26, 
             signal: int = 9) -> pd.DataFrame:
        """MACD indicator"""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    @staticmethod
    def bollinger_bands(series: pd.Series, 
                       period: int = 20, 
                       std_dev: float = 2) -> pd.DataFrame:
        """Bollinger Bands"""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        
        return pd.DataFrame({
            'middle': sma,
            'upper': sma + (std_dev * std),
            'lower': sma - (std_dev * std)
        })
```

### 4. Risk Management

**Position Sizing**
```python
class PositionSizer:
    """Calculate optimal position sizes"""
    
    @staticmethod
    def kelly_criterion(win_prob: float, 
                       win_loss_ratio: float, 
                       kelly_fraction: float = 0.25) -> float:
        """Kelly Criterion position sizing"""
        q = 1 - win_prob  # Probability of loss
        f = (win_prob * win_loss_ratio - q) / win_loss_ratio
        return max(0, f * kelly_fraction)  # Use fraction of Kelly
    
    @staticmethod
    def fixed_fractional(portfolio_value: float,
                        risk_percent: float = 0.02,
                        stop_loss_percent: float = 0.05) -> float:
        """Fixed fractional position sizing"""
        risk_amount = portfolio_value * risk_percent
        position_size = risk_amount / stop_loss_percent
        return position_size
    
    @staticmethod
    def volatility_adjusted(portfolio_value: float,
                           target_volatility: float,
                           asset_volatility: float,
                           max_leverage: float = 1.0) -> float:
        """Size positions based on volatility"""
        position_fraction = target_volatility / asset_volatility
        position_fraction = min(position_fraction, max_leverage)
        return portfolio_value * position_fraction
```

**Stop Loss Management**
```python
class StopLossManager:
    """Manage stop losses for positions"""
    
    def __init__(self, default_stop_percent: float = 0.05):
        self.default_stop_percent = default_stop_percent
        self.stop_levels = {}
    
    def calculate_initial_stop(self, 
                             entry_price: float,
                             atr: Optional[float] = None,
                             support_level: Optional[float] = None) -> float:
        """Calculate initial stop loss"""
        if atr:
            # ATR-based stop
            return entry_price - (2 * atr)
        elif support_level:
            # Support-based stop
            return support_level * 0.99
        else:
            # Percentage-based stop
            return entry_price * (1 - self.default_stop_percent)
    
    def update_trailing_stop(self,
                           symbol: str,
                           current_price: float,
                           trail_percent: float = 0.03):
        """Update trailing stop loss"""
        if symbol in self.stop_levels:
            current_stop = self.stop_levels[symbol]
            new_stop = current_price * (1 - trail_percent)
            self.stop_levels[symbol] = max(current_stop, new_stop)
```

### 5. Strategy Optimization

**Parameter Optimization**
```python
class StrategyOptimizer:
    """Optimize strategy parameters"""
    
    async def grid_search(self,
                         strategy_class: Type[BaseStrategy],
                         param_grid: Dict[str, List[Any]],
                         historical_data: pd.DataFrame,
                         metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """Grid search optimization"""
        best_params = None
        best_score = -float('inf')
        
        # Generate all parameter combinations
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())
        
        for param_values in param_combinations:
            params = dict(zip(param_names, param_values))
            
            # Create strategy instance
            strategy = strategy_class(**params)
            
            # Run backtest
            backtest_result = await self.run_backtest(
                strategy, 
                historical_data
            )
            
            # Evaluate metric
            score = getattr(backtest_result, metric)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'metric': metric
        }
    
    async def walk_forward_analysis(self,
                                   strategy: BaseStrategy,
                                   data: pd.DataFrame,
                                   train_periods: int = 252,
                                   test_periods: int = 63,
                                   step_size: int = 21) -> List[BacktestResult]:
        """Walk-forward optimization"""
        results = []
        
        for i in range(0, len(data) - train_periods - test_periods, step_size):
            # Training data
            train_start = i
            train_end = i + train_periods
            train_data = data.iloc[train_start:train_end]
            
            # Optimize on training data
            optimized_params = await self.optimize_on_period(
                strategy, 
                train_data
            )
            
            # Test data
            test_start = train_end
            test_end = test_start + test_periods
            test_data = data.iloc[test_start:test_end]
            
            # Run on test data
            strategy.params = optimized_params
            result = await self.run_backtest(strategy, test_data)
            results.append(result)
        
        return results
```

## Strategy Combinations

### Multi-Strategy Portfolio
```python
class MultiStrategyPortfolio:
    """Combine multiple strategies"""
    
    def __init__(self, strategies: List[Tuple[BaseStrategy, float]]):
        """
        Args:
            strategies: List of (strategy, weight) tuples
        """
        self.strategies = strategies
        self.normalize_weights()
    
    def normalize_weights(self):
        """Ensure weights sum to 1"""
        total_weight = sum(weight for _, weight in self.strategies)
        self.strategies = [
            (strategy, weight / total_weight) 
            for strategy, weight in self.strategies
        ]
    
    async def generate_signals(self,
                              market_data: MarketData,
                              portfolio: Portfolio,
                              timestamp: datetime) -> List[Signal]:
        """Combine signals from all strategies"""
        combined_signals = {}
        
        for strategy, weight in self.strategies:
            signals = await strategy.generate_signals(
                market_data, 
                portfolio, 
                timestamp
            )
            
            # Aggregate signals by symbol
            for signal in signals:
                if signal.symbol not in combined_signals:
                    combined_signals[signal.symbol] = {
                        'buy_strength': 0,
                        'sell_strength': 0
                    }
                
                if signal.direction == "BUY":
                    combined_signals[signal.symbol]['buy_strength'] += (
                        signal.signal_strength * weight
                    )
                elif signal.direction == "SELL":
                    combined_signals[signal.symbol]['sell_strength'] += (
                        abs(signal.signal_strength) * weight
                    )
        
        # Generate final signals
        final_signals = []
        for symbol, strengths in combined_signals.items():
            net_strength = strengths['buy_strength'] - strengths['sell_strength']
            
            if abs(net_strength) > 0.1:  # Threshold
                final_signals.append(Signal(
                    symbol=symbol,
                    direction="BUY" if net_strength > 0 else "SELL",
                    signal_strength=net_strength
                ))
        
        return final_signals
```

## Performance Analysis

### Backtest Metrics
```python
@dataclass
class BacktestResult:
    """Complete backtest results"""
    # Returns
    total_return: float
    annual_return: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Other metrics
    exposure_time: float
    turnover: float
    
    # Time series
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
```

## Best Practices

### 1. Strategy Development
- Start simple, add complexity gradually
- Use out-of-sample testing
- Account for transaction costs
- Consider market impact
- Validate on multiple time periods

### 2. Risk Management
- Never risk more than 2% per trade
- Use correlation-based position limits
- Implement portfolio-level stops
- Monitor concentration risk
- Track regime changes

### 3. Implementation
- Use limit orders when possible
- Implement smart order routing
- Monitor slippage carefully
- Have failsafe mechanisms
- Log all trading decisions

### 4. Monitoring
- Track strategy performance daily
- Compare to expectations
- Monitor for style drift
- Check for data issues
- Review risk metrics

## Future Enhancements

1. **Machine Learning Integration**
   - Feature engineering pipeline
   - Model training framework
   - Online learning capabilities
   - Ensemble methods

2. **Advanced Order Types**
   - Iceberg orders
   - TWAP/VWAP execution
   - Conditional orders
   - Basket orders

3. **Multi-Asset Support**
   - Currency hedging
   - Cross-asset strategies
   - Options strategies
   - Futures spreads

4. **Real-time Execution**
   - FIX protocol support
   - Co-location setup
   - Latency optimization
   - Hardware acceleration