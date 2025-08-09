# Trading Strategies Architecture

## Core Design Principles

### 1. Derived Metrics Are Calculated On-Demand

**Key Principle**: All technical indicators and derived metrics (SMA, EMA, RSI, MACD, etc.) are computed on-demand from raw data, never pre-calculated or stored.

**Rationale**:
- **Flexibility**: Strategies can use different parameters (e.g., SMA 20 vs SMA 50) without needing pre-computation
- **Adaptability**: Works with any data source - historical, simulated, or synthetic
- **Accuracy**: Always uses the latest calculation methods and avoids stale data
- **Storage Efficiency**: No need to store thousands of indicator combinations

**Implementation**:
```python
# WRONG - Don't expect pre-calculated indicators
signals = market_data['sma_20']  # ❌ No pre-calculated indicators exist

# CORRECT - Calculate indicators from raw OHLCV data
def calculate_indicators(self, historical_data: pd.DataFrame) -> Dict[str, pd.Series]:
    indicators = {}
    indicators['sma_20'] = historical_data['close'].rolling(20).mean()
    indicators['sma_50'] = historical_data['close'].rolling(50).mean()
    indicators['rsi'] = self.calculate_rsi(historical_data['close'], 14)
    return indicators
```

### 2. Strategy Interface

Strategies receive:
1. **Historical Data**: Raw OHLCV data for indicator calculation
2. **Current Market Data**: Today's prices, volume, etc.
3. **Portfolio State**: Current positions, cash, P&L
4. **Trading Day**: Current simulation date

```python
def generate_signals(
    self,
    market_data: Dict[str, Dict],      # Current day's data for each symbol
    portfolio: Portfolio,              # Current portfolio state
    trading_day: date,                # Current simulation date
    historical_data: Dict[str, pd.DataFrame]  # Historical OHLCV for indicators
) -> List[Signal]:
    """Generate trading signals based on current state and calculated indicators."""
```

### 3. Data Flow

```
Raw Data (OHLCV) → Strategy → Calculate Indicators → Generate Signals → Execute Trades
                      ↑                                        ↓
                      └──────── Portfolio State ←──────────────┘
```

## Strategy Types

### 1. Signal-Based Strategies
- Use technical indicators calculated from price/volume data
- Examples: Moving Average Crossover, RSI Oversold, Bollinger Bands

### 2. Event-Based Strategies
- React to market events (earnings, economic releases, news)
- Examples: Earnings momentum, FOMC announcement trades

### 3. Time-Based Strategies
- Trade based on time patterns
- Examples: Opening range breakout, end-of-month rebalancing

### 4. ML-Based Strategies
- Use machine learning models trained on historical patterns
- Models predict on-demand using current market features

## Indicator Calculation Guidelines

### 1. Always Request Sufficient History
```python
# For 200-day SMA, request at least 200+ days of history
lookback_days = max(200, self.max_indicator_period) + buffer_days
historical_data = await market_replay.get_price_series(
    symbol,
    trading_day - timedelta(days=lookback_days),
    trading_day
)
```

### 2. Handle Missing Data Gracefully
```python
def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    if len(prices) < period:
        return pd.Series(index=prices.index, dtype=float)  # Return NaN series
    return prices.rolling(window=period, min_periods=period).mean()
```

### 3. Common Indicators Library
Create reusable indicator calculations:
```python
class TechnicalIndicators:
    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        
    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        
    @staticmethod
    def macd(prices: pd.Series, fast=12, slow=26, signal=9) -> Dict[str, pd.Series]:
        """MACD with signal line and histogram"""
```

## Example Strategy Implementation

```python
class GoldenCrossStrategy(SignalBasedStrategy):
    def __init__(self):
        self.sma_short = 50
        self.sma_long = 200
        
    async def generate_signals(self, market_data, portfolio, trading_day, historical_data):
        signals = []
        
        for symbol in self.universe:
            # Get historical prices
            hist = historical_data.get(symbol)
            if hist is None or len(hist) < self.sma_long:
                continue
                
            # Calculate indicators on-demand
            sma_50 = hist['close'].rolling(self.sma_short).mean()
            sma_200 = hist['close'].rolling(self.sma_long).mean()
            
            # Current values
            current_sma_50 = sma_50.iloc[-1]
            current_sma_200 = sma_200.iloc[-1]
            prev_sma_50 = sma_50.iloc[-2]
            prev_sma_200 = sma_200.iloc[-2]
            
            # Check for golden cross
            if prev_sma_50 <= prev_sma_200 and current_sma_50 > current_sma_200:
                signals.append({
                    'symbol': symbol,
                    'direction': 'BUY',
                    'quantity': self.calculate_position_size(symbol, market_data[symbol]['close']),
                    'reason': 'Golden Cross'
                })
                
            # Check for death cross
            elif prev_sma_50 >= prev_sma_200 and current_sma_50 < current_sma_200:
                if symbol in portfolio.positions:
                    signals.append({
                        'symbol': symbol,
                        'direction': 'SELL',
                        'quantity': portfolio.positions[symbol].quantity,
                        'reason': 'Death Cross'
                    })
                    
        return signals
```

## Benefits of On-Demand Calculation

1. **Strategy Flexibility**: Each strategy can use custom indicator parameters
2. **Backtesting Accuracy**: No look-ahead bias from pre-calculated values
3. **Memory Efficiency**: Only calculate what's needed, when needed
4. **Algorithm Updates**: Easy to improve indicator calculations without data migration
5. **Multi-Timeframe**: Same strategy can work on different timeframes
6. **Alternative Data**: Can incorporate non-price data (sentiment, fundamentals) dynamically

## Performance Optimization

While indicators are calculated on-demand, we can optimize:

1. **Caching Within Simulation**: Cache calculations for the current trading day
2. **Vectorized Operations**: Use NumPy/Pandas vectorized operations
3. **Incremental Updates**: For streaming data, update indicators incrementally
4. **Parallel Computation**: Calculate indicators for multiple symbols in parallel

```python
# Example: Cached indicator calculation
class CachedIndicatorStrategy:
    def __init__(self):
        self._indicator_cache = {}
        
    def get_indicators(self, symbol: str, data: pd.DataFrame, trading_day: date) -> Dict:
        cache_key = (symbol, trading_day)
        if cache_key not in self._indicator_cache:
            self._indicator_cache[cache_key] = self.calculate_indicators(data)
        return self._indicator_cache[cache_key]
```

## Summary

The on-demand calculation approach ensures our trading system remains flexible, accurate, and maintainable. Strategies are responsible for calculating their own indicators from raw data, making them portable across different data sources and simulation environments.