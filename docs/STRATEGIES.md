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

## Performance Analysis: Computational Complexity

### Executive Summary
**Slowest Strategy: Single ML Prediction Strategy** - Takes 10-20x longer than simple strategies due to feature engineering and model inference.
<!-- Note: Ensemble ML Strategy excluded temporarily - would be even slower with 10-50x overhead -->

### Performance Ranking (Slowest to Fastest)

<!--
#### 1. 🐌 **Ensemble ML Strategy** (SLOWEST)
**Time Complexity: O(n × m × f × p)**
- n = number of data points
- m = number of models in ensemble (typically 5-10)
- f = number of features (100-200)
- p = prediction complexity per model

**Why it's slowest:**
- Feature Engineering: 200+ features including technical indicators, statistical features, microstructure metrics
- Multiple Model Inference: XGBoost, Random Forest, Neural Network, LSTM, Statistical models
- Ensemble Voting: Weighted averaging, confidence calculation, consensus building
- Post-processing: Calibration, risk overlay, position sizing

**Estimated Time per Decision:**
- Feature calculation: 50-100ms
- Model inference: 20-40ms × 5 models = 100-200ms
- Ensemble voting: 10-20ms
- **Total: 160-320ms per symbol**
- For 500 symbols: **80-160 seconds per timestamp**
-->

#### 1. 🐢 **Single ML Prediction Strategy** (Currently SLOWEST)
**Time Complexity: O(n × f × p)**

**Operations:**
- Feature Engineering: 100+ features
- Single Model Inference: One model prediction with confidence scoring
- Signal Generation: Threshold application and position sizing

**Estimated Time: 55-95ms per symbol** (27-47 seconds for 500 symbols)

#### 2. 🦥 **Composite Strategy**
**Time Complexity: O(n × s × c)**
- s = number of sub-strategies (typically 3-7)

**Operations:**
- Run multiple sub-strategies in parallel
- Signal aggregation and consensus calculation
- Portfolio optimization across strategies

**Estimated Time: 70ms per symbol** (35 seconds for 500 symbols)

#### 3. 🐕 **Event-Based Strategy**
**Time Complexity: O(n × e × w)**
- e = number of events, w = event window size

**Operations:**
- Event detection and proximity checking
- Pre/post event pattern matching
- Event-specific position management

**Estimated Time: 20-30ms per symbol** (10-15 seconds for 500 symbols)

#### 4. 🐎 **Complex Signal-Based Strategy** (Multi-indicator)
**Time Complexity: O(n × i)**
- i = number of indicators (20-50)

**Operations:**
- Calculate multiple indicators (MA, RSI, MACD, Bollinger, etc.)
- Evaluate complex entry/exit rules
- Cross-indicator confirmation

**Estimated Time: 20-35ms per symbol** (10-17 seconds for 500 symbols)

#### 5. 🐆 **Time-Based Strategy**
**Time Complexity: O(n × t)**

**Operations:**
- Time window checks (session, trading hours)
- Simple signal logic (gap detection, session momentum)

**Estimated Time: 7-13ms per symbol** (3.5-6.5 seconds for 500 symbols)

#### 6. 🚀 **Simple Signal-Based Strategy** (FASTEST)
**Time Complexity: O(n)**

**Examples:** Golden Cross, RSI Mean Reversion, Single MA Crossover

**Operations:**
- Calculate 1-3 indicators
- Simple rule evaluation

**Estimated Time: 3-6ms per symbol** (1.5-3 seconds for 500 symbols)

### Performance Bottlenecks

#### ML Strategies:
1. **Feature Engineering**: Rolling windows, correlations, pattern recognition
2. **Model Inference**: Tree traversal, matrix multiplication, sequence processing
3. **Ensemble Overhead**: Multiple model coordination and voting

#### Signal-Based Strategies:
1. **Indicator Calculation**: Complex indicators with long lookbacks
2. **Rule Evaluation**: Multiple condition checking

#### All Strategies:
1. **I/O Operations**: Database queries, data fetching
2. **Memory Access**: Cache misses with large lookbacks

### Optimization Strategies

#### For ML Strategies:
```python
# Pre-compute features in Metrics Pipeline
# Use ONNX for faster inference
# Quantize models (INT8)
# GPU acceleration for neural networks
# Parallel model execution in ensemble
```

#### For Signal-Based Strategies:
```python
# Pre-calculate indicators in Metrics Pipeline
# Use vectorized operations (NumPy)
# Implement incremental updates
# Cache calculations within trading day
```

### Real-Time vs Batch Performance

#### Real-Time Trading:
- **HFT (< 1ms)**: Only simple signal strategies
- **Day Trading (< 100ms)**: Simple to medium complexity signals
- **Swing Trading (< 1s)**: Most strategies except ensemble ML
- **Position Trading (< 10s)**: All strategies viable

#### Batch Backtesting Throughput:
<!-- - **Ensemble ML**: 100-500 symbols/minute -->
- **Single ML**: 500-1000 symbols/minute
- **Composite**: 1000-2000 symbols/minute
- **Signal-Based**: 5000-10000 symbols/minute
- **Simple Signal**: 10000-20000 symbols/minute

### Benchmarking Results

**Test Setup:** 500 symbols, 252 trading days, 1-minute bars, 8-core CPU

| Strategy Type | Per Decision | Full Backtest (252 days) |
|--------------|--------------|-------------------------|
| <!-- Ensemble ML | 160-320ms | 92-184 days | -->
| Single ML | 55-95ms | 31-54 days |
| Composite | 70ms | 40 days |
| Event-Based | 20-30ms | 11-17 days |
| Complex Signal | 20-35ms | 11-20 days |
| Time-Based | 7-13ms | 4-7.5 days |
| Simple Signal | 3-6ms | 1.7-3.3 days |

### Recommendations

1. **For Speed-Critical Applications:**
   - Use tiered approach: Fast filter → Medium analysis → Slow confirmation
   - Pre-compute everything possible in Metrics Pipeline
   - Parallelize across symbols, not strategies

2. **For Accuracy-Critical Applications:**
   - Use ensemble ML despite speed penalty
   - Run on reduced universe (top 100 vs 500 stocks)
   - Use longer time frames (daily vs minute bars)

3. **For Production Systems:**
   - ML strategies for daily/weekly positions
   - Simple strategies for intraday/HFT
   - Implement async processing with queues for ML strategies



   # Event-Based Strategy Requirements

## Current Status ❌
Event-based strategies are NOT currently functional because:
- No event data in the metrics pipeline
- No dividend/earnings dates in test scenarios
- Event strategies return 0% (no trades executed)

## Required Components to Enable Event Strategies

### 1. Event Data Pipeline
**Priority: HIGH**

#### Data Sources Needed:
- **Dividend Data**
  - Ex-dividend dates
  - Dividend amounts
  - Payment dates
  - Historical dividend consistency
  
- **Earnings Data**
  - Announcement dates & times (BMO/AMC)
  - Analyst estimates
  - Historical surprises
  - Guidance updates

- **Economic Events**
  - FOMC meetings
  - CPI/Jobs reports
  - GDP releases

#### Implementation Steps:A
```python
# Add to Data Pipeline (Stage 1)
class EventDataIngestion:
    - fetch_dividend_calendar()
    - fetch_earnings_calendar()
    - fetch_economic_calendar()
    - store_events_to_db()
```

### 2. Metrics Pipeline Enhancement
**Priority: HIGH**

Add event-aware metrics calculation:
```python
# Add to Metrics Pipeline (Stage 3)
- days_to_next_dividend
- days_to_next_earnings
- dividend_yield
- earnings_surprise_history
- implied_volatility (if options data available)
```

### 3. Assessment Pipeline Updates
**Priority: MEDIUM**

Modify backtesting engine to:
- Load event data alongside price data
- Pass events to strategy.generate_signals()
- Track event-based performance metrics

### 4. Database Schema

```sql
-- Add events schema
CREATE TABLE events.dividend_events (
    symbol VARCHAR(10),
    ex_date DATE,
    payment_date DATE,
    amount DECIMAL(10,4),
    yield_pct DECIMAL(5,2)
);

CREATE TABLE events.earnings_events (
    symbol VARCHAR(10),
    announcement_date TIMESTAMP,
    fiscal_quarter VARCHAR(10),
    eps_estimate DECIMAL(10,4),
    eps_actual DECIMAL(10,4),
    revenue_estimate BIGINT,
    revenue_actual BIGINT
);
```

### 5. Strategy Enhancements

Update event strategies to use real data:
```python
# Current (not working)
EventTrigger(
    event_type=EventType.DIVIDEND,
    pre_event_days=5,
    entry_conditions={'dividend_yield': 0.03}
)

# Future (with data)
EventTrigger(
    event_type=EventType.DIVIDEND,
    pre_event_days=5,
    entry_conditions={
        'dividend_yield': lambda x: x > 0.03,
        'days_to_ex_date': lambda x: x <= 5,
        'historical_capture_success': lambda x: x > 0.7
    }
)
```

## Symbols That Will Benefit

### Dividend Capture Candidates
| Symbol | Yield | Frequency | Best Strategy |
|--------|-------|-----------|---------------|
| JNJ | 2.7% | Quarterly | Dividend Capture |
| KO | 3.1% | Quarterly | Dividend Capture |
| O | 5.5% | Monthly | Monthly Dividend |
| XOM | 3.5% | Quarterly | Energy Dividend |

### Earnings Volatility Plays
| Symbol | Avg Move | IV Rank | Best Strategy |
|--------|----------|---------|---------------|
| NVDA | ±8% | High | Earnings Straddle |
| TSLA | ±7% | High | Pre-Earnings Momentum |
| NFLX | ±10% | Very High | Earnings Breakout |
| META | ±6% | Medium | Earnings Surprise |

## Implementation Timeline

### Phase 1: Data Collection (Week 1-2)
- [ ] Set up dividend data feed
- [ ] Set up earnings data feed
- [ ] Create event storage tables
- [ ] Historical data backfill

### Phase 2: Pipeline Integration (Week 3-4)
- [ ] Add event ingestion to Data Pipeline
- [ ] Calculate event metrics in Metrics Pipeline
- [ ] Update test data generator
- [ ] Verify data quality

### Phase 3: Strategy Testing (Week 5-6)
- [ ] Update EventBasedStrategy class
- [ ] Add event data to backtesting
- [ ] Test dividend capture on JNJ
- [ ] Test earnings play on NVDA

### Phase 4: Production (Week 7-8)
- [ ] Full assessment with events
- [ ] Performance validation
- [ ] Documentation
- [ ] Launch event strategies

## Success Metrics

When complete, we should see:
- Dividend strategy: 3-5% annual return from captures
- Earnings strategy: 15-20% return from volatility
- Clear recommendations for event-driven symbols
- Proper risk management around events

## API/Data Sources to Consider

1. **Yahoo Finance** - Free dividend calendar
2. **Alpha Vantage** - Free tier includes events
3. **Polygon.io** - Comprehensive event data (paid)
4. **IEX Cloud** - Good earnings calendar
5. **FRED** - Economic events

## Testing Approach

```python
# Generate synthetic event data for testing
def generate_test_events():
    return {
        'AAPL': {
            'dividends': quarterly_dates(0.24),
            'earnings': quarterly_dates()
        },
        'JNJ': {
            'dividends': quarterly_dates(1.19),
            'earnings': quarterly_dates()
        }
    }
```

## Notes

- Event strategies are HIGH VALUE - they provide uncorrelated returns
- Critical for symbols like JNJ, XOM (dividends) and NVDA, TSLA (earnings)
- Can significantly improve Sharpe ratio when combined with trend strategies
- Must have accurate event timing - even 1 day off ruins the strategy

---
*Created: 2025-08-13*
*Status: PLANNED - Not yet implemented*
*Priority: HIGH - Enables new strategy class*