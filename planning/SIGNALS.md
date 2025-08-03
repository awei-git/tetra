# Signals Module Documentation

## Overview
The Signals module is responsible for computing a comprehensive set of technical, statistical, and machine learning signals from time series data. It serves as the foundation for all trading strategies by providing standardized, reusable signal calculations.

## Architecture

### Module Structure
```
src/signals/
├── __init__.py
├── base.py                      # Base classes and interfaces
│   ├── SignalComputer          # Main signal computation engine
│   ├── Signal                  # Signal data structure
│   └── SignalConfig            # Configuration for signals
│
├── technical/
│   ├── __init__.py
│   ├── trend.py                # Trend indicators (SMA, EMA, etc.)
│   ├── momentum.py             # Momentum indicators (RSI, MACD, etc.)
│   ├── volatility.py           # Volatility indicators (BB, ATR, etc.)
│   ├── volume.py               # Volume indicators (OBV, VWAP, etc.)
│   └── patterns.py             # Pattern recognition
│
├── statistical/
│   ├── __init__.py
│   ├── returns.py              # Return calculations
│   ├── correlation.py          # Correlation analysis
│   ├── zscore.py               # Statistical normalization
│   └── regime.py               # Market regime detection
│
├── ml/
│   ├── __init__.py
│   ├── wrapper.py              # ML model wrapper interface
│   ├── features.py             # Feature engineering
│   └── predictions.py          # Model predictions
│
└── utils/
    ├── __init__.py
    ├── validation.py           # Data validation
    └── performance.py          # Signal computation optimization
```

## Core Components

### 1. Base Signal Computer

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Any

class SignalComputer:
    """Main engine for computing all signals"""
    
    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self.technical = TechnicalSignals(config)
        self.statistical = StatisticalSignals(config)
        self.ml = MLSignals(config) if config.enable_ml else None
        
    def compute_all(self, 
                   data: pd.DataFrame,
                   symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute all configured signals for given data.
        
        Args:
            data: OHLCV data with MultiIndex (symbol, date) or single symbol
            symbols: Specific symbols to compute (None = all)
            
        Returns:
            DataFrame with all computed signals
        """
        # Validate input data
        self._validate_data(data)
        
        # Initialize result DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # Compute technical signals
        tech_signals = self.technical.compute(data)
        signals = pd.concat([signals, tech_signals], axis=1)
        
        # Compute statistical signals
        stat_signals = self.statistical.compute(data)
        signals = pd.concat([signals, stat_signals], axis=1)
        
        # Compute ML signals if enabled
        if self.ml and self.config.enable_ml:
            ml_signals = self.ml.compute(data, signals)
            signals = pd.concat([signals, ml_signals], axis=1)
        
        # Add metadata
        signals.attrs['computed_at'] = pd.Timestamp.now()
        signals.attrs['config'] = self.config.to_dict()
        
        return signals
    
    def compute_incremental(self, 
                           data: pd.DataFrame,
                           previous_signals: pd.DataFrame) -> pd.DataFrame:
        """Efficiently compute signals for new data"""
        # Only compute for new rows
        new_data = data[~data.index.isin(previous_signals.index)]
        
        if new_data.empty:
            return previous_signals
            
        # Compute signals for new data with lookback
        lookback_data = data.iloc[-self.config.max_lookback:]
        new_signals = self.compute_all(lookback_data)
        
        # Merge with previous signals
        return pd.concat([previous_signals, new_signals.loc[new_data.index]])
```

### 2. Technical Indicators

```python
class TechnicalSignals:
    """Compute technical indicators"""
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical signals"""
        signals = pd.DataFrame(index=data.index)
        
        # Price-based signals
        signals['sma_20'] = self.sma(data['close'], 20)
        signals['sma_50'] = self.sma(data['close'], 50)
        signals['sma_200'] = self.sma(data['close'], 200)
        signals['ema_12'] = self.ema(data['close'], 12)
        signals['ema_26'] = self.ema(data['close'], 26)
        
        # Momentum indicators
        signals['rsi_14'] = self.rsi(data['close'], 14)
        signals['macd'], signals['macd_signal'], signals['macd_hist'] = self.macd(data['close'])
        signals['stoch_k'], signals['stoch_d'] = self.stochastic(data)
        signals['williams_r'] = self.williams_r(data)
        signals['roc_10'] = self.rate_of_change(data['close'], 10)
        
        # Volatility indicators
        signals['bb_upper'], signals['bb_middle'], signals['bb_lower'] = self.bollinger_bands(data['close'])
        signals['bb_position'] = (data['close'] - signals['bb_lower']) / (signals['bb_upper'] - signals['bb_lower'])
        signals['atr_14'] = self.atr(data, 14)
        signals['volatility_20'] = data['close'].pct_change().rolling(20).std()
        
        # Volume indicators
        signals['volume_sma_20'] = self.sma(data['volume'], 20)
        signals['volume_ratio'] = data['volume'] / signals['volume_sma_20']
        signals['obv'] = self.on_balance_volume(data)
        signals['vwap'] = self.vwap(data)
        signals['mfi_14'] = self.money_flow_index(data, 14)
        
        # Support/Resistance
        signals['resistance_1'] = data['high'].rolling(20).max()
        signals['support_1'] = data['low'].rolling(20).min()
        signals['pivot'] = (data['high'] + data['low'] + data['close']) / 3
        
        return signals
```

### 3. Statistical Signals

```python
class StatisticalSignals:
    """Compute statistical signals"""
    
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute all statistical signals"""
        signals = pd.DataFrame(index=data.index)
        
        # Return calculations
        signals['returns_1d'] = data['close'].pct_change()
        signals['returns_5d'] = data['close'].pct_change(5)
        signals['returns_20d'] = data['close'].pct_change(20)
        signals['log_returns'] = np.log(data['close']).diff()
        
        # Statistical measures
        signals['zscore_20'] = self.zscore(data['close'], 20)
        signals['zscore_50'] = self.zscore(data['close'], 50)
        signals['skew_20'] = signals['returns_1d'].rolling(20).skew()
        signals['kurtosis_20'] = signals['returns_1d'].rolling(20).kurt()
        
        # Correlation and beta
        if 'SPY' in data.columns:  # If market data available
            signals['correlation_spy_60'] = self.rolling_correlation(
                data['close'], data['SPY']['close'], 60
            )
            signals['beta_spy_60'] = self.rolling_beta(
                data['close'], data['SPY']['close'], 60
            )
        
        # Volatility measures
        signals['realized_vol_20'] = self.realized_volatility(signals['returns_1d'], 20)
        signals['vol_of_vol'] = signals['realized_vol_20'].rolling(20).std()
        
        # Microstructure
        signals['high_low_ratio'] = data['high'] / data['low']
        signals['close_to_high'] = (data['high'] - data['close']) / (data['high'] - data['low'])
        signals['intraday_momentum'] = (data['close'] - data['open']) / data['open']
        
        # Regime detection
        signals['trend_strength'] = self.trend_strength(data['close'], 50)
        signals['market_regime'] = self.detect_regime(signals)
        
        return signals
```

### 4. ML Signal Integration

```python
class MLSignals:
    """Integrate ML model predictions as signals"""
    
    def __init__(self, config: SignalConfig):
        self.config = config
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        
    def load_models(self, model_registry: Dict[str, Any]):
        """Load pre-trained models"""
        for name, model_path in model_registry.items():
            self.models[name] = joblib.load(model_path)
    
    def compute(self, 
               data: pd.DataFrame, 
               technical_signals: pd.DataFrame) -> pd.DataFrame:
        """Generate ML predictions as signals"""
        signals = pd.DataFrame(index=data.index)
        
        # Engineer features from raw data and technical signals
        features = self.feature_engineer.create_features(data, technical_signals)
        
        # Generate predictions for each model
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                # Classification model
                predictions = model.predict_proba(features)
                signals[f'ml_{model_name}_buy_prob'] = predictions[:, 1]
                signals[f'ml_{model_name}_sell_prob'] = predictions[:, 0]
            else:
                # Regression model
                signals[f'ml_{model_name}_prediction'] = model.predict(features)
        
        # Ensemble predictions
        if len(self.models) > 1:
            signals['ml_ensemble_score'] = self._ensemble_predictions(signals)
        
        return signals
```

## Signal Catalog

### Technical Indicators (25+)

| Category | Signal | Description | Parameters |
|----------|--------|-------------|------------|
| **Trend** | SMA | Simple Moving Average | periods: 20, 50, 200 |
| | EMA | Exponential Moving Average | periods: 12, 26 |
| | WMA | Weighted Moving Average | period: 20 |
| | KAMA | Kaufman Adaptive MA | period: 30 |
| **Momentum** | RSI | Relative Strength Index | period: 14 |
| | MACD | Moving Average Convergence Divergence | fast: 12, slow: 26, signal: 9 |
| | Stochastic | Stochastic Oscillator | k: 14, d: 3 |
| | Williams %R | Williams Percent Range | period: 14 |
| | ROC | Rate of Change | period: 10 |
| | CCI | Commodity Channel Index | period: 20 |
| **Volatility** | Bollinger Bands | Price bands based on std dev | period: 20, std: 2 |
| | ATR | Average True Range | period: 14 |
| | Keltner Channels | ATR-based channels | period: 20, atr_mult: 2 |
| | Donchian Channels | High/Low channels | period: 20 |
| **Volume** | OBV | On Balance Volume | - |
| | VWAP | Volume Weighted Average Price | - |
| | MFI | Money Flow Index | period: 14 |
| | Accumulation/Distribution | A/D Line | - |
| | CMF | Chaikin Money Flow | period: 20 |

### Statistical Signals (15+)

| Signal | Description | Use Case |
|--------|-------------|----------|
| Returns | 1d, 5d, 20d, 60d returns | Momentum strategies |
| Z-Score | Standardized price deviation | Mean reversion |
| Correlation | Rolling correlation with market | Risk management |
| Beta | Market sensitivity | Portfolio construction |
| Skewness | Return distribution asymmetry | Risk assessment |
| Kurtosis | Tail risk measure | Black swan detection |
| Volatility | Realized, GARCH, Vol-of-vol | Option strategies |
| Microstructure | High-low ratio, close position | Intraday patterns |
| Regime | Trend, sideways, volatile | Strategy selection |

### ML Signals (Configurable)

| Model Type | Output | Example Use |
|------------|--------|-------------|
| Direction Classifier | Buy/Sell/Hold probability | Entry signals |
| Return Predictor | Next period return forecast | Position sizing |
| Volatility Forecast | Future volatility estimate | Risk management |
| Regime Classifier | Market state prediction | Strategy switching |
| Anomaly Detector | Unusual pattern score | Risk alerts |

## Configuration

```python
@dataclass
class SignalConfig:
    """Configuration for signal computation"""
    
    # Technical indicators
    enable_technical: bool = True
    technical_indicators: List[str] = field(default_factory=lambda: ['all'])
    
    # Statistical signals  
    enable_statistical: bool = True
    statistical_indicators: List[str] = field(default_factory=lambda: ['all'])
    
    # ML signals
    enable_ml: bool = False
    ml_models: Dict[str, str] = field(default_factory=dict)
    
    # Computation settings
    max_lookback: int = 252  # Maximum lookback period
    min_data_points: int = 50  # Minimum data for computation
    parallel_compute: bool = True
    chunk_size: int = 10000  # For batch processing
    
    # Performance
    cache_signals: bool = True
    cache_ttl: int = 3600  # Cache time-to-live in seconds
```

## Usage Examples

### Basic Usage
```python
# Initialize signal computer
signal_computer = SignalComputer()

# Load market data
data = pd.read_sql("SELECT * FROM market_data.ohlcv WHERE symbol = 'AAPL'", engine)

# Compute all signals
signals = signal_computer.compute_all(data)

# Access specific signals
print(signals['rsi_14'].tail())
print(signals['macd_hist'].tail())
```

### Multi-Symbol Computation
```python
# Load data for multiple symbols
symbols = ['AAPL', 'MSFT', 'GOOGL']
data = load_multi_symbol_data(symbols)

# Compute signals for all symbols
signals = signal_computer.compute_all(data, symbols=symbols)

# Signals will have MultiIndex (symbol, date)
apple_rsi = signals.loc['AAPL', 'rsi_14']
```

### Incremental Updates
```python
# Initial computation
historical_signals = signal_computer.compute_all(historical_data)

# Update with new data
new_signals = signal_computer.compute_incremental(
    new_data, 
    historical_signals
)
```

### Custom Configuration
```python
# Configure specific signals
config = SignalConfig(
    technical_indicators=['sma_20', 'rsi_14', 'macd'],
    statistical_indicators=['returns_1d', 'zscore_20'],
    enable_ml=True,
    ml_models={'direction': 'models/xgboost_direction.pkl'}
)

signal_computer = SignalComputer(config)
```

## Performance Optimization

### 1. Vectorized Computation
- Use NumPy/Pandas vectorized operations
- Avoid loops where possible
- Leverage pandas rolling window functions

### 2. Parallel Processing
```python
def compute_parallel(self, data: pd.DataFrame, n_jobs: int = -1):
    """Compute signals in parallel for multiple symbols"""
    from joblib import Parallel, delayed
    
    # Split by symbol
    symbol_groups = data.groupby(level=0)
    
    # Compute in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(self.compute_single_symbol)(group) 
        for name, group in symbol_groups
    )
    
    return pd.concat(results)
```

### 3. Caching Strategy
```python
class SignalCache:
    """Cache computed signals"""
    
    def __init__(self, ttl: int = 3600):
        self.cache = {}
        self.ttl = ttl
        
    def get(self, key: str) -> Optional[pd.DataFrame]:
        if key in self.cache:
            signals, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return signals
        return None
        
    def set(self, key: str, signals: pd.DataFrame):
        self.cache[key] = (signals, time.time())
```

### 4. Incremental Computation
- Only compute new data points
- Reuse previous calculations where possible
- Update rolling statistics efficiently

## Integration Points

### 1. With Simulator
```python
# Simulator provides data
market_data = simulator.get_historical_data(symbols, date_range)

# Compute signals
signals = signal_computer.compute_all(market_data)
```

### 2. With Strategies
```python
# Strategy receives computed signals
class MyStrategy:
    def generate_orders(self, signals: pd.DataFrame):
        if signals['rsi_14'].iloc[-1] < 30:
            return Order('BUY', ...)
```

### 3. With Performance Module
```python
# Backtester uses signals
backtester.run(
    strategy=strategy,
    market_data=data,
    signals=signals  # Pre-computed signals
)
```

## Testing Strategy

### 1. Unit Tests
- Test each indicator calculation
- Verify edge cases (insufficient data, NaN handling)
- Compare with reference implementations

### 2. Integration Tests
- Test signal computation pipeline
- Verify multi-symbol handling
- Test incremental updates

### 3. Performance Tests
- Benchmark computation speed
- Test memory usage with large datasets
- Verify caching effectiveness

## Future Enhancements

1. **Real-time Signals**
   - WebSocket integration for live data
   - Streaming signal computation
   - Low-latency optimizations

2. **Advanced Indicators**
   - Market microstructure signals
   - Order flow indicators
   - Social sentiment integration

3. **GPU Acceleration**
   - CUDA kernels for parallel computation
   - GPU-accelerated ML inference
   - Batch processing optimization

4. **Signal Quality Metrics**
   - Signal stability analysis
   - Predictive power assessment
   - Feature importance ranking

5. **Auto-ML Integration**
   - Automated feature selection
   - Online model updates
   - Ensemble optimization