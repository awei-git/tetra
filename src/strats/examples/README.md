# Strategy Configuration Examples

This directory contains YAML configuration files for defining trading strategies without writing code.

## Available Strategy Types

### 1. Signal-Based Strategies (`signal_based`)
Strategies that trade based on technical indicators and signals.

**Examples:**
- `signal_based.yaml` - RSI mean reversion strategy
- `golden_cross.yaml` - Classic 50/200 SMA crossover
- `bollinger_bands.yaml` - Mean reversion using Bollinger Bands
- `turtle_trading.yaml` - Trend-following breakout system

**Key Components:**
- Signal rules with entry/exit conditions
- Operators: `GREATER_THAN`, `LESS_THAN`, `CROSSES_ABOVE`, `CROSSES_BELOW`, `BETWEEN`
- Position sizing and risk management
- Multiple rules can be combined

### 2. Event-Based Strategies (`event_based`)
Strategies that trade around market events.

**Examples:**
- `event_based.yaml` - Trading earnings, dividends, and FOMC events

**Key Components:**
- Event triggers (EARNINGS, DIVIDEND, FOMC, etc.)
- Pre/post event trading windows
- Entry/exit conditions specific to events

### 3. Time-Based Strategies (`time_based`)
Strategies with specific trading windows and schedules.

**Examples:**
- `time_based.yaml` - Intraday momentum with specific time windows

**Key Components:**
- Trading windows with start/end times
- Session types (REGULAR, PREMARKET, AFTERHOURS)
- Force close rules for end of day

### 4. ML-Based Strategies (`ml_based`)
Strategies using machine learning models for predictions.

**Examples:**
- `ml_based.yaml` - XGBoost model with prediction thresholds

**Key Components:**
- Model configuration and path
- Prediction and confidence thresholds
- Feature specifications
- Ensemble options

### 5. Composite Strategies (`composite`)
Strategies that combine multiple sub-strategies.

**Examples:**
- `composite.yaml` - Multi-strategy portfolio with weighted voting

**Key Components:**
- Component strategies with weights
- Combination modes (WEIGHTED, MAJORITY, UNANIMOUS)
- Dynamic weight adaptation
- Market regime adjustments

## Configuration Structure

### Basic Parameters
All strategies share common base parameters:
```yaml
strategy_type: signal_based  # Type of strategy
name: Strategy Name          # Unique identifier
description: What it does    # Human-readable description

parameters:
  initial_capital: 100000    # Starting capital
  position_size: 0.1         # Position size (10%)
  max_positions: 5           # Max concurrent positions
  commission: 0.001          # Trading costs
```

### Signal Conditions
For signal-based strategies:
```yaml
entry_conditions:
  - signal_name: rsi_14      # Indicator name
    operator: LESS_THAN      # Comparison operator
    value: 30                # Threshold value or another signal
    lookback: 1              # Periods to look back
    weight: 1.0              # Importance weight
```

### Available Operators
- `GREATER_THAN` (>)
- `LESS_THAN` (<)
- `EQUAL` (=)
- `GREATER_EQUAL` (>=)
- `LESS_EQUAL` (<=)
- `NOT_EQUAL` (!=)
- `CROSSES_ABOVE` (crosses above)
- `CROSSES_BELOW` (crosses below)
- `BETWEEN` (within range)
- `OUTSIDE` (outside range)

### Common Signals/Indicators
- Price: `close`, `open`, `high`, `low`
- Moving Averages: `sma_20`, `sma_50`, `sma_200`, `ema_12`, `ema_26`
- Momentum: `rsi_14`, `macd`, `macd_signal`, `macd_histogram`
- Volatility: `atr_14`, `bb_upper`, `bb_middle`, `bb_lower`
- Volume: `volume`, `volume_sma_20`, `volume_ratio`
- Custom: `highest_20`, `lowest_10`, `vwap`, `adx_14`

## Usage

### Loading a Strategy from Config

```python
from src.strats.config_loader import load_strategy_from_yaml

# Load a single strategy
strategy = load_strategy_from_yaml('src/strats/examples/golden_cross.yaml')

# Use in backtesting
backtester.add_strategy(strategy)
results = backtester.run()
```

### Creating Custom Configurations

1. Copy an existing example as a template
2. Modify the parameters and rules
3. Test with small capital first
4. Validate signals are available in your data

### Best Practices

1. **Start Simple**: Begin with basic rules and add complexity gradually
2. **Test Thoroughly**: Backtest across different market conditions
3. **Risk Management**: Always include stop losses and position limits
4. **Document**: Add comments explaining the strategy logic
5. **Version Control**: Keep track of configuration changes

## Validation

Before using a configuration:
1. Check all referenced signals exist in your data
2. Verify operator logic (e.g., CROSSES_ABOVE needs historical data)
3. Ensure position sizes don't exceed capital
4. Test with paper trading first

## Advanced Features

### Dynamic Parameters
Some parameters can reference market conditions:
```yaml
position_size: dynamic  # Will be calculated based on volatility
stop_loss: 2_atr       # 2x Average True Range
```

### Custom Functions
Reference Python functions for complex logic:
```yaml
entry_strategy: custom_entry_function  # Name of Python function
exit_strategy: custom_exit_function
```

### Market Filters
Apply strategies only to specific symbols or conditions:
```yaml
symbol_filter: ["AAPL", "MSFT", "GOOGL"]
market_cap_min: 10000000000  # $10B minimum
volume_min: 1000000          # $1M daily volume
```

## Troubleshooting

### Common Issues

1. **"Signal not found"**: Ensure the signal name matches exactly with available indicators
2. **"Invalid operator"**: Check operator is spelled correctly in uppercase
3. **"Position size too large"**: Reduce position_size or increase initial_capital
4. **"No trades generated"**: Conditions may be too restrictive, try relaxing thresholds

### Debugging Tips

- Enable verbose logging to see signal evaluations
- Start with one rule at a time
- Use backtesting visualization to see entry/exit points
- Check data quality and availability for required periods

## Contributing

To add new example strategies:
1. Create a descriptive YAML file
2. Include comprehensive comments
3. Test the configuration
4. Add documentation to this README