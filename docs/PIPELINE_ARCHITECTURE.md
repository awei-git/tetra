# Tetra Pipeline Architecture Documentation

## Overview

Tetra uses a 4-stage pipeline architecture for comprehensive trading strategy assessment:

```
DATA → SCENARIOS → METRICS → ASSESSMENT
```

Each pipeline builds upon the output of the previous stage, creating a complete workflow from raw market data to strategy performance rankings.

## Pipeline Status Summary

| Pipeline | Status | Output | Location |
|----------|--------|--------|----------|
| Data | ✅ GOOD | 433,244 OHLCV records | PostgreSQL/TimescaleDB |
| Scenarios | ✅ GOOD | 32 market scenarios | `data/scenarios/scenario_metadata.json` |
| Metrics | ✅ GOOD | 24 metrics files | `data/metrics/*_metrics.parquet` |
| Assessment | ✅ GOOD | Strategy rankings | `data/assessment/` |

## 1. Data Pipeline

**Purpose**: Ingest historical market data from external sources into the database.

### Key Components
- **Source**: Polygon API (primary), yfinance (fallback)
- **Storage**: PostgreSQL with TimescaleDB extension
- **Schema**: `market_data.ohlcv` table (hypertable)

### Output
- **Records**: 433,244 OHLCV data points
- **Symbols**: 154 stocks from market universe
- **Date Range**: 2015-08-02 to present
- **Fields**: timestamp, symbol, open, high, low, close, volume

### Run Command
```bash
./bin/pipelines/run_data_pipeline.sh
```

### Database Access
```python
import asyncpg
conn = await asyncpg.connect('postgresql://tetra_user:tetra_password@localhost:5432/tetra')
count = await conn.fetchval('SELECT COUNT(*) FROM market_data.ohlcv')
```

## 2. Scenarios Pipeline

**Purpose**: Define market scenarios representing different conditions (bull markets, crashes, recoveries, etc.).

### Key Components
- **Scenario Types**: Historical, stress, regime-based
- **Definition**: Start/end dates, market characteristics
- **Storage**: JSON metadata file

### Output
- **File**: `data/scenarios/scenario_metadata.json`
- **Count**: 32 scenarios defined
- **Coverage**: Bull markets, bear markets, crashes, recoveries, volatility events

### Scenario Examples
- COVID-19 Market Crash (2020)
- AI Boom 2023
- Fed Tightening Cycle 2022-2023
- Silicon Valley Bank Collapse
- Trump Election Rally 2016

### Run Command
```bash
./bin/pipelines/run_scenarios_pipeline.sh
```

### Loading Scenarios
```python
import json
with open('data/scenarios/scenario_metadata.json') as f:
    data = json.load(f)
    scenarios = data['scenarios']  # List of 32 scenario dicts
```

**Note**: 6 scenarios are skipped in metrics generation because their dates fall before 2015 (outside available data range).

## 3. Metrics Pipeline

**Purpose**: Calculate technical indicators and metrics for each scenario-symbol combination.

### Key Components
- **Input**: OHLCV data + scenario definitions
- **Calculations**: 140+ technical indicators
- **Output Format**: Parquet files per scenario

### Output
- **Files**: 24 metrics files (6 scenarios skipped due to date range)
- **Location**: `data/metrics/{scenario_name}_metrics.parquet`
- **Indicators**: RSI, SMA, EMA, MACD, Bollinger Bands, ATR, volume metrics, etc.

### File Naming Convention
```python
filename = f"{scenario_name.replace(' ', '_').replace('-', '_')}_metrics.parquet"
```

### Run Command
```bash
./bin/pipelines/run_metrics_pipeline.sh
```

### Loading Metrics
```python
import pandas as pd
df = pd.read_parquet('data/metrics/AI_Boom_2023_metrics.parquet')
# DataFrame with columns: symbol, date, close, rsi_14, sma_20, etc.
```

### Available Indicators (Sample)
- Price: open, high, low, close, vwap
- Moving Averages: sma_10, sma_20, sma_50, sma_200, ema_12, ema_26
- Momentum: rsi_14, macd, macd_signal, roc_10, stochastic_k
- Volatility: atr_14, bollinger bands (bb_upper, bb_middle, bb_lower)
- Volume: obv, volume_mean_20, volume_ratio, cmf_20
- And 100+ more...

## 4. Assessment Pipeline

**Purpose**: Backtest all strategies across all scenarios and symbols to produce comprehensive performance rankings.

### Key Components
- **Strategies**: 11 trading strategies (buy_and_hold, golden_cross, rsi_reversion, etc.)
- **Backtesting**: Real strategy execution with signal generation
- **Combinations**: EVERY strategy × EVERY scenario × EVERY symbol

### Process
1. **Data Gathering**: Load scenarios, symbols, strategies, and metrics
2. **Strategy Loading**: Instantiate strategy implementations
3. **Backtest Execution**: Run 53,856 backtests (11 × 153 × 26 valid scenarios)
4. **Performance Analysis**: Calculate returns, Sharpe, drawdown, etc.
5. **Ranking**: Rank strategies by overall performance
6. **Storage**: Save results to database and files

### Output
- **Location**: `data/assessment/`
- **Rankings**: Overall, by category, by regime
- **Metrics**: Total return, Sharpe ratio, max drawdown, win rate
- **Trade Details**: Entry/exit points, P&L per trade

### Run Command
```bash
./bin/pipelines/run_assessment_pipeline.sh
```

### Backtest Implementation
The assessment pipeline uses comprehensive backtest execution that:
- Iterates through each historical data point
- Calls strategy's `generate_signals()` method
- Executes BUY/SELL trades based on signals
- Tracks cash, positions, and equity curve
- Calculates realistic performance metrics

## Data Flow

```mermaid
graph LR
    A[Polygon/Yahoo API] -->|OHLCV| B[Data Pipeline]
    B -->|PostgreSQL| C[OHLCV Table]
    
    D[Scenario Definitions] -->|JSON| E[Scenarios Pipeline]
    E -->|scenario_metadata.json| F[Scenarios]
    
    C --> G[Metrics Pipeline]
    F --> G
    G -->|Parquet Files| H[Technical Indicators]
    
    H --> I[Assessment Pipeline]
    F --> I
    J[Strategy Definitions] --> I
    I -->|Rankings| K[Results]
```

## File Structure

```
data/
├── scenarios/
│   ├── scenario_metadata.json      # 32 scenario definitions
│   ├── scenario_summary.txt        # Human-readable summary
│   └── scenario_timeseries.parquet # Time series data
├── metrics/
│   ├── AI_Boom_2023_metrics.parquet
│   ├── COVID_19_Market_Crash_metrics.parquet
│   └── ... (24 files total)
└── assessment/
    ├── strategy_rankings.json
    ├── backtest_results.parquet
    └── performance_summary.json
```

## Important Notes

### Data Availability
- Market data starts from 2015-08-02
- 6 scenarios have dates before 2015 and are skipped:
  - Dot-Com Bubble Peak
  - QE3 Rally 2012-2013
  - Global Financial Crisis 2008
  - GFC 2008-2009 Full Cycle
  - Dot-Com Bubble Full Cycle
  - Flash Crash Scenario

### Performance Considerations
- Assessment pipeline processes 53,856 backtests
- Typical runtime: 30-60 minutes depending on hardware
- CPU intensive due to comprehensive strategy execution
- Memory usage: ~4-5GB during peak processing

### Strategy Count
Currently 11 strategies defined in `src/definitions/strategies.py`:
1. Buy and Hold
2. Dollar Cost Averaging
3. Golden Cross
4. Trend Following
5. Mean Reversion
6. RSI Strategy
7. Momentum
8. Dual Momentum
9. Volatility Targeting
10. ML Ensemble
11. Balanced Portfolio

Additional strategies can be found in `src/strats/benchmark.py` but need to be registered in the DEFAULT_STRATEGIES dictionary to be included in assessment.

## Running All Pipelines

To run the complete pipeline sequence:

```bash
# 1. Ingest market data
./bin/pipelines/run_data_pipeline.sh

# 2. Generate scenarios (usually run once)
./bin/pipelines/run_scenarios_pipeline.sh

# 3. Calculate metrics for all scenarios
./bin/pipelines/run_metrics_pipeline.sh

# 4. Run comprehensive assessment
./bin/pipelines/run_assessment_pipeline.sh
```

## Monitoring Pipeline Status

Check pipeline outputs:
```bash
# Data pipeline - check database
psql -U tetra_user -d tetra -c "SELECT COUNT(*) FROM market_data.ohlcv;"

# Scenarios pipeline - check JSON file
ls -la data/scenarios/scenario_metadata.json

# Metrics pipeline - check parquet files
ls -la data/metrics/*.parquet | wc -l

# Assessment pipeline - check if running
ps aux | grep assessment_pipeline
```

## Troubleshooting

### Missing Metrics Files
If metrics files are missing for certain scenarios, check if their date ranges fall within available data (2015-present).

### Assessment Pipeline Taking Too Long
The assessment pipeline is CPU-intensive. Normal runtime is 30-60 minutes for 53,856 backtests. Monitor progress via log files in `/tmp/tetra_assessment_pipeline_*.log`.

### Module Not Found Errors
Scenarios are stored in JSON files, not Python modules. Load them using:
```python
import json
with open('data/scenarios/scenario_metadata.json') as f:
    scenarios = json.load(f)['scenarios']
```

## Conclusion

All four pipelines are functioning correctly:
- ✅ **Data Pipeline**: Successfully ingesting market data
- ✅ **Scenarios Pipeline**: Generating comprehensive market scenarios
- ✅ **Metrics Pipeline**: Computing technical indicators
- ✅ **Assessment Pipeline**: Evaluating strategies with real execution logic

The system is ready for production use and provides comprehensive strategy assessment across diverse market conditions.