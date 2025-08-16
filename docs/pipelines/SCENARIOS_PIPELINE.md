# Scenarios Pipeline Documentation

## Overview

The Scenarios Pipeline is Stage 2 of the Tetra platform's data processing architecture. It generates comprehensive market scenarios from raw historical data, providing diverse testing environments for robust strategy evaluation. The pipeline creates 131 scenarios including historical events, stochastic simulations, and stress tests, storing them efficiently with daily overwrite to minimize storage usage.

## Purpose

- **Create diverse testing environments** for robust strategy evaluation
- **Identify regime-specific performance** by categorizing market conditions
- **Enable stress testing** through synthetic scenarios
- **Support Monte Carlo analysis** via stochastic simulations
- **Provide reproducible scenarios** for consistent backtesting

## Architecture

### Pipeline Structure
```
src/pipelines/
├── scenarios_pipeline/
│   ├── __init__.py
│   ├── pipeline.py              # ScenariosPipeline implementation
│   ├── main.py                  # CLI entry point
│   └── steps/
│       ├── data_loading.py      # Load raw market data
│       ├── historical_scenario.py # Historical event extraction
│       ├── stochastic_scenario.py # Monte Carlo/Bootstrap generation
│       ├── stress_scenario.py    # Stress test scenarios
│       ├── scenario_storage.py   # File-based storage with daily overwrite
│       └── scenario_aggregation.py # Combine all scenario types
```

### Execution Schedule
- **Daily at 7:30 PM**: Runs automatically via launchd after data pipeline completes
- **Storage**: Fixed filenames for daily overwrite (< 1MB total)
- **Output**: 131 scenarios across 5 categories

## Scenario Types

### 1. Historical Event Scenarios

Extract specific historical periods with significant market events:

```python
EVENT_SCENARIOS = {
    "covid_crash": {
        "start": "2020-02-20",
        "end": "2020-04-30",
        "description": "COVID-19 market crash and initial recovery",
        "volatility_multiplier": 3.5,
        "key_events": ["circuit_breakers", "fed_intervention"]
    },
    "svb_collapse": {
        "start": "2023-03-08",
        "end": "2023-03-31",
        "description": "Regional banking crisis",
        "affected_sectors": ["financials", "regional_banks"],
        "volatility_multiplier": 2.0
    },
    "gme_squeeze": {
        "start": "2021-01-11",
        "end": "2021-02-10",
        "description": "GameStop short squeeze",
        "affected_symbols": ["GME", "AMC", "BB", "NOK"],
        "volatility_multiplier": 5.0
    },
    "fed_taper_2022": {
        "start": "2022-03-01",
        "end": "2022-12-31",
        "description": "Fed rate hike cycle",
        "rate_environment": "rising",
        "volatility_multiplier": 1.8
    }
}
```

### 2. Market Regime Scenarios

Identify and extract different market regimes:

```python
REGIME_TYPES = {
    "bull_market": {
        "criteria": {
            "sma_50_above_200": True,
            "positive_momentum": True,
            "low_volatility": True
        },
        "min_duration_days": 90
    },
    "bear_market": {
        "criteria": {
            "sma_50_below_200": True,
            "negative_momentum": True,
            "high_volatility": True
        },
        "min_duration_days": 60
    },
    "high_volatility": {
        "criteria": {
            "vix_above": 25,
            "daily_moves_above": 0.02
        },
        "min_duration_days": 30
    },
    "ranging_market": {
        "criteria": {
            "low_trend_strength": True,
            "price_in_range": True
        },
        "min_duration_days": 45
    }
}
```

### 3. Stochastic Scenarios

Generate synthetic scenarios using statistical methods:

#### Monte Carlo Simulation
```python
monte_carlo_config = {
    "num_scenarios": 1000,
    "time_horizon_days": 252,
    "return_model": "historical_distribution",
    "correlation_matrix": "rolling_60d",
    "volatility_scaling": "GARCH"
}
```

#### Bootstrap Resampling
```python
bootstrap_config = {
    "num_scenarios": 500,
    "block_size_days": 20,  # Block bootstrap for autocorrelation
    "sampling_method": "circular_block",
    "preserve_seasonality": True
}
```

### 4. Stress Test Scenarios

Create extreme but plausible scenarios:

```python
STRESS_SCENARIOS = {
    "flash_crash": {
        "duration_minutes": 30,
        "max_drawdown": -0.10,
        "recovery_time_hours": 24,
        "affected_symbols": "all"
    },
    "correlation_breakdown": {
        "normal_correlation": 0.7,
        "stress_correlation": -0.3,
        "duration_days": 5
    },
    "liquidity_crisis": {
        "volume_reduction": 0.8,
        "spread_widening": 5.0,
        "duration_days": 10
    },
    "sector_rotation": {
        "from_sector": "technology",
        "to_sector": "utilities",
        "rotation_speed_days": 20,
        "magnitude": 0.15
    }
}
```

## Storage Implementation

### File-Based Storage with Daily Overwrite

The pipeline uses efficient file-based storage that overwrites daily to minimize disk usage:

```python
# Storage structure
data/scenarios/
├── scenario_metadata.json      # Scenario definitions and parameters
├── scenario_timeseries.parquet # Time series data (if saved)
└── scenario_summary.txt        # Human-readable summary
```

### Storage Format

#### Metadata (JSON)
```json
{
  "generated_at": "2025-08-13T12:49:35",
  "scenarios": [
    {
      "name": "COVID-19 Market Crash",
      "scenario_type": "CRISIS",
      "start_date": "2020-02-20",
      "end_date": "2020-04-30",
      "expected_return": -0.35,
      "volatility_multiplier": 3.5,
      "metadata": {...}
    }
  ]
}
```

#### Time Series (Parquet)
- Columnar format for efficient compression
- Schema: scenario_name, symbol, date, returns, prices
- Optional: Only saved with --save-timeseries flag

### Loading Scenarios

```python
from src.pipelines.scenarios_pipeline.steps.scenario_storage import ScenarioLoaderStep

# Load cached scenarios
loader = ScenarioLoaderStep()
scenarios = await loader.run()
```

## Pipeline Workflow

### 1. Data Loading
```python
# Load required historical data
data = await load_market_data(
    symbols=universe_symbols,
    start_date=config.lookback_start,
    end_date=config.lookback_end
)
```

### 2. Scenario Generation
```python
# Generate scenarios based on configuration
scenarios = []

# Historical events
for event_name, event_config in EVENT_SCENARIOS.items():
    scenario = generate_historical_scenario(data, event_config)
    scenarios.append(scenario)

# Market regimes
regimes = detect_market_regimes(data, REGIME_TYPES)
scenarios.extend(regimes)

# Stochastic scenarios
if config.include_stochastic:
    mc_scenarios = generate_monte_carlo(data, monte_carlo_config)
    scenarios.extend(mc_scenarios)

# Stress tests
if config.include_stress:
    stress_scenarios = generate_stress_tests(data, STRESS_SCENARIOS)
    scenarios.extend(stress_scenarios)
```

### 3. Scenario Validation
```python
# Validate each scenario
for scenario in scenarios:
    # Check data completeness
    validate_data_coverage(scenario)
    
    # Verify statistical properties
    validate_statistical_properties(scenario)
    
    # Ensure realistic constraints
    validate_market_constraints(scenario)
```

### 4. Storage
```python
# Store scenario definitions and data
for scenario in validated_scenarios:
    # Store definition
    scenario_id = await store_scenario_definition(scenario)
    
    # Store time series data if needed
    if scenario.has_adjusted_data:
        await store_scenario_timeseries(scenario_id, scenario.data)
    
    # Store symbol mappings
    await store_scenario_symbols(scenario_id, scenario.symbols)
```

## CLI Usage

### Automated Execution

```bash
# Run via scheduled task (daily at 7:30 PM)
./bin/pipelines/run_scenarios_pipeline.sh

# Manual execution
uv run python -m src.pipelines.scenarios_pipeline.main \
    --type all \
    --include-full-cycles \
    --num-scenarios 100 \
    --severity severe \
    --save-timeseries \
    --parallel
```

### Command Options

```bash
# Generate specific scenario types
python -m src.pipelines.scenarios_pipeline.main --type historical
python -m src.pipelines.scenarios_pipeline.main --type stochastic --num-scenarios 50
python -m src.pipelines.scenarios_pipeline.main --type stress --severity severe

# Include full market cycles (adds 8 scenarios)
python -m src.pipelines.scenarios_pipeline.main --include-full-cycles

# Save time series data (increases storage)
python -m src.pipelines.scenarios_pipeline.main --save-timeseries
```

### Advanced Options

```bash
# Specify symbols universe
python -m src.pipelines.scenarios_pipeline.main \
    --symbols SPY,QQQ,IWM,DIA \
    --type all

# Custom date range
python -m src.pipelines.scenarios_pipeline.main \
    --start-date 2015-01-01 \
    --end-date 2024-12-31 \
    --type historical

# Parallel processing
python -m src.pipelines.scenarios_pipeline.main \
    --type all \
    --parallel \
    --workers 8

# Dry run (no database writes)
python -m src.pipelines.scenarios_pipeline.main \
    --type all \
    --dry-run
```

## Configuration

### Pipeline Configuration (config/scenarios_pipeline.yaml)

```yaml
scenarios_pipeline:
  # Scenario generation settings
  generation:
    historical:
      enabled: true
      events: ["covid_crash", "svb_collapse", "gme_squeeze"]
      include_context_days: 30  # Days before/after event
    
    regime:
      enabled: true
      min_regime_days: 60
      regime_types: ["bull", "bear", "high_vol", "ranging"]
    
    stochastic:
      enabled: true
      monte_carlo:
        num_scenarios: 1000
        confidence_levels: [0.95, 0.99]
      bootstrap:
        num_scenarios: 500
        block_size: 20
    
    stress:
      enabled: true
      scenarios: ["flash_crash", "correlation_breakdown", "liquidity_crisis"]
      severity_levels: ["moderate", "severe", "extreme"]
  
  # Data settings
  data:
    lookback_years: 10
    symbols_universe: "sp500"  # or custom list
    
  # Performance settings
  performance:
    parallel_workers: 8
    batch_size: 100
    cache_enabled: true
```

## Integration with Pipeline Architecture

### Pipeline Flow
```
Stage 1: Data Pipeline (7:00 PM)
  ↓
Stage 2: Scenarios Pipeline (7:30 PM) ← You are here
  ↓
Stage 3: Metrics Pipeline (Next stage - pre-calculates indicators)
  ↓
Stage 4: Assessment Pipeline (Backtests strategies)
```

### Input Dependencies
- **Data Pipeline**: Market data from `market_data.ohlcv` table
- **Economic Data**: Fed rates, VIX for scenario calibration

### Output for Metrics Pipeline
The Metrics Pipeline (Stage 3) will:
1. Load scenarios from `data/scenarios/` directory
2. Calculate all technical indicators and statistical metrics for each scenario
3. Store pre-computed metrics for fast strategy backtesting

This separation ensures:
- **Scenarios Pipeline**: Defines WHAT market conditions to test
- **Metrics Pipeline**: Calculates HOW assets behave (indicators/metrics)
- **Assessment Pipeline**: Evaluates strategy performance

## Performance Considerations

### Optimization Strategies

1. **Parallel Generation**: Generate independent scenarios in parallel
2. **Batch Processing**: Process symbols in batches for memory efficiency
3. **Incremental Updates**: Only generate new scenarios when data changes
4. **Caching**: Cache intermediate calculations (correlations, volatilities)

### Resource Requirements

```yaml
# Typical resource usage
memory:
  historical: 2GB
  regime: 4GB
  stochastic_1000: 8GB
  stress: 2GB

processing_time:
  historical: 5 minutes
  regime: 15 minutes
  stochastic_1000: 60 minutes
  stress: 10 minutes

storage:
  definitions: 10MB
  time_series_per_scenario: 500MB (if storing adjusted prices)
```

## Monitoring and Validation

### Quality Metrics

```python
# Scenario quality checks
quality_metrics = {
    "data_coverage": 0.95,  # Minimum required coverage
    "statistical_validity": {
        "return_distribution": "within_3_sigma",
        "correlation_stability": 0.8
    },
    "scenario_diversity": {
        "min_unique_scenarios": 50,
        "regime_coverage": ["bull", "bear", "neutral"]
    }
}
```

### Monitoring Dashboard

Key metrics to track:
- Number of scenarios generated by type
- Data coverage per scenario
- Generation time and resource usage
- Scenario diversity metrics
- Validation failure rates

## API Endpoints

```python
# FastAPI routes
@router.get("/scenarios")
async def list_scenarios(
    type: Optional[str] = None,
    active: bool = True
) -> List[ScenarioDefinition]:
    """List all available scenarios"""

@router.post("/scenarios/generate")
async def generate_scenarios(
    config: ScenarioGenerationConfig
) -> ScenarioGenerationResult:
    """Trigger scenario generation"""

@router.get("/scenarios/{scenario_id}")
async def get_scenario(
    scenario_id: int
) -> ScenarioDetail:
    """Get detailed scenario information"""

@router.get("/scenarios/{scenario_id}/timeseries")
async def get_scenario_data(
    scenario_id: int,
    symbol: Optional[str] = None
) -> pd.DataFrame:
    """Get scenario time series data"""
```

## Best Practices

1. **Version Control**: Version scenarios for reproducibility
2. **Documentation**: Document assumptions for each scenario type
3. **Validation**: Always validate scenarios before use
4. **Incremental Generation**: Generate scenarios incrementally as new data arrives
5. **Archival**: Archive old scenarios but keep definitions for reference

## Troubleshooting

### Common Issues

1. **Memory errors during stochastic generation**
   - Reduce number of scenarios
   - Use batch processing
   - Increase system memory

2. **Regime detection finds no regimes**
   - Adjust regime criteria thresholds
   - Increase lookback period
   - Check data quality

3. **Scenario validation failures**
   - Review validation criteria
   - Check for data gaps
   - Verify statistical assumptions

## Scenario Details

### Current Implementation (131 Scenarios)

```
BOOTSTRAP (50 scenarios):
  • Block bootstrap with 20-day blocks
  • Preserves autocorrelation structure
  • Generated from 2-year historical window

MONTE_CARLO (50 scenarios):
  • Geometric Brownian Motion simulation
  • Uses historical volatility and drift
  • 252-day forward projections

HISTORICAL EVENTS (19 scenarios):
Bull Markets (8):
  • AI Boom 2023
  • Dot-Com Bubble Peak
  • Fed Pivot Rally 2023
  • Manufacturing Renaissance 2025
  • Post-COVID Recovery Rally
  • QE3 Rally 2012-2013
  • Trump Election Rally 2016

Crisis Events (3):
  • COVID-19 Market Crash
  • Global Financial Crisis 2008
  • Silicon Valley Bank Collapse

Full Cycles (8):
  • Banking Crisis 2023 Full Cycle
  • COVID-19 Full Cycle (Crash + Recovery)
  • Dot-Com Bubble Full Cycle
  • Fed Tightening Cycle 2022-2023
  • GFC 2008-2009 Full Cycle
  • Tariff Cycle 2025-2026 (projected)
  • Trump Era Full Term

STRESS TESTS (12 scenarios):
  • Credit Spread Widening
  • Derivatives Market Disruption
  • Flash Crash Scenario
  • Geopolitical Crisis
  • Market Liquidity Crisis
  • Rate Shock +500bp
  • Stagflation Environment
  • Systemic Cyber Attack
  • Tariff Implementation 2025
  • Tech Bubble 2.0 Burst
  • USD Crisis
  • Volatility Spike
```

## Future Enhancements

- **Regime-switching models**: Hidden Markov Model scenarios
- **Cross-asset correlation breaks**: Multi-asset stress scenarios
- **Intraday scenarios**: High-frequency event modeling
- **Climate risk scenarios**: ESG and transition risk events
- **Crypto contagion**: Digital asset spillover effects