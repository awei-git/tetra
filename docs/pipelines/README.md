# Tetra Pipeline Architecture

## Overview

The Tetra platform implements a 4-stage pipeline architecture that systematically processes market data from raw ingestion through strategy evaluation. Each pipeline stage has a single, well-defined responsibility, enabling modularity, scalability, and maintainability.

## Pipeline Stages

### 🔄 Data Flow
```
[External APIs] → [Data Pipeline] → [Scenarios Pipeline] → [Metrics Pipeline] → [Assessment Pipeline]
```

### 1. [Data Pipeline](DATA_PIPELINE.md)
**Stage 1: Raw Data Ingestion**
- Fetches market data, economic indicators, events, and news from external sources
- Runs daily at 7 PM ET (configurable)
- Stores raw, unprocessed data in PostgreSQL/TimescaleDB
- Ensures data quality and completeness

### 2. [Scenarios Pipeline](SCENARIOS_PIPELINE.md)
**Stage 2: Market Scenario Generation**
- Creates different market environments for comprehensive testing
- Generates historical periods, market regimes, and simulated scenarios
- Defines scenario metadata including time ranges and volatility adjustments
- Enables testing strategies across diverse market conditions

### 3. [Metrics Pipeline](METRICS_PIPELINE.md)
**Stage 3: Derived Metrics Calculation**
- Pre-calculates technical indicators, statistical metrics, and ML features
- Computes metrics for each scenario independently
- Caches results for fast strategy evaluation
- Eliminates redundant calculations during backtesting

### 4. [Assessment Pipeline](ASSESSMENT_PIPELINE.md)
**Stage 4: Strategy Evaluation**
- Loads strategies from YAML configurations
- Backtests strategies across all scenarios using pre-calculated metrics
- Calculates performance metrics and rankings
- Generates comprehensive reports for strategy selection

## Quick Start

### Running Individual Pipelines

```bash
# Stage 1: Ingest today's data
python -m src.pipelines.data_pipeline.main --mode daily

# Stage 2: Generate scenarios
python -m src.pipelines.scenarios_pipeline.main --type historical

# Stage 3: Calculate metrics for scenarios
python -m src.pipelines.metrics_pipeline.main --scenario all

# Stage 4: Assess strategies
python -m src.pipelines.assessment_pipeline.main --strategies all
```

### Running Full Pipeline Chain

```bash
# Run all stages in sequence
./bin/pipelines/run_full_pipeline.sh

# Run with specific date range
./bin/pipelines/run_full_pipeline.sh --start-date 2024-01-01 --end-date 2024-12-31
```

## Pipeline Orchestration

See [ORCHESTRATION.md](ORCHESTRATION.md) for details on:
- Pipeline dependencies and execution order
- Scheduling configuration
- Error handling and recovery
- Performance optimization

## Database Schema

Each pipeline stage interacts with specific database schemas:

```sql
-- Stage 1: Raw data storage
market_data.*      -- OHLCV, symbols, exchanges
economic_data.*    -- Economic indicators, releases
events.*           -- Earnings, dividends, economic events
news.*            -- Articles, sentiment

-- Stage 2: Scenario definitions
scenarios.*        -- Scenario metadata, parameters

-- Stage 3: Calculated metrics
derived.*          -- Technical indicators, ML features (with scenario_id)

-- Stage 4: Strategy results
strategies.*       -- Backtest results, performance metrics
```

## Benefits of This Architecture

### 1. **Separation of Concerns**
Each pipeline has a single, clear responsibility, making the system easier to understand and maintain.

### 2. **Performance Optimization**
- Pre-calculated metrics eliminate redundant computation
- Parallel processing within each stage
- Efficient data storage with TimescaleDB

### 3. **Flexibility**
- Easy to add new data sources to Stage 1
- Simple to define new scenarios in Stage 2
- Extensible metric calculations in Stage 3
- Pluggable strategy definitions in Stage 4

### 4. **Reproducibility**
- Scenarios are versioned and stored
- Metrics are cached with scenario context
- Consistent backtesting across time

### 5. **Scalability**
- Each stage can be scaled independently
- Parallel execution within stages
- Distributed processing ready

## Configuration

Pipeline configuration is managed through:
- `config/pipelines.yaml` - Global pipeline settings
- Environment variables - Override specific parameters
- Command-line arguments - Runtime configuration

## Monitoring

Each pipeline provides:
- Detailed logging with rotation
- Progress tracking and ETA
- Error reporting with context
- Performance metrics

## Development

### Adding a New Pipeline Stage

1. Create pipeline module in `src/pipelines/your_pipeline/`
2. Implement `Pipeline` base class
3. Define pipeline steps extending `PipelineStep`
4. Add configuration to `config/pipelines.yaml`
5. Create documentation in `docs/pipelines/`
6. Add orchestration rules

### Testing Pipelines

```bash
# Run pipeline tests
pytest tests/pipelines/ -v

# Test specific pipeline
pytest tests/pipelines/test_data_pipeline.py -v

# Integration tests
pytest tests/integration/test_pipeline_chain.py -v
```

## Troubleshooting

### Common Issues

1. **Pipeline fails with database connection error**
   - Check PostgreSQL is running: `docker ps | grep postgres`
   - Verify connection settings in `.env`

2. **Metrics pipeline runs slowly**
   - Check if scenarios are too large
   - Verify database indexes are present
   - Consider increasing parallel workers

3. **Assessment pipeline memory issues**
   - Reduce batch size in configuration
   - Enable disk-based caching
   - Use subset of scenarios for testing

## Complete Documentation

For a comprehensive guide to the entire pipeline system including:
- Architecture principles and data flow
- Database schema and storage
- Running and monitoring pipelines
- WebGUI integration
- Troubleshooting guide

See: **[Pipeline System Documentation](PIPELINE_SYSTEM.md)**

## Next Steps

- Read individual pipeline documentation for detailed information
- Review [ORCHESTRATION.md](ORCHESTRATION.md) for scheduling setup
- Check example configurations in `config/examples/`
- Explore strategy definitions in `src/strats/examples/`