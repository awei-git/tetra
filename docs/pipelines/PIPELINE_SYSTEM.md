# Tetra Pipeline System Documentation

## Overview

The Tetra pipeline system consists of four interconnected data pipelines that run sequentially to ingest market data, generate scenarios, calculate metrics, and assess strategy performance. All pipelines store their results in PostgreSQL for display via the WebGUI.

## Architecture Principles

### Data Flow
```
Data Pipeline → Scenarios Pipeline → Metrics Pipeline → Assessment Pipeline → Database → WebGUI
```

### Key Principles
1. **Pipelines compute, Database stores** - All calculations happen in pipelines
2. **WebGUI displays only** - Frontend/backend APIs only fetch and format data
3. **No mock data** - All data must come from real sources
4. **Sequential execution** - Each pipeline depends on the previous one's output

## Pipeline Components

### 1. Data Pipeline
**Location**: `src/pipelines/data_pipeline/`
**Schedule**: Daily at 7:00 PM
**Purpose**: Ingests raw market data from external sources

**Data Sources**:
- Market data (OHLCV) from Polygon/Yahoo Finance
- Economic indicators from FRED
- News articles from NewsAPI
- Event data from various sources

**Output**: Raw data stored in `market_data`, `economic_data`, `news`, and `events` schemas

### 2. Scenarios Pipeline
**Location**: `src/pipelines/scenarios_pipeline/`
**Schedule**: Daily at 7:30 PM
**Purpose**: Generates market scenarios for backtesting

**Scenario Types**:
- Historical scenarios (COVID crash, 2008 crisis, etc.)
- Stress scenarios (rate shocks, liquidity crises)
- Stochastic scenarios (Monte Carlo simulations)

**Output**: Scenario timeseries in `data/scenarios/` and metadata in database

### 3. Metrics Pipeline
**Location**: `src/pipelines/metrics_pipeline/`
**Schedule**: Daily at 8:00 PM
**Purpose**: Calculates performance metrics for each scenario

**Metric Categories**:
- Statistical metrics (returns, volatility, correlations)
- Technical indicators (RSI, MACD, Bollinger Bands)
- ML metrics (predictions, feature importance)
- Event-based metrics (earnings impact, economic releases)

**Output**: Metrics stored in `data/metrics/` as parquet files

### 4. Assessment Pipeline
**Location**: `src/pipelines/assessment_pipeline/`
**Schedule**: Daily at 8:30 PM
**Purpose**: Backtests all strategies across all scenarios and ranks them

**Process**:
1. Loads all strategy definitions from `src/strats/`
2. Tests each strategy-symbol combination on all scenarios
3. Calculates performance metrics (Sharpe, returns, drawdown)
4. Ranks strategies by risk-adjusted score
5. Stores results in `assessment` schema

**Output**: 
- `assessment.backtest_results` - Individual backtest results
- `assessment.strategy_rankings` - Aggregated rankings
- `assessment.scenario_performance` - Best performers per scenario
- `assessment.pipeline_runs` - Execution history

## Database Schema

### Assessment Schema Tables

```sql
assessment.backtest_results
├── strategy_name     -- Strategy identifier
├── strategy_category -- Category (momentum, mean_reversion, etc.)
├── scenario_name     -- Market scenario tested
├── symbol           -- Trading symbol
├── total_return     -- Total return percentage
├── sharpe_ratio     -- Risk-adjusted return
├── max_drawdown     -- Maximum drawdown
├── score           -- Composite performance score
└── run_date        -- Pipeline execution date

assessment.strategy_rankings
├── strategy_name    -- Strategy identifier
├── overall_rank    -- Overall ranking position
├── category_rank   -- Rank within category
├── avg_score      -- Average score across all tests
├── avg_return     -- Average return
├── avg_sharpe     -- Average Sharpe ratio
└── run_date       -- Pipeline execution date

assessment.scenario_performance
├── scenario_name     -- Market scenario
├── top_strategy     -- Best performing strategy
├── top_return      -- Best return achieved
├── avg_return      -- Average return across strategies
└── strategies_tested -- Number of strategies tested

assessment.pipeline_runs
├── pipeline_name    -- Pipeline identifier
├── start_time      -- Execution start time
├── end_time        -- Execution end time
├── status          -- running/success/failed
└── records_processed -- Number of records processed
```

## Running Pipelines

### Manual Execution

```bash
# Run individual pipelines
cd /Users/angwei/Repos/tetra
./bin/run_data_pipeline.sh
./bin/run_scenarios_pipeline.sh
./bin/run_metrics_pipeline.sh
./bin/run_assessment_pipeline.sh

# Or run all pipelines in sequence
./bin/run_all_pipelines.sh
```

### Via API

```python
# Trigger pipeline via API
POST /api/pipeline/trigger/data
POST /api/pipeline/trigger/scenarios
POST /api/pipeline/trigger/metrics
POST /api/pipeline/trigger/assessment
```

### Scheduled Execution

Pipelines run automatically via launchd:
```bash
# Check schedule status
launchctl list | grep tetra

# View logs
tail -f /tmp/tetra_data_pipeline_*.log
tail -f /tmp/tetra_assessment_pipeline_*.log
```

## Pipeline Monitoring

### Via WebGUI

Navigate to the Monitor tab to see:
- Pipeline execution status
- Last run times and durations
- Error messages if any
- Records processed

### Via API

```python
# Get pipeline status
GET /api/pipeline/status

# Get execution history
GET /api/pipeline/history/{pipeline_name}

# Check pipeline health
GET /api/pipeline/health
```

## Adding New Strategies

1. **Define strategy** in `src/strats/`:
   ```python
   class MyStrategy(BaseStrategy):
       def generate_signals(self, data):
           # Implementation
   ```

2. **Register in DEFAULT_STRATEGIES** in `src/definitions/strategies.py`:
   ```python
   DEFAULT_STRATEGIES = {
       'my_strategy': StrategyConfig(
           name='my_strategy',
           category='momentum',
           # ... configuration
       )
   }
   ```

3. **Run assessment pipeline** to test and rank:
   ```bash
   ./bin/run_assessment_pipeline.sh
   ```

4. **View results** in WebGUI Strategies tab

## Important Notes

### No Calculations in WebGUI
- **NEVER** calculate metrics in frontend or backend APIs
- **NEVER** generate mock data
- **ALWAYS** fetch from database
- If data is missing, run the appropriate pipeline

### Data Freshness
- Pipelines run daily after market close
- Historical data updated incrementally
- Scenarios regenerated with latest data
- Rankings reflect most recent market conditions

### Performance Considerations
- Assessment pipeline tests ~1000+ strategy-symbol-scenario combinations
- Typical runtime: 30-60 minutes for full assessment
- Results cached in database for fast retrieval
- WebGUI queries optimized with indexes

## Troubleshooting

### Missing Data in WebGUI

1. Check if pipelines have run:
   ```sql
   SELECT * FROM assessment.pipeline_runs ORDER BY start_time DESC LIMIT 10;
   ```

2. Check for data in assessment tables:
   ```sql
   SELECT COUNT(*) FROM assessment.backtest_results;
   SELECT COUNT(*) FROM assessment.strategy_rankings;
   ```

3. If empty, run pipelines:
   ```bash
   ./bin/run_assessment_pipeline.sh
   ```

### Pipeline Failures

1. Check logs:
   ```bash
   tail -n 100 /tmp/tetra_assessment_pipeline_*.log
   ```

2. Check pipeline status:
   ```bash
   curl http://localhost:8000/api/pipeline/health
   ```

3. Clear stuck runs:
   ```sql
   UPDATE assessment.pipeline_runs 
   SET status = 'failed', error_message = 'Manually terminated'
   WHERE status = 'running' AND start_time < NOW() - INTERVAL '2 hours';
   ```

### Performance Issues

1. Check database indexes:
   ```sql
   SELECT * FROM pg_indexes WHERE schemaname = 'assessment';
   ```

2. Analyze query performance:
   ```sql
   EXPLAIN ANALYZE SELECT * FROM assessment.backtest_results WHERE ...;
   ```

3. Vacuum and analyze tables:
   ```sql
   VACUUM ANALYZE assessment.backtest_results;
   VACUUM ANALYZE assessment.strategy_rankings;
   ```

## Development Workflow

1. **Modify pipeline code** in `src/pipelines/`
2. **Test locally** with small dataset
3. **Run pipeline** to generate data
4. **Verify in database** that data is stored correctly
5. **Check WebGUI** displays data properly
6. **No WebGUI calculations** - if needed, add to pipeline

## Contact

For issues or questions about the pipeline system, check:
- Logs in `/tmp/tetra_*_pipeline_*.log`
- Database tables in `assessment` schema
- Pipeline status in WebGUI Monitor tab