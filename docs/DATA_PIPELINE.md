# Data Pipeline Documentation

## Overview
The Tetra data pipeline is a modular, extensible system designed to ingest financial data from multiple sources, perform quality checks, and store it in TimescaleDB for analysis. The pipeline runs daily at 5 AM ET and supports both incremental updates and historical backfills.

## Architecture

### Pipeline Structure
```
src/pipelines/
├── base.py                     # Abstract base classes
│   ├── Pipeline               # Main orchestrator
│   ├── PipelineStep          # Step interface
│   └── PipelineContext       # Shared context
│
├── data_pipeline/
│   ├── pipeline.py            # DataPipeline implementation
│   ├── main.py               # CLI entry point
│   └── steps/
│       ├── market_data.py     # OHLCV data ingestion
│       ├── economic_data.py   # Economic indicators
│       ├── news_sentiment.py  # News and sentiment
│       ├── event_data.py      # Corporate events
│       └── data_quality.py    # Coverage analysis
│
scripts/
├── daily_update.py            # Simple daily updater
├── fill_missing_august.py     # Gap filler utility
└── update_economic_august.py  # Economic data updater
```

### Design Principles

1. **Unified Design**: Daily updates are just backfills with `start_date = end_date = today`
2. **Modular Steps**: Each data source is a separate step that can run independently
3. **Parallel Execution**: Steps run concurrently where possible
4. **Graceful Degradation**: Partial failures don't stop the entire pipeline
5. **Comprehensive Logging**: Detailed logs for debugging and monitoring

## Data Sources

### 1. Market Data (Polygon.io + YFinance)

**Primary: Polygon.io**
- Free tier: 5 API calls/minute
- Provides: OHLCV, trades, quotes
- Coverage: US stocks and ETFs
- Rate limiting: Automatic with exponential backoff

**Backup: YFinance**
- No rate limits
- Used when Polygon fails
- Provides: OHLCV data only
- Less reliable but good fallback

**Volume Scaling Fix**
```python
# Polygon returns volume in shares, we store in millions
if provider == "polygon":
    volume = volume / 1_000_000
```

### 2. Economic Data (FRED)

**Federal Reserve Economic Data**
- 58 indicators tracked
- Key metrics: GDP, CPI, unemployment, interest rates
- Update frequency: Daily to quarterly
- No rate limiting

**Indicators Configuration**
```python
INDICATORS = [
    ("GDPC1", "Real GDP", UpdateFrequency.QUARTERLY),
    ("CPIAUCSL", "CPI Urban", UpdateFrequency.MONTHLY),
    ("UNRATE", "Unemployment Rate", UpdateFrequency.MONTHLY),
    ("DFF", "Federal Funds Rate", UpdateFrequency.DAILY),
    # ... 54 more
]
```

### 3. News & Sentiment

**NewsAPI**
- Real-time financial news
- 1000 requests/day (free tier)
- Keyword and source filtering
- Sentiment analysis via NLTK

**AlphaVantage**
- Market sentiment indicators
- News sentiment scores
- 5 API calls/minute

### 4. Event Data

**Financial Calendar Events**
- Earnings releases
- Economic data releases
- Dividend announcements
- Fed meetings
- Options expirations

## Pipeline Execution

### Daily Update Mode

The daily pipeline runs every morning at 5 AM ET via launchd:

```bash
# Simple daily update script
python scripts/daily_update.py
```

**Daily Update Logic**
1. Identify symbols with missing recent data
2. Group symbols by how far behind they are
3. Fetch data in batches with appropriate date ranges
4. Handle rate limiting gracefully
5. Use YFinance for failed symbols

### Backfill Mode

For historical data or gap filling:

```bash
# Backfill last 30 days
python -m src.pipelines.data_pipeline.main --mode=backfill --days=30

# Backfill specific date range
python -m src.pipelines.data_pipeline.main --mode=backfill \
    --start-date=2024-01-01 --end-date=2024-12-31

# Skip certain steps
python -m src.pipelines.data_pipeline.main --mode=daily \
    --skip-steps news_sentiment event_data
```

### Gap Detection

The pipeline automatically detects and fills gaps:

```python
async def detect_gaps(symbol: str) -> List[DateRange]:
    """Detect missing data periods for a symbol"""
    query = """
    WITH date_series AS (
        SELECT generate_series(
            '2015-01-01'::date,
            CURRENT_DATE,
            '1 day'::interval
        )::date AS expected_date
    ),
    actual_data AS (
        SELECT DISTINCT DATE(timestamp) as data_date
        FROM market_data.ohlcv
        WHERE symbol = :symbol
    )
    SELECT 
        MIN(expected_date) as gap_start,
        MAX(expected_date) as gap_end
    FROM date_series
    LEFT JOIN actual_data ON expected_date = data_date
    WHERE data_date IS NULL
    GROUP BY (expected_date - ROW_NUMBER() OVER (ORDER BY expected_date))
    """
```

## Data Quality

### Coverage Metrics

The pipeline tracks coverage for each symbol:
- Total records
- Date range
- Missing days
- Quality score (excellent/good/fair/poor)

### Quality Checks

1. **OHLCV Validation**
   - High ≥ Low
   - High ≥ Open, Close
   - Low ≤ Open, Close
   - Volume ≥ 0

2. **Duplicate Prevention**
   - UNIQUE constraint on (symbol, timestamp)
   - ON CONFLICT DO UPDATE for updates

3. **Gap Detection**
   - Identify missing trading days
   - Account for weekends/holidays
   - Alert on extended gaps

## Scheduling

### macOS launchd Configuration

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" 
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.tetra.daily-pipeline</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/angwei/.pyenv/shims/python</string>
        <string>/Users/angwei/Repos/tetra/scripts/daily_update.py</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>5</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/tmp/tetra-daily.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/tetra-daily.err</string>
</dict>
</plist>
```

### Manual Execution

```bash
# Load the job
launchctl load ~/Library/LaunchAgents/com.tetra.daily-pipeline.plist

# Trigger immediately
launchctl start com.tetra.daily-pipeline

# Check status
launchctl list | grep tetra
```

## Error Handling

### Retry Logic
- Exponential backoff for rate limits
- Maximum 3 retries per request
- Failed symbols tracked and retried with backup provider

### Partial Failures
- Each step returns success/failure status
- Pipeline continues even if some steps fail
- Failed steps logged for manual review

### Monitoring
- Logs written to `/tmp/tetra-*.log`
- Email alerts for critical failures (planned)
- Slack integration (planned)

## Performance Optimization

### Batch Processing
- Market data: 50 symbols per batch
- Economic data: 10 indicators per batch
- News: 100 articles per batch

### Async Operations
- All database operations use async SQLAlchemy
- Concurrent API requests where allowed
- Connection pooling for database

### Caching
- Recent data cached in memory during run
- Frequently accessed data in Redis (planned)
- CDN for static market data (planned)

## Troubleshooting

### Common Issues

1. **Rate Limit Exceeded**
   ```
   Error: 429 Too Many Requests
   Solution: Reduce batch size or add delays
   ```

2. **Missing Data After Update**
   ```
   Check: Symbol still shows old date
   Solution: Run fill_missing script for specific symbol
   ```

3. **Database Connection Failed**
   ```
   Error: connection refused
   Solution: Check Docker container is running
   ```

### Debug Commands

```bash
# Check last run
tail -f /tmp/tetra-daily.log

# Test specific symbol
python -c "from src.ingestion.data_ingester import DataIngester; ..."

# Verify data in database
psql -h localhost -p 5432 -U postgres -d tetra \
  -c "SELECT MAX(timestamp) FROM market_data.ohlcv WHERE symbol='AAPL'"
```

## Data Storage Location

### Important: Database Architecture
The Tetra platform stores all data directly in a **PostgreSQL database with TimescaleDB extension**. There are **no raw data files** - all data from APIs is processed and stored directly in the database.

### Database Location
- **Primary Database**: PostgreSQL running in Docker container `tetra-postgres`
- **Data Volume**: Docker volume `tetra_postgres_data` mounted at `/var/lib/postgresql/data`
- **Connection**: `postgresql://tetra_user:tetra_password@localhost:5432/tetra`
- **Port**: 5432 (ensure no local PostgreSQL conflicts)

### Data Flow
1. **APIs** (Polygon.io, FRED, NewsAPI, etc.) → 
2. **Python Pipeline** (data validation & transformation) → 
3. **PostgreSQL/TimescaleDB** (persistent storage)

### Accessing the Database
```bash
# Connect to the Docker PostgreSQL
docker exec -it tetra-postgres psql -U tetra_user -d tetra

# View data statistics
SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM market_data.ohlcv;
```

### Troubleshooting
If data appears incorrect, check:
1. Ensure Docker container is running: `docker ps | grep tetra-postgres`
2. Verify no local PostgreSQL is conflicting on port 5432
3. Check which database the backend is connecting to

## Future Enhancements

1. **Real-time Data**
   - WebSocket integration for live prices
   - Kafka for event streaming
   - Redis for hot data

2. **Additional Sources**
   - Interactive Brokers for execution data
   - Crypto exchanges (Binance, Coinbase)
   - Social media sentiment

3. **Advanced Features**
   - Data versioning
   - Audit trails
   - Data lineage tracking
   - Automated data quality reports