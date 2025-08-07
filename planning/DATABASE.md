# Database Infrastructure Documentation

## Overview
Tetra uses PostgreSQL 15 with the TimescaleDB extension for optimal time-series data performance. The database runs in a Docker container and is designed to handle millions of financial data points with efficient querying and storage.

## Technology Stack

- **Database**: PostgreSQL 15.x
- **Extension**: TimescaleDB 2.x (time-series optimization)
- **Container**: Docker (timescale/timescaledb:latest-pg15)
- **ORM**: SQLAlchemy 2.0 with async support
- **Connection Pool**: asyncpg
- **Migration Tool**: Alembic

## Docker Configuration

### Container Setup
```yaml
# docker-compose.yml
services:
  postgres:
    image: timescale/timescaledb:latest-pg15
    container_name: tetra-postgres
    environment:
      POSTGRES_DB: tetra
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - tetra_postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  tetra_postgres_data:
```

### Connection Details

**ACTUAL PRODUCTION CREDENTIALS:**
```
Host: localhost
Port: 5432
Database: tetra
Username: tetra_user
Password: tetra_password
URL: postgresql://tetra_user:tetra_password@localhost:5432/tetra
```

**Note:** The Docker container is configured with these credentials in docker-compose.yml

### ⚠️ Important: PostgreSQL Instance Conflicts

The system may have **two PostgreSQL instances** that can cause connection issues:

1. **Docker PostgreSQL** (Primary - Contains complete data)
   - Container: `tetra-postgres`
   - Data: **367,371 records** (2015-2025)
   - Location: Docker volume `tetra_postgres_data`
   - This is where the pipeline writes data

2. **Local PostgreSQL** (May cause conflicts)
   - Installation: Homebrew (`/opt/homebrew/var/postgresql@15`)
   - Data: May contain partial or outdated data
   - Issue: If running on port 5432, it intercepts connections

### Verifying Correct Connection

```bash
# Check which PostgreSQL is running on port 5432
lsof -i :5432

# Connect to Docker PostgreSQL specifically
docker exec -it tetra-postgres psql -U tetra_user -d tetra

# Check data in Docker PostgreSQL
docker exec tetra-postgres psql -U tetra_user -d tetra -c \
  "SELECT COUNT(*) as records, MIN(timestamp)::date as from, MAX(timestamp)::date as to FROM market_data.ohlcv;"

# If local PostgreSQL is running, check it separately
psql -h localhost -U tetra_user -d tetra -c \
  "SELECT COUNT(*) as records, MIN(timestamp)::date as from, MAX(timestamp)::date as to FROM market_data.ohlcv;"
```

### Fixing Connection Issues

If the backend connects to the wrong database:

1. **Option 1: Stop local PostgreSQL**
   ```bash
   brew services stop postgresql@15
   ```

2. **Option 2: Change Docker PostgreSQL port**
   ```yaml
   # In docker-compose.yml
   ports:
     - "5433:5432"  # Use port 5433 externally
   ```
   Then update connection strings to use port 5433

3. **Option 3: Ensure Docker starts first**
   ```bash
   # Stop all PostgreSQL instances
   brew services stop postgresql@15
   docker-compose down
   
   # Start only Docker PostgreSQL
   docker-compose up -d postgres
   ```

## Schema Design

### Schema Organization
```sql
-- Market data schema
CREATE SCHEMA IF NOT EXISTS market_data;

-- Economic indicators
CREATE SCHEMA IF NOT EXISTS economic_data;

-- News and sentiment
CREATE SCHEMA IF NOT EXISTS news;

-- Financial events
CREATE SCHEMA IF NOT EXISTS events;

-- Derived/calculated data
CREATE SCHEMA IF NOT EXISTS derived;

-- Trading strategies
CREATE SCHEMA IF NOT EXISTS strategies;

-- Execution records
CREATE SCHEMA IF NOT EXISTS execution;
```

### Core Tables

#### market_data.ohlcv (Hypertable)
```sql
CREATE TABLE market_data.ohlcv (
    id UUID DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open NUMERIC(20, 8) NOT NULL CHECK (open > 0),
    high NUMERIC(20, 8) NOT NULL CHECK (high > 0),
    low NUMERIC(20, 8) NOT NULL CHECK (low > 0),
    close NUMERIC(20, 8) NOT NULL CHECK (close > 0),
    volume BIGINT NOT NULL CHECK (volume >= 0),
    vwap NUMERIC(20, 8),
    trades_count INTEGER,
    timeframe VARCHAR(10) NOT NULL,
    source VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, timestamp),
    CONSTRAINT uq_ohlcv_symbol_time UNIQUE (symbol, timestamp)
);

-- Convert to hypertable
SELECT create_hypertable('market_data.ohlcv', 'timestamp', 
    chunk_time_interval => INTERVAL '7 days');

-- Create indexes
CREATE INDEX idx_ohlcv_symbol_time ON market_data.ohlcv (symbol, timestamp DESC);
CREATE INDEX idx_ohlcv_timestamp ON market_data.ohlcv (timestamp DESC);

-- Add constraints
ALTER TABLE market_data.ohlcv 
    ADD CONSTRAINT chk_high_low CHECK (high >= low),
    ADD CONSTRAINT chk_high_prices CHECK (high >= open AND high >= close),
    ADD CONSTRAINT chk_low_prices CHECK (low <= open AND low <= close);
```

#### economic_data.economic_data
```sql
CREATE TABLE economic_data.economic_data (
    id UUID DEFAULT gen_random_uuid(),
    symbol VARCHAR(50) NOT NULL,  -- FRED symbol (e.g., GDPC1)
    date TIMESTAMPTZ NOT NULL,
    value NUMERIC(20, 8) NOT NULL,
    is_preliminary BOOLEAN DEFAULT FALSE,
    revision_date TIMESTAMPTZ,
    source VARCHAR(50) NOT NULL DEFAULT 'FRED',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, date),
    CONSTRAINT uq_econ_symbol_date UNIQUE (symbol, date)
);

CREATE INDEX idx_econ_symbol_date ON economic_data.economic_data (symbol, date DESC);
CREATE INDEX idx_econ_date ON economic_data.economic_data (date DESC);
```

#### news.news_articles
```sql
CREATE TABLE news.news_articles (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    content TEXT,
    url TEXT UNIQUE NOT NULL,
    source VARCHAR(100),
    author VARCHAR(200),
    published_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_news_published ON news.news_articles (published_at DESC);
CREATE INDEX idx_news_source ON news.news_articles (source);
```

#### events.event_data
```sql
CREATE TABLE events.event_data (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    event_datetime TIMESTAMPTZ NOT NULL,
    event_name VARCHAR(200) NOT NULL,
    description TEXT,
    impact_level INTEGER CHECK (impact_level BETWEEN 1 AND 4),
    symbol VARCHAR(20),
    currency VARCHAR(10),
    country VARCHAR(10),
    actual NUMERIC(20, 8),
    forecast NUMERIC(20, 8),
    previous NUMERIC(20, 8),
    source VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_events_datetime ON events.event_data (event_datetime DESC);
CREATE INDEX idx_events_symbol ON events.event_data (symbol);
CREATE INDEX idx_events_type ON events.event_data (event_type);
```

## TimescaleDB Features

### Hypertables
```sql
-- Convert OHLCV to hypertable with 7-day chunks
SELECT create_hypertable('market_data.ohlcv', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE);

-- Enable compression
ALTER TABLE market_data.ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'timestamp DESC'
);

-- Add compression policy (compress chunks older than 30 days)
SELECT add_compression_policy('market_data.ohlcv', INTERVAL '30 days');
```

### Continuous Aggregates
```sql
-- Daily OHLCV from intraday data
CREATE MATERIALIZED VIEW market_data.ohlcv_daily
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('1 day', timestamp) AS day,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    avg(vwap) AS vwap
FROM market_data.ohlcv
GROUP BY symbol, day
WITH NO DATA;

-- Add refresh policy
SELECT add_continuous_aggregate_policy('market_data.ohlcv_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');
```

### Data Retention
```sql
-- Add retention policy (keep 2 years of detailed data)
SELECT add_retention_policy('market_data.ohlcv', INTERVAL '2 years');

-- Archive older data to compressed format
CREATE TABLE market_data.ohlcv_archive AS 
SELECT * FROM market_data.ohlcv 
WHERE timestamp < NOW() - INTERVAL '2 years';
```

## Connection Management

### AsyncPG Configuration
```python
# src/db/base.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Create async session factory
async_session = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)
```

### Connection Pool Settings
- **pool_size**: 20 connections
- **max_overflow**: 10 additional connections
- **pool_timeout**: 30 seconds
- **pool_recycle**: 3600 seconds (1 hour)

## Query Optimization

### Common Query Patterns

```sql
-- Latest price for a symbol
SELECT * FROM market_data.ohlcv 
WHERE symbol = 'AAPL' 
ORDER BY timestamp DESC 
LIMIT 1;

-- Daily aggregates with time_bucket
SELECT 
    time_bucket('1 day', timestamp) AS day,
    symbol,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume
FROM market_data.ohlcv
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY day, symbol
ORDER BY day DESC;

-- Gap detection
WITH date_series AS (
    SELECT generate_series(
        DATE '2024-01-01',
        CURRENT_DATE,
        INTERVAL '1 day'
    )::date AS trading_date
),
existing_data AS (
    SELECT DISTINCT DATE(timestamp) as data_date
    FROM market_data.ohlcv
    WHERE symbol = 'AAPL'
)
SELECT trading_date
FROM date_series
LEFT JOIN existing_data ON trading_date = data_date
WHERE data_date IS NULL
  AND EXTRACT(DOW FROM trading_date) NOT IN (0, 6);
```

### Performance Tips

1. **Use time-based queries**: Always include timestamp in WHERE clause
2. **Leverage indexes**: Symbol + timestamp queries are optimized
3. **Use continuous aggregates**: For repeated aggregate queries
4. **Batch inserts**: Use INSERT with multiple VALUES or COPY
5. **Connection pooling**: Reuse connections for better performance

## Backup & Recovery

### Automated Backups
```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR="/backups/postgres"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_NAME="tetra"

# Create backup
docker exec tetra-postgres pg_dump -U postgres -d $DB_NAME | \
    gzip > $BACKUP_DIR/tetra_$TIMESTAMP.sql.gz

# Keep only last 30 days
find $BACKUP_DIR -name "tetra_*.sql.gz" -mtime +30 -delete
```

### Point-in-Time Recovery
```bash
# Enable WAL archiving
ALTER SYSTEM SET wal_level = replica;
ALTER SYSTEM SET archive_mode = on;
ALTER SYSTEM SET archive_command = 'cp %p /archive/%f';
```

### Restore Procedure
```bash
# Restore from backup
gunzip < /backups/tetra_20240101_050000.sql.gz | \
    docker exec -i tetra-postgres psql -U postgres -d tetra
```

## Monitoring

### Key Metrics
```sql
-- Database size
SELECT pg_database_size('tetra') / 1024 / 1024 AS size_mb;

-- Table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Chunk information
SELECT 
    hypertable_name,
    chunk_name,
    pg_size_pretty(total_bytes) AS size,
    range_start,
    range_end
FROM timescaledb_information.chunks
WHERE hypertable_name = 'ohlcv'
ORDER BY range_start DESC;

-- Compression stats
SELECT 
    hypertable_name,
    chunk_name,
    compression_status,
    pg_size_pretty(before_compression_total_bytes) AS original,
    pg_size_pretty(after_compression_total_bytes) AS compressed
FROM timescaledb_information.chunks
WHERE compression_status = 'Compressed';
```

### Health Checks
```python
# Health check endpoint
@router.get("/health/database")
async def database_health():
    try:
        async with get_session() as session:
            result = await session.execute(text("SELECT 1"))
            return {"status": "healthy", "timestamp": datetime.utcnow()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## Security

### User Management
```sql
-- Create read-only user for analytics
CREATE USER tetra_reader WITH PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE tetra TO tetra_reader;
GRANT USAGE ON SCHEMA market_data, economic_data TO tetra_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA market_data, economic_data TO tetra_reader;

-- Create application user with limited permissions
CREATE USER tetra_app WITH PASSWORD 'app_password';
GRANT CONNECT ON DATABASE tetra TO tetra_app;
GRANT USAGE ON ALL SCHEMAS IN DATABASE tetra TO tetra_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN DATABASE tetra TO tetra_app;
```

### Connection Security
- Use SSL for remote connections
- Rotate passwords regularly
- Limit connections by IP (pg_hba.conf)
- Use connection pooling to limit concurrent connections

## Troubleshooting

### Common Issues

1. **Out of shared memory**
   ```sql
   ALTER SYSTEM SET max_locks_per_transaction = 128;
   ALTER SYSTEM SET max_connections = 200;
   ```

2. **Slow queries**
   ```sql
   -- Enable query logging
   ALTER SYSTEM SET log_min_duration_statement = 1000;
   
   -- Check slow queries
   SELECT * FROM pg_stat_statements 
   ORDER BY total_time DESC 
   LIMIT 10;
   ```

3. **Disk space issues**
   ```bash
   # Check chunk sizes
   docker exec tetra-postgres psql -U postgres -d tetra \
     -c "SELECT pg_size_pretty(pg_database_size('tetra'));"
   
   # Manual compression
   SELECT compress_chunk(c) 
   FROM show_chunks('market_data.ohlcv') c 
   WHERE c < NOW() - INTERVAL '7 days';
   ```

## Future Enhancements

1. **Partitioning Strategy**
   - Partition by year for very old data
   - Separate hot/cold storage

2. **Replication**
   - Streaming replication for HA
   - Read replicas for analytics

3. **Performance**
   - Column store for analytics
   - GPU acceleration for aggregates

4. **Integration**
   - Kafka connect for streaming
   - Apache Superset for visualization