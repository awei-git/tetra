# Tetra Trading Platform

A comprehensive quantitative trading platform built with Python, FastAPI, PostgreSQL/TimescaleDB, and Kafka.

## Features

- **Market Data Ingestion**: Automated data collection from multiple providers (Polygon.io, Finnhub, etc.)
- **Time-Series Storage**: Optimized storage using TimescaleDB for historical market data
- **RESTful API**: FastAPI-based API for data access and strategy management
- **Event Processing**: Kafka-based event streaming for real-time data
- **Extensible Architecture**: Easy to add new data sources and strategies

## Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Make (optional, for convenience commands)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tetra.git
cd tetra
```

2. Create and configure secrets:
```bash
cp config/secrets.yml.template config/secrets.yml
# Edit config/secrets.yml with your API keys
```

3. Install dependencies:
```bash
make install
# or
pip install -r requirements.txt
```

4. Start the infrastructure:
```bash
make db-up
# or
docker-compose up -d postgres redis kafka zookeeper
```

5. Run database migrations:
```bash
# First time only - create the migration
./scripts/create_migration.sh

# Apply migrations
make migrate
# or
alembic upgrade head

# Initialize TimescaleDB features
python scripts/init_db.py
```

6. Start the API server:
```bash
make run
# or
uvicorn src.api.main:app --reload
```

The API will be available at http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Kafka UI: http://localhost:8080

## Project Structure

```
tetra/
├── src/
│   ├── api/            # FastAPI application and routers
│   ├── clients/        # External API clients
│   ├── db/            # Database models and connections
│   ├── ingestion/     # Data ingestion pipeline
│   ├── models/        # Pydantic models for validation
│   └── utils/         # Utility functions
├── config/            # Configuration files
├── docker/            # Docker configurations
├── alembic/           # Database migrations
├── scripts/           # Utility scripts
└── tests/             # Test suite
```

## Usage

### Ingesting Market Data

```python
from src.ingestion.data_ingester import DataIngester

# Create ingester
ingester = DataIngester(provider="polygon")

# Backfill historical data
symbols = ["AAPL", "MSFT", "GOOGL"]
stats = await ingester.backfill_historical_data(
    symbols=symbols,
    days_back=30,
    timeframe="1d"
)
```

### API Endpoints

- `GET /api/v1/health` - Health check
- `GET /api/v1/market-data/ohlcv/{symbol}` - Get OHLCV data
- `GET /api/v1/market-data/latest/{symbol}` - Get latest price
- `GET /api/v1/events` - Get market events

### Running the Scheduler

To run automated data ingestion:

```python
from src.ingestion.scheduler import IngestionScheduler

scheduler = IngestionScheduler()
scheduler.start()
```

## Development

### Running Tests
```bash
make test
# or
pytest tests/ -v
```

### Code Formatting
```bash
make format
# or
black src/ tests/
```

### Linting
```bash
make lint
# or
ruff check src/ tests/
```

## Configuration

Configuration is managed through:
- `config/config.py` - Application settings
- `config/secrets.yml` - API keys and sensitive data (gitignored)
- Environment variables - Override any setting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

[Your License Here]