# Tetra WebGUI Backend

FastAPI backend for the Tetra WebGUI, providing database monitoring and LLM-powered natural language SQL queries.

## Features

- **Database Monitoring API**: Real-time coverage statistics across all schemas
- **LLM-Powered SQL Generation**: Convert natural language to SQL using Claude
- **Query Validation**: Ensures only safe SELECT queries are executed
- **WebSocket Support**: Real-time updates for monitoring dashboard
- **Schema Registry**: Provides context to LLM for accurate SQL generation

## Setup

1. **Install dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your database and API credentials
```

3. **Run the server**:
```bash
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

### Monitor Endpoints
- `GET /api/monitor/coverage` - Get data coverage for all schemas
- `GET /api/monitor/schemas` - Get database schema information
- `GET /api/monitor/stats/{schema}` - Get detailed stats for a schema

### Chat Endpoints
- `POST /api/chat/query` - Process natural language query
- `GET /api/chat/history` - Get query history
- `POST /api/chat/save` - Save query to history
- `POST /api/chat/validate` - Validate SQL without executing

### WebSocket
- `WS /ws/monitor` - Real-time monitoring updates

## Environment Variables

- `DATABASE_URL`: PostgreSQL connection string
- `ANTHROPIC_API_KEY`: Claude API key for SQL generation
- `LLM_PROVIDER`: LLM provider (anthropic or openai)
- `LLM_MODEL`: Model to use for SQL generation
- `SECRET_KEY`: Secret key for JWT tokens

## Development

```bash
# Run tests
pytest

# Format code
black app/

# Type checking
mypy app/
```

## Architecture

```
app/
├── routers/        # API endpoints
├── services/       # Business logic
├── models/         # Pydantic models
└── config.py       # Configuration
```