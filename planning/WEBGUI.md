# Web GUI & LLM Chat Documentation

## Overview
The Tetra Web GUI provides a modern interface for monitoring data coverage, exploring the database through natural language queries, and visualizing financial data. Built with Vue.js 3 and integrated with GPT-4 for intelligent query generation.

## Architecture

### Frontend Stack
- **Framework**: Vue.js 3 with Composition API
- **Styling**: Tailwind CSS
- **State Management**: Pinia
- **HTTP Client**: Axios
- **WebSocket**: Native WebSocket API
- **Charts**: Chart.js (planned)
- **Code Highlighting**: Prism.js

### Backend Stack
- **Framework**: FastAPI
- **Database**: AsyncPG for PostgreSQL
- **LLM**: OpenAI GPT-4
- **WebSocket**: FastAPI WebSocket
- **CORS**: Enabled for development

## Project Structure

```
frontend/
├── src/
│   ├── views/
│   │   ├── DataMonitor.vue      # Coverage dashboard
│   │   └── DatabaseChat.vue     # LLM chat interface
│   ├── components/
│   │   ├── CoverageCard.vue     # Data source cards
│   │   ├── SymbolModal.vue      # Symbol details modal
│   │   ├── ChatMessage.vue      # Chat message component
│   │   └── QueryResults.vue     # SQL results table
│   ├── services/
│   │   ├── api.js               # API client setup
│   │   └── websocket.js         # WebSocket connection
│   ├── router/
│   │   └── index.js             # Vue Router config
│   └── assets/
│       └── styles/
│           └── main.css         # Global styles

backend/
├── app/
│   ├── routers/
│   │   ├── monitor.py           # Coverage endpoints
│   │   └── chat.py              # LLM chat endpoints
│   ├── services/
│   │   ├── database.py          # DB connection pool
│   │   ├── llm.py               # LLM integration
│   │   ├── query_executor.py    # Safe SQL execution
│   │   └── schema_registry.py   # DB schema for LLM
│   └── main.py                  # FastAPI app
```

## Key Features

### 1. Data Coverage Monitor

**Real-time Dashboard**
- Coverage cards for each data source
- Symbol count and date ranges
- Record counts with formatting
- Quality indicators (good/fair/poor)
- WebSocket updates every 30 seconds

**Symbol Details Modal**
- Drill down to individual symbols
- Coverage quality metrics
- Missing days calculation
- Trading days vs calendar days
- Data source attribution

**Implementation**
```vue
<!-- DataMonitor.vue -->
<template>
  <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
    <CoverageCard 
      v-for="topic in topics" 
      :key="topic.name"
      :topic="topic"
      @show-details="showSymbolDetails"
    />
  </div>
</template>

<script setup>
const fetchCoverage = async () => {
  const data = await monitorAPI.getCoverage()
  topics.value = transformToTopics(data)
}

// WebSocket for real-time updates
const ws = createWebSocket((data) => {
  if (data.type === 'coverage_update') {
    updateTopic(data.topic, data.update)
  }
})
</script>
```

### 2. LLM Chat Interface

**Natural Language Queries**
- Convert questions to SQL using GPT-4
- Schema-aware query generation
- SQL syntax highlighting
- Safe query execution (read-only)
- Results in table format

**Query Examples**
- "Show me AAPL price last 30 days"
- "What's the GDP growth rate?"
- "Compare SPY and QQQ performance"
- "Find stocks with highest volume today"

**Safety Features**
- Only SELECT queries allowed
- 30-second timeout
- 1000 row limit
- No sensitive data exposure

**Implementation**
```javascript
// Chat query flow
const handleQuery = async (query) => {
  // 1. Send to LLM for SQL generation
  const response = await chatAPI.sendQuery(query)
  
  // 2. Display generated SQL
  showSQL(response.sql)
  
  // 3. Execute query
  const results = await chatAPI.executeQuery(response.sql)
  
  // 4. Display results
  showResults(results.data, results.columns)
  
  // 5. Save to history
  saveToHistory(query, response.sql, results)
}
```

### 3. Schema Registry

**Purpose**: Provide database context to LLM for accurate query generation

**Schema Information**
```python
# schema_registry.py
SCHEMAS = {
    "market_data": {
        "ohlcv": {
            "description": "Price and volume data",
            "columns": {
                "symbol": "Stock ticker",
                "timestamp": "Date/time of data",
                "open": "Opening price",
                "high": "Highest price",
                "low": "Lowest price", 
                "close": "Closing price",
                "volume": "Trading volume"
            }
        }
    },
    "economic_data": {
        "economic_data": {
            "description": "Economic indicators",
            "columns": {
                "symbol": "Indicator code (e.g., GDPC1)",
                "date": "Date of data point",
                "value": "Indicator value"
            }
        }
    }
}
```

**LLM Context**
```python
def get_schema_context() -> str:
    """Format schema for LLM understanding"""
    context = []
    for schema, tables in SCHEMAS.items():
        for table, info in tables.items():
            context.append(f"{schema}.{table}: {info['description']}")
            for col, desc in info['columns'].items():
                context.append(f"  - {col}: {desc}")
    return "\n".join(context)
```

## API Endpoints

### Monitor API

```python
# GET /api/monitor/coverage
# Returns overall coverage statistics
{
    "market_data": {
        "record_count": 500000,
        "date_range": {"start": "2015-01-01", "end": "2025-08-01"},
        "symbols": 153,
        "status": "healthy"
    },
    "economic_data": {...},
    "news": {...},
    "events": {...}
}

# GET /api/monitor/symbols/market_data
# Returns symbol-level details
{
    "symbols": [
        {
            "symbol": "AAPL",
            "record_count": 2500,
            "date_range": {...},
            "coverage_quality": "good",
            "missing_days": 5
        }
    ]
}
```

### Chat API

```python
# POST /api/chat/query
# Request:
{
    "query": "Show AAPL price last 30 days"
}

# Response:
{
    "sql": "SELECT * FROM market_data.ohlcv WHERE symbol = 'AAPL' AND timestamp >= CURRENT_DATE - INTERVAL '30 days' ORDER BY timestamp DESC",
    "data": [...],
    "columns": ["symbol", "timestamp", "open", "high", "low", "close", "volume"],
    "analysis": "AAPL has shown strong performance...",
    "row_count": 22
}
```

## Deployment

### Development Mode

```bash
# Frontend (port 5173)
cd frontend
npm install
npm run dev

# Backend (port 8000)
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Production Deployment

**Frontend Build**
```bash
npm run build
# Outputs to dist/ folder
# Serve with nginx or similar
```

**Backend with Gunicorn**
```bash
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

**Nginx Configuration**
```nginx
server {
    listen 80;
    server_name tetra.example.com;

    # Frontend
    location / {
        root /var/www/tetra/dist;
        try_files $uri $uri/ /index.html;
    }

    # API proxy
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # WebSocket proxy
    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Security Considerations

### Frontend Security
- Input sanitization for queries
- XSS prevention with Vue.js
- HTTPS in production
- Content Security Policy headers

### Backend Security
- SQL injection prevention
- Read-only database user
- API rate limiting
- CORS configuration
- Authentication (planned)

### LLM Security
- Schema filtering (no sensitive tables)
- Query validation before execution
- Result size limits
- No system table access

## Performance Optimization

### Frontend Performance
- Lazy loading for routes
- Virtual scrolling for large tables
- Debounced search inputs
- Memoized computed properties
- WebSocket reconnection logic

### Backend Performance
- Connection pooling
- Query result caching (planned)
- Async database operations
- Batch API requests
- CDN for static assets

## User Experience

### Responsive Design
- Mobile-first approach
- Breakpoints: sm (640px), md (768px), lg (1024px)
- Touch-friendly interfaces
- Adaptive layouts

### Dark Mode
- System preference detection
- Manual toggle
- Consistent color scheme
- High contrast for readability

### Accessibility
- ARIA labels
- Keyboard navigation
- Screen reader support
- Color contrast compliance

## Future Enhancements

### 1. Advanced Visualizations
- Time-series charts with Chart.js
- Candlestick charts for OHLCV
- Correlation heatmaps
- Portfolio performance graphs

### 2. Query Builder
- Visual query builder
- Drag-and-drop interface
- Pre-built query templates
- Query sharing

### 3. Local LLM Integration
- Ollama integration for privacy
- On-premise model deployment
- Hybrid cloud/local approach
- Custom model fine-tuning

### 4. Export Features
- CSV/Excel export
- PDF reports
- API for external tools
- Scheduled exports

### 5. Collaboration
- Shared queries
- Comments on results
- Team workspaces
- Audit trails

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Check CORS settings
   - Verify WebSocket proxy
   - Check firewall rules

2. **LLM Query Generation Errors**
   - Verify OpenAI API key
   - Check rate limits
   - Review schema registry

3. **Large Result Sets**
   - Implement pagination
   - Add result streaming
   - Increase timeouts

### Debug Tools

```javascript
// Enable debug logging
localStorage.setItem('debug', 'true')

// Check WebSocket status
console.log(ws.readyState)

// Monitor API calls
window.axios.interceptors.request.use(config => {
  console.log('API Request:', config)
  return config
})
```

## Development Guidelines

### Code Style
- Vue 3 Composition API
- TypeScript (migration planned)
- ESLint + Prettier
- Conventional commits

### Testing
- Vitest for unit tests
- Cypress for E2E tests
- Mock API responses
- Visual regression tests

### Documentation
- Component documentation
- API documentation
- User guides
- Video tutorials