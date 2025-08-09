# Frontend Architecture and Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Database Monitor & LLM Chat](#database-monitor--llm-chat)
5. [Strategies Tab Integration](#strategies-tab-integration)
6. [Connection Troubleshooting](#connection-troubleshooting)
7. [Development Guide](#development-guide)

---

## Overview

The Tetra frontend is a modern web application built with Vue.js 3 and Tailwind CSS, providing:
- Real-time database monitoring and coverage visualization
- Natural language SQL queries through LLM integration
- Trading strategy performance analysis and visualization
- Interactive dashboards for market data exploration

### Technology Stack
- **Framework**: Vue.js 3 with Composition API
- **Styling**: Tailwind CSS
- **Build Tool**: Vite
- **State Management**: Pinia (planned)
- **Charts**: Chart.js
- **API Client**: Axios
- **WebSocket**: Native WebSocket for real-time updates

---

## Architecture

### Directory Structure
```
frontend/
├── src/
│   ├── views/
│   │   ├── DataMonitor.vue          # Main dashboard with coverage cards
│   │   ├── DatabaseChat.vue         # LLM chat interface
│   │   └── StrategiesEnhanced.vue   # Strategy analysis dashboard
│   ├── components/
│   │   ├── CoverageCard.vue         # Individual data topic cards
│   │   ├── SymbolTable.vue          # Detailed symbol coverage
│   │   ├── ChatInterface.vue        # Chat UI component
│   │   ├── SqlDisplay.vue           # SQL syntax highlighting
│   │   └── NavBar.vue               # Navigation component
│   ├── services/
│   │   ├── api.js                   # Axios API client
│   │   └── websocket.js             # Real-time updates
│   ├── router/
│   │   └── index.js                 # Vue Router config
│   └── App.vue                      # Root component
├── vite.config.js                   # Vite configuration
├── package.json                     # Dependencies
└── tailwind.config.js              # Tailwind CSS config
```

### Data Flow
```
Frontend Components
       ↓
   Axios API Client
       ↓
Backend FastAPI (port 8000)
       ↓
PostgreSQL/TimescaleDB
```

### Configuration
Frontend runs on port 3000 (configured in vite.config.js):
```javascript
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 3000,
    strictPort: true,
    host: '0.0.0.0'
  }
})
```

---

## Components

### Core Components

#### 1. DataMonitor.vue
Main dashboard showing data coverage statistics:
- Market Data coverage (symbols, date ranges, record count)
- Economic Data indicators
- News & Sentiment analysis
- Events tracking

#### 2. DatabaseChat.vue
Natural language interface to query the database:
- Converts natural language to SQL using GPT-4
- Displays generated SQL with syntax highlighting
- Shows query results in table format
- Maintains query history in localStorage

#### 3. StrategiesEnhanced.vue
Comprehensive strategy analysis dashboard with:
- Performance heatmap grid
- Time series analysis
- Risk metrics
- Market scenario testing
- Portfolio recommendations

---

## Database Monitor & LLM Chat

### Implementation Status ✅

#### Frontend Features
1. **Real-time Coverage Cards**
   - Market Data: 365,836 records, 153 symbols (2015-2025)
   - Economic Data: 50,363 records, 15 indicators
   - News: 150,000+ articles with sentiment
   - Events: 50,000+ financial events

2. **LLM Chat Interface**
   - Natural language to SQL conversion
   - SQL syntax highlighting with Prism.js
   - Query execution with results display
   - Table/chart visualization toggle
   - Query history in localStorage
   - Schema context provided to LLM

3. **WebSocket Integration**
   - Real-time updates for coverage statistics
   - Live query status updates
   - Connection status indicator

#### Backend Integration

**API Endpoints**:
```python
# Monitor endpoints
GET /api/monitor/coverage          # Overall data coverage
GET /api/monitor/schemas           # Database schema info
GET /api/monitor/stats/{schema}    # Schema statistics
GET /api/monitor/symbols/{schema}  # Symbol-level details

# Chat endpoints
POST /api/chat/query              # Convert NL to SQL
POST /api/chat/execute            # Execute SQL query
GET  /api/chat/schema-context     # Get schema for LLM

# WebSocket
WS /ws/monitor                    # Real-time updates
```

#### Safety & Performance
- Read-only database connections
- SQL injection prevention
- Query timeout limits (30s)
- Result size limits (1000 rows)
- Connection pooling with asyncpg

---

## Strategies Tab Integration

### Overview
The enhanced Strategies tab provides comprehensive strategy performance analysis with:

1. **Performance Grid (Heatmap)**
   - Shows strategy performance across different symbols
   - Configurable metrics: Return, Sharpe Ratio, Max Drawdown, Total Trades
   - Color-coded cells for quick visual analysis
   - Interactive tooltips with detailed information

2. **Time Series Analysis**
   - Rolling window performance visualization
   - Multi-strategy comparison charts
   - Cumulative return tracking
   - Statistical metrics for each time period

3. **Risk Analysis**
   - Comprehensive risk metrics table
   - Risk-return scatter plot
   - Drawdown analysis
   - Recovery time tracking
   - VaR and CVaR calculations

4. **Market Scenario Testing**
   - Performance under different market conditions
   - Bull/Bear market analysis
   - Volatility and crash scenarios
   - Strategy rankings by scenario

5. **Portfolio Recommendations**
   - Conservative, Balanced, and Aggressive portfolios
   - Optimal strategy allocation percentages
   - Expected returns and risk metrics
   - Implementation guidelines

### Component Structure
```vue
<template>
  <!-- Header with controls -->
  <div class="header">
    <select v-model="selectedTimeWindow"> <!-- Time window selector -->
    <select v-model="selectedSymbol">      <!-- Symbol filter -->
    <button @click="runAnalysis">          <!-- Run new analysis -->
  </div>

  <!-- Performance overview cards -->
  <div class="overview-section">
    <!-- Top performer, Most consistent, Best risk-adjusted, etc. -->
  </div>

  <!-- Tab navigation -->
  <div class="tab-navigation">
    <!-- Performance Grid, Time Series, Risk, Scenarios, Recommendations -->
  </div>

  <!-- Tab content -->
  <div class="tab-content">
    <!-- Dynamic content based on active tab -->
  </div>
</template>
```

### API Integration

**Key endpoints**:
1. **GET /api/strategies/analysis** - Comprehensive analysis data
2. **GET /api/strategies/performance-grid** - Performance heatmap data
3. **GET /api/strategies/time-series/{strategy}** - Time series data
4. **GET /api/strategies/risk-metrics** - Risk analysis metrics
5. **GET /api/strategies/scenario-analysis** - Market scenario performance
6. **POST /api/strategies/run-analysis** - Trigger new analysis

### Data Structure Examples

**Performance Grid Data**:
```json
{
  "grid": {
    "buy_and_hold": {
      "SPY": {
        "return": 0.234,
        "sharpe": 1.07,
        "drawdown": -0.187,
        "trades": 0
      }
    }
  },
  "symbols": ["SPY", "QQQ", "IWM"],
  "strategies": ["buy_and_hold", "momentum_factor"]
}
```

**Time Series Data**:
```json
{
  "time_series": {
    "dates": ["2024-01-01", "2024-01-08", ...],
    "returns": [0.012, -0.005, 0.023, ...],
    "sharpe": [1.2, 0.8, 1.5, ...],
    "cumulative_return": [1.012, 1.007, 1.030, ...]
  },
  "rolling_stats": {
    "avg_return": 0.015,
    "volatility": 0.18,
    "best_period": 0.054,
    "worst_period": -0.032
  }
}
```

---

## Connection Troubleshooting

### Common Issues and Solutions

#### 1. "Error loading data - Network Error"

This is typically a CORS issue where the backend rejects requests from the frontend.

**Quick Fix Steps**:

1. **Check frontend port**:
   ```bash
   # Look at the Vite output or browser URL
   # Common ports: 3000, 5173, 5174, 5175, 5176, 5177
   ```

2. **Update backend CORS configuration**:
   
   Edit `/backend/.env`:
   ```env
   # CORS Origins - Add ALL possible frontend ports
   CORS_ORIGINS=["http://localhost:3000","http://localhost:5173","http://localhost:5174","http://localhost:5175","http://localhost:5176","http://localhost:5177"]
   ```

   Also update `/backend/app/config.py`:
   ```python
   # CORS
   cors_origins: list[str] = [
       "http://localhost:5173",  # Vue dev server
       "http://localhost:5174",  # Vue preview
       "http://localhost:3000",  # Alternative frontend
       "http://localhost:5175",  # Additional Vite ports
       "http://localhost:5176",
       "http://localhost:5177",
   ]
   ```

3. **Restart backend completely**:
   ```bash
   # Kill existing backend
   pkill -f "uvicorn app.main:app"
   
   # Start backend again
   cd /Users/angwei/Repos/tetra/backend
   source venv/bin/activate
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Hard refresh frontend**:
   - Mac: Cmd + Shift + R
   - Windows/Linux: Ctrl + Shift + R

#### 2. Database Connection Issues

1. **Check PostgreSQL container**:
   ```bash
   docker ps | grep postgres
   # Should show: tetra-postgres container
   ```

2. **If not running, start it**:
   ```bash
   docker-compose up -d postgres
   ```

3. **Verify connection**:
   ```bash
   PGPASSWORD=tetra_password psql -h localhost -p 5432 -U tetra_user -d tetra -c "SELECT current_database();"
   ```

#### 3. Frontend Shows Blank/White Screen

1. **Check App.vue background**:
   ```vue
   <!-- Should be dark for current theme -->
   <div id="app" class="min-h-screen bg-gray-900">
   ```

2. **Update NavBar.vue to match**:
   ```vue
   <nav class="bg-gray-800 shadow-lg border-b border-gray-700">
   ```

#### 4. Multiple Vite Processes

Kill all Vite processes and restart:
```bash
pkill -f "vite"
# Or kill specific ports:
for port in 3000 5173 5174 5175 5176 5177; do 
  lsof -ti:$port | xargs kill -9 2>/dev/null || true
done
```

### Complete Service Restart

When nothing else works:
```bash
# 1. Stop all services
tetra-stop  # or manually:
pkill -f "uvicorn"
pkill -f "vite"

# 2. Clear port conflicts
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# 3. Ensure database is running
docker ps | grep postgres || docker-compose up -d postgres

# 4. Start services
tetra  # or manually:
/Users/angwei/Repos/tetra/bin/launch_services.sh

# 5. Verify services
curl http://localhost:8000/health  # Should return JSON
curl http://localhost:3000         # Should return HTML
```

### Debugging Tools

**Check Backend Logs**:
```bash
tail -f /tmp/tetra-backend.log
```

**Check Frontend Logs**:
```bash
tail -f /tmp/tetra-frontend.log
```

**Test CORS Configuration**:
```bash
# Test preflight request
curl -X OPTIONS \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: GET" \
  -H "Access-Control-Request-Headers: Content-Type" \
  -i http://localhost:8000/api/monitor/coverage

# Should see: access-control-allow-origin: http://localhost:3000
```

**Browser Developer Tools**:
1. Open Developer Tools (F12)
2. Go to Network tab
3. Look for failed requests (red)
4. Check Console for JavaScript errors
5. In failed requests, check Response Headers for CORS errors

---

## Development Guide

### Setup

1. **Install Dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Start Development Server**:
   ```bash
   npm run dev
   # Runs on http://localhost:3000
   ```

3. **Build for Production**:
   ```bash
   npm run build
   npm run preview
   ```

### Adding New Components

1. **Create Component File**:
   ```vue
   <!-- src/components/MyComponent.vue -->
   <template>
     <div class="my-component">
       <!-- Component template -->
     </div>
   </template>

   <script setup>
   import { ref, computed } from 'vue'
   // Component logic
   </script>

   <style scoped>
   /* Component styles */
   </style>
   ```

2. **Register Route** (if needed):
   ```javascript
   // src/router/index.js
   {
     path: '/my-route',
     name: 'MyRoute',
     component: () => import('../views/MyView.vue')
   }
   ```

3. **Add API Integration**:
   ```javascript
   // src/services/api.js
   export const myApiCall = async (params) => {
     const response = await api.get('/api/my-endpoint', { params })
     return response.data
   }
   ```

### Style Guidelines

- Use Tailwind CSS utility classes
- Dark theme by default (bg-gray-900, text-gray-100)
- Consistent spacing (p-4, m-2, etc.)
- Responsive design (sm:, md:, lg: breakpoints)
- Interactive states (hover:, focus:, active:)

### Performance Considerations

1. **Lazy Loading**: Use dynamic imports for large components
2. **Caching**: API responses cached for 1 hour where appropriate
3. **Pagination**: Large datasets are paginated
4. **WebSocket**: Use for real-time updates instead of polling
5. **Chart Optimization**: Limit data points for smooth rendering

### Testing

```bash
# Run unit tests
npm run test:unit

# Run e2e tests
npm run test:e2e

# Run linter
npm run lint
```

### Deployment

The frontend is deployed alongside the backend using Docker:
```dockerfile
# Build stage
FROM node:18 as build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
```

### Quick Reference

**Service URLs**:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Database: postgresql://tetra_user:tetra_password@localhost:5432/tetra

**Key Files**:
- Backend config: `/backend/.env`, `/backend/app/config.py`
- Frontend config: `/frontend/vite.config.js`
- API service: `/frontend/src/services/api.js`
- Launch script: `/bin/launch_services.sh`

**Aliases** (after sourcing ~/.zshrc):
- `tetra` - Launch all services
- `tetra-stop` - Stop all services
- `tetra-restart` - Restart all services
- `tetra-logs` - View logs
- `tetra-status` - Check running processes

### Next Steps

1. **Add Export Features**:
   - CSV/Excel export for grids
   - PDF reports generation
   - Chart image downloads

2. **Enhanced Visualizations**:
   - 3D surface plots for multi-dimensional analysis
   - Correlation matrices
   - Monte Carlo simulations

3. **Real-time Updates**:
   - WebSocket integration for live strategy performance
   - Automatic refresh on new data
   - Push notifications for alerts

4. **User Customization**:
   - Save custom analysis configurations
   - Create strategy watchlists
   - Set performance alerts