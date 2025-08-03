# Tetra - Comprehensive Quantitative Trading Platform

## Project Vision
Build a comprehensive quantitative trading platform that combines data ingestion, market simulation, strategy development, machine learning, and automated execution into a unified system.

## Quick Links
- [Database Infrastructure](DATABASE.md)
- [Data Pipeline](DATA_PIPELINE.md) 
- [Web GUI & Monitoring](WEBGUI.md)
- [Market Simulator](SIMULATOR.md)
- [Trading Strategies](STRATEGY.md)
- [Machine Learning](ML_MODELS.md)
- [Reporting System](REPORTING.md)
- [Trade Execution](EXECUTION.md)
- [Risk Management](RISK_MANAGEMENT.md)
- [System Architecture](ARCHITECTURE.md)
- [Deployment Guide](DEPLOYMENT.md)

## Implementation Status

### ✅ Completed Components

| Component | Description | Key Features |
|-----------|-------------|--------------|
| **Database Infrastructure** | PostgreSQL + TimescaleDB in Docker | • Time-series optimization<br>• Automatic partitioning<br>• Compression policies |
| **Data Pipeline** | Multi-source data ingestion | • 153 symbols tracked<br>• 10 years history<br>• Daily updates at 5 AM |
| **Market Data** | Polygon.io & YFinance integration | • OHLCV data<br>• Volume scaling fixed<br>• Gap detection |
| **Economic Data** | FRED API integration | • 58 indicators<br>• GDP, CPI, unemployment<br>• Auto-updated |
| **News & Sentiment** | NewsAPI + AlphaVantage | • Real-time news<br>• Sentiment analysis<br>• Symbol tagging |
| **Web Dashboard** | Vue.js monitoring interface | • Coverage visualization<br>• Real-time updates<br>• WebSocket support |
| **LLM Chat** | Natural language SQL queries | • GPT-4 integration<br>• Schema-aware<br>• Safe query execution |
| **Automation** | Daily pipeline scheduling | • macOS launchd<br>• Error handling<br>• Logging |

### 🚧 In Progress

| Component | Description | Current Status |
|-----------|-------------|----------------|
| **Historical Simulator** | Replay market conditions | Design phase - architecture planned |
| **Strategy Engine** | Trading strategy framework | Requirements gathering |
| **ML Models** | Predictive analytics | Data preparation phase |

### 📋 Planned Components

| Component | Description | Priority |
|-----------|-------------|----------|
| **Stochastic Simulator** | Monte Carlo simulations | High |
| **Event Simulator** | Market shock modeling | High |
| **Risk Management** | Portfolio risk analytics | High |
| **Execution Module** | Broker integration | Medium |
| **Reporting System** | SOD/EOD reports | Medium |
| **Mobile App** | iOS/Android monitoring | Low |
| **Crypto Integration** | Binance, CoinGecko APIs | Low |

## System Metrics

- **Data Volume**: 500K+ OHLCV records
- **Symbols Tracked**: 153 (ETFs, stocks, indices)
- **Economic Indicators**: 58 from FRED
- **Update Frequency**: Daily at 5 AM ET
- **Historical Coverage**: 10 years (2015-2025)
- **Database Size**: ~2GB compressed

## Technology Stack

| Layer | Technology | Status |
|-------|------------|--------|
| Database | PostgreSQL 15 + TimescaleDB | ✅ Production |
| Backend API | FastAPI + SQLAlchemy | ✅ Production |
| Data Pipeline | Python async/await | ✅ Production |
| Frontend | Vue.js 3 + Tailwind CSS | ✅ Production |
| LLM Integration | OpenAI GPT-4 | ✅ Production |
| Scheduling | macOS launchd | ✅ Production |
| ML Framework | scikit-learn, TensorFlow | 📋 Planned |
| Message Queue | Kafka | 📋 Planned |
| Monitoring | Prometheus + Grafana | 📋 Planned |

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Sources   │     │   Simulators    │     │    Strategies   │
│ • Polygon.io    │     │ • Historical    │     │ • Pairs Trading │
│ • FRED          │────▶│ • Stochastic    │────▶│ • Momentum      │
│ • NewsAPI       │     │ • Event-based   │     │ • Mean Reversion│
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TimescaleDB Database                         │
│  • Market Data  • Economic Data  • Events  • Strategies         │
└─────────────────────────────────────────────────────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   FastAPI Backend    │     │   ML Models     │     │   Execution     │
│ • REST APIs          │     │ • Predictions   │     │ • IB Gateway    │
│ • WebSocket          │────▶│ • Optimization  │────▶│ • Order Mgmt    │
│ • LLM Integration    │     │ • Backtesting   │     │ • Risk Checks   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                                │
         ▼                                                ▼
┌─────────────────┐                              ┌─────────────────┐
│   Vue.js GUI    │                              │    Reports      │
│ • Monitoring    │                              │ • SOD/EOD       │
│ • Chat Interface│                              │ • Performance   │
│ • Visualizations│                              │ • Risk Metrics  │
└─────────────────┘                              └─────────────────┘
```

## Project Timeline

### Phase 1: Foundation (Completed ✅)
- Database setup with TimescaleDB
- Data pipeline for market/economic data
- Basic web interface with monitoring
- LLM chat integration

### Phase 2: Simulation & Strategy (Current 🚧)
- Historical market simulator
- Trading strategy framework
- Backtesting engine
- Performance analytics

### Phase 3: Intelligence (Q3 2025)
- ML prediction models
- Strategy optimization
- Risk management system
- Advanced analytics

### Phase 4: Execution (Q4 2025)
- Broker integration
- Automated trading
- Real-time monitoring
- Production deployment

## Key Design Decisions

1. **TimescaleDB over vanilla PostgreSQL**: Superior time-series performance with automatic partitioning
2. **FastAPI + Vue.js**: Modern async Python backend with reactive frontend
3. **Separate schemas**: Logical data separation for maintainability
4. **launchd over cron**: Better macOS integration and reliability
5. **GPT-4 for SQL**: Natural language interface for complex queries
6. **Modular pipeline**: Easy to extend with new data sources

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/awei-git/tetra.git
   cd tetra
   ```

2. **Start the database**
   ```bash
   docker-compose up -d postgres
   ```

3. **Run the data pipeline**
   ```bash
   python scripts/daily_update.py
   ```

4. **Start the backend**
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

5. **Start the frontend**
   ```bash
   cd frontend
   npm run dev
   ```

## Contributing

See [ARCHITECTURE.md](ARCHITECTURE.md) for system design details and contribution guidelines.