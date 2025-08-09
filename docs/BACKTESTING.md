# Backtesting Implementation Details

## Overview
The backtesting system has been fully implemented and integrated with the simulator for data access. The architecture follows the correct flow: **Pipeline → Database → Simulator → Backtester**.

## Architecture

### Data Flow
1. **Data Pipeline** ingests raw market data into the database
2. **Simulator** (via MarketReplay) reads data from the database
3. **Backtesting Engine** uses the simulator to access market data
4. **Strategies** generate trading signals based on market data
5. **Portfolio** tracks positions and calculates P&L

### Key Components

#### 1. Backtesting Engine (`src/backtesting/engine.py`)
- Main orchestrator for running backtests
- Manages portfolio, execution, and metrics calculation
- Supports configurable parameters (commission, slippage, position limits)

#### 2. Data Handler (`src/backtesting/data_handler.py`)
- Completely rewritten to use the simulator's MarketReplay
- Provides synchronous interface to async simulator methods
- Handles data caching and efficient retrieval

#### 3. Portfolio Management (`src/backtesting/portfolio.py`)
- Tracks positions, cash, and portfolio value
- Calculates realized and unrealized P&L
- Enforces risk limits and position sizing

#### 4. Metrics Calculator (`src/backtesting/metrics.py`)
- Calculates comprehensive performance metrics:
  - Returns (total, annualized)
  - Risk metrics (volatility, Sharpe, Sortino, max drawdown)
  - Trade statistics (win rate, profit factor, avg win/loss)
  - Risk-adjusted metrics (Calmar, information ratio)

#### 5. Strategy Base (`src/strategies/base.py`)
- Abstract base class for all strategies
- Defines interface: `should_enter()`, `should_exit()`, `generate_signals()`
- Supports both long and short positions

## Database Structure

### Models Organization
```
src/
├── db/
│   ├── __init__.py
│   └── base.py          # Database connection and session management
└── models/
    ├── pydantic/        # Data validation models
    │   ├── base.py
    │   ├── market_data.py
    │   ├── economic_data.py
    │   ├── event_data.py
    │   └── news_sentiment.py
    └── sqlalchemy/      # ORM models
        ├── market_data.py
        ├── economic_data.py
        ├── event_data.py
        └── news_sentiment.py
```

### Key Database Models
- **OHLCVModel**: Market price data (open, high, low, close, volume)
- **EventDataModel**: Corporate and economic events
- **EconomicIndicatorModel**: Economic indicators and metrics
- **NewsArticleModel/NewsSentimentModel**: News and sentiment data

## Strategy Framework

### Implemented Strategy Types

#### 1. Simple Strategies (`scripts/test_simple_strategies.py`)
- **SimpleTrendFollowing**: Follow price trends
- **SimpleRangeTrading**: Trade within defined ranges
- **SimpleDollarCostAverage**: Regular periodic investments
- **SimpleRotation**: Rotate between top performers

#### 2. Technical Strategies (`scripts/test_technical_strategies.py`)
- **MovingAverageCrossStrategy**: MA crossover signals
- **RSIMeanReversionStrategy**: Trade RSI oversold/overbought
- **BollingerBandStrategy**: Trade band breakouts
- **MACDStrategy**: MACD signal line crossovers
- **CompositeIndicatorStrategy**: Combine multiple indicators

#### 3. Portfolio Strategies (`scripts/test_portfolio_strategies.py`)
- **EqualWeightPortfolioStrategy**: Maintain equal weights
- **MomentumPortfolioStrategy**: Hold top momentum stocks
- **RiskParityStrategy**: Weight by inverse volatility
- **SectorRotationStrategy**: Rotate between sectors
- **DynamicHedgeStrategy**: Dynamic risk management

### Strategy Interface
```python
class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, data, events=None):
        """Generate trading signals from market data"""
        pass
    
    @abstractmethod
    def should_enter(self, symbol, timestamp, bar_data, signals, events):
        """Determine if should enter position"""
        return enter_bool, position_side, position_size
    
    @abstractmethod
    def should_exit(self, position, timestamp, bar_data, signals, events):
        """Determine if should exit position"""
        return exit_bool
```

## Testing Framework

### Test Scripts
1. **`simple_backtest_no_signals.py`**: Basic buy-and-hold test
2. **`test_simple_strategies.py`**: Test simple trading strategies
3. **`test_strategies.py`**: Test momentum and mean reversion strategies
4. **`test_technical_strategies.py`**: Test indicator-based strategies
5. **`test_portfolio_strategies.py`**: Test portfolio-level strategies
6. **`run_all_strategy_tests.py`**: Comprehensive test suite

### Example Results
```
STRATEGY PERFORMANCE SUMMARY
================================================================================
Strategy                      Return     Sharpe      MaxDD     Trades    WinRate
--------------------------------------------------------------------------------
Trend Following (5d)          0.71%       0.68     1.62%          8    37.50%
Range Trading                 1.32%       1.12     1.55%          6    16.67%
Dollar Cost Average           1.72%       1.85     1.01%          0     0.00%
Simple Rotation (Top 2)      -0.92%      -1.45     2.13%          3    66.67%
```

## Performance Optimizations

1. **Data Caching**: MarketReplay caches frequently accessed data
2. **Batch Processing**: Load data in chunks to manage memory
3. **Efficient Calculations**: Vectorized operations for metrics
4. **Lazy Loading**: Only load data when needed

## Configuration

### BacktestConfig Parameters
```python
@dataclass
class BacktestConfig:
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0001  # 0.01%
    min_trade_size: float = 100.0
    max_positions: int = 10
    max_position_pct: float = 0.2  # 20% max per position
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    benchmark: Optional[str] = "SPY"
    risk_free_rate: float = 0.02  # 2% annual
    data_frequency: str = "1d"
    warmup_periods: int = 100
    calculate_metrics_every: int = 20
```

## Integration with Simulator

### MarketReplay Integration
The DataHandler now uses the simulator's MarketReplay class:
- Async methods wrapped for sync backtesting
- Efficient data loading with preload option
- Support for multiple timeframes
- Automatic handling of trading calendar

### Data Access Pattern
```python
# DataHandler internally uses:
loop.run_until_complete(
    self.market_replay.load_data(symbols, start_date, end_date, preload=True)
)

# Then retrieves data:
day_data = loop.run_until_complete(
    self.market_replay.get_market_data(symbols, current_date)
)
```

## Future Enhancements

1. **Multi-timeframe Support**: Allow strategies to use multiple timeframes
2. **Options Support**: Add options trading capabilities
3. **Portfolio Optimization**: Implement mean-variance optimization
4. **Machine Learning Integration**: Support ML-based strategies
5. **Real-time Paper Trading**: Connect to live data feeds
6. **Advanced Risk Management**: VaR, CVaR, stress testing
7. **Strategy Parameter Optimization**: Grid search, genetic algorithms
8. **Event-driven Backtesting**: Support for event-based strategies

## Usage Example

```python
from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.strategies.base import BaseStrategy, PositionSide

# Define strategy
class MyStrategy(BaseStrategy):
    def generate_signals(self, data, events=None):
        return pd.DataFrame()
    
    def should_enter(self, symbol, timestamp, bar_data, signals, events):
        # Strategy logic
        return True, PositionSide.LONG, 0.1
    
    def should_exit(self, position, timestamp, bar_data, signals, events):
        # Exit logic
        return position.unrealized_pnl > 100

# Configure and run
config = BacktestConfig(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 31),
    initial_capital=100000
)

engine = BacktestEngine(config)
report = engine.run(
    strategy=MyStrategy,
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    signal_computer=None
)

print(f"Total Return: {report.total_return:.2%}")
print(f"Sharpe Ratio: {report.sharpe_ratio:.2f}")
```