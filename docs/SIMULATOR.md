# Market Simulator Documentation

## Overview
The Tetra Market Simulator provides a framework for backtesting trading strategies using historical data, with plans to extend to stochastic and event-based simulations. The simulator can replay market conditions from specific time periods or events, track portfolio performance, and generate comprehensive metrics.

## Architecture

### Module Structure
```
src/simulators/
├── __init__.py
├── base.py                      # Abstract base classes
│   ├── BaseSimulator           # Common simulator interface
│   ├── SimulationResult        # Standardized results
│   └── SimulationConfig        # Configuration
│
├── historical/
│   ├── __init__.py
│   ├── simulator.py            # Historical market replay
│   ├── event_periods.py        # Pre-defined events
│   └── market_replay.py        # Data replay engine
│
├── portfolio/
│   ├── __init__.py
│   ├── portfolio.py            # Portfolio management
│   ├── position.py             # Individual positions
│   ├── transaction.py          # Trade records
│   └── cash_manager.py         # Cash and margin
│
├── metrics/
│   ├── __init__.py
│   ├── performance.py          # Return calculations
│   ├── risk.py                 # Risk metrics
│   └── attribution.py          # Performance attribution
│
├── events/
│   ├── __init__.py
│   ├── corporate_actions.py    # Dividends, splits
│   └── market_events.py        # Halts, circuit breakers
│
└── utils/
    ├── __init__.py
    ├── data_loader.py          # Efficient data loading
    └── time_utils.py           # Trading calendar
```

## Core Components

### 1. Base Simulator

**BaseSimulator (Abstract)**
```python
from abc import ABC, abstractmethod
from datetime import date
from typing import Dict, List, Optional

class BaseSimulator(ABC):
    """Abstract base class for all simulators"""
    
    @abstractmethod
    async def run_simulation(
        self,
        portfolio: Portfolio,
        start_date: date,
        end_date: date,
        config: SimulationConfig
    ) -> SimulationResult:
        """Run simulation for given period"""
        pass
    
    @abstractmethod
    async def get_market_data(
        self,
        symbols: List[str],
        date: date
    ) -> Dict[str, MarketData]:
        """Get market data for specific date"""
        pass
```

**SimulationConfig**
```python
@dataclass
class SimulationConfig:
    """Configuration for simulations"""
    # Execution settings
    slippage_bps: float = 10  # basis points
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    
    # Market impact
    market_impact_model: str = "linear"  # linear, sqrt, none
    impact_coefficient: float = 0.1
    
    # Risk limits
    max_position_size: float = 0.1  # 10% of portfolio
    max_leverage: float = 1.0
    
    # Data settings
    use_adjusted_prices: bool = True
    include_dividends: bool = True
    
    # Performance
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = 0.02
```

### 2. Historical Simulator

**Core Implementation**
```python
class HistoricalSimulator(BaseSimulator):
    """Simulate trading using historical market data"""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.market_calendar = TradingCalendar()
        
    async def simulate_period(
        self,
        portfolio: Portfolio,
        start_date: date,
        end_date: date,
        strategy: Optional[Strategy] = None
    ) -> SimulationResult:
        """Simulate portfolio over a date range"""
        
        trading_days = self.market_calendar.get_trading_days(
            start_date, end_date
        )
        
        results = SimulationResult()
        
        for trading_day in trading_days:
            # Load market data
            market_data = await self.data_loader.load_day(
                trading_day, 
                portfolio.get_symbols()
            )
            
            # Update portfolio values
            portfolio.mark_to_market(market_data)
            
            # Execute strategy if provided
            if strategy:
                signals = strategy.generate_signals(
                    portfolio, 
                    market_data, 
                    trading_day
                )
                await self._execute_signals(
                    portfolio, 
                    signals, 
                    market_data
                )
            
            # Record daily snapshot
            results.record_snapshot(
                trading_day, 
                portfolio.get_snapshot()
            )
            
            # Process corporate actions
            await self._process_corporate_actions(
                portfolio, 
                trading_day
            )
        
        # Calculate final metrics
        results.calculate_metrics()
        return results
```

**Event Period Simulation**
```python
async def simulate_event(
    self,
    event_name: str,
    portfolio: Portfolio,
    strategy: Optional[Strategy] = None
) -> SimulationResult:
    """Simulate a pre-defined market event"""
    
    event_period = EVENT_PERIODS[event_name]
    
    # Add context before event
    start_date = event_period.start_date - timedelta(days=30)
    
    return await self.simulate_period(
        portfolio,
        start_date,
        event_period.end_date,
        strategy
    )
```

### 3. Event Periods

**Pre-defined Market Events**
```python
EVENT_PERIODS = {
    "covid_crash": EventPeriod(
        name="COVID-19 Market Crash",
        start_date=date(2020, 2, 20),
        end_date=date(2020, 4, 30),
        description="Initial COVID panic and recovery",
        key_dates={
            date(2020, 3, 12): "First circuit breaker",
            date(2020, 3, 16): "Second circuit breaker", 
            date(2020, 3, 18): "Third circuit breaker",
            date(2020, 3, 23): "Market bottom"
        }
    ),
    
    "svb_collapse": EventPeriod(
        name="Silicon Valley Bank Collapse",
        start_date=date(2023, 3, 8),
        end_date=date(2023, 3, 31),
        description="Regional banking crisis"
    ),
    
    "gme_squeeze": EventPeriod(
        name="GameStop Short Squeeze",
        start_date=date(2021, 1, 11),
        end_date=date(2021, 2, 10),
        description="Retail-driven short squeeze"
    ),
    
    "fed_taper_2022": EventPeriod(
        name="Fed Rate Hike Cycle",
        start_date=date(2022, 3, 1),
        end_date=date(2022, 12, 31),
        description="Aggressive rate increases"
    ),
    
    "trump_election": EventPeriod(
        name="Trump Election 2016",
        start_date=date(2016, 11, 1),
        end_date=date(2016, 12, 31),
        description="Election surprise and rally"
    ),
    
    "financial_crisis": EventPeriod(
        name="2008 Financial Crisis",
        start_date=date(2008, 9, 1),
        end_date=date(2009, 3, 31),
        description="Global financial meltdown"
    )
}
```

### 4. Portfolio Management

**Portfolio Class**
```python
class Portfolio:
    """Manage positions, cash, and performance"""
    
    def __init__(
        self, 
        initial_cash: float,
        base_currency: str = "USD"
    ):
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.transactions: List[Transaction] = []
        self.base_currency = base_currency
        self._initial_value = initial_cash
        
    def add_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        commission: float = 0
    ) -> Position:
        """Add or update a position"""
        
        transaction = Transaction(
            symbol=symbol,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            commission=commission,
            transaction_type="BUY"
        )
        
        self.transactions.append(transaction)
        self.cash -= (quantity * price + commission)
        
        if symbol in self.positions:
            self.positions[symbol].add_shares(quantity, price)
        else:
            self.positions[symbol] = Position(
                symbol, quantity, price, timestamp
            )
            
        return self.positions[symbol]
    
    def get_total_value(self, market_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        position_value = sum(
            pos.get_market_value(market_prices.get(symbol, pos.last_price))
            for symbol, pos in self.positions.items()
        )
        return self.cash + position_value
    
    def get_returns(self, market_prices: Dict[str, float]) -> float:
        """Calculate total return percentage"""
        current_value = self.get_total_value(market_prices)
        return (current_value - self._initial_value) / self._initial_value
```

**Position Class**
```python
class Position:
    """Individual position tracking"""
    
    def __init__(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        entry_time: datetime
    ):
        self.symbol = symbol
        self.quantity = quantity
        self.cost_basis = quantity * entry_price
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.last_price = entry_price
        
    def add_shares(self, quantity: float, price: float):
        """Add to existing position"""
        self.cost_basis += quantity * price
        self.quantity += quantity
        self.entry_price = self.cost_basis / self.quantity
        
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        self.last_price = current_price
        market_value = self.quantity * current_price
        return market_value - self.cost_basis
```

### 5. Performance Metrics

**Metric Calculations**
```python
class PerformanceMetrics:
    """Calculate portfolio performance metrics"""
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / periods_per_year
        return np.sqrt(periods_per_year) * (
            excess_returns.mean() / excess_returns.std()
        )
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate maximum drawdown and duration"""
        cumulative = (1 + equity_curve).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # Find recovery
        recovery_idx = None
        if max_dd_idx < len(cumulative) - 1:
            post_dd = cumulative[max_dd_idx:]
            recovery_mask = post_dd >= running_max[max_dd_idx]
            if recovery_mask.any():
                recovery_idx = recovery_mask.idxmax()
        
        return {
            "max_drawdown": max_dd,
            "max_dd_date": max_dd_idx,
            "recovery_date": recovery_idx,
            "duration_days": (recovery_idx - max_dd_idx).days if recovery_idx else None
        }
    
    @staticmethod
    def calculate_win_rate(trades: List[Transaction]) -> Dict[str, float]:
        """Calculate win/loss statistics"""
        pnls = [t.realized_pnl for t in trades if t.realized_pnl is not None]
        
        if not pnls:
            return {}
            
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        return {
            "win_rate": len(wins) / len(pnls),
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "profit_factor": sum(wins) / abs(sum(losses)) if losses else float('inf')
        }
```

### 6. Corporate Actions

**Handling Dividends and Splits**
```python
class CorporateActionHandler:
    """Process dividends, splits, and other actions"""
    
    async def process_dividends(
        self,
        portfolio: Portfolio,
        date: date
    ):
        """Process dividend payments"""
        dividends = await self.fetch_dividends(
            portfolio.get_symbols(), 
            date
        )
        
        for div in dividends:
            if div.symbol in portfolio.positions:
                position = portfolio.positions[div.symbol]
                dividend_amount = position.quantity * div.amount
                portfolio.cash += dividend_amount
                
                # Record as transaction
                portfolio.transactions.append(
                    Transaction(
                        symbol=div.symbol,
                        quantity=0,
                        price=div.amount,
                        timestamp=date,
                        transaction_type="DIVIDEND"
                    )
                )
    
    async def process_splits(
        self,
        portfolio: Portfolio,
        date: date
    ):
        """Process stock splits"""
        splits = await self.fetch_splits(
            portfolio.get_symbols(),
            date
        )
        
        for split in splits:
            if split.symbol in portfolio.positions:
                position = portfolio.positions[split.symbol]
                # Adjust quantity and cost basis
                position.quantity *= split.ratio
                position.entry_price /= split.ratio
```

## Usage Examples

### Basic Historical Simulation
```python
# Initialize simulator
simulator = HistoricalSimulator(DataLoader())

# Create portfolio
portfolio = Portfolio(initial_cash=100000)

# Add initial positions
portfolio.add_position("SPY", 100, 400.0, datetime.now())
portfolio.add_position("AAPL", 50, 150.0, datetime.now())

# Run simulation
result = await simulator.simulate_period(
    portfolio,
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 31)
)

# View results
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
```

### Event-Based Simulation
```python
# Simulate COVID crash
result = await simulator.simulate_event(
    "covid_crash",
    portfolio
)

# Compare different strategies
strategies = [
    BuyAndHoldStrategy(),
    StopLossStrategy(stop_loss=0.05),
    RebalancingStrategy(target_weights={"SPY": 0.6, "TLT": 0.4})
]

results = []
for strategy in strategies:
    portfolio_copy = copy.deepcopy(portfolio)
    result = await simulator.simulate_event(
        "covid_crash",
        portfolio_copy,
        strategy
    )
    results.append(result)

# Compare performance
compare_results(results)
```

## Future Extensions

### 1. Stochastic Simulator
```python
class StochasticSimulator(BaseSimulator):
    """Monte Carlo simulation with random walks"""
    
    def __init__(self, model: PriceModel):
        self.model = model  # GBM, Jump Diffusion, etc.
        
    async def run_simulation(
        self,
        portfolio: Portfolio,
        num_paths: int = 1000,
        time_horizon: int = 252
    ) -> List[SimulationResult]:
        """Run Monte Carlo simulation"""
        pass
```

### 2. Event Simulator
```python
class EventSimulator(BaseSimulator):
    """Simulate market shocks and jumps"""
    
    def inject_event(
        self,
        base_scenario: Scenario,
        event: MarketEvent,
        timing: datetime
    ) -> Scenario:
        """Inject shock into base scenario"""
        pass
```

### 3. Multi-Asset Features
- Currency conversions
- Cross-asset correlations
- International markets
- Futures and options

### 4. Advanced Metrics
- Factor attribution
- Risk decomposition
- Scenario analysis
- Stress testing

## Performance Considerations

### Data Loading
- Cache frequently accessed data
- Use columnar storage for time series
- Implement lazy loading
- Batch database queries

### Computation
- Vectorize calculations with NumPy
- Use Numba for hot loops
- Parallelize Monte Carlo paths
- GPU acceleration for large portfolios

### Memory Management
- Stream large datasets
- Use generators for time series
- Implement data windowing
- Clear unused references

## Testing Strategy

### Unit Tests
```python
def test_portfolio_add_position():
    portfolio = Portfolio(10000)
    position = portfolio.add_position("AAPL", 10, 150.0, datetime.now())
    
    assert portfolio.cash == 8500  # 10000 - (10 * 150)
    assert position.quantity == 10
    assert position.cost_basis == 1500
```

### Integration Tests
- Test with real historical data
- Verify corporate action handling
- Check metric calculations
- Validate event periods

### Performance Tests
- Large portfolio simulation
- High-frequency data
- Monte Carlo scalability
- Memory usage profiling

## Best Practices

1. **Data Quality**
   - Validate historical data
   - Handle missing data points
   - Adjust for survivorship bias
   - Check for data anomalies

2. **Realistic Simulation**
   - Include transaction costs
   - Model market impact
   - Account for slippage
   - Respect position limits

3. **Risk Management**
   - Implement stop losses
   - Monitor leverage
   - Track correlation
   - Stress test strategies

4. **Performance Analysis**
   - Use appropriate benchmarks
   - Calculate risk-adjusted returns
   - Analyze drawdown periods
   - Track execution quality