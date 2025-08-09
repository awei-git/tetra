# Performance Assessment Module Documentation

## Overview
The Performance module provides comprehensive backtesting capabilities and performance metrics calculation. It runs strategies through historical or simulated data, tracks execution, and produces detailed analytics to evaluate strategy effectiveness.

## Architecture

### Module Structure
```
src/performance/
├── __init__.py
├── backtester.py               # Main backtesting engine
│   ├── Backtester              # Core backtest runner
│   ├── BacktestConfig          # Configuration
│   └── ExecutionEngine         # Order execution simulation
│
├── metrics/
│   ├── __init__.py
│   ├── returns.py              # Return calculations
│   ├── risk.py                 # Risk metrics (Sharpe, Sortino, etc.)
│   ├── drawdown.py             # Drawdown analysis
│   ├── trading.py              # Trading metrics (win rate, etc.)
│   └── attribution.py          # Performance attribution
│
├── tracking/
│   ├── __init__.py
│   ├── portfolio_tracker.py    # Track portfolio state
│   ├── trade_log.py            # Log all trades
│   ├── snapshot.py             # Daily snapshots
│   └── cash_flow.py            # Track cash movements
│
├── analysis/
│   ├── __init__.py
│   ├── statistical.py          # Statistical analysis
│   ├── regime.py               # Performance by market regime
│   ├── factor.py               # Factor analysis
│   └── optimization.py         # Parameter optimization
│
├── reporting/
│   ├── __init__.py
│   ├── report_generator.py     # Generate reports
│   ├── visualizations.py       # Charts and plots
│   ├── templates/              # Report templates
│   └── export.py               # Export to various formats
│
└── utils/
    ├── __init__.py
    ├── costs.py                # Transaction cost models
    ├── slippage.py             # Slippage models
    └── benchmark.py            # Benchmark comparisons
```

## Core Components

### 1. Backtesting Engine

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import date, datetime
import pandas as pd
import numpy as np

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    
    # Dates
    start_date: date
    end_date: date
    
    # Capital
    initial_capital: float = 100000
    currency: str = 'USD'
    
    # Execution
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    slippage_model: str = 'fixed'  # 'fixed', 'linear', 'square_root'
    slippage_bps: float = 10  # basis points
    
    # Market impact
    market_impact_model: str = 'none'  # 'none', 'linear', 'almgren_chriss'
    impact_coefficient: float = 0.1
    
    # Constraints
    allow_shorts: bool = False
    max_leverage: float = 1.0
    
    # Data
    frequency: str = 'daily'  # 'daily', 'hourly', 'minute'
    use_adjusted_prices: bool = True
    
    # Reporting
    benchmark_symbol: Optional[str] = 'SPY'
    risk_free_rate: float = 0.02
    save_snapshots: bool = True
    verbose: bool = True

class Backtester:
    """Main backtesting engine"""
    
    def __init__(self, 
                 simulator: BaseSimulator,
                 config: BacktestConfig):
        self.simulator = simulator
        self.config = config
        self.execution_engine = ExecutionEngine(config)
        self.portfolio_tracker = PortfolioTracker(config.initial_capital)
        self.trade_log = TradeLog()
        self.snapshots = []
        
    async def run(self,
                  strategy: Strategy,
                  signal_computer: SignalComputer) -> BacktestResult:
        """
        Run complete backtest.
        
        Args:
            strategy: Trading strategy to test
            signal_computer: Signal computation engine
            
        Returns:
            BacktestResult with all metrics and analysis
        """
        # Initialize
        await self._initialize()
        
        # Get trading days
        trading_days = self._get_trading_days()
        
        if self.config.verbose:
            print(f"Running backtest from {self.config.start_date} to {self.config.end_date}")
            print(f"Trading days: {len(trading_days)}")
        
        # Main backtest loop
        for i, trading_day in enumerate(trading_days):
            # Get market data
            market_data = await self.simulator.get_market_data(
                strategy.universe,
                trading_day
            )
            
            # Compute signals
            signals = signal_computer.compute_all(market_data)
            
            # Update portfolio with latest prices
            self.portfolio_tracker.update_prices(market_data)
            
            # Generate orders
            orders = strategy.generate_orders(
                signals,
                self.portfolio_tracker.portfolio,
                trading_day
            )
            
            # Execute orders
            fills = await self.execution_engine.execute_orders(
                orders,
                market_data,
                self.portfolio_tracker.portfolio
            )
            
            # Update portfolio with fills
            for fill in fills:
                self.portfolio_tracker.process_fill(fill)
                self.trade_log.record_trade(fill)
            
            # Take snapshot
            if self.config.save_snapshots:
                snapshot = self.portfolio_tracker.take_snapshot(trading_day)
                self.snapshots.append(snapshot)
            
            # Progress update
            if self.config.verbose and i % 20 == 0:
                self._print_progress(i, len(trading_days), trading_day)
        
        # Generate result
        result = self._generate_result()
        
        if self.config.verbose:
            print("\nBacktest complete!")
            print(result.summary())
        
        return result
    
    async def _initialize(self):
        """Initialize backtest components"""
        # Preload data if needed
        await self.simulator.initialize()
        
        # Load benchmark data
        if self.config.benchmark_symbol:
            self.benchmark_data = await self._load_benchmark_data()
    
    def _generate_result(self) -> 'BacktestResult':
        """Generate comprehensive backtest result"""
        # Create equity curve
        equity_curve = pd.Series(
            {s.timestamp: s.total_value for s in self.snapshots}
        )
        
        # Calculate metrics
        metrics_calculator = MetricsCalculator(self.config)
        
        result = BacktestResult(
            config=self.config,
            equity_curve=equity_curve,
            trades=self.trade_log.get_all_trades(),
            snapshots=self.snapshots,
            
            # Returns
            total_return=metrics_calculator.calculate_total_return(equity_curve),
            annual_return=metrics_calculator.calculate_annual_return(equity_curve),
            
            # Risk metrics
            sharpe_ratio=metrics_calculator.calculate_sharpe_ratio(equity_curve),
            sortino_ratio=metrics_calculator.calculate_sortino_ratio(equity_curve),
            calmar_ratio=metrics_calculator.calculate_calmar_ratio(equity_curve),
            
            # Drawdown
            max_drawdown=metrics_calculator.calculate_max_drawdown(equity_curve),
            drawdown_duration=metrics_calculator.calculate_drawdown_duration(equity_curve),
            
            # Trading metrics
            win_rate=metrics_calculator.calculate_win_rate(self.trade_log),
            profit_factor=metrics_calculator.calculate_profit_factor(self.trade_log),
            avg_win_loss_ratio=metrics_calculator.calculate_avg_win_loss_ratio(self.trade_log),
            
            # Other metrics
            total_trades=len(self.trade_log.trades),
            turnover=metrics_calculator.calculate_turnover(self.trade_log, equity_curve),
            
            # Benchmark comparison
            benchmark_metrics=metrics_calculator.calculate_benchmark_metrics(
                equity_curve, 
                self.benchmark_data
            ) if self.config.benchmark_symbol else None
        )
        
        return result

class ExecutionEngine:
    """Simulate order execution with realistic fills"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.slippage_model = self._create_slippage_model(config.slippage_model)
        self.impact_model = self._create_impact_model(config.market_impact_model)
        
    async def execute_orders(self,
                           orders: List[Order],
                           market_data: pd.DataFrame,
                           portfolio: Portfolio) -> List[Fill]:
        """Execute orders and return fills"""
        fills = []
        
        for order in orders:
            # Check if order can be executed
            if not self._can_execute(order, market_data, portfolio):
                continue
            
            # Calculate execution price with slippage
            base_price = self._get_execution_price(order, market_data)
            slippage = self.slippage_model.calculate(order, market_data)
            impact = self.impact_model.calculate(order, market_data)
            
            execution_price = base_price * (1 + slippage + impact)
            
            # Calculate commission
            commission = self._calculate_commission(order)
            
            # Create fill
            fill = Fill(
                order=order,
                execution_price=execution_price,
                commission=commission,
                timestamp=market_data.index[0],
                slippage=slippage,
                market_impact=impact
            )
            
            fills.append(fill)
        
        return fills
    
    def _can_execute(self, order: Order, market_data: pd.DataFrame, portfolio: Portfolio) -> bool:
        """Check if order can be executed"""
        # Check if symbol is in market data
        if order.symbol not in market_data.index:
            return False
        
        # Check if enough cash for buy orders
        if order.action == 'BUY':
            required_cash = order.quantity * market_data.loc[order.symbol, 'close']
            if required_cash > portfolio.cash:
                return False
        
        # Check if position exists for sell orders
        if order.action == 'SELL':
            position = portfolio.get_position(order.symbol)
            if not position or position.quantity < order.quantity:
                return False
        
        return True
```

### 2. Metrics Calculation

```python
class MetricsCalculator:
    """Calculate performance metrics"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
    def calculate_total_return(self, equity_curve: pd.Series) -> float:
        """Calculate total return"""
        if len(equity_curve) < 2:
            return 0.0
        return (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
    
    def calculate_annual_return(self, equity_curve: pd.Series) -> float:
        """Calculate annualized return"""
        if len(equity_curve) < 2:
            return 0.0
            
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if days <= 0:
            return 0.0
            
        total_return = self.calculate_total_return(equity_curve)
        return (1 + total_return) ** (365.25 / days) - 1
    
    def calculate_sharpe_ratio(self, equity_curve: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        returns = equity_curve.pct_change().dropna()
        
        if len(returns) < 2:
            return 0.0
            
        # Annualized return and volatility
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        
        if annual_vol == 0:
            return 0.0
            
        return (annual_return - self.config.risk_free_rate) / annual_vol
    
    def calculate_sortino_ratio(self, equity_curve: pd.Series) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        returns = equity_curve.pct_change().dropna()
        
        if len(returns) < 2:
            return 0.0
            
        # Downside returns only
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')  # No downside
            
        annual_return = returns.mean() * 252
        downside_vol = downside_returns.std() * np.sqrt(252)
        
        if downside_vol == 0:
            return float('inf')
            
        return (annual_return - self.config.risk_free_rate) / downside_vol
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """Calculate maximum drawdown and related metrics"""
        cumulative = (1 + equity_curve.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # Find peak before drawdown
        peak_idx = running_max[:max_dd_idx].idxmax()
        
        # Find recovery
        recovery_idx = None
        if max_dd_idx < drawdown.index[-1]:
            post_dd = cumulative[max_dd_idx:]
            recovery_mask = post_dd >= running_max[max_dd_idx]
            if recovery_mask.any():
                recovery_idx = recovery_mask.idxmax()
        
        return {
            'max_drawdown': max_dd,
            'peak_date': peak_idx,
            'valley_date': max_dd_idx,
            'recovery_date': recovery_idx,
            'duration_days': (recovery_idx - peak_idx).days if recovery_idx else None
        }
    
    def calculate_win_rate(self, trade_log: TradeLog) -> float:
        """Calculate percentage of winning trades"""
        trades = trade_log.get_closed_trades()
        
        if not trades:
            return 0.0
            
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        return winning_trades / len(trades)
    
    def calculate_profit_factor(self, trade_log: TradeLog) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        trades = trade_log.get_closed_trades()
        
        if not trades:
            return 0.0
            
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
            
        return gross_profit / gross_loss
```

### 3. Portfolio Tracking

```python
class PortfolioTracker:
    """Track portfolio state during backtest"""
    
    def __init__(self, initial_capital: float):
        self.portfolio = Portfolio(initial_capital)
        self.history = []
        
    def update_prices(self, market_data: pd.DataFrame):
        """Update portfolio with latest market prices"""
        for symbol in self.portfolio.positions:
            if symbol in market_data.index:
                price = market_data.loc[symbol, 'close']
                self.portfolio.positions[symbol].update_price(price)
    
    def process_fill(self, fill: Fill):
        """Process a trade fill"""
        if fill.order.action == 'BUY':
            self.portfolio.add_position(
                symbol=fill.order.symbol,
                quantity=fill.order.quantity,
                price=fill.execution_price,
                commission=fill.commission
            )
        elif fill.order.action == 'SELL':
            self.portfolio.reduce_position(
                symbol=fill.order.symbol,
                quantity=fill.order.quantity,
                price=fill.execution_price,
                commission=fill.commission
            )
    
    def take_snapshot(self, timestamp: datetime) -> PortfolioSnapshot:
        """Take portfolio snapshot"""
        positions = {}
        for symbol, position in self.portfolio.positions.items():
            positions[symbol] = {
                'quantity': position.quantity,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl
            }
        
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.portfolio.cash,
            positions=positions,
            total_value=self.portfolio.total_value,
            leverage=self.portfolio.leverage
        )
        
        self.history.append(snapshot)
        return snapshot

class TradeLog:
    """Log all trades for analysis"""
    
    def __init__(self):
        self.trades = []
        self.open_positions = {}
        
    def record_trade(self, fill: Fill):
        """Record a trade"""
        trade = Trade(
            symbol=fill.order.symbol,
            action=fill.order.action,
            quantity=fill.order.quantity,
            price=fill.execution_price,
            commission=fill.commission,
            timestamp=fill.timestamp
        )
        
        self.trades.append(trade)
        
        # Track P&L for closed trades
        if fill.order.action == 'BUY':
            if fill.order.symbol not in self.open_positions:
                self.open_positions[fill.order.symbol] = []
            self.open_positions[fill.order.symbol].append(trade)
            
        elif fill.order.action == 'SELL':
            if fill.order.symbol in self.open_positions:
                # FIFO matching
                remaining_qty = fill.order.quantity
                total_pnl = 0
                
                while remaining_qty > 0 and self.open_positions[fill.order.symbol]:
                    open_trade = self.open_positions[fill.order.symbol][0]
                    
                    if open_trade.quantity <= remaining_qty:
                        # Close entire position
                        pnl = (fill.execution_price - open_trade.price) * open_trade.quantity
                        pnl -= (open_trade.commission + fill.commission * open_trade.quantity / fill.order.quantity)
                        
                        total_pnl += pnl
                        remaining_qty -= open_trade.quantity
                        self.open_positions[fill.order.symbol].pop(0)
                    else:
                        # Partial close
                        pnl = (fill.execution_price - open_trade.price) * remaining_qty
                        pnl -= fill.commission
                        
                        total_pnl += pnl
                        open_trade.quantity -= remaining_qty
                        remaining_qty = 0
                
                trade.pnl = total_pnl
```

### 4. Analysis Tools

```python
class PerformanceAnalyzer:
    """Analyze backtest performance"""
    
    def __init__(self, result: BacktestResult):
        self.result = result
        
    def analyze_by_period(self) -> pd.DataFrame:
        """Analyze performance by time period"""
        equity_curve = self.result.equity_curve
        returns = equity_curve.pct_change()
        
        periods = {
            'Daily': returns,
            'Weekly': returns.resample('W').apply(lambda x: (1 + x).prod() - 1),
            'Monthly': returns.resample('M').apply(lambda x: (1 + x).prod() - 1),
            'Quarterly': returns.resample('Q').apply(lambda x: (1 + x).prod() - 1),
            'Yearly': returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        }
        
        analysis = pd.DataFrame()
        for period, period_returns in periods.items():
            analysis.loc['Mean Return', period] = period_returns.mean()
            analysis.loc['Std Dev', period] = period_returns.std()
            analysis.loc['Sharpe', period] = period_returns.mean() / period_returns.std() if period_returns.std() > 0 else 0
            analysis.loc['Best', period] = period_returns.max()
            analysis.loc['Worst', period] = period_returns.min()
            analysis.loc['% Positive', period] = (period_returns > 0).mean()
        
        return analysis
    
    def analyze_by_market_regime(self, regime_classifier) -> pd.DataFrame:
        """Analyze performance by market regime"""
        regimes = regime_classifier.classify(self.result.equity_curve.index)
        returns = self.result.equity_curve.pct_change()
        
        regime_analysis = pd.DataFrame()
        for regime in regimes.unique():
            regime_returns = returns[regimes == regime]
            
            regime_analysis.loc['Days', regime] = len(regime_returns)
            regime_analysis.loc['Total Return', regime] = (1 + regime_returns).prod() - 1
            regime_analysis.loc['Avg Daily Return', regime] = regime_returns.mean()
            regime_analysis.loc['Volatility', regime] = regime_returns.std()
            regime_analysis.loc['Sharpe', regime] = regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0
            regime_analysis.loc['Max Drawdown', regime] = self._calculate_max_dd(regime_returns)
        
        return regime_analysis
    
    def analyze_trade_distribution(self) -> Dict[str, Any]:
        """Analyze trade P&L distribution"""
        trades = [t for t in self.result.trades if hasattr(t, 'pnl') and t.pnl is not None]
        
        if not trades:
            return {}
            
        pnls = [t.pnl for t in trades]
        
        return {
            'count': len(pnls),
            'mean': np.mean(pnls),
            'median': np.median(pnls),
            'std': np.std(pnls),
            'skew': stats.skew(pnls),
            'kurtosis': stats.kurtosis(pnls),
            'percentiles': {
                '5%': np.percentile(pnls, 5),
                '25%': np.percentile(pnls, 25),
                '75%': np.percentile(pnls, 75),
                '95%': np.percentile(pnls, 95)
            },
            'largest_win': max(pnls),
            'largest_loss': min(pnls)
        }
```

### 5. Reporting and Visualization

```python
class BacktestReport:
    """Generate comprehensive backtest reports"""
    
    def __init__(self, result: BacktestResult):
        self.result = result
        self.analyzer = PerformanceAnalyzer(result)
        
    def generate_html_report(self) -> str:
        """Generate HTML report"""
        template = self._load_template('backtest_report.html')
        
        # Prepare data
        summary_stats = self._generate_summary_stats()
        charts = self._generate_charts()
        trade_analysis = self.analyzer.analyze_trade_distribution()
        period_analysis = self.analyzer.analyze_by_period()
        
        # Render template
        return template.render(
            summary=summary_stats,
            charts=charts,
            trades=trade_analysis,
            periods=period_analysis,
            config=self.result.config
        )
    
    def _generate_charts(self) -> Dict[str, str]:
        """Generate chart images"""
        charts = {}
        
        # Equity curve
        fig, ax = plt.subplots(figsize=(12, 6))
        self.result.equity_curve.plot(ax=ax, label='Strategy')
        if self.result.benchmark_metrics:
            self.result.benchmark_metrics['equity_curve'].plot(ax=ax, label='Benchmark')
        ax.set_title('Equity Curve')
        ax.set_ylabel('Portfolio Value')
        ax.legend()
        charts['equity_curve'] = self._fig_to_base64(fig)
        
        # Drawdown
        fig, ax = plt.subplots(figsize=(12, 4))
        drawdown = self._calculate_drawdown_series()
        drawdown.plot(ax=ax, color='red', alpha=0.7)
        ax.fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.3)
        ax.set_title('Drawdown')
        ax.set_ylabel('Drawdown %')
        charts['drawdown'] = self._fig_to_base64(fig)
        
        # Returns distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        returns = self.result.equity_curve.pct_change().dropna()
        returns.hist(bins=50, ax=ax, alpha=0.7)
        ax.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        ax.set_title('Returns Distribution')
        ax.set_xlabel('Daily Return')
        ax.legend()
        charts['returns_dist'] = self._fig_to_base64(fig)
        
        # Monthly returns heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        monthly_returns = self._calculate_monthly_returns()
        sns.heatmap(monthly_returns, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=ax)
        ax.set_title('Monthly Returns Heatmap')
        charts['monthly_heatmap'] = self._fig_to_base64(fig)
        
        return charts

class BacktestResult:
    """Container for backtest results"""
    
    def __init__(self, **kwargs):
        # Store all metrics
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def summary(self) -> str:
        """Generate summary string"""
        return f"""
Backtest Results Summary
========================
Period: {self.config.start_date} to {self.config.end_date}
Initial Capital: ${self.config.initial_capital:,.2f}

Performance Metrics:
------------------
Total Return: {self.total_return:.2%}
Annual Return: {self.annual_return:.2%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Sortino Ratio: {self.sortino_ratio:.2f}
Max Drawdown: {self.max_drawdown['max_drawdown']:.2%}

Trading Statistics:
-----------------
Total Trades: {self.total_trades}
Win Rate: {self.win_rate:.2%}
Profit Factor: {self.profit_factor:.2f}
Average Win/Loss: {self.avg_win_loss_ratio:.2f}

Risk Metrics:
-----------
Volatility: {self._calculate_volatility():.2%}
Downside Deviation: {self._calculate_downside_deviation():.2%}
Max Drawdown Duration: {self.drawdown_duration['duration_days']} days
"""
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert key metrics to DataFrame"""
        metrics = {
            'Total Return': f"{self.total_return:.2%}",
            'Annual Return': f"{self.annual_return:.2%}",
            'Sharpe Ratio': f"{self.sharpe_ratio:.2f}",
            'Sortino Ratio': f"{self.sortino_ratio:.2f}",
            'Max Drawdown': f"{self.max_drawdown['max_drawdown']:.2%}",
            'Win Rate': f"{self.win_rate:.2%}",
            'Profit Factor': f"{self.profit_factor:.2f}",
            'Total Trades': self.total_trades
        }
        
        return pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
```

## Usage Examples

### Basic Backtesting
```python
# Configure backtest
config = BacktestConfig(
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 31),
    initial_capital=100000,
    commission_per_share=0.005,
    slippage_bps=10
)

# Create components
signal_computer = SignalComputer()
strategy = MomentumStrategy(universe=['AAPL', 'MSFT', 'GOOGL'])
simulator = HistoricalSimulator()

# Run backtest
backtester = Backtester(simulator, config)
result = await backtester.run(strategy, signal_computer)

# View results
print(result.summary())
```

### Advanced Analysis
```python
# Analyze results
analyzer = PerformanceAnalyzer(result)

# Performance by period
period_analysis = analyzer.analyze_by_period()
print(period_analysis)

# Trade distribution
trade_dist = analyzer.analyze_trade_distribution()
print(f"Average trade P&L: ${trade_dist['mean']:.2f}")
print(f"Trade P&L skew: {trade_dist['skew']:.2f}")

# Generate report
report = BacktestReport(result)
html_report = report.generate_html_report()
with open('backtest_report.html', 'w') as f:
    f.write(html_report)
```

### Parameter Optimization
```python
# Grid search optimization
param_grid = {
    'lookback': [10, 20, 30],
    'holding_period': [3, 5, 10],
    'top_n': [5, 10, 15]
}

optimizer = ParameterOptimizer(backtester)
best_params, results = await optimizer.grid_search(
    strategy_class=MomentumStrategy,
    param_grid=param_grid,
    metric='sharpe_ratio'
)

print(f"Best parameters: {best_params}")
print(f"Best Sharpe: {results[str(best_params)].sharpe_ratio:.2f}")
```

### Walk-Forward Analysis
```python
# Walk-forward optimization
walk_forward = WalkForwardAnalyzer(
    train_periods=252,  # 1 year training
    test_periods=63,    # 3 months testing
    step_size=21        # Monthly steps
)

wf_results = await walk_forward.analyze(
    strategy_class=MomentumStrategy,
    signal_computer=signal_computer,
    optimizer=optimizer,
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31)
)

# Plot out-of-sample performance
walk_forward.plot_results(wf_results)
```

## Performance Optimization

### 1. Efficient Data Loading
```python
class CachedDataLoader:
    """Cache market data for faster backtesting"""
    
    def __init__(self, cache_size: int = 1000000):
        self.cache = {}
        self.cache_size = cache_size
        
    async def get_data(self, symbols: List[str], date: date) -> pd.DataFrame:
        key = (tuple(symbols), date)
        
        if key in self.cache:
            return self.cache[key]
            
        # Load from database
        data = await self._load_from_db(symbols, date)
        
        # Update cache
        if len(self.cache) < self.cache_size:
            self.cache[key] = data
            
        return data
```

### 2. Vectorized Metrics
```python
def calculate_metrics_vectorized(equity_curve: pd.Series) -> Dict[str, float]:
    """Calculate all metrics in one pass"""
    returns = equity_curve.pct_change().dropna()
    
    # Pre-calculate common values
    mean_return = returns.mean()
    std_return = returns.std()
    downside_returns = returns[returns < 0]
    
    # Calculate all metrics
    metrics = {
        'total_return': (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0],
        'sharpe_ratio': mean_return / std_return * np.sqrt(252) if std_return > 0 else 0,
        'sortino_ratio': mean_return / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0,
        'max_drawdown': calculate_max_drawdown_vectorized(equity_curve),
        'volatility': std_return * np.sqrt(252)
    }
    
    return metrics
```

### 3. Parallel Backtesting
```python
async def run_parallel_backtests(
    strategies: List[Strategy],
    config: BacktestConfig,
    n_workers: int = 4
) -> List[BacktestResult]:
    """Run multiple backtests in parallel"""
    
    async def run_single(strategy):
        backtester = Backtester(simulator, config)
        return await backtester.run(strategy, signal_computer)
    
    # Run in parallel
    tasks = [run_single(strategy) for strategy in strategies]
    results = await asyncio.gather(*tasks)
    
    return results
```

## Integration Points

### 1. With Simulator
```python
# Backtester uses simulator for market data
backtester = Backtester(
    simulator=HistoricalSimulator(),  # or StochasticSimulator()
    config=config
)
```

### 2. With Signals
```python
# Pre-compute signals for efficiency
all_signals = signal_computer.compute_all(historical_data)
backtester.run_with_precomputed_signals(strategy, all_signals)
```

### 3. With Strategies
```python
# Test multiple strategies
strategies = [
    MomentumStrategy(),
    MeanReversionStrategy(),
    PairsTradingStrategy()
]

for strategy in strategies:
    result = await backtester.run(strategy, signal_computer)
    results[strategy.name] = result
```

## Testing Strategy

### 1. Unit Tests
- Test metric calculations
- Verify trade execution logic
- Test portfolio tracking

### 2. Integration Tests
- Full backtest with known results
- Verify data flow
- Test report generation

### 3. Performance Tests
- Benchmark backtest speed
- Memory usage profiling
- Optimization scalability

## Future Enhancements

1. **Real-time Backtesting**
   - Stream processing for live strategies
   - Rolling window backtests
   - Online metric updates

2. **Advanced Analytics**
   - Factor attribution
   - Style analysis
   - Regime-based performance

3. **Machine Learning Integration**
   - Feature importance for signals
   - Strategy parameter optimization
   - Performance prediction

4. **Risk Analytics**
   - VaR and CVaR calculations
   - Stress testing
   - Scenario analysis

5. **Execution Analysis**
   - Slippage analysis
   - Market impact modeling
   - Optimal execution algorithms