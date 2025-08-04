"""Core backtesting engine."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Type
import pandas as pd
import numpy as np
from tqdm import tqdm

from ..strategies.base import BaseStrategy, StrategyResult, Trade, PositionSide
from ..signals.base.signal_computer import SignalComputer
from .portfolio import Portfolio, PortfolioState
from .execution import ExecutionEngine, Order, OrderType
from .metrics import MetricsCalculator, PerformanceReport
from .data_handler import DataHandler
from ..simulators.historical import HistoricalSimulator
from ..simulators.base import SimulationConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0001  # 0.01%
    min_trade_size: float = 100.0
    max_positions: int = 10
    use_adjusted_close: bool = True
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    risk_free_rate: float = 0.02  # 2% annual
    benchmark: Optional[str] = "SPY"
    
    # Execution settings
    fill_at_close: bool = True  # Fill orders at close price
    allow_partial_fills: bool = True
    
    # Data settings
    warmup_periods: int = 100  # Periods needed for indicator calculation
    data_frequency: str = "1d"  # 1m, 5m, 1h, 1d
    
    # Risk settings
    max_drawdown_pct: Optional[float] = None  # Stop if drawdown exceeds
    position_size_limit: float = 0.2  # Max 20% per position
    
    # Performance settings
    calculate_metrics_every: int = 20  # Calculate metrics every N bars
    save_trades: bool = True
    save_positions: bool = True
    save_signals: bool = False
    
    def validate(self):
        """Validate configuration."""
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if self.commission < 0 or self.commission > 0.1:
            raise ValueError("Commission must be between 0 and 10%")


class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(self, config: BacktestConfig):
        """Initialize backtesting engine.
        
        Args:
            config: Backtest configuration
        """
        config.validate()
        self.config = config
        
        # Core components
        self.portfolio = Portfolio(initial_capital=config.initial_capital)
        self.execution_engine = ExecutionEngine(
            commission=config.commission,
            slippage=config.slippage
        )
        self.metrics_calculator = MetricsCalculator(risk_free_rate=config.risk_free_rate)
        self.data_handler = DataHandler()
        
        # State tracking
        self.current_datetime: Optional[datetime] = None
        self.current_prices: Dict[str, float] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.position_history: List[PortfolioState] = []
        self.signals_history: Dict[str, pd.DataFrame] = {}
        
        # Performance tracking
        self.bars_processed = 0
        self.last_metrics_calc = 0
        
    def run(self,
            strategy: Type[BaseStrategy],
            symbols: List[str],
            signal_computer: Optional[SignalComputer] = None) -> PerformanceReport:
        """Run backtest for a strategy.
        
        Args:
            strategy: Strategy class to test
            symbols: List of symbols to trade
            signal_computer: Optional signal computer for technical/ML signals
            
        Returns:
            Performance report with metrics and results
        """
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        logger.info(f"Trading symbols: {symbols}")
        
        # Initialize strategy
        strategy_instance = strategy(
            initial_capital=self.config.initial_capital,
            commission=self.config.commission,
            slippage=self.config.slippage,
            max_positions=self.config.max_positions
        )
        
        # Load data
        logger.info("Loading market data...")
        market_data = self._load_data(symbols)
        
        # Load event data if strategy uses events
        event_data = None
        if hasattr(strategy_instance, 'event_triggers'):
            logger.info("Loading event data...")
            event_data = self._load_event_data(symbols)
        
        # Main backtest loop
        logger.info("Running backtest...")
        # Group by timestamp for proper iteration
        for timestamp, group_data in tqdm(market_data.groupby(level='timestamp'), total=len(market_data.index.get_level_values('timestamp').unique())):
            self.current_datetime = timestamp
            self.bars_processed += 1
            
            # Update current prices
            self._update_prices_from_group(group_data)
            
            # Skip warmup period
            if self.bars_processed <= self.config.warmup_periods:
                continue
            
            # Compute signals if signal computer provided
            signals = None
            if signal_computer:
                signals = self._compute_signals(signal_computer, symbols, timestamp)
            
            # Get events for current timestamp
            events = None
            if event_data is not None:
                events = self._get_current_events(event_data, timestamp)
            
            # Update portfolio with current prices
            self.portfolio.update_prices(self.current_prices)
            
            # Check for exits
            self._check_exits(strategy_instance, group_data, signals, events)
            
            # Check for entries
            self._check_entries(strategy_instance, symbols, group_data, signals, events)
            
            # Record state
            self._record_state()
            
            # Calculate metrics periodically
            if self.bars_processed % self.config.calculate_metrics_every == 0:
                self._update_metrics()
            
            # Check risk limits
            if self._check_risk_limits():
                logger.warning("Risk limits breached, stopping backtest")
                break
        
        # Final calculations
        logger.info("Calculating final metrics...")
        return self._generate_report(strategy_instance)
    
    def _load_data(self, symbols: List[str]) -> pd.DataFrame:
        """Load market data for symbols."""
        return self.data_handler.load_market_data(
            symbols=symbols,
            start_date=self.config.start_date - timedelta(days=self.config.warmup_periods),
            end_date=self.config.end_date,
            frequency=self.config.data_frequency
        )
    
    def _load_event_data(self, symbols: List[str]) -> pd.DataFrame:
        """Load event data for symbols."""
        return self.data_handler.load_event_data(
            symbols=symbols,
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )
    
    def _update_prices_from_group(self, group_data: pd.DataFrame):
        """Update current prices from grouped data."""
        for symbol in group_data.index.get_level_values('symbol').unique():
            symbol_data = group_data.xs(symbol, level='symbol')
            if self.config.use_adjusted_close and 'adj_close' in symbol_data.columns:
                self.current_prices[symbol] = symbol_data['adj_close'].iloc[0]
            else:
                self.current_prices[symbol] = symbol_data['close'].iloc[0]
    
    def _compute_signals(self, 
                        signal_computer: SignalComputer,
                        symbols: List[str],
                        timestamp: datetime) -> pd.DataFrame:
        """Compute signals for current timestamp."""
        signals = {}
        
        for symbol in symbols:
            # Get historical data for signal computation
            lookback = signal_computer.get_required_lookback()
            historical_data = self.data_handler.get_historical_data(
                symbol=symbol,
                end_date=timestamp,
                periods=lookback
            )
            
            if len(historical_data) >= lookback:
                symbol_signals = signal_computer.compute(historical_data)
                signals[symbol] = symbol_signals.iloc[-1]  # Latest signals
        
        return pd.DataFrame(signals).T
    
    def _get_current_events(self, 
                           event_data: pd.DataFrame,
                           timestamp: datetime) -> pd.DataFrame:
        """Get events for current timestamp."""
        if timestamp in event_data.index:
            return event_data.loc[timestamp]
        return pd.DataFrame()
    
    def _check_exits(self,
                    strategy: BaseStrategy,
                    bar_data: pd.DataFrame,
                    signals: Optional[pd.DataFrame],
                    events: Optional[pd.DataFrame]):
        """Check for position exits."""
        for symbol, position in list(self.portfolio.positions.items()):
            if symbol in bar_data.index.get_level_values('symbol'):
                symbol_data = bar_data.xs(symbol, level='symbol').iloc[0]
                symbol_signals = signals.loc[symbol] if signals is not None and symbol in signals.index else None
                symbol_events = events[events['symbol'] == symbol] if events is not None and 'symbol' in events else None
                
                if strategy.should_exit(position, self.current_datetime, symbol_data, symbol_signals, symbol_events):
                    # Create exit order
                    order = Order(
                        symbol=symbol,
                        quantity=position.quantity,
                        order_type=OrderType.MARKET,
                        side='sell' if (position.side.value if hasattr(position.side, 'value') else position.side) == 'long' else 'buy',
                        timestamp=self.current_datetime
                    )
                    
                    # Execute order
                    fill = self.execution_engine.execute(
                        order=order,
                        price=self.current_prices[symbol],
                        portfolio=self.portfolio
                    )
                    
                    if fill:
                        # Record trade
                        trade = self._create_trade_from_fill(fill, position)
                        self.trades.append(trade)
                        strategy.state.closed_trades.append(trade)
    
    def _check_entries(self,
                      strategy: BaseStrategy,
                      symbols: List[str],
                      bar_data: pd.DataFrame,
                      signals: Optional[pd.DataFrame],
                      events: Optional[pd.DataFrame]):
        """Check for new position entries."""
        for symbol in symbols:
            if symbol in self.portfolio.positions:
                continue  # Already have position
            
            if symbol not in bar_data.index.get_level_values('symbol'):
                continue  # No data for symbol
            
            symbol_data = bar_data.xs(symbol, level='symbol').iloc[0]
            symbol_signals = signals.loc[symbol] if signals is not None and symbol in signals.index else None
            symbol_events = events[events['symbol'] == symbol] if events is not None and 'symbol' in events else None
            
            should_enter, side, size = strategy.should_enter(
                symbol, self.current_datetime, symbol_data, symbol_signals, symbol_events
            )
            
            if should_enter:
                # Calculate position size
                position_value = self.portfolio.total_value * size
                quantity = position_value / self.current_prices[symbol]
                
                # Check position limits
                if position_value > self.portfolio.total_value * self.config.position_size_limit:
                    quantity = (self.portfolio.total_value * self.config.position_size_limit) / self.current_prices[symbol]
                
                # Create entry order
                order = Order(
                    symbol=symbol,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    side='buy' if (side.value if hasattr(side, 'value') else side) == 'long' else 'sell',
                    timestamp=self.current_datetime
                )
                
                # Execute order
                fill = self.execution_engine.execute(
                    order=order,
                    price=self.current_prices[symbol],
                    portfolio=self.portfolio
                )
                
                if fill:
                    # Update strategy state
                    position = self.portfolio.positions[symbol]
                    position.metadata['entry_rule'] = getattr(strategy, 'last_triggered_rule', 'manual')
                    position.metadata['entry_time'] = self.current_datetime
                    strategy.state.positions[symbol] = position
    
    def _record_state(self):
        """Record current portfolio state."""
        # Record equity
        self.equity_curve.append((self.current_datetime, self.portfolio.total_value))
        
        # Record positions if configured
        if self.config.save_positions:
            self.position_history.append(self.portfolio.get_state())
    
    def _update_metrics(self):
        """Update performance metrics."""
        equity_df = pd.DataFrame(self.equity_curve, columns=['datetime', 'equity'])
        equity_df.set_index('datetime', inplace=True)
        
        # Calculate basic metrics
        returns = equity_df['equity'].pct_change().dropna()
        self.metrics_calculator.update(returns)
    
    def _check_risk_limits(self) -> bool:
        """Check if risk limits are breached."""
        if self.config.max_drawdown_pct:
            current_drawdown = self.metrics_calculator.max_drawdown
            if current_drawdown > self.config.max_drawdown_pct:
                return True
        return False
    
    def _create_trade_from_fill(self, fill: Dict[str, Any], position) -> Trade:
        """Create trade object from fill."""
        trade = Trade(
            symbol=fill['symbol'],
            side=position.side,
            quantity=fill['quantity'],
            entry_price=position.avg_price,
            entry_time=position.metadata.get('entry_time', self.current_datetime),
            exit_price=fill['price'],
            exit_time=fill['timestamp'],
            commission=fill['commission'],
            slippage=fill['slippage']
        )
        # Calculate P&L
        if position.side == PositionSide.LONG or (hasattr(position.side, 'value') and position.side.value == 'long'):
            trade.pnl = (fill['price'] - position.avg_price) * fill['quantity'] - fill['commission'] - fill['slippage']
        else:  # SHORT
            trade.pnl = (position.avg_price - fill['price']) * fill['quantity'] - fill['commission'] - fill['slippage']
        
        trade.pnl_percent = trade.pnl / (position.avg_price * fill['quantity']) * 100 if position.avg_price > 0 else 0
        
        return trade
    
    def _generate_report(self, strategy: BaseStrategy) -> PerformanceReport:
        """Generate final performance report."""
        # Convert equity curve to series
        equity_df = pd.DataFrame(self.equity_curve, columns=['datetime', 'equity'])
        equity_df.set_index('datetime', inplace=True)
        
        # Get benchmark data if specified
        benchmark_data = None
        if self.config.benchmark:
            benchmark_data = self.data_handler.load_market_data(
                symbols=[self.config.benchmark],
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                frequency=self.config.data_frequency
            )
        
        # Generate report
        report = self.metrics_calculator.generate_report(
            equity_curve=equity_df['equity'],
            trades=self.trades,
            positions=self.position_history,
            benchmark=benchmark_data,
            initial_capital=self.config.initial_capital
        )
        
        # Add strategy-specific metrics
        report.strategy_metrics = strategy.calculate_metrics()
        
        # Add configuration
        report.config = self.config
        
        return report