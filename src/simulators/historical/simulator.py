"""Historical market simulator implementation."""

import asyncio
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any

from ..base import BaseSimulator, SimulationConfig, SimulationResult, SimulationSnapshot
from ..portfolio import Portfolio, TransactionType
from .market_replay import MarketReplay
from .event_periods import EVENT_PERIODS, get_overlapping_events
from ..utils.trading_calendar import TradingCalendar


class HistoricalSimulator(BaseSimulator):
    """Simulate trading using historical market data."""
    
    def __init__(
        self, 
        config: Optional[SimulationConfig] = None,
        cache_days: int = 30
    ):
        """
        Initialize historical simulator.
        
        Args:
            config: Simulation configuration
            cache_days: Number of days to cache market data
        """
        super().__init__(config)
        self.market_replay = MarketReplay(cache_days)
        self.calendar = TradingCalendar()
        
    async def _initialize(self) -> None:
        """Initialize simulator components."""
        # Any initialization needed
        pass
    
    async def run_simulation(
        self,
        portfolio: Portfolio,
        start_date: date,
        end_date: date, 
        strategy: Optional[Any] = None,
        preload_data: bool = True,
        progress_callback: Optional[Any] = None
    ) -> SimulationResult:
        """
        Run historical simulation.
        
        Args:
            portfolio: Portfolio to simulate
            start_date: Simulation start date
            end_date: Simulation end date
            strategy: Optional trading strategy
            preload_data: Whether to preload all data
            progress_callback: Optional callback for progress updates
            
        Returns:
            SimulationResult with performance metrics
        """
        await self.initialize()
        
        # Get trading days
        trading_days = self.calendar.get_trading_days(start_date, end_date)
        if not trading_days:
            raise ValueError(f"No trading days between {start_date} and {end_date}")
        
        # Get all symbols we need data for
        symbols = list(portfolio.get_symbols())
        if strategy and hasattr(strategy, 'universe'):
            symbols.extend(strategy.universe)
        symbols = list(set(symbols))  # Remove duplicates
        
        # Add benchmark if needed
        if self.config.benchmark_symbol not in symbols:
            symbols.append(self.config.benchmark_symbol)
        
        # Load market data
        await self.market_replay.load_data(
            symbols, 
            start_date - timedelta(days=60),  # Extra for indicators
            end_date,
            preload=preload_data
        )
        
        # Initialize result
        initial_value = portfolio.get_total_value({})
        result = SimulationResult(
            start_date=start_date,
            end_date=end_date,
            initial_value=initial_value,
            final_value=initial_value  # Will be updated at the end
        )
        
        # Check for overlapping events
        events = get_overlapping_events(start_date, end_date)
        
        # Main simulation loop
        for i, trading_day in enumerate(trading_days):
            # Get market data for the day
            market_data = await self.market_replay.get_market_data(
                symbols, 
                trading_day
            )
            
            # Convert to price dict for portfolio
            market_prices = {
                symbol: data['close'] 
                for symbol, data in market_data.items()
            }
            
            # Update portfolio values
            portfolio.mark_to_market(market_prices)
            
            # Execute strategy if provided
            if strategy:
                signals = await self._execute_strategy(
                    strategy,
                    portfolio,
                    market_data,
                    trading_day
                )
                
                # Process signals
                if signals and self.config.verbose:
                    print(f"Day {trading_day}: Generated {len(signals)} signals")
                    
                for signal in signals:
                    await self._execute_signal(
                        portfolio,
                        signal,
                        market_data,
                        trading_day
                    )
            
            # Process corporate actions
            await self._process_corporate_actions(
                portfolio,
                trading_day
            )
            
            # Record daily snapshot
            snapshot = self._create_snapshot(
                portfolio,
                market_prices,
                trading_day
            )
            result.snapshots.append(snapshot)
            
            # Progress callback
            if progress_callback:
                progress = (i + 1) / len(trading_days)
                await progress_callback(progress, trading_day)
        
        # Final calculations
        result.final_value = portfolio.get_total_value(market_prices)
        
        # Add trades from portfolio transactions
        result.trades = self._extract_trades_from_transactions(portfolio.transactions)
        
        # For buy and hold strategies, count open positions as trades too
        if len(result.trades) == 0 and len(portfolio.transactions) > 0:
            # Count transactions as trades for strategies that don't close positions
            result.total_trades = len(portfolio.transactions)
        
        # Calculate all metrics including trade statistics
        result.calculate_metrics(self.config.risk_free_rate)
        
        # Add benchmark comparison if available
        if self.config.benchmark_symbol in market_prices:
            await self._calculate_benchmark_metrics(result)
        
        return result
    
    def _extract_trades_from_transactions(self, transactions: List[Any]) -> List[Dict[str, Any]]:
        """Extract completed trades from transactions."""
        trades = []
        position_tracker = {}  # Track open positions by symbol
        
        for tx in transactions:
            symbol = tx.symbol
            
            if tx.transaction_type.name == "BUY":
                # Track opening position
                if symbol not in position_tracker:
                    position_tracker[symbol] = []
                position_tracker[symbol].append({
                    'entry_time': tx.timestamp,
                    'entry_price': tx.price,
                    'quantity': tx.quantity,
                    'entry_commission': tx.commission
                })
            
            elif tx.transaction_type.name == "SELL":
                # Match with open positions (FIFO)
                if symbol in position_tracker and position_tracker[symbol]:
                    # Close positions FIFO
                    remaining_qty = tx.quantity
                    while remaining_qty > 0 and position_tracker[symbol]:
                        position = position_tracker[symbol][0]
                        
                        # Calculate how much to close
                        close_qty = min(remaining_qty, position['quantity'])
                        
                        # Create trade record
                        trade = {
                            'symbol': symbol,
                            'entry_date': position['entry_time'],
                            'exit_date': tx.timestamp,
                            'quantity': close_qty,
                            'entry_price': position['entry_price'],
                            'exit_price': tx.price,
                            'entry_commission': position['entry_commission'] * (close_qty / position['quantity']),
                            'exit_commission': tx.commission * (close_qty / tx.quantity),
                            'pnl': (tx.price - position['entry_price']) * close_qty - 
                                   (position['entry_commission'] * (close_qty / position['quantity']) + 
                                    tx.commission * (close_qty / tx.quantity))
                        }
                        trades.append(trade)
                        
                        # Update position
                        position['quantity'] -= close_qty
                        remaining_qty -= close_qty
                        
                        # Remove if fully closed
                        if position['quantity'] <= 0:
                            position_tracker[symbol].pop(0)
        
        return trades
    
    async def simulate_event(
        self,
        event_name: str,
        portfolio: Portfolio,
        strategy: Optional[Any] = None,
        context_days: int = 30
    ) -> SimulationResult:
        """
        Simulate a pre-defined market event.
        
        Args:
            event_name: Name of event from EVENT_PERIODS
            portfolio: Portfolio to simulate
            strategy: Optional trading strategy
            context_days: Days before event to include
            
        Returns:
            SimulationResult for the event period
        """
        if event_name not in EVENT_PERIODS:
            raise ValueError(f"Unknown event: {event_name}")
            
        event = EVENT_PERIODS[event_name]
        
        # Add context before event
        start_date = event.start_date - timedelta(days=context_days)
        
        # Run simulation with event metadata
        result = await self.run_simulation(
            portfolio,
            start_date,
            event.end_date,
            strategy
        )
        
        # Add event information to result
        result.metadata = {
            'event_name': event.name,
            'event_description': event.description,
            'volatility_multiplier': event.volatility_multiplier,
            'key_dates': event.key_dates
        }
        
        return result
    
    async def get_market_data(
        self,
        symbols: List[str],
        date: date
    ) -> Dict[str, Any]:
        """Get market data for specific date."""
        return await self.market_replay.get_market_data(symbols, date)
    
    async def _execute_strategy(
        self,
        strategy: Any,
        portfolio: Portfolio,
        market_data: Dict[str, Dict],
        trading_day: date
    ) -> List[Any]:
        """Execute trading strategy."""
        # Get historical data for indicators
        historical_data = {}
        # Get all symbols from market data if strategy universe is empty or small
        symbols_to_load = set(strategy.universe) if hasattr(strategy, 'universe') else set()
        symbols_to_load.update(market_data.keys())
        
        # Import technical indicators calculator
        from src.ml.technical_indicators import TechnicalIndicators
        
        for symbol in symbols_to_load:
            # Get full OHLCV data, not just price series
            df = await self.market_replay._load_symbol_data(
                symbol,
                trading_day - timedelta(days=252),  # 1 year of history
                trading_day
            )
            if not df.empty:
                # Calculate technical indicators
                df_with_indicators = TechnicalIndicators.calculate_all_indicators(df, symbol)
                historical_data[symbol] = df_with_indicators
        
        # Calculate indicators
        if hasattr(strategy, 'calculate_indicators'):
            strategy.indicators = strategy.calculate_indicators(historical_data)
        
        # Generate signals
        if hasattr(strategy, 'generate_signals'):
            if asyncio.iscoroutinefunction(strategy.generate_signals):
                signals = await strategy.generate_signals(
                    market_data,
                    portfolio,
                    trading_day,
                    historical_data
                )
            else:
                signals = strategy.generate_signals(
                    market_data,
                    portfolio,
                    trading_day,
                    historical_data
                )
            return signals
        
        return []
    
    async def _execute_signal(
        self,
        portfolio: Portfolio,
        signal: Any,
        market_data: Dict[str, Dict],
        trading_day: date
    ) -> None:
        """Execute a trading signal."""
        # Handle both dict and object signals
        symbol = signal.get('symbol') if isinstance(signal, dict) else signal.symbol
        direction = signal.get('direction') if isinstance(signal, dict) else signal.direction
        quantity = signal.get('quantity', 100) if isinstance(signal, dict) else (signal.quantity or 100)
        
        if symbol not in market_data:
            return
            
        market_info = market_data[symbol]
        
        # Apply slippage
        execution_price = self.apply_slippage(
            market_info['close'],
            direction == "BUY"
        )
        
        # Calculate commission
        commission = self.calculate_commission(
            quantity,
            execution_price
        )
        
        # Determine quantity if not specified
        if not quantity:
            if direction == "BUY":
                # Calculate based on available cash and position limits
                available_cash = portfolio.cash - commission
                max_position_value = portfolio.get_total_value(market_data) * self.config.max_position_size
                position_value = min(available_cash, max_position_value)
                quantity = int(position_value / execution_price)
            else:
                # Sell entire position
                position = portfolio.get_position(symbol)
                if position:
                    quantity = abs(position.quantity)
                else:
                    return
        
        # Execute trade
        try:
            if direction == "BUY":
                portfolio.add_position(
                    symbol=symbol,
                    quantity=quantity,
                    price=execution_price,
                    timestamp=datetime.combine(trading_day, datetime.min.time()),
                    commission=commission,
                    order_id=signal.get('order_id') if isinstance(signal, dict) else getattr(signal, 'order_id', None)
                )
            elif direction == "SELL":
                portfolio.add_position(
                    symbol=symbol,
                    quantity=-quantity,
                    price=execution_price,
                    timestamp=datetime.combine(trading_day, datetime.min.time()),
                    commission=commission,
                    order_id=signal.get('order_id') if isinstance(signal, dict) else getattr(signal, 'order_id', None)
                )
        except ValueError as e:
            # Log failed trades
            if self.config.verbose:
                print(f"Trade failed: {e}")
    
    async def _process_corporate_actions(
        self,
        portfolio: Portfolio,
        trading_day: date
    ) -> None:
        """Process dividends and splits."""
        symbols = list(portfolio.get_symbols())
        
        if self.config.include_dividends:
            dividends = await self.market_replay.get_dividends(symbols, trading_day)
            for div in dividends:
                portfolio.process_dividend(
                    symbol=div['symbol'],
                    amount_per_share=div['amount'],
                    timestamp=datetime.combine(trading_day, datetime.min.time())
                )
        
        if self.config.include_splits:
            splits = await self.market_replay.get_splits(symbols, trading_day)
            for split in splits:
                portfolio.process_split(
                    symbol=split['symbol'],
                    split_ratio=split['ratio'],
                    timestamp=datetime.combine(trading_day, datetime.min.time())
                )
    
    def _create_snapshot(
        self,
        portfolio: Portfolio,
        market_prices: Dict[str, float],
        trading_day: date
    ) -> SimulationSnapshot:
        """Create daily portfolio snapshot."""
        snapshot_data = portfolio.get_snapshot(market_prices)
        
        return SimulationSnapshot(
            timestamp=datetime.combine(trading_day, datetime.min.time()),
            total_value=snapshot_data['total_value'],
            cash=snapshot_data['cash'],
            positions_value=snapshot_data['positions_value'],
            positions=snapshot_data['positions']
        )
    
    async def _calculate_benchmark_metrics(self, result: SimulationResult) -> None:
        """Calculate benchmark comparison metrics."""
        # Get benchmark prices
        benchmark_series = await self.market_replay.get_price_series(
            self.config.benchmark_symbol,
            result.start_date,
            result.end_date
        )
        
        if benchmark_series.empty:
            return
            
        # Calculate benchmark return
        benchmark_return = (benchmark_series.iloc[-1] - benchmark_series.iloc[0]) / benchmark_series.iloc[0]
        result.benchmark_return = benchmark_return
        
        # Calculate alpha
        result.alpha = result.total_return - benchmark_return
        
        # Calculate beta and information ratio would require more sophisticated analysis
        # Leaving as None for now
    
    async def _cleanup(self) -> None:
        """Clean up resources."""
        self.market_replay.clear_cache()