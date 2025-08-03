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
        result = SimulationResult(
            start_date=start_date,
            end_date=end_date,
            initial_value=portfolio.get_total_value({})
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
        result.calculate_metrics(self.config.risk_free_rate)
        
        # Add benchmark comparison if available
        if self.config.benchmark_symbol in market_prices:
            await self._calculate_benchmark_metrics(result)
        
        return result
    
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
        for symbol in strategy.universe:
            series = await self.market_replay.get_price_series(
                symbol,
                trading_day - timedelta(days=252),  # 1 year of history
                trading_day
            )
            if not series.empty:
                historical_data[symbol] = series
        
        # Calculate indicators
        if hasattr(strategy, 'calculate_indicators'):
            strategy.indicators = strategy.calculate_indicators(historical_data)
        
        # Generate signals
        if hasattr(strategy, 'generate_signals'):
            if asyncio.iscoroutinefunction(strategy.generate_signals):
                signals = await strategy.generate_signals(
                    market_data,
                    portfolio,
                    trading_day
                )
            else:
                signals = strategy.generate_signals(
                    market_data,
                    portfolio,
                    trading_day
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
        if signal.symbol not in market_data:
            return
            
        market_info = market_data[signal.symbol]
        
        # Apply slippage
        execution_price = self.apply_slippage(
            market_info['close'],
            signal.direction == "BUY"
        )
        
        # Calculate commission
        commission = self.calculate_commission(
            signal.quantity or 100,
            execution_price
        )
        
        # Determine quantity if not specified
        if not signal.quantity:
            if signal.direction == "BUY":
                # Calculate based on available cash and position limits
                available_cash = portfolio.cash - commission
                max_position_value = portfolio.get_total_value(market_data) * self.config.max_position_size
                position_value = min(available_cash, max_position_value)
                signal.quantity = int(position_value / execution_price)
            else:
                # Sell entire position
                position = portfolio.get_position(signal.symbol)
                if position:
                    signal.quantity = abs(position.quantity)
                else:
                    return
        
        # Execute trade
        try:
            if signal.direction == "BUY":
                portfolio.add_position(
                    symbol=signal.symbol,
                    quantity=signal.quantity,
                    price=execution_price,
                    timestamp=datetime.combine(trading_day, datetime.min.time()),
                    commission=commission,
                    order_id=getattr(signal, 'order_id', None)
                )
            elif signal.direction == "SELL":
                portfolio.add_position(
                    symbol=signal.symbol,
                    quantity=-signal.quantity,
                    price=execution_price,
                    timestamp=datetime.combine(trading_day, datetime.min.time()),
                    commission=commission,
                    order_id=getattr(signal, 'order_id', None)
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