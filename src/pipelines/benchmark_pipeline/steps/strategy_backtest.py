"""Strategy backtest execution step for benchmark pipeline."""

from typing import Dict, Any, List
import asyncio
from datetime import datetime, timedelta
import time

from src.pipelines.base import PipelineStep, PipelineContext
from src.utils.logging import logger
from src.simulators.historical import HistoricalSimulator
from src.strats.base import BaseStrategy


class StrategyBacktestStep(PipelineStep[Dict[str, Any]]):
    """Execute strategy backtests using the historical simulator."""
    
    def __init__(self):
        super().__init__(
            name="StrategyBacktest",
            description="Run strategies through historical simulation"
        )
    
    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Execute strategies through the simulator with dynamic symbol assignment."""
        strategies = context.data.get("strategies", {})
        simulator = context.data.get("simulator")
        symbols = context.data.get("symbols", [])
        parallel = context.data.get("parallel", 4)
        
        if not simulator:
            return {"status": "failed", "error": "No simulator available"}
        
        if not symbols:
            return {"status": "failed", "error": "No symbols to test"}
        
        logger.info(f"Starting backtest for {len(strategies)} strategies on {len(symbols)} symbols")
        
        # First, analyze symbols to determine best strategy for each
        symbol_strategy_map = await self._analyze_symbols_and_assign_strategies(symbols, strategies, context)
        
        start_time = time.time()
        backtest_results = {}
        
        # Create strategy-symbol pairs to test
        test_pairs = []
        for symbol, strategy_name in symbol_strategy_map.items():
            if strategy_name in strategies:
                test_pairs.append((f"{strategy_name}_{symbol}", strategies[strategy_name], symbol))
        
        logger.info(f"Created {len(test_pairs)} strategy-symbol pairs for testing")
        
        # Run backtests in parallel batches
        for i in range(0, len(test_pairs), parallel):
            batch = test_pairs[i:i + parallel]
            
            # Create tasks for parallel execution
            tasks = []
            for pair_name, strategy, symbol in batch:
                task = self._run_single_backtest(pair_name, strategy, simulator, context, symbol)
                tasks.append(task)
            
            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for j, (pair_name, strategy, symbol) in enumerate(batch):
                result = batch_results[j]
                if isinstance(result, Exception):
                    logger.error(f"Backtest failed for {pair_name}: {result}")
                    backtest_results[pair_name] = {"status": "failed", "error": str(result)}
                else:
                    backtest_results[pair_name] = result
            
            logger.info(f"Completed batch {i//parallel + 1}/{(len(test_pairs) + parallel - 1)//parallel}")
        
        # Store results in context
        context.data["backtest_results"] = backtest_results
        context.data["total_backtests"] = len(strategies)
        
        # Calculate summary statistics
        successful = [r for r in backtest_results.values() if r.get("status") == "success"]
        failed = len(backtest_results) - len(successful)
        
        execution_time = time.time() - start_time
        context.data["execution_time"] = execution_time
        
        result = {
            "status": "success",
            "strategies_tested": len(strategies),
            "successful": len(successful),
            "failed": failed,
            "execution_time": round(execution_time, 2),
            "avg_time_per_strategy": round(execution_time / len(strategies), 2) if strategies else 0
        }
        
        logger.info(f"Backtest complete: {result}")
        return result
    
    async def _analyze_symbols_and_assign_strategies(self, symbols: List[str], strategies: Dict[str, BaseStrategy], 
                                                    context: PipelineContext) -> Dict[str, str]:
        """Analyze each symbol's characteristics and assign most suitable strategy."""
        import pandas as pd
        import numpy as np
        
        symbol_strategy_map = {}
        simulator = context.data.get("simulator")
        
        if not simulator:
            logger.error("No simulator available for symbol analysis")
            # Default assignment
            default_strategy = list(strategies.keys())[0] if strategies else "buy_and_hold"
            return {symbol: default_strategy for symbol in symbols}
        
        end_date = context.data.get("end_date")
        start_date = end_date - timedelta(days=365)  # 1 year of data for analysis
        
        for symbol in symbols:
            try:
                # Use simulator to get historical data
                price_series = await simulator.market_replay.get_price_series(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Convert to dictionary format
                if price_series is not None and not price_series.empty:
                    # Get OHLCV data for the period
                    data = []
                    for date in price_series.index:
                        market_data = await simulator.get_market_data([symbol], date)
                        if symbol in market_data:
                            data.append({
                                'timestamp': date,
                                'open': market_data[symbol].get('open', price_series.loc[date]),
                                'high': market_data[symbol].get('high', price_series.loc[date]),
                                'low': market_data[symbol].get('low', price_series.loc[date]),
                                'close': market_data[symbol].get('close', price_series.loc[date]),
                                'volume': market_data[symbol].get('volume', 0)
                            })
                else:
                    data = None
                
                if data is None or len(data) < 50:  # Need minimum data
                    logger.warning(f"Insufficient data for {symbol}, using default strategy")
                    symbol_strategy_map[symbol] = "buy_and_hold"
                    continue
                
                # Convert to dataframe for analysis
                df = pd.DataFrame(data)
                df['returns'] = df['close'].pct_change()
                
                # Calculate key characteristics
                volatility = df['returns'].std() * np.sqrt(252)
                
                # Trend strength (using simple moving average)
                df['sma_20'] = df['close'].rolling(20).mean()
                df['sma_50'] = df['close'].rolling(50).mean()
                trend_strength = (df['close'].iloc[-1] - df['sma_50'].iloc[-1]) / df['sma_50'].iloc[-1] if not pd.isna(df['sma_50'].iloc[-1]) else 0
                
                # Mean reversion indicator (RSI)
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
                
                # Volume trend
                avg_volume_recent = df['volume'].rolling(20).mean().iloc[-1]
                avg_volume_long = df['volume'].rolling(50).mean().iloc[-1]
                volume_trend = avg_volume_recent / avg_volume_long if avg_volume_long > 0 else 1
                
                # Market cap estimation (rough estimate based on price and average volume)
                avg_price = df['close'].mean()
                is_large_cap = avg_price > 100 and df['volume'].mean() > 5_000_000
                
                # Assign strategy based on characteristics - using actual strategy names from benchmark.py
                if symbol in ['SPY', 'QQQ', 'IWM', 'DIA']:  # Index ETFs
                    strategy_name = "buy_and_hold"
                elif symbol in ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB']:  # Sector ETFs
                    strategy_name = "sector_rotation"
                elif '-USD' in symbol or symbol in ['BTC', 'ETH']:  # Crypto
                    strategy_name = "crypto"
                elif symbol in ['VXX', 'VIXY', 'SVXY', 'UVXY']:  # Volatility ETFs
                    strategy_name = "volatility"
                elif volatility < 0.15 and abs(trend_strength) < 0.05:  # Low volatility, sideways
                    strategy_name = "bollinger_bands"
                elif volatility > 0.4:  # High volatility
                    if current_rsi < 30 or current_rsi > 70:
                        strategy_name = "rsi_reversion"
                    else:
                        strategy_name = "turtle_trading"  # Good for volatile trends
                elif trend_strength > 0.1 and volume_trend > 1.2:  # Strong uptrend with volume
                    strategy_name = "volume_momentum"
                elif trend_strength > 0.1:  # Strong uptrend
                    strategy_name = "golden_cross"
                elif trend_strength < -0.1:  # Strong downtrend
                    strategy_name = "all_weather"  # Defensive strategy
                elif is_large_cap and volatility < 0.25:  # Large cap with moderate volatility
                    strategy_name = "mega_cap_momentum"
                elif volume_trend > 1.5:  # High volume activity
                    strategy_name = "macd_crossover"
                else:
                    # Default based on general characteristics
                    if volatility > 0.25:
                        strategy_name = "momentum_factor"
                    else:
                        strategy_name = "dual_momentum"
                
                # Make sure we have this strategy
                if strategy_name not in strategies:
                    # Fallback to a basic strategy we know exists
                    logger.warning(f"Strategy {strategy_name} not found, using buy_and_hold for {symbol}")
                    strategy_name = "buy_and_hold"
                
                symbol_strategy_map[symbol] = strategy_name
                logger.info(f"Assigned {strategy_name} to {symbol} (vol={volatility:.2%}, trend={trend_strength:.2%}, RSI={current_rsi:.0f})")
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                symbol_strategy_map[symbol] = "buy_and_hold"  # Default fallback
        
        return symbol_strategy_map
    
    async def _run_single_backtest(self, name: str, strategy: BaseStrategy, 
                                  simulator: HistoricalSimulator, context: PipelineContext, symbol: str = None) -> Dict[str, Any]:
        """Run a single strategy backtest."""
        logger.info(f"Starting backtest for strategy: {name}")
        
        try:
            # Get dates from context (stored by SimulationSetup)
            from src.simulators.portfolio import Portfolio
            
            # Create portfolio for this strategy
            portfolio = Portfolio(
                initial_cash=simulator.config.starting_cash,
                base_currency=simulator.config.base_currency
            )
            
            # If a specific symbol is provided, configure strategy for it
            if symbol and hasattr(strategy, 'set_symbols'):
                strategy.set_symbols([symbol])
            
            # Run the backtest
            backtest_start = time.time()
            
            # Get start/end dates from the context
            start_date = context.data.get("start_date")
            end_date = context.data.get("end_date")
            
            results = await simulator.run_simulation(
                portfolio=portfolio,
                start_date=start_date,
                end_date=end_date,
                strategy=strategy
            )
            backtest_time = time.time() - backtest_start
            
            # Extract key metrics
            metrics = {
                "status": "success",
                "strategy_name": name,
                "symbol": symbol,
                "backtest_time": round(backtest_time, 2),
                "final_value": results.final_value,
                "total_return": results.total_return,
                "annualized_return": results.annual_return,  # Fixed: was annualized_return
                "sharpe_ratio": results.sharpe_ratio,
                "max_drawdown": results.max_drawdown,
                "volatility": results.volatility,
                "total_trades": results.total_trades,
                "win_rate": results.win_rate,
                "profit_factor": results.profit_factor,
                "avg_win": results.avg_win,
                "avg_loss": results.avg_loss,
                "best_trade": getattr(results, 'best_trade', None),
                "worst_trade": getattr(results, 'worst_trade', None),
                "equity_curve": results.equity_curve.to_dict() if hasattr(results, 'equity_curve') else None,
                "trades": results.trades if hasattr(results, 'trades') else []
            }
            
            logger.info(f"Completed backtest for {name}: Return={metrics['total_return']:.2%}, "
                       f"Sharpe={metrics['sharpe_ratio']:.2f}, Trades={metrics['total_trades']}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in backtest for {name}: {e}")
            return {
                "status": "failed",
                "strategy_name": name,
                "error": str(e)
            }