"""Simulation setup step for benchmark pipeline."""

from typing import Dict, Any, List
from datetime import datetime, timedelta

from src.pipelines.base import PipelineStep, PipelineContext
from src.utils.logging import logger
from src.simulators.historical import HistoricalSimulator
from src.simulators.base import SimulationConfig
from src.data_definitions.market_universe import MarketUniverse


class SimulationSetupStep(PipelineStep[Dict[str, Any]]):
    """Setup historical simulator for strategy backtesting."""
    
    def __init__(self):
        super().__init__(
            name="SimulationSetup",
            description="Initialize historical simulator with proper configuration"
        )
    
    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Setup simulator for the backtest period."""
        start_date = context.data["backtest_start"]
        end_date = context.data["backtest_end"]
        universe_filter = context.data.get("universe_filter", "core")
        
        logger.info(f"Setting up simulator from {start_date} to {end_date} for universe: {universe_filter}")
        
        # Get symbols based on universe filter
        symbols = self._get_universe_symbols(universe_filter)
        context.data["symbols"] = symbols
        
        # Create simulator configuration
        config = SimulationConfig(
            starting_cash=100000,  # Standard initial capital
            commission_per_share=0.005,  # $0.005 per share
            slippage_bps=5,   # 5 basis points slippage
            benchmark_symbol="SPY",  # Use SPY as benchmark
            use_adjusted_prices=True,  # Handle splits/dividends
            allow_fractional_shares=True,  # Allow fractional shares
            max_position_size=0.2  # 20% max per position
        )
        
        # Store config and dates in context
        context.data["simulator_config"] = config
        context.data["start_date"] = start_date
        context.data["end_date"] = end_date
        
        # Create simulator instance
        try:
            simulator = HistoricalSimulator(config)
            await simulator.initialize()
            
            # Store the simulator and related data
            context.data["simulator"] = simulator
            context.data["symbols"] = symbols
            
            # For the benchmark pipeline, we'll pass dates to the simulation
            data_stats = {
                "trading_days": (end_date - start_date).days,
                "coverage_by_symbol": {s: 1.0 for s in symbols}
            }
            
            context.data["simulator"] = simulator
            context.data["data_stats"] = data_stats
            
            result = {
                "status": "success",
                "symbols_configured": len(symbols),
                "date_range": {"start": str(start_date), "end": str(end_date)},
                "trading_days": data_stats.get("trading_days", 0),
                "data_coverage": data_stats.get("coverage_by_symbol", {}),
                "initial_capital": config.starting_cash,
                "costs": {
                    "commission": config.commission_per_share,
                    "slippage": config.slippage_bps
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to setup simulator: {e}")
            result = {
                "status": "failed",
                "error": str(e)
            }
        
        logger.info(f"Simulation setup complete: {result}")
        return result
    
    def _get_universe_symbols(self, universe_filter: str) -> List[str]:
        """Get symbols based on universe filter."""
        if universe_filter == "core":
            # Core symbols for benchmark testing
            return [
                'SPY', 'QQQ', 'IWM',  # Index ETFs
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',  # Top tech
                'JPM', 'BAC', 'WFC',  # Banks
                'JNJ', 'PFE', 'UNH',  # Healthcare
                'XOM', 'CVX',  # Energy
                'BTC-USD', 'ETH-USD'  # Crypto
            ]
        elif universe_filter == "all":
            return MarketUniverse.get_all_symbols()
        elif universe_filter == "large_cap":
            return MarketUniverse.LARGE_CAP_STOCKS[:20]
        elif universe_filter == "tech":
            return MarketUniverse.LARGE_CAP_STOCKS[:10] + MarketUniverse.AI_INFRASTRUCTURE_STOCKS
        elif universe_filter == "crypto":
            return MarketUniverse.CRYPTO_SYMBOLS
        else:
            # Default to core
            return self._get_universe_symbols("core")