"""Strategy collection step for benchmark pipeline."""

from typing import Dict, Any, List
from datetime import datetime

from src.pipelines.base import PipelineStep, PipelineContext
from src.utils.logging import logger
from src.strats.benchmark import (
    get_all_benchmarks,
    get_core_benchmarks,
    get_benchmarks_by_style,
    get_benchmark_strategy
)


class StrategyCollectionStep(PipelineStep[Dict[str, Any]]):
    """Collect and initialize benchmark strategies for testing."""
    
    def __init__(self):
        super().__init__(
            name="StrategyCollection",
            description="Gather benchmark strategies for backtesting"
        )
    
    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Collect strategies based on filters."""
        strategy_filters = context.data.get("strategy_filters", None)
        mode = context.data.get("mode", "daily")
        
        logger.info(f"Collecting strategies with filters: {strategy_filters}")
        
        # Determine which strategies to run
        if strategy_filters:
            # Specific strategies requested
            strategies = {}
            for name in strategy_filters:
                try:
                    strategies[name] = get_benchmark_strategy(name)
                except ValueError as e:
                    logger.warning(f"Strategy {name} not found: {e}")
        elif mode == "daily":
            # Daily run - use all benchmarks for comprehensive testing
            strategies = get_all_benchmarks()
        else:
            # Full backtest - use all strategies
            strategies = get_all_benchmarks()
        
        # Store strategies in context
        context.data["strategies"] = strategies
        context.data["strategies_tested"] = len(strategies)
        
        # Get strategy categories for analysis
        categories = {}
        styles = get_benchmarks_by_style()
        for style, strategy_names in styles.items():
            for name in strategy_names:
                if name in strategies:
                    categories[name] = style
        
        context.data["strategy_categories"] = categories
        
        result = {
            "status": "success",
            "strategies_collected": len(strategies),
            "strategy_names": list(strategies.keys()),
            "categories": categories
        }
        
        logger.info(f"Collected {len(strategies)} strategies: {list(strategies.keys())}")
        return result