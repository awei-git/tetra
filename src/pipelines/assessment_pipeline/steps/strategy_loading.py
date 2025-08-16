"""Step 2: Load and initialize strategy implementations."""

import logging
from typing import Dict, List, Any
from pathlib import Path
import importlib
import inspect

from src.pipelines.base import PipelineStep, PipelineContext
from src.strats.base import BaseStrategy
from src.strats.signal_based import SignalBasedStrategy
from src.strats.ml_based import MLPredictionStrategy
from src.strats.benchmark import buy_and_hold_strategy, golden_cross_strategy

logger = logging.getLogger(__name__)


class StrategyLoadingStep(PipelineStep):
    """Load and initialize all strategy implementations."""
    
    def __init__(self):
        super().__init__("StrategyLoading")
        self.strategy_classes = {}
        self._register_builtin_strategies()
    
    def _register_builtin_strategies(self):
        """Register built-in strategy classes and factory functions."""
        self.strategy_classes = {
            'signal_based': SignalBasedStrategy,
            'ml_based': MLPredictionStrategy,
        }
        self.strategy_factories = {
            'buy_and_hold': buy_and_hold_strategy,
            'golden_cross': golden_cross_strategy,
        }
    
    async def execute(self, context: PipelineContext) -> None:
        """Load and initialize strategy implementations."""
        logger.info("Loading strategy implementations")
        
        # Get strategy configurations from context
        strategy_configs = context.data.get('strategy_configs', [])
        if not strategy_configs:
            raise ValueError("No strategy configurations found in context")
        
        # Initialize strategy instances
        strategies = []
        failed_strategies = []
        
        for config in strategy_configs:
            try:
                strategy = await self._initialize_strategy(config)
                if strategy:
                    strategies.append({
                        'name': config['name'],
                        'instance': strategy,
                        'config': config,
                        'category': config.get('category', 'unknown')
                    })
                    logger.debug(f"Loaded strategy: {config['name']}")
            except Exception as e:
                logger.error(f"Failed to load strategy {config['name']}: {e}")
                failed_strategies.append(config['name'])
        
        # Store results in context
        context.data['strategies'] = strategies
        context.data['failed_strategies'] = failed_strategies
        
        logger.info(f"Successfully loaded {len(strategies)} strategies")
        if failed_strategies:
            logger.warning(f"Failed to load {len(failed_strategies)} strategies: {failed_strategies}")
    
    async def _initialize_strategy(self, config: Dict[str, Any]) -> BaseStrategy:
        """Initialize a strategy instance from configuration."""
        strategy_name = config['name']
        category = config.get('category', 'signal_based')
        parameters = config.get('parameters', {})
        
        # Check if this is a factory function
        if strategy_name in self.strategy_factories:
            # Call factory function to get strategy instance
            try:
                strategy = self.strategy_factories[strategy_name]()
                return strategy
            except Exception as e:
                logger.error(f"Failed to create {strategy_name} from factory: {e}")
                raise
        
        # Map category to strategy class
        if category in ['passive', 'benchmark']:
            # Use buy_and_hold factory for passive strategies
            if 'buy_and_hold' in self.strategy_factories:
                strategy = self.strategy_factories['buy_and_hold']()
            else:
                # Create a simple signal-based strategy for passive
                # For now, just use buy and hold
                strategy = self.strategy_factories.get('buy_and_hold', buy_and_hold_strategy)()
        elif category in ['ml_based', 'machine_learning']:
            # For ML strategies, just use a simple strategy for now
            # ML strategies need trained models which we don't have
            strategy = self.strategy_factories.get('buy_and_hold', buy_and_hold_strategy)()
        else:
            # For signal-based strategies, create with proper parameters
            # Since we don't have signal rules defined, use a factory function if available
            # or default to buy and hold
            factory_name = strategy_name.replace('_strategy', '')
            if factory_name in self.strategy_factories:
                strategy = self.strategy_factories[factory_name]()
            else:
                # Default to buy and hold for undefined strategies
                strategy = self.strategy_factories.get('buy_and_hold', buy_and_hold_strategy)()
        
        return strategy
    
    async def _load_custom_strategies(self, directory: Path) -> Dict[str, type]:
        """Load custom strategy classes from a directory."""
        custom_strategies = {}
        
        if not directory.exists():
            return custom_strategies
        
        for file_path in directory.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
            
            try:
                # Import the module
                spec = importlib.util.spec_from_file_location(
                    file_path.stem, 
                    file_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find strategy classes in the module
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseStrategy) and 
                        obj != BaseStrategy):
                        custom_strategies[name.lower()] = obj
                        logger.debug(f"Loaded custom strategy: {name}")
                        
            except Exception as e:
                logger.error(f"Failed to load strategies from {file_path}: {e}")
        
        return custom_strategies