"""
Strategy configuration loader for YAML-based strategy definitions.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import logging

from .signal_based import (
    SignalBasedStrategy, SignalRule, SignalCondition, 
    ConditionOperator, PositionSide
)
from .event_based import (
    EventBasedStrategy, EventTrigger, EventType, EventImpact
)
from .time_based import (
    TimeBasedStrategy, TradingWindow, TradingSchedule, 
    SessionType, TimeFrame
)
from .composite import (
    CompositeStrategy, StrategyWeight, CombinationMode
)
from .ml_based import MLPredictionStrategy, EnsembleMLStrategy

logger = logging.getLogger(__name__)


class StrategyConfigLoader:
    """Loads and instantiates strategies from YAML/JSON configuration files."""
    
    # Map string operators to enum values
    OPERATOR_MAP = {
        "GREATER_THAN": ConditionOperator.GREATER_THAN,
        ">": ConditionOperator.GREATER_THAN,
        "LESS_THAN": ConditionOperator.LESS_THAN,
        "<": ConditionOperator.LESS_THAN,
        "EQUAL": ConditionOperator.EQUAL,
        "=": ConditionOperator.EQUAL,
        "==": ConditionOperator.EQUAL,
        "GREATER_EQUAL": ConditionOperator.GREATER_EQUAL,
        ">=": ConditionOperator.GREATER_EQUAL,
        "LESS_EQUAL": ConditionOperator.LESS_EQUAL,
        "<=": ConditionOperator.LESS_EQUAL,
        "NOT_EQUAL": ConditionOperator.NOT_EQUAL,
        "!=": ConditionOperator.NOT_EQUAL,
        "CROSSES_ABOVE": ConditionOperator.CROSSES_ABOVE,
        "CROSSES_BELOW": ConditionOperator.CROSSES_BELOW,
        "BETWEEN": ConditionOperator.BETWEEN,
        "OUTSIDE": ConditionOperator.OUTSIDE,
    }
    
    # Map string position sides to enum values
    POSITION_SIDE_MAP = {
        "LONG": PositionSide.LONG,
        "SHORT": PositionSide.SHORT,
        "FLAT": PositionSide.FLAT,
    }
    
    # Map string event types to enum values
    EVENT_TYPE_MAP = {
        "EARNINGS": EventType.EARNINGS,
        "ECONOMIC": EventType.ECONOMIC,
        "DIVIDEND": EventType.DIVIDEND,
        "SPLIT": EventType.SPLIT,
        "FOMC": EventType.FOMC,
        "ECB": EventType.ECB,
        "NEWS": EventType.NEWS,
        "CUSTOM": EventType.CUSTOM,
    }
    
    # Map string event impacts to enum values
    EVENT_IMPACT_MAP = {
        "HIGH": EventImpact.HIGH,
        "MEDIUM": EventImpact.MEDIUM,
        "LOW": EventImpact.LOW,
        "UNKNOWN": EventImpact.UNKNOWN,
    }
    
    # Map string session types to enum values
    SESSION_TYPE_MAP = {
        "PREMARKET": SessionType.PREMARKET,
        "REGULAR": SessionType.REGULAR,
        "AFTERHOURS": SessionType.AFTERHOURS,
        "OVERNIGHT": SessionType.OVERNIGHT,
        "ASIAN": SessionType.ASIAN,
        "EUROPEAN": SessionType.EUROPEAN,
        "US": SessionType.US,
    }
    
    # Map string combination modes to enum values
    COMBINATION_MODE_MAP = {
        "UNANIMOUS": CombinationMode.UNANIMOUS,
        "MAJORITY": CombinationMode.MAJORITY,
        "WEIGHTED": CombinationMode.WEIGHTED,
        "SEQUENTIAL": CombinationMode.SEQUENTIAL,
        "ADAPTIVE": CombinationMode.ADAPTIVE,
    }
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> Any:
        """
        Load a strategy from a YAML or JSON configuration file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            Instantiated strategy object
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Load configuration
        with open(file_path, 'r') as f:
            if file_path.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif file_path.suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Determine strategy type and create appropriate instance
        strategy_type = config.get('strategy_type')
        
        if strategy_type == 'signal_based':
            return cls._create_signal_based_strategy(config)
        elif strategy_type == 'event_based':
            return cls._create_event_based_strategy(config)
        elif strategy_type == 'time_based':
            return cls._create_time_based_strategy(config)
        elif strategy_type == 'composite':
            return cls._create_composite_strategy(config, file_path.parent)
        elif strategy_type == 'ml_based':
            return cls._create_ml_based_strategy(config)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    @classmethod
    def _create_signal_based_strategy(cls, config: Dict[str, Any]) -> SignalBasedStrategy:
        """Create a signal-based strategy from configuration."""
        # Extract parameters
        params = config.get('parameters', {})
        
        # Create signal rules
        signal_rules = []
        for rule_config in config.get('signal_rules', []):
            # Create entry conditions
            entry_conditions = []
            for cond_config in rule_config.get('entry_conditions', []):
                condition = SignalCondition(
                    signal_name=cond_config['signal_name'],
                    operator=cls.OPERATOR_MAP[cond_config['operator']],
                    value=cond_config['value'],
                    lookback=cond_config.get('lookback', 1),
                    weight=cond_config.get('weight', 1.0)
                )
                entry_conditions.append(condition)
            
            # Create exit conditions
            exit_conditions = []
            for cond_config in rule_config.get('exit_conditions', []):
                condition = SignalCondition(
                    signal_name=cond_config['signal_name'],
                    operator=cls.OPERATOR_MAP[cond_config['operator']],
                    value=cond_config['value'],
                    lookback=cond_config.get('lookback', 1),
                    weight=cond_config.get('weight', 1.0)
                )
                exit_conditions.append(condition)
            
            # Create rule
            rule = SignalRule(
                name=rule_config['name'],
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
                position_side=cls.POSITION_SIDE_MAP[rule_config['position_side']],
                position_size_factor=rule_config.get('position_size_factor', 1.0),
                stop_loss=rule_config.get('stop_loss'),
                take_profit=rule_config.get('take_profit'),
                time_limit=rule_config.get('time_limit'),
                require_all=rule_config.get('require_all', True)
            )
            signal_rules.append(rule)
        
        # Create strategy
        strategy = SignalBasedStrategy(
            name=config['name'],
            signal_rules=signal_rules,
            signal_weights=config.get('signal_weights', {}),
            confirmation_required=params.get('confirmation_required', 1),
            initial_capital=params.get('initial_capital', 100000),
            position_size=params.get('position_size', 0.1),
            max_positions=params.get('max_positions', 5),
            commission=params.get('commission', 0.001)
        )
        
        return strategy
    
    @classmethod
    def _create_event_based_strategy(cls, config: Dict[str, Any]) -> EventBasedStrategy:
        """Create an event-based strategy from configuration."""
        params = config.get('parameters', {})
        
        # Create event triggers
        event_triggers = []
        for trigger_config in config.get('event_triggers', []):
            trigger = EventTrigger(
                event_type=cls.EVENT_TYPE_MAP[trigger_config['event_type']],
                symbol=trigger_config.get('symbol'),
                impact=cls.EVENT_IMPACT_MAP[trigger_config.get('impact', 'UNKNOWN')],
                pre_event_days=trigger_config.get('pre_event_days', 5),
                post_event_days=trigger_config.get('post_event_days', 3),
                entry_conditions=trigger_config.get('entry_conditions', {}),
                exit_conditions=trigger_config.get('exit_conditions', {}),
                metadata=trigger_config.get('metadata', {})
            )
            event_triggers.append(trigger)
        
        # Create strategy
        strategy = EventBasedStrategy(
            name=config['name'],
            event_triggers=event_triggers,
            initial_capital=params.get('initial_capital', 100000),
            position_size=params.get('position_size', 0.1),
            max_positions=params.get('max_positions', 10),
            commission=params.get('commission', 0.001)
        )
        
        return strategy
    
    @classmethod
    def _create_time_based_strategy(cls, config: Dict[str, Any]) -> TimeBasedStrategy:
        """Create a time-based strategy from configuration."""
        from datetime import time, datetime
        
        params = config.get('parameters', {})
        schedule_config = config.get('trading_schedule', {})
        
        # Create trading windows
        windows = []
        for window_config in schedule_config.get('windows', []):
            # Parse time strings
            start_time = time.fromisoformat(window_config['start_time'])
            end_time = time.fromisoformat(window_config['end_time'])
            
            window = TradingWindow(
                start_time=start_time,
                end_time=end_time,
                days_of_week=window_config.get('days_of_week', list(range(5))),
                timezone=window_config.get('timezone', 'America/New_York'),
                session_type=cls.SESSION_TYPE_MAP[window_config.get('session_type', 'REGULAR')]
            )
            windows.append(window)
        
        # Parse blackout dates
        blackout_dates = set()
        for date_str in schedule_config.get('blackout_dates', []):
            blackout_dates.add(datetime.fromisoformat(date_str).date())
        
        # Parse force close time
        force_close_time = time.fromisoformat(schedule_config.get('force_close_time', '15:45'))
        
        # Create trading schedule
        schedule = TradingSchedule(
            windows=windows,
            blackout_dates=blackout_dates,
            max_trades_per_day=schedule_config.get('max_trades_per_day', 10),
            max_trades_per_window=schedule_config.get('max_trades_per_window', 3),
            force_close_end_of_day=schedule_config.get('force_close_end_of_day', True),
            force_close_time=force_close_time
        )
        
        # Create strategy
        strategy = TimeBasedStrategy(
            name=config['name'],
            trading_schedule=schedule,
            initial_capital=params.get('initial_capital', 100000),
            position_size=params.get('position_size', 0.1),
            max_positions=params.get('max_positions', 5),
            commission=params.get('commission', 0.001)
        )
        
        return strategy
    
    @classmethod
    def _create_composite_strategy(cls, config: Dict[str, Any], base_path: Path) -> CompositeStrategy:
        """Create a composite strategy from configuration."""
        params = config.get('parameters', {})
        
        # Load component strategies
        strategy_weights = []
        for strat_config in config.get('strategies', []):
            # Load strategy from referenced config file
            if 'strategy_config' in strat_config:
                strategy_path = base_path / strat_config['strategy_config']
                strategy = cls.load_from_file(strategy_path)
            else:
                # Strategy might be defined inline
                strategy = cls._create_strategy_from_inline_config(strat_config)
            
            # Create strategy weight
            weight = StrategyWeight(
                strategy=strategy,
                weight=strat_config.get('weight', 1.0),
                min_confidence=strat_config.get('min_confidence', 0.0),
                enabled=strat_config.get('enabled', True),
                performance_weight=strat_config.get('performance_weight', 0.0)
            )
            strategy_weights.append(weight)
        
        # Create strategy
        strategy = CompositeStrategy(
            name=config['name'],
            strategies=strategy_weights,
            combination_mode=cls.COMBINATION_MODE_MAP[config.get('combination_mode', 'WEIGHTED')],
            min_agreement=config.get('min_agreement', 0.6),
            adapt_weights=config.get('adapt_weights', False),
            performance_window=config.get('performance_window', 100),
            initial_capital=params.get('initial_capital', 100000),
            position_size=params.get('position_size', 0.05),
            max_positions=params.get('max_positions', 20),
            commission=params.get('commission', 0.001)
        )
        
        return strategy
    
    @classmethod
    def _create_ml_based_strategy(cls, config: Dict[str, Any]) -> Union[MLPredictionStrategy, EnsembleMLStrategy]:
        """Create an ML-based strategy from configuration."""
        params = config.get('parameters', {})
        ml_config = config.get('ml_config', {})
        
        # Check if ensemble is enabled
        ensemble = ml_config.get('ensemble', {})
        if ensemble.get('enabled', False):
            # Create ensemble strategy
            strategy = EnsembleMLStrategy(
                name=config['name'],  # Pass name to constructor
                models_dir=ensemble.get('models_dir', 'output/ml_pipeline/models'),
                model_names=ensemble.get('model_names'),
                voting_method=ensemble.get('voting_method', 'weighted'),
                prediction_threshold=ml_config.get('prediction_threshold', 0.002),
                confidence_threshold=ml_config.get('confidence_threshold', 0.6),
                position_size_pct=params.get('position_size', 0.1),
                max_positions=params.get('max_positions', 10),
                initial_capital=params.get('initial_capital', 100000),
                commission=params.get('commission', 0.001)
            )
        else:
            # Create single model strategy
            strategy = MLPredictionStrategy(
                name=config['name'],  # Pass name to constructor
                model_path=ml_config.get('model_path'),
                model_name=ml_config.get('model_name', 'xgboost_return_1d'),
                prediction_threshold=ml_config.get('prediction_threshold', 0.002),
                confidence_threshold=ml_config.get('confidence_threshold', 0.6),
                position_size_pct=params.get('position_size', 0.1),
                max_positions=params.get('max_positions', 10),
                initial_capital=params.get('initial_capital', 100000),
                position_size=params.get('position_size', 0.1),
                commission=params.get('commission', 0.001)
            )
        return strategy
    
    @classmethod
    def _create_strategy_from_inline_config(cls, config: Dict[str, Any]) -> Any:
        """Create a strategy from inline configuration (used in composite strategies)."""
        # This would handle inline strategy definitions
        # For now, return None - would need full implementation
        logger.warning("Inline strategy definitions not yet implemented")
        return None


def load_strategy_from_yaml(file_path: Union[str, Path]) -> Any:
    """
    Convenience function to load a strategy from a YAML file.
    
    Args:
        file_path: Path to the YAML configuration file
        
    Returns:
        Instantiated strategy object
    """
    return StrategyConfigLoader.load_from_file(file_path)


def load_all_strategies_from_directory(directory: Union[str, Path]) -> Dict[str, Any]:
    """
    Load all strategy configurations from a directory.
    
    Args:
        directory: Path to directory containing YAML/JSON configs
        
    Returns:
        Dictionary mapping strategy names to strategy objects
    """
    directory = Path(directory)
    strategies = {}
    
    # Load all YAML and JSON files
    for file_path in directory.glob("*.yaml"):
        try:
            strategy = StrategyConfigLoader.load_from_file(file_path)
            strategies[strategy.name] = strategy
            logger.info(f"Loaded strategy: {strategy.name}")
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
    
    for file_path in directory.glob("*.yml"):
        try:
            strategy = StrategyConfigLoader.load_from_file(file_path)
            strategies[strategy.name] = strategy
            logger.info(f"Loaded strategy: {strategy.name}")
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
    
    for file_path in directory.glob("*.json"):
        try:
            strategy = StrategyConfigLoader.load_from_file(file_path)
            strategies[strategy.name] = strategy
            logger.info(f"Loaded strategy: {strategy.name}")
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
    
    return strategies


def validate_strategy_config(file_path: Union[str, Path]) -> bool:
    """
    Validate a strategy configuration file without instantiating.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Try to load the strategy
        strategy = StrategyConfigLoader.load_from_file(file_path)
        
        # Basic validation
        if not hasattr(strategy, 'name'):
            logger.error("Strategy missing name")
            return False
        
        logger.info(f"Configuration valid for strategy: {strategy.name}")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


if __name__ == "__main__":
    # Test loading strategies
    import sys
    
    if len(sys.argv) > 1:
        # Load specific strategy
        file_path = sys.argv[1]
        strategy = load_strategy_from_yaml(file_path)
        print(f"Loaded strategy: {strategy.name}")
        print(f"Type: {type(strategy).__name__}")
        print(f"Parameters: {strategy.__dict__}")
    else:
        # Load all examples
        examples_dir = Path(__file__).parent / "examples"
        strategies = load_all_strategies_from_directory(examples_dir)
        print(f"Loaded {len(strategies)} strategies:")
        for name, strategy in strategies.items():
            print(f"  - {name}: {type(strategy).__name__}")