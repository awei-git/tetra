"""Centralized definitions for Tetra trading platform."""

# Data definitions
from .economic_indicators import EconomicIndicators
from .market_universe import MarketUniverse

# Trading configurations
from .trading import (
    TradingConstants,
    SimulationConfig,
    PositionSizingConfig,
    RiskManagementConfig,
    OrderConfig,
    DEFAULT_SIMULATION,
    DEFAULT_POSITION_SIZING,
    DEFAULT_RISK_MANAGEMENT,
    CONSERVATIVE_CONFIG,
    AGGRESSIVE_CONFIG,
    BALANCED_CONFIG
)

# Metric definitions
from .metrics import (
    METRIC_GROUPS,
    DEFAULT_SCORING,
    METRIC_THRESHOLDS,
    get_all_metrics,
    get_required_metrics,
    get_metric_info
)

# Pipeline configurations
from .pipelines import (
    PipelineConfig,
    DataPipelineConfig,
    ScenariosPipelineConfig,
    MetricsPipelineConfig,
    AssessmentPipelineConfig,
    MLPipelineConfig,
    DEFAULT_PIPELINES,
    PIPELINE_SCHEDULES,
    get_pipeline_config,
    get_pipeline_schedule
)

# Strategy definitions
from .strategies import (
    StrategyCategory,
    StrategyConfig,
    StrategyRankingCriteria,
    DEFAULT_STRATEGIES,
    DEFAULT_RANKING,
    get_strategy_config,
    get_strategies_by_category
)

__all__ = [
    # Data
    "EconomicIndicators",
    "MarketUniverse",
    # Trading
    "TradingConstants",
    "SimulationConfig",
    "PositionSizingConfig",
    "RiskManagementConfig",
    "OrderConfig",
    "DEFAULT_SIMULATION",
    "DEFAULT_POSITION_SIZING",
    "DEFAULT_RISK_MANAGEMENT",
    "CONSERVATIVE_CONFIG",
    "AGGRESSIVE_CONFIG",
    "BALANCED_CONFIG",
    # Metrics
    "METRIC_GROUPS",
    "DEFAULT_SCORING",
    "METRIC_THRESHOLDS",
    "get_all_metrics",
    "get_required_metrics",
    "get_metric_info",
    # Pipelines
    "PipelineConfig",
    "DataPipelineConfig",
    "ScenariosPipelineConfig",
    "MetricsPipelineConfig",
    "AssessmentPipelineConfig",
    "MLPipelineConfig",
    "DEFAULT_PIPELINES",
    "PIPELINE_SCHEDULES",
    "get_pipeline_config",
    "get_pipeline_schedule",
    # Strategies
    "StrategyCategory",
    "StrategyConfig",
    "StrategyRankingCriteria",
    "DEFAULT_STRATEGIES",
    "DEFAULT_RANKING",
    "get_strategy_config",
    "get_strategies_by_category"
]