"""Pipeline configurations for Tetra platform."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path


# ==================== PIPELINE CONFIGURATIONS ====================

@dataclass
class PipelineConfig:
    """Base configuration for all pipelines."""
    name: str
    parallel_workers: int = 8
    batch_size: int = 10
    cache_enabled: bool = True
    log_level: str = "INFO"
    save_to_db: bool = True
    output_dir: Path = field(default_factory=lambda: Path("data"))
    
    def __post_init__(self):
        """Ensure output directory exists."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class DataPipelineConfig(PipelineConfig):
    """Configuration for Data Pipeline (Stage 1)."""
    name: str = "DataPipeline"
    
    # Data sources
    enable_market_data: bool = True
    enable_economic_data: bool = True
    enable_news_sentiment: bool = True
    enable_event_data: bool = True
    
    # Market data settings
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "IWM"])
    lookback_days: int = 365
    data_provider: str = "polygon"
    
    # Economic data settings
    economic_indicators: List[str] = field(default_factory=lambda: [
        "GDP", "CPI", "UNRATE", "DFF", "DGS10"
    ])
    
    # News settings
    news_sources: List[str] = field(default_factory=lambda: ["newsapi", "benzinga"])
    news_lookback_hours: int = 24
    
    # Quality checks
    run_quality_checks: bool = True
    min_data_completeness: float = 0.95
    
    # Storage
    output_dir: Path = field(default_factory=lambda: Path("data/raw"))


@dataclass
class ScenariosPipelineConfig(PipelineConfig):
    """Configuration for Scenarios Pipeline (Stage 2)."""
    name: str = "ScenariosPipeline"
    
    # Scenario generation
    scenario_types: List[str] = field(default_factory=lambda: [
        "historical", "monte_carlo", "stress_test", "regime_based"
    ])
    
    # Historical scenarios
    historical_periods: List[str] = field(default_factory=lambda: [
        "2008_financial_crisis",
        "2020_covid_crash",
        "2022_bear_market"
    ])
    
    # Monte Carlo settings
    num_monte_carlo_paths: int = 1000
    monte_carlo_horizon_days: int = 252
    
    # Stress test parameters
    stress_test_shocks: Dict[str, float] = field(default_factory=lambda: {
        "market_crash": -0.30,
        "volatility_spike": 2.0,
        "interest_rate_shock": 0.02
    })
    
    # Regime detection
    regime_lookback_days: int = 60
    regime_methods: List[str] = field(default_factory=lambda: ["hmm", "threshold"])
    
    # Storage
    output_dir: Path = field(default_factory=lambda: Path("data/scenarios"))
    save_format: str = "parquet"


@dataclass
class MetricsPipelineConfig(PipelineConfig):
    """Configuration for Metrics Pipeline (Stage 3)."""
    name: str = "MetricsPipeline"
    
    # Metric calculation
    calculate_technical: bool = True
    calculate_statistical: bool = True
    calculate_ml_features: bool = True
    
    # Technical indicators
    technical_indicators: List[str] = field(default_factory=lambda: [
        "SMA", "EMA", "RSI", "MACD", "BB", "ATR", "ADX"
    ])
    indicator_periods: List[int] = field(default_factory=lambda: [
        5, 10, 20, 50, 100, 200
    ])
    
    # Statistical metrics
    statistical_windows: List[int] = field(default_factory=lambda: [
        20, 60, 120, 252
    ])
    calculate_correlations: bool = True
    calculate_cointegration: bool = False
    
    # ML features
    feature_engineering: bool = True
    feature_selection: bool = True
    max_features: int = 100
    
    # Caching
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path("cache/metrics"))
    
    # Storage
    output_dir: Path = field(default_factory=lambda: Path("data/metrics"))


@dataclass
class AssessmentPipelineConfig(PipelineConfig):
    """Configuration for Assessment Pipeline (Stage 4)."""
    name: str = "AssessmentPipeline"
    
    # Backtesting settings
    initial_capital: float = 100000
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    
    # Strategy evaluation
    strategies: List[str] = field(default_factory=lambda: [
        "Buy and Hold",
        "Golden Cross",
        "Mean Reversion",
        "Momentum",
        "RSI Strategy"
    ])
    
    # Symbol universe
    symbols: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "DIA", "GLD", "TLT"
    ])
    lookback_days: int = 500
    
    # Risk parameters
    risk_free_rate: float = 0.02
    max_drawdown_limit: float = 0.30
    position_sizing: str = "equal_weight"
    
    # Ranking configuration
    ranking_method: str = "weighted_score"
    ranking_weights: Dict[str, float] = field(default_factory=lambda: {
        "sharpe_ratio": 0.30,
        "total_return": 0.20,
        "max_drawdown": 0.20,
        "win_rate": 0.15,
        "profit_factor": 0.10,
        "sqn": 0.05
    })
    
    # Minimum thresholds
    min_sharpe: float = 0.5
    max_acceptable_drawdown: float = -0.25
    min_trades: int = 10
    
    # Storage
    output_dir: Path = field(default_factory=lambda: Path("data/assessment"))
    save_rankings: bool = True
    save_backtests: bool = True


@dataclass
class MLPipelineConfig(PipelineConfig):
    """Configuration for ML Pipeline."""
    name: str = "MLPipeline"
    
    # Model settings
    models: List[str] = field(default_factory=lambda: [
        "xgboost", "lightgbm", "catboost", "lstm"
    ])
    
    # Training settings
    train_test_split: float = 0.8
    validation_split: float = 0.1
    cross_validation_folds: int = 5
    
    # Feature settings
    feature_engineering: bool = True
    feature_scaling: str = "standard"  # standard, minmax, robust
    handle_missing: str = "interpolate"  # drop, interpolate, fill
    
    # Hyperparameter tuning
    hyperparameter_tuning: bool = True
    tuning_method: str = "bayesian"  # grid, random, bayesian
    tuning_iterations: int = 100
    
    # Model evaluation
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "sharpe", "returns"
    ])
    
    # Ensemble
    use_ensemble: bool = True
    ensemble_method: str = "voting"  # voting, stacking, blending
    
    # Storage
    output_dir: Path = field(default_factory=lambda: Path("models"))
    save_models: bool = True
    model_versioning: bool = True


# ==================== PIPELINE DEFAULTS ====================

DEFAULT_PIPELINES = {
    "data": DataPipelineConfig(),
    "scenarios": ScenariosPipelineConfig(),
    "metrics": MetricsPipelineConfig(),
    "assessment": AssessmentPipelineConfig(),
    "ml": MLPipelineConfig()
}


# ==================== PIPELINE SCHEDULES ====================

@dataclass
class PipelineSchedule:
    """Schedule configuration for pipeline execution."""
    pipeline_name: str
    enabled: bool = True
    schedule_type: str = "cron"  # cron, interval, once
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    run_at_startup: bool = False
    retry_on_failure: bool = True
    max_retries: int = 3
    timeout_seconds: int = 3600


PIPELINE_SCHEDULES = [
    PipelineSchedule(
        pipeline_name="data",
        enabled=True,
        schedule_type="cron",
        cron_expression="0 19 * * 1-5",  # 7 PM weekdays
        run_at_startup=False
    ),
    PipelineSchedule(
        pipeline_name="scenarios",
        enabled=True,
        schedule_type="cron",
        cron_expression="30 19 * * 1-5",  # 7:30 PM weekdays
        run_at_startup=False
    ),
    PipelineSchedule(
        pipeline_name="metrics",
        enabled=True,
        schedule_type="cron",
        cron_expression="0 20 * * 1-5",  # 8 PM weekdays
        run_at_startup=False
    ),
    PipelineSchedule(
        pipeline_name="assessment",
        enabled=True,
        schedule_type="cron",
        cron_expression="30 20 * * 1-5",  # 8:30 PM weekdays
        run_at_startup=False
    ),
    PipelineSchedule(
        pipeline_name="ml",
        enabled=False,  # Manual trigger only
        schedule_type="cron",
        cron_expression="0 2 * * 6",  # 2 AM Saturday
        run_at_startup=False
    )
]


# ==================== PIPELINE DEPENDENCIES ====================

PIPELINE_DEPENDENCIES = {
    "data": [],  # No dependencies
    "scenarios": ["data"],  # Depends on data pipeline
    "metrics": ["scenarios"],  # Depends on scenarios pipeline
    "assessment": ["metrics"],  # Depends on metrics pipeline
    "ml": ["assessment"]  # Depends on assessment pipeline
}


# ==================== HELPER FUNCTIONS ====================

def get_pipeline_config(pipeline_name: str) -> Optional[PipelineConfig]:
    """Get configuration for a specific pipeline."""
    return DEFAULT_PIPELINES.get(pipeline_name)


def get_pipeline_schedule(pipeline_name: str) -> Optional[PipelineSchedule]:
    """Get schedule for a specific pipeline."""
    for schedule in PIPELINE_SCHEDULES:
        if schedule.pipeline_name == pipeline_name:
            return schedule
    return None


def get_pipeline_dependencies(pipeline_name: str) -> List[str]:
    """Get dependencies for a specific pipeline."""
    return PIPELINE_DEPENDENCIES.get(pipeline_name, [])


def validate_pipeline_order(pipelines: List[str]) -> bool:
    """Validate that pipelines are in correct dependency order."""
    for i, pipeline in enumerate(pipelines):
        deps = get_pipeline_dependencies(pipeline)
        for dep in deps:
            if dep not in pipelines[:i]:
                return False
    return True