"""Strategy definitions and configurations for Tetra platform."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Literal
from enum import Enum


# ==================== STRATEGY CATEGORIES ====================

class StrategyCategory(Enum):
    """Strategy categories for classification."""
    PASSIVE = "passive"
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    ARBITRAGE = "arbitrage"
    ML_BASED = "ml_based"
    COMPOSITE = "composite"
    MARKET_MAKING = "market_making"
    EVENT_DRIVEN = "event_driven"


# ==================== STRATEGY CONFIGURATION ====================

@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    
    # Identity
    name: str
    category: StrategyCategory
    description: str = ""
    version: str = "1.0.0"
    
    # Trading symbols
    primary_symbol: str = "SPY"
    alternative_symbols: List[str] = field(default_factory=list)
    symbol_selection: Literal["fixed", "dynamic", "sector_rotation"] = "fixed"
    
    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Risk management
    max_position_size: float = 0.10
    stop_loss: Optional[float] = 0.02
    take_profit: Optional[float] = None
    max_holding_period: Optional[int] = None
    
    # Execution
    entry_signal_threshold: float = 0.0
    exit_signal_threshold: float = 0.0
    rebalance_frequency: Literal["daily", "weekly", "monthly", "signal_based"] = "signal_based"
    
    # Constraints
    min_price: float = 5.0  # Avoid penny stocks
    min_volume: float = 1000000  # Minimum daily volume
    max_positions: int = 10
    
    # Features required
    required_indicators: List[str] = field(default_factory=list)
    required_data_points: int = 200
    
    # Performance targets
    target_sharpe: float = 1.0
    target_annual_return: float = 0.12
    max_acceptable_drawdown: float = 0.20
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    author: str = "system"
    created_date: str = ""
    last_modified: str = ""
    
    def validate(self) -> bool:
        """Validate strategy configuration."""
        if self.max_position_size > 1.0 or self.max_position_size <= 0:
            raise ValueError(f"Invalid max_position_size: {self.max_position_size}")
        
        if self.stop_loss and (self.stop_loss <= 0 or self.stop_loss >= 1):
            raise ValueError(f"Invalid stop_loss: {self.stop_loss}")
        
        if self.take_profit and self.take_profit <= 0:
            raise ValueError(f"Invalid take_profit: {self.take_profit}")
        
        if self.max_positions <= 0:
            raise ValueError(f"Invalid max_positions: {self.max_positions}")
        
        return True


# ==================== DEFAULT STRATEGY CONFIGURATIONS ====================

DEFAULT_STRATEGIES = {
    
    # Passive Strategies
    "buy_and_hold": StrategyConfig(
        name="Buy and Hold",
        category=StrategyCategory.PASSIVE,
        description="Simple buy and hold strategy",
        primary_symbol="SPY",
        alternative_symbols=["QQQ", "IWM", "DIA"],
        parameters={"rebalance": "never"},
        stop_loss=None,
        max_holding_period=None,
        required_indicators=[],
        required_data_points=1
    ),
    
    "dollar_cost_averaging": StrategyConfig(
        name="Dollar Cost Averaging",
        category=StrategyCategory.PASSIVE,
        description="Regular periodic investments",
        primary_symbol="SPY",
        parameters={"investment_frequency": "monthly", "amount": 1000},
        rebalance_frequency="monthly"
    ),
    
    # Trend Following Strategies
    "golden_cross": StrategyConfig(
        name="Golden Cross",
        category=StrategyCategory.TREND_FOLLOWING,
        description="Buy when 50-day MA crosses above 200-day MA",
        primary_symbol="IWM",
        alternative_symbols=["SPY", "QQQ"],
        parameters={"fast_ma": 50, "slow_ma": 200},
        required_indicators=["SMA_50", "SMA_200"],
        required_data_points=200,
        stop_loss=0.05
    ),
    
    "trend_following": StrategyConfig(
        name="Trend Following",
        category=StrategyCategory.TREND_FOLLOWING,
        description="Follow established trends using moving averages",
        primary_symbol="QQQ",
        parameters={"ma_period": 20, "atr_multiplier": 2},
        required_indicators=["SMA_20", "ATR"],
        stop_loss=0.03
    ),
    
    # Mean Reversion Strategies
    "mean_reversion": StrategyConfig(
        name="Mean Reversion",
        category=StrategyCategory.MEAN_REVERSION,
        description="Trade reversions to mean using Bollinger Bands",
        primary_symbol="SPY",
        alternative_symbols=["IWM", "DIA"],
        parameters={"lookback": 20, "z_score": 2, "bb_period": 20},
        required_indicators=["BB_20", "SMA_20"],
        stop_loss=0.02,
        take_profit=0.05
    ),
    
    "rsi_strategy": StrategyConfig(
        name="RSI Strategy",
        category=StrategyCategory.MEAN_REVERSION,
        description="Trade oversold/overbought conditions using RSI",
        primary_symbol="IWM",
        alternative_symbols=["SPY", "XLF"],
        parameters={"period": 14, "oversold": 30, "overbought": 70},
        required_indicators=["RSI_14"],
        stop_loss=0.03
    ),
    
    # Momentum Strategies
    "momentum": StrategyConfig(
        name="Momentum",
        category=StrategyCategory.MOMENTUM,
        description="Buy high momentum stocks",
        primary_symbol="QQQ",
        alternative_symbols=["SPY", "IWM"],
        parameters={"lookback": 60, "rebalance_days": 20},
        required_indicators=["ROC_60"],
        rebalance_frequency="monthly"
    ),
    
    "dual_momentum": StrategyConfig(
        name="Dual Momentum",
        category=StrategyCategory.MOMENTUM,
        description="Combine absolute and relative momentum",
        primary_symbol="SPY",
        parameters={
            "lookback": 252,
            "rebalance_frequency": "monthly",
            "cash_proxy": "BIL"
        },
        required_indicators=["ROC_252"],
        rebalance_frequency="monthly"
    ),
    
    # Volatility Strategies
    "volatility_targeting": StrategyConfig(
        name="Volatility Targeting",
        category=StrategyCategory.VOLATILITY,
        description="Adjust position size based on volatility",
        primary_symbol="SPY",
        parameters={
            "target_volatility": 0.15,
            "lookback": 20,
            "max_leverage": 1.5
        },
        required_indicators=["ATR", "HV_20"]
    ),
    
    # ML-Based Strategies
    "ml_ensemble": StrategyConfig(
        name="ML Ensemble",
        category=StrategyCategory.ML_BASED,
        description="Machine learning ensemble predictions",
        primary_symbol="SPY",
        parameters={
            "models": ["xgboost", "lightgbm", "catboost"],
            "prediction_horizon": 5,
            "confidence_threshold": 0.6
        },
        required_indicators=["SMA_20", "RSI_14", "MACD", "ATR"],
        required_data_points=500
    ),
    
    # Composite Strategies
    "balanced_portfolio": StrategyConfig(
        name="Balanced Portfolio",
        category=StrategyCategory.COMPOSITE,
        description="60/40 stocks/bonds allocation",
        primary_symbol="SPY",
        alternative_symbols=["TLT"],
        parameters={
            "stock_allocation": 0.6,
            "bond_allocation": 0.4,
            "rebalance_threshold": 0.05
        },
        rebalance_frequency="monthly"
    )
}


# ==================== STRATEGY RANKING ====================

@dataclass
class StrategyRankingCriteria:
    """Criteria for ranking strategies."""
    
    # Weights for different metrics
    sharpe_weight: float = 0.30
    return_weight: float = 0.20
    drawdown_weight: float = 0.20
    consistency_weight: float = 0.15
    win_rate_weight: float = 0.10
    regime_adaptability_weight: float = 0.05
    
    # Minimum thresholds
    min_sharpe: float = 0.5
    min_annual_return: float = 0.05
    max_drawdown: float = -0.30
    min_win_rate: float = 0.40
    min_trades: int = 10
    
    # Regime weights
    bull_market_weight: float = 0.3
    bear_market_weight: float = 0.3
    sideways_market_weight: float = 0.2
    high_volatility_weight: float = 0.2
    
    def calculate_score(self, metrics: Dict[str, float]) -> float:
        """Calculate ranking score for a strategy."""
        score = 0.0
        
        # Performance score
        score += metrics.get("sharpe_ratio", 0) * self.sharpe_weight
        score += metrics.get("annual_return", 0) * self.return_weight * 100
        
        # Risk score
        drawdown = abs(metrics.get("max_drawdown", 0))
        if drawdown > 0:
            score += (1 / (1 + drawdown)) * self.drawdown_weight * 20
        
        # Consistency score
        score += metrics.get("win_rate", 0) * self.win_rate_weight * 20
        score += metrics.get("consistency_score", 0) * self.consistency_weight * 10
        
        # Regime adaptability
        regime_score = (
            metrics.get("bull_performance", 0) * self.bull_market_weight +
            metrics.get("bear_performance", 0) * self.bear_market_weight +
            metrics.get("sideways_performance", 0) * self.sideways_market_weight +
            metrics.get("high_vol_performance", 0) * self.high_volatility_weight
        )
        score += regime_score * self.regime_adaptability_weight
        
        return score
    
    def meets_minimum_requirements(self, metrics: Dict[str, float]) -> bool:
        """Check if strategy meets minimum requirements."""
        if metrics.get("sharpe_ratio", 0) < self.min_sharpe:
            return False
        if metrics.get("annual_return", 0) < self.min_annual_return:
            return False
        if metrics.get("max_drawdown", 0) < self.max_drawdown:
            return False
        if metrics.get("win_rate", 0) < self.min_win_rate:
            return False
        if metrics.get("total_trades", 0) < self.min_trades:
            return False
        return True


# Default ranking criteria
DEFAULT_RANKING = StrategyRankingCriteria()


# ==================== HELPER FUNCTIONS ====================

def get_strategy_config(strategy_name: str) -> Optional[StrategyConfig]:
    """Get configuration for a specific strategy."""
    # Normalize name
    normalized_name = strategy_name.lower().replace(" ", "_")
    return DEFAULT_STRATEGIES.get(normalized_name)


def get_strategies_by_category(category: StrategyCategory) -> List[StrategyConfig]:
    """Get all strategies in a specific category."""
    strategies = []
    for config in DEFAULT_STRATEGIES.values():
        if config.category == category:
            strategies.append(config)
    return strategies


def get_required_indicators(strategies: List[str]) -> List[str]:
    """Get all required indicators for a list of strategies."""
    indicators = set()
    for strategy_name in strategies:
        config = get_strategy_config(strategy_name)
        if config:
            indicators.update(config.required_indicators)
    return list(indicators)


def validate_strategy_compatibility(strategy1: str, strategy2: str) -> bool:
    """Check if two strategies can be combined."""
    config1 = get_strategy_config(strategy1)
    config2 = get_strategy_config(strategy2)
    
    if not config1 or not config2:
        return False
    
    # Check for conflicting parameters
    if config1.rebalance_frequency != config2.rebalance_frequency:
        return False
    
    # Check for symbol overlap
    symbols1 = set([config1.primary_symbol] + config1.alternative_symbols)
    symbols2 = set([config2.primary_symbol] + config2.alternative_symbols)
    if symbols1 & symbols2:  # If there's overlap
        return False
    
    return True