"""Comprehensive metric definitions for Tetra platform."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


# ==================== PERFORMANCE METRICS ====================

CORE_PERFORMANCE_METRICS = {
    "total_return": {
        "description": "Total return over the period",
        "formula": "(final_value / initial_value) - 1",
        "unit": "percentage"
    },
    "annualized_return": {
        "description": "Annualized return",
        "formula": "((1 + total_return) ^ (365/days)) - 1",
        "unit": "percentage"
    },
    "volatility": {
        "description": "Standard deviation of returns",
        "formula": "std(returns) * sqrt(252)",
        "unit": "percentage"
    },
    "downside_deviation": {
        "description": "Standard deviation of negative returns",
        "formula": "std(negative_returns) * sqrt(252)",
        "unit": "percentage"
    }
}

RISK_ADJUSTED_METRICS = {
    "sharpe_ratio": {
        "description": "Risk-adjusted return",
        "formula": "(return - risk_free_rate) / volatility",
        "unit": "ratio",
        "good_threshold": 1.0,
        "excellent_threshold": 2.0
    },
    "sortino_ratio": {
        "description": "Downside risk-adjusted return",
        "formula": "(return - risk_free_rate) / downside_deviation",
        "unit": "ratio",
        "good_threshold": 1.5,
        "excellent_threshold": 2.5
    },
    "calmar_ratio": {
        "description": "Return to drawdown ratio",
        "formula": "annualized_return / max_drawdown",
        "unit": "ratio",
        "good_threshold": 1.0,
        "excellent_threshold": 3.0
    },
    "information_ratio": {
        "description": "Active return to tracking error",
        "formula": "(return - benchmark_return) / tracking_error",
        "unit": "ratio"
    },
    "treynor_ratio": {
        "description": "Excess return per unit of systematic risk",
        "formula": "(return - risk_free_rate) / beta",
        "unit": "ratio"
    }
}

RISK_METRICS = {
    "max_drawdown": {
        "description": "Maximum peak-to-trough decline",
        "formula": "min((value - peak) / peak)",
        "unit": "percentage",
        "acceptable_threshold": -0.20,
        "warning_threshold": -0.30
    },
    "avg_drawdown": {
        "description": "Average of all drawdown periods",
        "formula": "mean(drawdowns)",
        "unit": "percentage"
    },
    "var_95": {
        "description": "95% Value at Risk",
        "formula": "percentile(returns, 5)",
        "unit": "percentage"
    },
    "cvar_95": {
        "description": "95% Conditional Value at Risk",
        "formula": "mean(returns | returns <= var_95)",
        "unit": "percentage"
    },
    "var_99": {
        "description": "99% Value at Risk",
        "formula": "percentile(returns, 1)",
        "unit": "percentage"
    },
    "cvar_99": {
        "description": "99% Conditional Value at Risk",
        "formula": "mean(returns | returns <= var_99)",
        "unit": "percentage"
    },
    "ulcer_index": {
        "description": "Measure of downside volatility",
        "formula": "sqrt(mean(drawdowns^2))",
        "unit": "index"
    },
    "omega_ratio": {
        "description": "Probability weighted ratio of gains vs losses",
        "formula": "sum(gains) / sum(losses)",
        "unit": "ratio",
        "good_threshold": 1.5
    },
    "tail_ratio": {
        "description": "Right tail to left tail ratio",
        "formula": "percentile(returns, 95) / abs(percentile(returns, 5))",
        "unit": "ratio"
    }
}

TRADE_QUALITY_METRICS = {
    "win_rate": {
        "description": "Percentage of winning trades",
        "formula": "winning_trades / total_trades",
        "unit": "percentage",
        "good_threshold": 0.50,
        "excellent_threshold": 0.60
    },
    "profit_factor": {
        "description": "Gross profit to gross loss ratio",
        "formula": "sum(winning_trades) / abs(sum(losing_trades))",
        "unit": "ratio",
        "good_threshold": 1.5,
        "excellent_threshold": 2.0
    },
    "expectancy": {
        "description": "Average expected profit per trade",
        "formula": "(win_rate * avg_win) - (loss_rate * avg_loss)",
        "unit": "currency"
    },
    "payoff_ratio": {
        "description": "Average win to average loss ratio",
        "formula": "avg_win / avg_loss",
        "unit": "ratio",
        "good_threshold": 1.5
    },
    "sqn": {
        "description": "System Quality Number (Van Tharp)",
        "formula": "sqrt(n) * expectancy / std(returns)",
        "unit": "score",
        "good_threshold": 2.0,
        "excellent_threshold": 3.0
    },
    "edge_ratio": {
        "description": "Trading edge relative to volatility",
        "formula": "(win_rate * avg_win - loss_rate * avg_loss) / std(returns)",
        "unit": "ratio"
    },
    "kelly_fraction": {
        "description": "Optimal position size by Kelly Criterion",
        "formula": "(win_rate * payoff_ratio - loss_rate) / payoff_ratio",
        "unit": "percentage",
        "max_recommended": 0.25
    }
}

EFFICIENCY_METRICS = {
    "avg_holding_period": {
        "description": "Average days per position",
        "formula": "mean(holding_periods)",
        "unit": "days"
    },
    "time_in_market": {
        "description": "Percentage of time with positions",
        "formula": "days_with_positions / total_days",
        "unit": "percentage"
    },
    "trade_frequency": {
        "description": "Number of trades per year",
        "formula": "total_trades / years",
        "unit": "trades/year"
    },
    "turnover": {
        "description": "Portfolio turnover rate",
        "formula": "total_traded_value / avg_portfolio_value",
        "unit": "ratio"
    },
    "max_consecutive_wins": {
        "description": "Longest winning streak",
        "formula": "max(consecutive_wins)",
        "unit": "count"
    },
    "max_consecutive_losses": {
        "description": "Longest losing streak",
        "formula": "max(consecutive_losses)",
        "unit": "count"
    },
    "recovery_factor": {
        "description": "Net profit to max drawdown ratio",
        "formula": "net_profit / abs(max_drawdown)",
        "unit": "ratio",
        "good_threshold": 2.0
    },
    "mar_ratio": {
        "description": "Managed Account Ratio",
        "formula": "cagr / abs(max_drawdown)",
        "unit": "ratio"
    }
}

REGIME_PERFORMANCE_METRICS = {
    "bull_market_return": {
        "description": "Annualized return in bull markets",
        "formula": "annualized_return(bull_periods)",
        "unit": "percentage"
    },
    "bear_market_return": {
        "description": "Annualized return in bear markets",
        "formula": "annualized_return(bear_periods)",
        "unit": "percentage"
    },
    "high_volatility_return": {
        "description": "Return in high volatility periods",
        "formula": "annualized_return(high_vol_periods)",
        "unit": "percentage"
    },
    "low_volatility_return": {
        "description": "Return in low volatility periods",
        "formula": "annualized_return(low_vol_periods)",
        "unit": "percentage"
    },
    "bull_market_sharpe": {
        "description": "Sharpe ratio during bull markets",
        "formula": "sharpe_ratio(bull_periods)",
        "unit": "ratio"
    },
    "bear_market_sharpe": {
        "description": "Sharpe ratio during bear markets",
        "formula": "sharpe_ratio(bear_periods)",
        "unit": "ratio"
    },
    "regime_adaptability": {
        "description": "Consistency across market regimes",
        "formula": "1 - std(regime_returns) / mean(regime_returns)",
        "unit": "score"
    }
}

STABILITY_METRICS = {
    "return_stability": {
        "description": "Consistency of rolling returns",
        "formula": "1 - std(rolling_returns) / mean(rolling_returns)",
        "unit": "score"
    },
    "rolling_sharpe_std": {
        "description": "Stability of Sharpe ratio over time",
        "formula": "std(rolling_sharpe)",
        "unit": "ratio"
    },
    "consistency_score": {
        "description": "Percentage of positive periods",
        "formula": "positive_periods / total_periods",
        "unit": "percentage"
    },
    "parameter_stability": {
        "description": "Sensitivity to parameter changes",
        "formula": "1 - std(param_results) / mean(param_results)",
        "unit": "score"
    }
}


# ==================== METRIC GROUPS ====================

@dataclass
class MetricGroup:
    """Group of related metrics."""
    name: str
    description: str
    metrics: Dict[str, Dict[str, Any]]
    weight_in_scoring: float = 1.0
    required_for_assessment: bool = True


METRIC_GROUPS = {
    "core_performance": MetricGroup(
        name="Core Performance",
        description="Basic performance metrics",
        metrics=CORE_PERFORMANCE_METRICS,
        weight_in_scoring=0.25,
        required_for_assessment=True
    ),
    "risk_adjusted": MetricGroup(
        name="Risk-Adjusted Returns",
        description="Risk-adjusted performance metrics",
        metrics=RISK_ADJUSTED_METRICS,
        weight_in_scoring=0.30,
        required_for_assessment=True
    ),
    "risk": MetricGroup(
        name="Risk Metrics",
        description="Risk and drawdown metrics",
        metrics=RISK_METRICS,
        weight_in_scoring=0.20,
        required_for_assessment=True
    ),
    "trade_quality": MetricGroup(
        name="Trade Quality",
        description="Trade execution quality metrics",
        metrics=TRADE_QUALITY_METRICS,
        weight_in_scoring=0.15,
        required_for_assessment=True
    ),
    "efficiency": MetricGroup(
        name="Efficiency",
        description="Trading efficiency metrics",
        metrics=EFFICIENCY_METRICS,
        weight_in_scoring=0.05,
        required_for_assessment=False
    ),
    "regime": MetricGroup(
        name="Regime Performance",
        description="Performance across market regimes",
        metrics=REGIME_PERFORMANCE_METRICS,
        weight_in_scoring=0.05,
        required_for_assessment=False
    )
}


# ==================== COMPOSITE SCORING ====================

@dataclass
class ScoringConfig:
    """Configuration for composite scoring."""
    sharpe_weight: float = 30.0
    return_weight: float = 100.0
    drawdown_penalty: float = 20.0
    win_rate_weight: float = 20.0
    profit_factor_weight: float = 10.0
    sqn_weight: float = 5.0
    profit_factor_cap: float = 3.0
    
    def calculate_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite score from metrics."""
        sharpe = metrics.get('sharpe_ratio', 0)
        total_return = metrics.get('total_return', 0)
        max_drawdown = abs(metrics.get('max_drawdown', 0))
        win_rate = metrics.get('win_rate', 0)
        profit_factor = min(metrics.get('profit_factor', 0), self.profit_factor_cap)
        sqn = metrics.get('sqn', 0)
        
        score = (
            sharpe * self.sharpe_weight +
            total_return * self.return_weight +
            (1 / (1 + max_drawdown)) * self.drawdown_penalty +
            win_rate * self.win_rate_weight +
            profit_factor * self.profit_factor_weight +
            sqn * self.sqn_weight
        )
        
        return score


# Default scoring configuration
DEFAULT_SCORING = ScoringConfig()


# ==================== METRIC THRESHOLDS ====================

METRIC_THRESHOLDS = {
    "minimum_acceptable": {
        "sharpe_ratio": 0.5,
        "max_drawdown": -0.30,
        "win_rate": 0.40,
        "profit_factor": 1.0,
        "total_trades": 10
    },
    "good_performance": {
        "sharpe_ratio": 1.0,
        "max_drawdown": -0.20,
        "win_rate": 0.50,
        "profit_factor": 1.5,
        "sqn": 2.0
    },
    "excellent_performance": {
        "sharpe_ratio": 2.0,
        "max_drawdown": -0.10,
        "win_rate": 0.60,
        "profit_factor": 2.0,
        "sqn": 3.0
    }
}


# ==================== HELPER FUNCTIONS ====================

def get_all_metrics() -> List[str]:
    """Get list of all metric names."""
    all_metrics = []
    for group in METRIC_GROUPS.values():
        all_metrics.extend(group.metrics.keys())
    return all_metrics


def get_required_metrics() -> List[str]:
    """Get list of required metrics for assessment."""
    required = []
    for group in METRIC_GROUPS.values():
        if group.required_for_assessment:
            required.extend(group.metrics.keys())
    return required


def get_metric_info(metric_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific metric."""
    for group in METRIC_GROUPS.values():
        if metric_name in group.metrics:
            return group.metrics[metric_name]
    return None