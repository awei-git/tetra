"""Trading configuration and constants for Tetra platform."""

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any
from datetime import time


# ==================== TRADING CONSTANTS ====================

class TradingConstants:
    """Global trading constants."""
    
    # Market Hours (Eastern Time)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    PRE_MARKET_OPEN = time(4, 0)
    AFTER_MARKET_CLOSE = time(20, 0)
    
    # Trading Days
    TRADING_DAYS_PER_YEAR = 252
    TRADING_HOURS_PER_DAY = 6.5
    
    # Default Risk Parameters
    DEFAULT_RISK_FREE_RATE = 0.02  # 2% annual
    DEFAULT_BENCHMARK = "SPY"
    
    # Position Limits
    MAX_POSITION_SIZE = 0.10  # 10% of portfolio
    MAX_LEVERAGE = 1.0
    MAX_SECTOR_EXPOSURE = 0.30  # 30% in one sector
    MIN_POSITION_SIZE = 0.01  # 1% of portfolio
    
    # Risk Limits
    MAX_DAILY_LOSS = 0.02  # 2% daily loss limit
    MAX_PORTFOLIO_RISK = 0.06  # 6% portfolio risk
    RISK_PER_TRADE = 0.02  # 2% risk per trade
    
    # Transaction Costs (defaults)
    DEFAULT_COMMISSION_PER_SHARE = 0.005
    DEFAULT_MIN_COMMISSION = 1.0
    DEFAULT_MAX_COMMISSION = 100.0
    DEFAULT_SLIPPAGE_BPS = 5  # basis points
    DEFAULT_SPREAD_BPS = 2
    
    # Market Impact
    LINEAR_IMPACT_COEFFICIENT = 0.1
    SQRT_IMPACT_COEFFICIENT = 0.05
    
    # Data Settings
    MIN_DATA_POINTS = 100  # Minimum data points for analysis
    LOOKBACK_DAYS_DEFAULT = 252  # 1 year
    
    # Currency
    BASE_CURRENCY = "USD"
    
    # Timezone
    MARKET_TIMEZONE = "America/New_York"


# ==================== SIMULATION CONFIGURATION ====================

@dataclass
class SimulationConfig:
    """Configuration for market simulations and backtesting."""
    
    # Execution settings
    slippage_bps: float = TradingConstants.DEFAULT_SLIPPAGE_BPS
    commission_per_share: float = TradingConstants.DEFAULT_COMMISSION_PER_SHARE
    min_commission: float = TradingConstants.DEFAULT_MIN_COMMISSION
    max_commission: float = TradingConstants.DEFAULT_MAX_COMMISSION
    spread_bps: float = TradingConstants.DEFAULT_SPREAD_BPS
    
    # Market impact
    market_impact_model: Literal["linear", "sqrt", "none"] = "linear"
    impact_coefficient: float = TradingConstants.LINEAR_IMPACT_COEFFICIENT
    
    # Risk limits
    max_position_size: float = TradingConstants.MAX_POSITION_SIZE
    max_leverage: float = TradingConstants.MAX_LEVERAGE
    max_sector_exposure: float = TradingConstants.MAX_SECTOR_EXPOSURE
    max_daily_loss: float = TradingConstants.MAX_DAILY_LOSS
    
    # Portfolio settings
    starting_cash: float = 100000
    base_currency: str = TradingConstants.BASE_CURRENCY
    allow_short_selling: bool = False
    allow_fractional_shares: bool = True
    rebalance_frequency: Literal["daily", "weekly", "monthly", "quarterly"] = "daily"
    
    # Data settings
    use_adjusted_prices: bool = True
    include_dividends: bool = True
    include_splits: bool = True
    
    # Performance calculation
    benchmark_symbol: str = TradingConstants.DEFAULT_BENCHMARK
    risk_free_rate: float = TradingConstants.DEFAULT_RISK_FREE_RATE
    
    # Timing
    market_open_time: str = "09:30"
    market_close_time: str = "16:00"
    timezone: str = TradingConstants.MARKET_TIMEZONE
    
    # Logging
    log_trades: bool = True
    log_daily_performance: bool = True
    verbose: bool = False
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if self.slippage_bps < 0:
            raise ValueError("Slippage cannot be negative")
        if self.commission_per_share < 0:
            raise ValueError("Commission cannot be negative")
        if self.max_position_size > 1.0:
            raise ValueError("Max position size cannot exceed 100%")
        if self.max_leverage < 1.0:
            raise ValueError("Max leverage cannot be less than 1.0")
        if self.risk_free_rate < 0:
            raise ValueError("Risk free rate cannot be negative")
        if self.starting_cash <= 0:
            raise ValueError("Starting cash must be positive")
    
    def __post_init__(self):
        """Validate after initialization."""
        self.validate()


# ==================== POSITION SIZING ====================

@dataclass
class PositionSizingConfig:
    """Configuration for position sizing strategies."""
    
    method: Literal["fixed", "equal_weight", "kelly", "risk_parity", "volatility_weighted"] = "equal_weight"
    
    # Fixed sizing
    fixed_size: float = 0.05  # 5% per position
    
    # Kelly criterion
    kelly_fraction_cap: float = 0.25  # Cap Kelly at 25%
    kelly_confidence: float = 0.5  # Reduce Kelly by 50% for safety
    
    # Risk parity
    target_risk: float = 0.01  # 1% risk per position
    
    # Volatility weighting
    volatility_lookback: int = 20  # Days for volatility calculation
    volatility_target: float = 0.15  # 15% annual volatility target
    
    # Limits
    min_position_size: float = TradingConstants.MIN_POSITION_SIZE
    max_position_size: float = TradingConstants.MAX_POSITION_SIZE
    max_positions: int = 20
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        signal_strength: float = 1.0,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        payoff_ratio: Optional[float] = None
    ) -> float:
        """Calculate position size based on method."""
        
        if self.method == "fixed":
            size = self.fixed_size
            
        elif self.method == "equal_weight":
            size = 1.0 / self.max_positions
            
        elif self.method == "kelly" and win_rate and payoff_ratio:
            kelly = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio
            kelly = max(0, min(kelly, self.kelly_fraction_cap))
            size = kelly * self.kelly_confidence
            
        elif self.method == "risk_parity" and volatility:
            size = self.target_risk / volatility
            
        elif self.method == "volatility_weighted" and volatility:
            size = self.volatility_target / volatility
            
        else:
            size = self.fixed_size  # Fallback
        
        # Apply signal strength
        size *= signal_strength
        
        # Apply limits
        size = max(self.min_position_size, min(size, self.max_position_size))
        
        return size


# ==================== RISK MANAGEMENT ====================

@dataclass
class RiskManagementConfig:
    """Configuration for risk management."""
    
    # Stop loss
    stop_loss_enabled: bool = True
    stop_loss_percentage: float = 0.02  # 2% stop loss
    trailing_stop_enabled: bool = False
    trailing_stop_percentage: float = 0.03  # 3% trailing stop
    
    # Take profit
    take_profit_enabled: bool = False
    take_profit_percentage: float = 0.10  # 10% take profit
    
    # Portfolio risk
    max_portfolio_risk: float = TradingConstants.MAX_PORTFOLIO_RISK
    max_correlation: float = 0.7  # Max correlation between positions
    max_concentration: float = 0.3  # Max 30% in single position
    
    # Drawdown control
    max_drawdown_limit: float = 0.20  # 20% max drawdown
    drawdown_reduction_factor: float = 0.5  # Reduce size by 50% in drawdown
    
    # Time-based
    max_holding_period: Optional[int] = None  # Days
    force_exit_on_expiry: bool = False
    
    def should_stop_loss(self, entry_price: float, current_price: float) -> bool:
        """Check if stop loss should trigger."""
        if not self.stop_loss_enabled:
            return False
        return (entry_price - current_price) / entry_price >= self.stop_loss_percentage
    
    def should_take_profit(self, entry_price: float, current_price: float) -> bool:
        """Check if take profit should trigger."""
        if not self.take_profit_enabled:
            return False
        return (current_price - entry_price) / entry_price >= self.take_profit_percentage


# ==================== ORDER TYPES ====================

@dataclass
class OrderConfig:
    """Configuration for order execution."""
    
    order_type: Literal["market", "limit", "stop", "stop_limit"] = "market"
    
    # Limit orders
    limit_price_offset: float = 0.0001  # 1 bp from mid
    limit_order_timeout: int = 300  # seconds
    
    # Smart routing
    use_smart_routing: bool = True
    split_large_orders: bool = True
    max_order_size: float = 10000  # shares
    
    # Execution algorithms
    execution_algo: Literal["twap", "vwap", "pov", "immediate"] = "immediate"
    algo_participation_rate: float = 0.1  # 10% of volume
    
    # Timing
    avoid_market_open: bool = True
    avoid_market_close: bool = True
    market_open_delay: int = 30  # minutes
    market_close_buffer: int = 15  # minutes


# ==================== DEFAULT CONFIGURATIONS ====================

DEFAULT_SIMULATION = SimulationConfig()
DEFAULT_POSITION_SIZING = PositionSizingConfig()
DEFAULT_RISK_MANAGEMENT = RiskManagementConfig()
DEFAULT_ORDER_CONFIG = OrderConfig()


# ==================== PRESET CONFIGURATIONS ====================

# Conservative configuration
CONSERVATIVE_CONFIG = {
    "simulation": SimulationConfig(
        max_position_size=0.05,
        max_leverage=1.0,
        allow_short_selling=False
    ),
    "position_sizing": PositionSizingConfig(
        method="fixed",
        fixed_size=0.02,
        max_positions=10
    ),
    "risk_management": RiskManagementConfig(
        stop_loss_percentage=0.01,
        max_drawdown_limit=0.10
    )
}

# Aggressive configuration
AGGRESSIVE_CONFIG = {
    "simulation": SimulationConfig(
        max_position_size=0.20,
        max_leverage=2.0,
        allow_short_selling=True
    ),
    "position_sizing": PositionSizingConfig(
        method="kelly",
        max_positions=5
    ),
    "risk_management": RiskManagementConfig(
        stop_loss_percentage=0.05,
        max_drawdown_limit=0.30
    )
}

# Balanced configuration
BALANCED_CONFIG = {
    "simulation": DEFAULT_SIMULATION,
    "position_sizing": DEFAULT_POSITION_SIZING,
    "risk_management": DEFAULT_RISK_MANAGEMENT
}