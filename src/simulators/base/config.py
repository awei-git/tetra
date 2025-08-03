"""Simulation configuration."""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class SimulationConfig:
    """Configuration for market simulations."""
    
    # Execution settings
    slippage_bps: float = 10  # basis points
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    max_commission: float = 100.0
    
    # Market impact
    market_impact_model: Literal["linear", "sqrt", "none"] = "linear"
    impact_coefficient: float = 0.1
    
    # Risk limits
    max_position_size: float = 0.1  # 10% of portfolio
    max_leverage: float = 1.0
    max_sector_exposure: float = 0.3  # 30% in one sector
    
    # Data settings
    use_adjusted_prices: bool = True
    include_dividends: bool = True
    include_splits: bool = True
    
    # Performance calculation
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = 0.02  # Annual rate
    
    # Simulation settings
    starting_cash: float = 100000
    base_currency: str = "USD"
    allow_short_selling: bool = False
    allow_fractional_shares: bool = True
    
    # Timing
    market_open_time: str = "09:30"
    market_close_time: str = "16:00"
    timezone: str = "America/New_York"
    
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