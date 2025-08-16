"""Abstract base class for all simulators."""

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Dict, List, Optional, Any

from ..portfolio import Portfolio
from src.definitions.trading import SimulationConfig
from .result import SimulationResult


class BaseSimulator(ABC):
    """Abstract base class for all market simulators."""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize the simulator.
        
        Args:
            config: Simulation configuration. If None, uses defaults.
        """
        self.config = config or SimulationConfig()
        self._is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize the simulator (e.g., load data, connect to DB)."""
        if not self._is_initialized:
            await self._initialize()
            self._is_initialized = True
            
    @abstractmethod
    async def _initialize(self) -> None:
        """Implementation-specific initialization."""
        pass
        
    @abstractmethod
    async def run_simulation(
        self,
        portfolio: Portfolio,
        start_date: date,
        end_date: date,
        strategy: Optional[Any] = None,
        **kwargs
    ) -> SimulationResult:
        """
        Run simulation for given period.
        
        Args:
            portfolio: Portfolio to simulate
            start_date: Simulation start date
            end_date: Simulation end date
            strategy: Optional trading strategy to execute
            **kwargs: Additional simulator-specific parameters
            
        Returns:
            SimulationResult containing performance metrics and history
        """
        pass
    
    @abstractmethod
    async def get_market_data(
        self,
        symbols: List[str],
        date: date
    ) -> Dict[str, Any]:
        """
        Get market data for specific date.
        
        Args:
            symbols: List of symbols to get data for
            date: Date to get data for
            
        Returns:
            Dictionary mapping symbols to market data
        """
        pass
    
    def apply_slippage(self, price: float, is_buy: bool) -> float:
        """
        Apply slippage to execution price.
        
        Args:
            price: Base execution price
            is_buy: True if buying, False if selling
            
        Returns:
            Adjusted price with slippage
        """
        slippage_factor = self.config.slippage_bps / 10000
        if is_buy:
            return price * (1 + slippage_factor)
        else:
            return price * (1 - slippage_factor)
    
    def calculate_commission(
        self, 
        shares: int, 
        price: float,
        order_type: str = "MARKET"
    ) -> float:
        """
        Calculate trading commission.
        
        Args:
            shares: Number of shares
            price: Price per share
            order_type: Type of order
            
        Returns:
            Commission amount
        """
        commission = shares * self.config.commission_per_share
        return max(commission, self.config.min_commission)
    
    def calculate_market_impact(
        self,
        shares: int,
        avg_volume: float,
        price: float
    ) -> float:
        """
        Calculate market impact of order.
        
        Args:
            shares: Number of shares to trade
            avg_volume: Average daily volume
            price: Current price
            
        Returns:
            Price impact as percentage
        """
        if avg_volume == 0 or self.config.market_impact_model == "none":
            return 0
            
        participation_rate = shares / avg_volume
        
        if self.config.market_impact_model == "linear":
            impact = participation_rate * self.config.impact_coefficient
        elif self.config.market_impact_model == "sqrt":
            impact = (participation_rate ** 0.5) * self.config.impact_coefficient
        else:
            impact = 0
            
        return min(impact, 0.05)  # Cap at 5% impact
    
    async def cleanup(self) -> None:
        """Clean up resources (close connections, etc)."""
        if self._is_initialized:
            await self._cleanup()
            self._is_initialized = False
            
    async def _cleanup(self) -> None:
        """Implementation-specific cleanup."""
        pass