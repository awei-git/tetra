"""Individual position tracking."""

from datetime import datetime
from typing import Optional


class Position:
    """Track individual stock position."""
    
    def __init__(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        entry_time: datetime,
        commission: float = 0
    ):
        """
        Initialize position.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares (negative for short)
            entry_price: Average entry price
            entry_time: Time of initial entry
            commission: Total commission paid
        """
        self.symbol = symbol
        self.quantity = quantity
        self.cost_basis = abs(quantity * entry_price) + commission
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.last_price = entry_price
        self.total_commission = commission
        self.realized_pnl = 0.0
        self._split_adjustment = 1.0
        
    def add_shares(
        self, 
        quantity: float, 
        price: float,
        commission: float = 0
    ) -> None:
        """
        Add shares to position (or reduce if negative).
        
        Args:
            quantity: Number of shares to add (negative to reduce)
            price: Execution price
            commission: Trading commission
        """
        # Handle position reduction
        if self.quantity > 0 and quantity < 0:  # Long position, selling
            # Calculate realized P&L
            reduction_ratio = min(abs(quantity) / self.quantity, 1.0)
            realized_cost = self.cost_basis * reduction_ratio
            realized_revenue = abs(quantity) * price - commission
            self.realized_pnl += realized_revenue - realized_cost
            
            # Update cost basis
            self.cost_basis *= (1 - reduction_ratio)
            
        elif self.quantity < 0 and quantity > 0:  # Short position, buying
            # Calculate realized P&L
            reduction_ratio = min(quantity / abs(self.quantity), 1.0)
            realized_cost = self.cost_basis * reduction_ratio
            realized_revenue = self.cost_basis * reduction_ratio - (quantity * price + commission)
            self.realized_pnl += realized_revenue
            
            # Update cost basis
            self.cost_basis *= (1 - reduction_ratio)
        
        # Handle position increase
        else:
            self.cost_basis += abs(quantity * price) + commission
            
        # Update quantity
        self.quantity += quantity
        
        # Update average entry price if position still exists
        if self.quantity != 0:
            self.entry_price = self.cost_basis / abs(self.quantity)
            
        # Update commission tracking
        self.total_commission += commission
        
    def update_price(self, current_price: float) -> None:
        """Update last known price."""
        self.last_price = current_price
        
    def get_market_value(self, current_price: Optional[float] = None) -> float:
        """
        Get current market value.
        
        Args:
            current_price: Current market price (uses last_price if None)
            
        Returns:
            Market value (negative for short positions)
        """
        price = current_price or self.last_price
        return self.quantity * price
    
    def get_unrealized_pnl(self, current_price: Optional[float] = None) -> float:
        """
        Calculate unrealized P&L.
        
        Args:
            current_price: Current market price
            
        Returns:
            Unrealized profit/loss
        """
        price = current_price or self.last_price
        
        if self.quantity > 0:  # Long position
            return (price * self.quantity) - self.cost_basis
        else:  # Short position
            return self.cost_basis - abs(price * self.quantity)
    
    def get_unrealized_pnl_percent(self, current_price: Optional[float] = None) -> float:
        """
        Calculate unrealized P&L percentage.
        
        Args:
            current_price: Current market price
            
        Returns:
            Unrealized P&L as percentage of cost basis
        """
        if self.cost_basis == 0:
            return 0
            
        unrealized_pnl = self.get_unrealized_pnl(current_price)
        return unrealized_pnl / self.cost_basis
    
    def get_total_pnl(self, current_price: Optional[float] = None) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.realized_pnl + self.get_unrealized_pnl(current_price)
    
    def apply_split(self, split_ratio: float) -> None:
        """
        Apply stock split to position.
        
        Args:
            split_ratio: Split ratio (e.g., 2.0 for 2:1 split)
        """
        self.quantity *= split_ratio
        self.entry_price /= split_ratio
        self.last_price /= split_ratio
        self._split_adjustment *= split_ratio
    
    def is_long(self) -> bool:
        """Check if this is a long position."""
        return self.quantity > 0
    
    def is_short(self) -> bool:
        """Check if this is a short position."""
        return self.quantity < 0
    
    def days_held(self, as_of: Optional[datetime] = None) -> int:
        """Calculate days position has been held."""
        end_date = as_of or datetime.now()
        return (end_date - self.entry_time).days
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"Position({self.symbol}: {self.quantity:.2f} shares @ "
                f"${self.entry_price:.2f}, P&L: ${self.get_total_pnl():.2f})")