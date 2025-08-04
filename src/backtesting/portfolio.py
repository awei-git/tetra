"""Portfolio management for backtesting."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from ..strategies.base import Position, PositionSide

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Snapshot of portfolio state at a point in time."""
    timestamp: datetime
    cash: float
    positions: Dict[str, Position]
    total_value: float
    leverage: float = 1.0
    margin_used: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'cash': self.cash,
            'total_value': self.total_value,
            'leverage': self.leverage,
            'margin_used': self.margin_used,
            'num_positions': len(self.positions),
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl
                }
                for symbol, pos in self.positions.items()
            }
        }


class Portfolio:
    """Portfolio manager for backtesting."""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 margin_enabled: bool = False,
                 margin_requirement: float = 0.5,  # 50% margin requirement
                 max_leverage: float = 2.0):
        """Initialize portfolio.
        
        Args:
            initial_capital: Starting capital
            margin_enabled: Whether to allow margin trading
            margin_requirement: Margin requirement (0.5 = 50%)
            max_leverage: Maximum allowed leverage
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.margin_enabled = margin_enabled
        self.margin_requirement = margin_requirement
        self.max_leverage = max_leverage
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.total_value = initial_capital
        self.buying_power = initial_capital
        
        # Performance tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        # Risk metrics
        self.current_leverage = 1.0
        self.margin_used = 0.0
        self.margin_available = initial_capital if margin_enabled else 0.0
    
    def update_prices(self, prices: Dict[str, float]):
        """Update portfolio with current market prices.
        
        Args:
            prices: Dictionary of symbol -> price
        """
        # Update position prices
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])
        
        # Recalculate portfolio value
        self._calculate_portfolio_value()
        
        # Update margin and leverage
        if self.margin_enabled:
            self._update_margin_status()
    
    def add_position(self, 
                    symbol: str,
                    side: PositionSide,
                    quantity: float,
                    price: float,
                    commission: float = 0.0,
                    slippage: float = 0.0) -> bool:
        """Add or update a position.
        
        Args:
            symbol: Symbol to trade
            side: Long or short
            quantity: Number of shares
            price: Execution price
            commission: Commission paid
            slippage: Slippage cost
            
        Returns:
            True if position was added/updated successfully
        """
        # Check if we have enough buying power
        required_capital = quantity * price * (1 + commission + slippage)
        if side == PositionSide.SHORT and not self.margin_enabled:
            logger.warning(f"Cannot short {symbol} without margin enabled")
            return False
        
        if required_capital > self.buying_power:
            logger.warning(f"Insufficient buying power for {symbol}: need {required_capital}, have {self.buying_power}")
            return False
        
        # Update or create position
        if symbol in self.positions:
            # Add to existing position
            position = self.positions[symbol]
            if position.side != side:
                logger.warning(f"Cannot add {side} position to existing {position.side} position for {symbol}")
                return False
            
            # Update position
            total_quantity = position.quantity + quantity
            position.avg_price = (position.avg_price * position.quantity + price * quantity) / total_quantity
            position.quantity = total_quantity
        else:
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                avg_price=price,
                current_price=price
            )
        
        # Update cash and costs
        self.cash -= required_capital
        self.total_commission += commission * quantity * price
        self.total_slippage += slippage * quantity * price
        
        # Recalculate portfolio
        self._calculate_portfolio_value()
        
        logger.debug(f"Added {side} position: {quantity} {symbol} @ {price}")
        return True
    
    def reduce_position(self,
                       symbol: str,
                       quantity: float,
                       price: float,
                       commission: float = 0.0,
                       slippage: float = 0.0) -> bool:
        """Reduce or close a position.
        
        Args:
            symbol: Symbol to trade
            quantity: Number of shares to sell/cover
            price: Execution price
            commission: Commission paid
            slippage: Slippage cost
            
        Returns:
            True if position was reduced successfully
        """
        if symbol not in self.positions:
            logger.warning(f"No position to reduce for {symbol}")
            return False
        
        position = self.positions[symbol]
        
        if quantity > position.quantity:
            logger.warning(f"Cannot reduce {quantity} shares of {symbol}, only have {position.quantity}")
            quantity = position.quantity
        
        # Calculate P&L
        if position.side == PositionSide.LONG:
            pnl = (price - position.avg_price) * quantity
        else:  # SHORT
            pnl = (position.avg_price - price) * quantity
        
        # Account for costs
        costs = (commission + slippage) * quantity * price
        pnl -= costs
        
        # Update position
        position.quantity -= quantity
        position.realized_pnl += pnl
        self.realized_pnl += pnl
        
        # Update cash
        proceeds = quantity * price * (1 - commission - slippage)
        self.cash += proceeds
        
        # Update costs
        self.total_commission += commission * quantity * price
        self.total_slippage += slippage * quantity * price
        
        # Remove position if fully closed
        if position.quantity == 0:
            del self.positions[symbol]
            logger.debug(f"Closed position: {symbol}")
        else:
            logger.debug(f"Reduced {position.side} position: {quantity} {symbol} @ {price}")
        
        # Recalculate portfolio
        self._calculate_portfolio_value()
        
        return True
    
    def close_position(self,
                      symbol: str,
                      price: float,
                      commission: float = 0.0,
                      slippage: float = 0.0) -> bool:
        """Close entire position.
        
        Args:
            symbol: Symbol to close
            price: Execution price
            commission: Commission rate
            slippage: Slippage rate
            
        Returns:
            True if position was closed successfully
        """
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        return self.reduce_position(symbol, position.quantity, price, commission, slippage)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def get_state(self) -> PortfolioState:
        """Get current portfolio state."""
        return PortfolioState(
            timestamp=datetime.now(),
            cash=self.cash,
            positions=self.positions.copy(),
            total_value=self.total_value,
            leverage=self.current_leverage,
            margin_used=self.margin_used
        )
    
    def _calculate_portfolio_value(self):
        """Calculate total portfolio value."""
        # Calculate position values
        position_value = sum(pos.market_value for pos in self.positions.values())
        
        # Calculate unrealized P&L
        self.unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # Total value
        self.total_value = self.cash + position_value
        
        # Update buying power
        if self.margin_enabled:
            self.buying_power = self.cash + (self.margin_available - self.margin_used)
        else:
            self.buying_power = self.cash
    
    def _update_margin_status(self):
        """Update margin and leverage calculations."""
        if not self.margin_enabled:
            return
        
        # Calculate total position value
        long_value = sum(
            pos.market_value for pos in self.positions.values() 
            if pos.side == PositionSide.LONG
        )
        short_value = sum(
            pos.market_value for pos in self.positions.values() 
            if pos.side == PositionSide.SHORT
        )
        
        # Calculate margin used
        self.margin_used = short_value * self.margin_requirement
        
        # Calculate leverage
        total_exposure = long_value + short_value
        if self.total_value > 0:
            self.current_leverage = total_exposure / self.total_value
        else:
            self.current_leverage = 0
        
        # Check margin call
        if self.current_leverage > self.max_leverage:
            logger.warning(f"Leverage {self.current_leverage:.2f} exceeds max {self.max_leverage}")
        
        # Update available margin
        self.margin_available = self.total_value * (self.max_leverage - 1)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        return {
            'total_value': self.total_value,
            'cash': self.cash,
            'buying_power': self.buying_power,
            'positions': len(self.positions),
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.realized_pnl + self.unrealized_pnl,
            'return_pct': ((self.total_value - self.initial_capital) / self.initial_capital) * 100,
            'leverage': self.current_leverage,
            'margin_used': self.margin_used,
            'commission_paid': self.total_commission,
            'slippage_cost': self.total_slippage
        }