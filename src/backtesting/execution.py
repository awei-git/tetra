"""Order execution engine for backtesting."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

from .portfolio import Portfolio

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Types of orders."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order representation."""
    symbol: str
    quantity: float
    order_type: OrderType
    side: str  # 'buy' or 'sell'
    timestamp: datetime
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    trailing_amount: Optional[float] = None  # For trailing stops
    time_in_force: str = "DAY"  # DAY, GTC, IOC, FOK
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_buy(self) -> bool:
        """Check if this is a buy order."""
        return self.side.lower() == 'buy'
    
    @property
    def is_sell(self) -> bool:
        """Check if this is a sell order."""
        return self.side.lower() == 'sell'
    
    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to fill."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_complete(self) -> bool:
        """Check if order is complete."""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                              OrderStatus.REJECTED, OrderStatus.EXPIRED]


@dataclass
class Fill:
    """Order fill information."""
    order_id: str
    symbol: str
    quantity: float
    price: float
    timestamp: datetime
    side: str
    commission: float = 0.0
    slippage: float = 0.0
    
    @property
    def value(self) -> float:
        """Get total fill value."""
        return self.quantity * self.price
    
    @property
    def total_cost(self) -> float:
        """Get total cost including commission and slippage."""
        return self.value + self.commission + self.slippage


class ExecutionEngine:
    """Simulates order execution for backtesting."""
    
    def __init__(self,
                 commission: float = 0.001,  # 0.1%
                 slippage: float = 0.0001,   # 0.01%
                 min_commission: float = 1.0,
                 price_impact: bool = True,
                 liquidity_model: Optional[str] = None):
        """Initialize execution engine.
        
        Args:
            commission: Commission rate (as decimal)
            slippage: Slippage rate (as decimal)
            min_commission: Minimum commission per trade
            price_impact: Whether to model price impact
            liquidity_model: Type of liquidity model to use
        """
        self.commission_rate = commission
        self.slippage_rate = slippage
        self.min_commission = min_commission
        self.price_impact = price_impact
        self.liquidity_model = liquidity_model
        
        # Order tracking
        self.pending_orders: List[Order] = []
        self.order_history: List[Order] = []
        self.fills: List[Fill] = []
        self.next_order_id = 1
    
    def submit_order(self, order: Order) -> str:
        """Submit an order for execution.
        
        Args:
            order: Order to submit
            
        Returns:
            Order ID
        """
        # Assign order ID
        order_id = f"ORD_{self.next_order_id:06d}"
        order.metadata['order_id'] = order_id
        self.next_order_id += 1
        
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            self.order_history.append(order)
            return order_id
        
        # Add to pending orders
        order.status = OrderStatus.SUBMITTED
        self.pending_orders.append(order)
        
        logger.debug(f"Submitted {order.side} order: {order.quantity} {order.symbol} @ {order.order_type.value}")
        return order_id
    
    def execute(self,
                order: Order,
                price: float,
                portfolio: Portfolio,
                volume: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Execute an order immediately.
        
        Args:
            order: Order to execute
            price: Current market price
            portfolio: Portfolio to update
            volume: Current volume (for liquidity modeling)
            
        Returns:
            Fill information if executed, None otherwise
        """
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            return None
        
        # Check if order should execute based on type
        execution_price = self._get_execution_price(order, price)
        if execution_price is None:
            return None
        
        # Apply slippage
        if order.is_buy:
            execution_price *= (1 + self.slippage_rate)
        else:
            execution_price *= (1 - self.slippage_rate)
        
        # Model price impact if enabled
        if self.price_impact and volume:
            impact = self._calculate_price_impact(order.quantity, volume)
            if order.is_buy:
                execution_price *= (1 + impact)
            else:
                execution_price *= (1 - impact)
        
        # Calculate commission
        commission = self._calculate_commission(order.quantity, execution_price)
        
        # Execute with portfolio
        if order.is_buy:
            # Buying/covering
            success = portfolio.add_position(
                symbol=order.symbol,
                side='long' if order.side == 'buy' else 'short',
                quantity=order.quantity,
                price=execution_price,
                commission=commission / (order.quantity * execution_price),
                slippage=self.slippage_rate
            )
        else:
            # Selling/shorting
            if order.symbol in portfolio.positions:
                success = portfolio.reduce_position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    price=execution_price,
                    commission=commission / (order.quantity * execution_price),
                    slippage=self.slippage_rate
                )
            else:
                # Short selling
                success = portfolio.add_position(
                    symbol=order.symbol,
                    side='short',
                    quantity=order.quantity,
                    price=execution_price,
                    commission=commission / (order.quantity * execution_price),
                    slippage=self.slippage_rate
                )
        
        if not success:
            order.status = OrderStatus.REJECTED
            return None
        
        # Create fill
        fill = Fill(
            order_id=order.metadata.get('order_id', 'UNKNOWN'),
            symbol=order.symbol,
            quantity=order.quantity,
            price=execution_price,
            timestamp=order.timestamp,
            side=order.side,
            commission=commission,
            slippage=abs(execution_price - price) * order.quantity
        )
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = execution_price
        order.commission = commission
        order.slippage = fill.slippage
        
        # Record fill
        self.fills.append(fill)
        self.order_history.append(order)
        
        logger.debug(f"Filled {order.side} order: {order.quantity} {order.symbol} @ {execution_price:.2f}")
        
        return {
            'symbol': order.symbol,
            'quantity': order.quantity,
            'price': execution_price,
            'side': order.side,
            'commission': commission,
            'slippage': fill.slippage,
            'timestamp': order.timestamp,
            'order_id': order.metadata.get('order_id', 'UNKNOWN')
        }
    
    def process_pending_orders(self,
                              market_data: Dict[str, Dict[str, float]],
                              portfolio: Portfolio,
                              timestamp: datetime) -> List[Fill]:
        """Process all pending orders.
        
        Args:
            market_data: Current market data (symbol -> OHLCV dict)
            portfolio: Portfolio to update
            timestamp: Current timestamp
            
        Returns:
            List of fills executed
        """
        fills = []
        orders_to_remove = []
        
        for order in self.pending_orders:
            symbol_data = market_data.get(order.symbol)
            if not symbol_data:
                continue
            
            # Check if order should execute
            should_execute, execution_price = self._check_order_execution(
                order, symbol_data, timestamp
            )
            
            if should_execute:
                # Execute order
                fill_info = self.execute(
                    order=order,
                    price=execution_price,
                    portfolio=portfolio,
                    volume=symbol_data.get('volume')
                )
                
                if fill_info:
                    fills.append(Fill(**fill_info))
                    orders_to_remove.append(order)
            
            # Check for expired orders
            elif order.time_in_force == "DAY" and timestamp.date() > order.timestamp.date():
                order.status = OrderStatus.EXPIRED
                orders_to_remove.append(order)
                self.order_history.append(order)
        
        # Remove executed/expired orders
        for order in orders_to_remove:
            self.pending_orders.remove(order)
        
        return fills
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if order was cancelled
        """
        for order in self.pending_orders:
            if order.metadata.get('order_id') == order_id:
                order.status = OrderStatus.CANCELLED
                self.pending_orders.remove(order)
                self.order_history.append(order)
                logger.debug(f"Cancelled order: {order_id}")
                return True
        return False
    
    def cancel_all_orders(self, symbol: Optional[str] = None):
        """Cancel all pending orders.
        
        Args:
            symbol: If specified, only cancel orders for this symbol
        """
        orders_to_cancel = []
        
        for order in self.pending_orders:
            if symbol is None or order.symbol == symbol:
                order.status = OrderStatus.CANCELLED
                orders_to_cancel.append(order)
                self.order_history.append(order)
        
        for order in orders_to_cancel:
            self.pending_orders.remove(order)
        
        logger.debug(f"Cancelled {len(orders_to_cancel)} orders")
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order parameters."""
        if order.quantity <= 0:
            logger.error(f"Invalid order quantity: {order.quantity}")
            return False
        
        if order.order_type == OrderType.LIMIT and order.price is None:
            logger.error("Limit order requires price")
            return False
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            logger.error("Stop order requires stop price")
            return False
        
        return True
    
    def _get_execution_price(self, order: Order, market_price: float) -> Optional[float]:
        """Get execution price based on order type."""
        if order.order_type == OrderType.MARKET:
            return market_price
        
        elif order.order_type == OrderType.LIMIT:
            if order.is_buy and market_price <= order.price:
                return min(market_price, order.price)
            elif order.is_sell and market_price >= order.price:
                return max(market_price, order.price)
        
        elif order.order_type == OrderType.STOP:
            if order.is_buy and market_price >= order.stop_price:
                return market_price
            elif order.is_sell and market_price <= order.stop_price:
                return market_price
        
        return None
    
    def _check_order_execution(self,
                              order: Order,
                              market_data: Dict[str, float],
                              timestamp: datetime) -> Tuple[bool, float]:
        """Check if order should execute with current market data."""
        # For backtesting, use high/low to check if order would have executed
        high = market_data.get('high', market_data['close'])
        low = market_data.get('low', market_data['close'])
        close = market_data['close']
        
        if order.order_type == OrderType.MARKET:
            return True, close
        
        elif order.order_type == OrderType.LIMIT:
            if order.is_buy and low <= order.price:
                return True, min(order.price, close)
            elif order.is_sell and high >= order.price:
                return True, max(order.price, close)
        
        elif order.order_type == OrderType.STOP:
            if order.is_buy and high >= order.stop_price:
                return True, max(order.stop_price, close)
            elif order.is_sell and low <= order.stop_price:
                return True, min(order.stop_price, close)
        
        return False, 0.0
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for a trade."""
        commission = quantity * price * self.commission_rate
        return max(commission, self.min_commission)
    
    def _calculate_price_impact(self, quantity: float, volume: float) -> float:
        """Calculate price impact based on order size relative to volume."""
        if volume <= 0:
            return 0.0
        
        # Simple square-root impact model
        participation_rate = quantity / volume
        impact = 0.1 * np.sqrt(participation_rate)  # 10% impact at 100% participation
        
        return min(impact, 0.05)  # Cap at 5% impact
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.fills:
            return {
                'total_orders': 0,
                'filled_orders': 0,
                'rejected_orders': 0,
                'fill_rate': 0.0,
                'avg_slippage': 0.0,
                'total_commission': 0.0
            }
        
        filled_orders = [o for o in self.order_history if o.status == OrderStatus.FILLED]
        rejected_orders = [o for o in self.order_history if o.status == OrderStatus.REJECTED]
        
        return {
            'total_orders': len(self.order_history),
            'filled_orders': len(filled_orders),
            'rejected_orders': len(rejected_orders),
            'fill_rate': len(filled_orders) / len(self.order_history) if self.order_history else 0,
            'avg_slippage': np.mean([f.slippage for f in self.fills]) if self.fills else 0,
            'total_commission': sum(f.commission for f in self.fills),
            'avg_commission': np.mean([f.commission for f in self.fills]) if self.fills else 0
        }