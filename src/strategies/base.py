"""Base strategy classes and types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TradeStatus(Enum):
    """Trade status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_price is None
    
    @property
    def duration(self) -> Optional[pd.Timedelta]:
        """Get trade duration."""
        if self.exit_time:
            return self.exit_time - self.entry_time
        return None
    
    def close(self, exit_price: float, exit_time: datetime):
        """Close the trade."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        
        # Calculate P&L
        if self.side == PositionSide.LONG:
            self.pnl = (exit_price - self.entry_price) * self.quantity - self.commission - self.slippage
        else:  # SHORT
            self.pnl = (self.entry_price - exit_price) * self.quantity - self.commission - self.slippage
        
        self.pnl_percent = (self.pnl / (self.entry_price * self.quantity)) * 100


@dataclass
class Position:
    """Current position in a symbol."""
    symbol: str
    side: PositionSide
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def market_value(self) -> float:
        """Get current market value."""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Get total cost basis."""
        return self.quantity * self.avg_price
    
    def update_price(self, price: float):
        """Update current price and unrealized P&L."""
        self.current_price = price
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (price - self.avg_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            self.unrealized_pnl = (self.avg_price - price) * self.quantity


@dataclass
class StrategyState:
    """Current state of the strategy."""
    timestamp: datetime
    positions: Dict[str, Position] = field(default_factory=dict)
    open_trades: List[Trade] = field(default_factory=list)
    closed_trades: List[Trade] = field(default_factory=list)
    cash: float = 100000.0  # Starting cash
    total_value: float = 100000.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_flat(self) -> bool:
        """Check if strategy has no positions."""
        return len(self.positions) == 0
    
    @property
    def total_positions(self) -> int:
        """Get total number of positions."""
        return len(self.positions)
    
    @property
    def total_trades(self) -> int:
        """Get total number of trades."""
        return len(self.closed_trades) + len(self.open_trades)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def update_market_values(self, prices: Dict[str, float]):
        """Update all position market values."""
        position_value = 0.0
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])
                position_value += position.market_value
        
        self.total_value = self.cash + position_value


@dataclass
class StrategyResult:
    """Results from strategy execution."""
    trades: List[Trade]
    equity_curve: pd.Series
    positions: pd.DataFrame
    metrics: Dict[str, Any]
    signals: Optional[pd.DataFrame] = None
    events: Optional[List[Dict[str, Any]]] = None


class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, 
                 name: str,
                 description: str = "",
                 initial_capital: float = 100000.0,
                 commission: float = 0.001,  # 0.1% commission
                 slippage: float = 0.0001,   # 0.01% slippage
                 max_positions: int = 10,
                 position_size: float = 0.1):  # 10% of capital per position
        self.name = name
        self.description = description
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_positions = max_positions
        self.position_size = position_size
        
        # Initialize state
        self.state = StrategyState(
            timestamp=datetime.now(),
            cash=initial_capital,
            total_value=initial_capital
        )
        
        # Risk management parameters
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.trailing_stop: Optional[float] = None
    
    @abstractmethod
    def generate_signals(self, 
                        data: pd.DataFrame,
                        signals: Optional[pd.DataFrame] = None,
                        events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate trading signals.
        
        Args:
            data: OHLCV data
            signals: Computed technical/ML signals
            events: Event data (earnings, economic releases, etc.)
            
        Returns:
            DataFrame with columns: signal (-1, 0, 1), position_size
        """
        pass
    
    @abstractmethod
    def should_enter(self, 
                    symbol: str,
                    timestamp: datetime,
                    data: pd.Series,
                    signals: Optional[pd.Series] = None,
                    events: Optional[pd.Series] = None) -> Tuple[bool, PositionSide, float]:
        """Determine if should enter position.
        
        Returns:
            Tuple of (should_enter, side, size)
        """
        pass
    
    @abstractmethod
    def should_exit(self,
                   position: Position,
                   timestamp: datetime,
                   data: pd.Series,
                   signals: Optional[pd.Series] = None,
                   events: Optional[pd.Series] = None) -> bool:
        """Determine if should exit position."""
        pass
    
    def calculate_position_size(self, 
                              symbol: str,
                              price: float,
                              volatility: Optional[float] = None) -> float:
        """Calculate position size based on risk management rules.
        
        Args:
            symbol: Symbol to trade
            price: Current price
            volatility: Optional volatility for risk-based sizing
            
        Returns:
            Number of shares/units to trade
        """
        # Fixed percentage allocation
        allocation = self.state.total_value * self.position_size
        
        # Volatility-based adjustment
        if volatility and volatility > 0:
            # Reduce size for high volatility
            volatility_adj = min(1.0, 0.02 / volatility)  # Target 2% volatility
            allocation *= volatility_adj
        
        # Calculate shares
        shares = allocation / price
        
        # Round to reasonable lot size
        if price > 100:
            shares = round(shares)
        elif price > 10:
            shares = round(shares, 1)
        else:
            shares = round(shares, 2)
        
        return max(1, shares)
    
    def execute_trade(self,
                     symbol: str,
                     side: PositionSide,
                     quantity: float,
                     price: float,
                     timestamp: datetime,
                     metadata: Optional[Dict[str, Any]] = None) -> Optional[Trade]:
        """Execute a trade.
        
        Args:
            symbol: Symbol to trade
            side: Buy or sell
            quantity: Number of shares
            price: Execution price
            timestamp: Trade timestamp
            metadata: Additional trade metadata
            
        Returns:
            Trade object if executed, None otherwise
        """
        # Check if we can trade
        if side != PositionSide.FLAT:
            # Check max positions
            if symbol not in self.state.positions and len(self.state.positions) >= self.max_positions:
                logger.warning(f"Max positions ({self.max_positions}) reached, skipping {symbol}")
                return None
            
            # Check capital
            required_capital = quantity * price * (1 + self.commission + self.slippage)
            if required_capital > self.state.cash:
                logger.warning(f"Insufficient capital for {symbol}: need {required_capital}, have {self.state.cash}")
                return None
        
        # Create trade
        trade = Trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price * (1 + self.slippage if side == PositionSide.LONG else 1 - self.slippage),
            entry_time=timestamp,
            commission=quantity * price * self.commission,
            slippage=quantity * price * self.slippage,
            metadata=metadata or {}
        )
        
        # Update position
        if side != PositionSide.FLAT:
            if symbol in self.state.positions:
                # Add to existing position
                position = self.state.positions[symbol]
                total_quantity = position.quantity + quantity
                position.avg_price = (position.avg_price * position.quantity + trade.entry_price * quantity) / total_quantity
                position.quantity = total_quantity
                position.trades.append(trade)
            else:
                # New position
                position = Position(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    avg_price=trade.entry_price,
                    current_price=trade.entry_price,
                    trades=[trade]
                )
                self.state.positions[symbol] = position
            
            # Update cash
            self.state.cash -= required_capital
            self.state.open_trades.append(trade)
        
        logger.info(f"Executed trade: {side.value} {quantity} {symbol} @ {trade.entry_price:.2f}")
        return trade
    
    def close_position(self,
                      symbol: str,
                      price: float,
                      timestamp: datetime,
                      reason: str = "signal") -> Optional[Trade]:
        """Close a position.
        
        Args:
            symbol: Symbol to close
            price: Exit price
            timestamp: Exit timestamp
            reason: Reason for closing
            
        Returns:
            Trade object if closed
        """
        position = self.state.positions.get(symbol)
        if not position:
            logger.warning(f"No position to close for {symbol}")
            return None
        
        # Close all trades in position
        for trade in position.trades:
            if trade.is_open:
                trade.close(price, timestamp)
                trade.metadata['exit_reason'] = reason
                self.state.open_trades.remove(trade)
                self.state.closed_trades.append(trade)
                
                # Update realized P&L
                position.realized_pnl += trade.pnl
        
        # Update cash
        exit_value = position.quantity * price * (1 - self.commission - self.slippage)
        self.state.cash += exit_value
        
        # Remove position
        del self.state.positions[symbol]
        
        logger.info(f"Closed position: {symbol} @ {price:.2f}, P&L: {position.realized_pnl:.2f}")
        return position.trades[-1] if position.trades else None
    
    def update_stops(self, position: Position, current_price: float):
        """Update stop loss and take profit levels.
        
        Args:
            position: Current position
            current_price: Current market price
        """
        # Trailing stop
        if self.trailing_stop and position.side == PositionSide.LONG:
            new_stop = current_price * (1 - self.trailing_stop)
            if 'stop_loss' not in position.metadata or new_stop > position.metadata['stop_loss']:
                position.metadata['stop_loss'] = new_stop
                logger.debug(f"Updated trailing stop for {position.symbol}: {new_stop:.2f}")
    
    def check_risk_limits(self, position: Position, current_price: float) -> bool:
        """Check if position hits risk limits.
        
        Returns:
            True if should exit position
        """
        # Stop loss
        if self.stop_loss:
            if position.side == PositionSide.LONG and current_price <= position.avg_price * (1 - self.stop_loss):
                logger.info(f"Stop loss triggered for {position.symbol}")
                return True
            elif position.side == PositionSide.SHORT and current_price >= position.avg_price * (1 + self.stop_loss):
                logger.info(f"Stop loss triggered for {position.symbol}")
                return True
        
        # Take profit
        if self.take_profit:
            if position.side == PositionSide.LONG and current_price >= position.avg_price * (1 + self.take_profit):
                logger.info(f"Take profit triggered for {position.symbol}")
                return True
            elif position.side == PositionSide.SHORT and current_price <= position.avg_price * (1 - self.take_profit):
                logger.info(f"Take profit triggered for {position.symbol}")
                return True
        
        # Dynamic stop loss
        if 'stop_loss' in position.metadata:
            if position.side == PositionSide.LONG and current_price <= position.metadata['stop_loss']:
                logger.info(f"Trailing stop triggered for {position.symbol}")
                return True
        
        return False
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate strategy performance metrics."""
        if not self.state.closed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_pnl': 0,
                'total_return': 0
            }
        
        # Calculate metrics
        pnls = [trade.pnl for trade in self.state.closed_trades if trade.pnl is not None]
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        
        metrics = {
            'total_trades': len(self.state.closed_trades),
            'win_rate': len(wins) / len(pnls) if pnls else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'profit_factor': sum(wins) / abs(sum(losses)) if losses else float('inf'),
            'total_pnl': sum(pnls),
            'total_return': sum(pnls) / self.initial_capital * 100
        }
        
        # Add more metrics as needed
        return metrics