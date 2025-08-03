"""Portfolio management class."""

from datetime import datetime
from typing import Dict, List, Optional, Set
import pandas as pd

from .position import Position
from .transaction import Transaction, TransactionType
from .cash_manager import CashManager


class Portfolio:
    """Manage positions, cash, and performance."""
    
    def __init__(
        self, 
        initial_cash: float,
        base_currency: str = "USD",
        allow_short: bool = False,
        allow_fractional: bool = True
    ):
        """
        Initialize portfolio.
        
        Args:
            initial_cash: Starting cash amount
            base_currency: Base currency for portfolio
            allow_short: Whether to allow short positions
            allow_fractional: Whether to allow fractional shares
        """
        self.cash_manager = CashManager(initial_cash, base_currency)
        self.positions: Dict[str, Position] = {}
        self.transactions: List[Transaction] = []
        self.allow_short = allow_short
        self.allow_fractional = allow_fractional
        self._initial_value = initial_cash
        self._creation_time = datetime.now()
        
    @property
    def cash(self) -> float:
        """Get current cash balance."""
        return self.cash_manager.get_balance()
    
    @property
    def initial_value(self) -> float:
        """Get initial portfolio value."""
        return self._initial_value
        
    def add_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        commission: float = 0,
        order_id: Optional[str] = None
    ) -> Position:
        """
        Add or update a position.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares (negative for short)
            price: Execution price
            timestamp: Transaction timestamp
            commission: Trading commission
            order_id: Optional order identifier
            
        Returns:
            Updated position
            
        Raises:
            ValueError: If trade violates portfolio rules
        """
        # Validate trade
        if not self.allow_short and quantity < 0:
            raise ValueError("Short selling not allowed")
            
        if not self.allow_fractional and quantity != int(quantity):
            raise ValueError("Fractional shares not allowed")
            
        # Calculate total cost
        gross_value = quantity * price
        total_cost = gross_value + commission
        
        # Check cash sufficiency for buys
        if quantity > 0 and total_cost > self.cash:
            raise ValueError(f"Insufficient cash: need ${total_cost:.2f}, have ${self.cash:.2f}")
        
        # Create transaction
        transaction = Transaction(
            symbol=symbol,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            commission=commission,
            transaction_type=TransactionType.BUY if quantity > 0 else TransactionType.SELL,
            order_id=order_id
        )
        
        # Update cash
        self.cash_manager.update_balance(-total_cost)
        
        # Update position
        if symbol in self.positions:
            self.positions[symbol].add_shares(quantity, price, commission)
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity, 
                entry_price=price,
                entry_time=timestamp,
                commission=commission
            )
        
        # Remove position if quantity is zero
        if self.positions[symbol].quantity == 0:
            del self.positions[symbol]
            
        # Record transaction
        self.transactions.append(transaction)
        
        return self.positions.get(symbol)
    
    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        commission: float = 0,
        order_id: Optional[str] = None
    ) -> Optional[Transaction]:
        """
        Close an entire position.
        
        Args:
            symbol: Stock symbol
            price: Execution price
            timestamp: Transaction timestamp
            commission: Trading commission
            order_id: Optional order identifier
            
        Returns:
            Closing transaction if position existed
        """
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        quantity = -position.quantity  # Opposite sign to close
        
        return self.add_position(
            symbol=symbol,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            commission=commission,
            order_id=order_id
        )
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def get_symbols(self) -> Set[str]:
        """Get all symbols with positions."""
        return set(self.positions.keys())
    
    def get_total_value(self, market_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            market_prices: Current market prices
            
        Returns:
            Total portfolio value
        """
        positions_value = sum(
            pos.get_market_value(market_prices.get(symbol, pos.last_price))
            for symbol, pos in self.positions.items()
        )
        return self.cash + positions_value
    
    def get_positions_value(self, market_prices: Dict[str, float]) -> float:
        """Get total value of all positions."""
        return sum(
            pos.get_market_value(market_prices.get(symbol, pos.last_price))
            for symbol, pos in self.positions.items()
        )
    
    def get_returns(self, market_prices: Dict[str, float]) -> float:
        """Calculate total return percentage."""
        current_value = self.get_total_value(market_prices)
        return (current_value - self._initial_value) / self._initial_value
    
    def mark_to_market(self, market_prices: Dict[str, float]) -> None:
        """Update position prices to current market."""
        for symbol, position in self.positions.items():
            if symbol in market_prices:
                position.update_price(market_prices[symbol])
    
    def get_snapshot(self, market_prices: Dict[str, float]) -> Dict:
        """Get current portfolio snapshot."""
        positions_data = {}
        for symbol, position in self.positions.items():
            current_price = market_prices.get(symbol, position.last_price)
            positions_data[symbol] = {
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': current_price,
                'cost_basis': position.cost_basis,
                'market_value': position.get_market_value(current_price),
                'unrealized_pnl': position.get_unrealized_pnl(current_price),
                'realized_pnl': position.realized_pnl
            }
            
        return {
            'timestamp': datetime.now(),
            'cash': self.cash,
            'positions_value': self.get_positions_value(market_prices),
            'total_value': self.get_total_value(market_prices),
            'positions': positions_data,
            'returns': self.get_returns(market_prices)
        }
    
    def process_dividend(
        self,
        symbol: str,
        amount_per_share: float,
        timestamp: datetime
    ) -> Optional[Transaction]:
        """
        Process dividend payment.
        
        Args:
            symbol: Stock symbol
            amount_per_share: Dividend per share
            timestamp: Payment date
            
        Returns:
            Dividend transaction if position exists
        """
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        dividend_amount = position.quantity * amount_per_share
        
        # Update cash
        self.cash_manager.update_balance(dividend_amount)
        
        # Create transaction
        transaction = Transaction(
            symbol=symbol,
            quantity=0,
            price=amount_per_share,
            timestamp=timestamp,
            commission=0,
            transaction_type=TransactionType.DIVIDEND,
            gross_amount=dividend_amount
        )
        
        self.transactions.append(transaction)
        return transaction
    
    def process_split(
        self,
        symbol: str,
        split_ratio: float,
        timestamp: datetime
    ) -> None:
        """
        Process stock split.
        
        Args:
            symbol: Stock symbol
            split_ratio: Split ratio (e.g., 2.0 for 2:1 split)
            timestamp: Split date
        """
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        position.apply_split(split_ratio)
        
        # Create transaction record
        transaction = Transaction(
            symbol=symbol,
            quantity=0,
            price=split_ratio,
            timestamp=timestamp,
            commission=0,
            transaction_type=TransactionType.SPLIT
        )
        
        self.transactions.append(transaction)
    
    def get_transaction_history(
        self, 
        symbol: Optional[str] = None,
        transaction_type: Optional[TransactionType] = None
    ) -> List[Transaction]:
        """
        Get filtered transaction history.
        
        Args:
            symbol: Filter by symbol
            transaction_type: Filter by transaction type
            
        Returns:
            List of matching transactions
        """
        transactions = self.transactions
        
        if symbol:
            transactions = [t for t in transactions if t.symbol == symbol]
            
        if transaction_type:
            transactions = [t for t in transactions if t.transaction_type == transaction_type]
            
        return transactions
    
    def to_dataframe(self, market_prices: Dict[str, float]) -> pd.DataFrame:
        """Convert positions to DataFrame."""
        data = []
        for symbol, position in self.positions.items():
            current_price = market_prices.get(symbol, position.last_price)
            data.append({
                'symbol': symbol,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': current_price,
                'cost_basis': position.cost_basis,
                'market_value': position.get_market_value(current_price),
                'unrealized_pnl': position.get_unrealized_pnl(current_price),
                'unrealized_pnl_pct': position.get_unrealized_pnl_percent(current_price),
                'weight': position.get_market_value(current_price) / self.get_total_value(market_prices)
            })
            
        return pd.DataFrame(data)