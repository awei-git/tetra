"""Cash and margin management."""

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class CashTransaction:
    """Record of cash movement."""
    amount: float
    timestamp: datetime
    description: str
    balance_after: float


class CashManager:
    """Manage portfolio cash and margin."""
    
    def __init__(
        self,
        initial_balance: float,
        base_currency: str = "USD",
        margin_enabled: bool = False,
        margin_rate: float = 0.05  # Annual margin interest rate
    ):
        """
        Initialize cash manager.
        
        Args:
            initial_balance: Starting cash balance
            base_currency: Base currency
            margin_enabled: Whether margin trading is allowed
            margin_rate: Annual interest rate for margin
        """
        self._balance = initial_balance
        self.base_currency = base_currency
        self.margin_enabled = margin_enabled
        self.margin_rate = margin_rate
        self.margin_used = 0.0
        self.transactions: List[CashTransaction] = []
        
        # Record initial deposit
        self._record_transaction(
            initial_balance, 
            datetime.now(), 
            "Initial deposit"
        )
    
    def get_balance(self) -> float:
        """Get current cash balance."""
        return self._balance
    
    def get_available_cash(self) -> float:
        """Get cash available for trading (including margin if enabled)."""
        if self.margin_enabled:
            # Typically 2:1 margin for stocks
            return self._balance * 2
        return max(0, self._balance)
    
    def update_balance(self, amount: float, description: str = "") -> None:
        """
        Update cash balance.
        
        Args:
            amount: Amount to add (positive) or subtract (negative)
            description: Optional transaction description
            
        Raises:
            ValueError: If insufficient funds
        """
        new_balance = self._balance + amount
        
        if not self.margin_enabled and new_balance < 0:
            raise ValueError(f"Insufficient funds: balance would be ${new_balance:.2f}")
        
        self._balance = new_balance
        
        # Track margin usage
        if new_balance < 0:
            self.margin_used = abs(new_balance)
        else:
            self.margin_used = 0
            
        self._record_transaction(amount, datetime.now(), description)
    
    def calculate_margin_interest(self, days: int) -> float:
        """
        Calculate margin interest owed.
        
        Args:
            days: Number of days margin was used
            
        Returns:
            Interest amount
        """
        if self.margin_used <= 0:
            return 0
            
        daily_rate = self.margin_rate / 365
        return self.margin_used * daily_rate * days
    
    def deposit(self, amount: float, timestamp: Optional[datetime] = None) -> None:
        """
        Deposit cash into account.
        
        Args:
            amount: Amount to deposit
            timestamp: Transaction timestamp
        """
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
            
        self._balance += amount
        self._record_transaction(
            amount, 
            timestamp or datetime.now(), 
            "Cash deposit"
        )
    
    def withdraw(self, amount: float, timestamp: Optional[datetime] = None) -> None:
        """
        Withdraw cash from account.
        
        Args:
            amount: Amount to withdraw
            timestamp: Transaction timestamp
            
        Raises:
            ValueError: If insufficient funds
        """
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
            
        if amount > self._balance:
            raise ValueError(f"Insufficient funds: attempting to withdraw ${amount:.2f}, "
                           f"available ${self._balance:.2f}")
        
        self._balance -= amount
        self._record_transaction(
            -amount,
            timestamp or datetime.now(),
            "Cash withdrawal"
        )
    
    def _record_transaction(
        self, 
        amount: float, 
        timestamp: datetime, 
        description: str
    ) -> None:
        """Record a cash transaction."""
        transaction = CashTransaction(
            amount=amount,
            timestamp=timestamp,
            description=description,
            balance_after=self._balance
        )
        self.transactions.append(transaction)
    
    def get_transaction_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[CashTransaction]:
        """
        Get cash transaction history.
        
        Args:
            start_date: Filter start date
            end_date: Filter end date
            
        Returns:
            List of cash transactions
        """
        transactions = self.transactions
        
        if start_date:
            transactions = [t for t in transactions if t.timestamp >= start_date]
            
        if end_date:
            transactions = [t for t in transactions if t.timestamp <= end_date]
            
        return transactions
    
    def get_summary(self) -> Dict[str, float]:
        """Get cash account summary."""
        return {
            'balance': self._balance,
            'available_cash': self.get_available_cash(),
            'margin_used': self.margin_used,
            'margin_available': self.get_available_cash() - self._balance if self.margin_enabled else 0,
            'total_deposits': sum(t.amount for t in self.transactions if t.amount > 0 and 'deposit' in t.description.lower()),
            'total_withdrawals': sum(abs(t.amount) for t in self.transactions if t.amount < 0 and 'withdrawal' in t.description.lower()),
        }