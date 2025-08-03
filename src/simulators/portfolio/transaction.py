"""Transaction record keeping."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import uuid


class TransactionType(Enum):
    """Types of transactions."""
    BUY = "BUY"
    SELL = "SELL"
    DIVIDEND = "DIVIDEND"
    SPLIT = "SPLIT"
    FEE = "FEE"
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"


@dataclass
class Transaction:
    """Record of a single transaction."""
    
    symbol: str
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0
    transaction_type: TransactionType = TransactionType.BUY
    order_id: Optional[str] = None
    transaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Additional fields
    gross_amount: Optional[float] = None
    net_amount: Optional[float] = None
    realized_pnl: Optional[float] = None
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate amounts after initialization."""
        if self.gross_amount is None:
            self.gross_amount = self.quantity * self.price
            
        if self.net_amount is None:
            if self.transaction_type in (TransactionType.BUY, TransactionType.SELL):
                if self.transaction_type == TransactionType.BUY:
                    self.net_amount = self.gross_amount + self.commission
                else:
                    self.net_amount = self.gross_amount - self.commission
            else:
                self.net_amount = self.gross_amount
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'transaction_id': self.transaction_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'commission': self.commission,
            'transaction_type': self.transaction_type.value,
            'gross_amount': self.gross_amount,
            'net_amount': self.net_amount,
            'realized_pnl': self.realized_pnl,
            'notes': self.notes,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """Create transaction from dictionary."""
        data = data.copy()
        
        # Convert timestamp
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            
        # Convert transaction type
        if isinstance(data['transaction_type'], str):
            data['transaction_type'] = TransactionType(data['transaction_type'])
            
        return cls(**data)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"Transaction({self.transaction_type.value} {self.quantity} "
                f"{self.symbol} @ ${self.price:.2f} on {self.timestamp.date()})")