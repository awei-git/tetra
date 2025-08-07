"""Trading signal class for simulator."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TradingSignal:
    """Trading signal with all necessary information."""
    symbol: str
    direction: str  # "BUY" or "SELL"
    quantity: Optional[int] = None
    order_type: str = "MARKET"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    metadata: Optional[dict] = None