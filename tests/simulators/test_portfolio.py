"""Tests for portfolio management functionality."""

import pytest
from datetime import datetime
from src.simulators.portfolio import Portfolio, Position, Transaction, TransactionType


class TestPortfolio:
    """Test Portfolio class."""
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        portfolio = Portfolio(initial_cash=100000)
        
        assert portfolio.cash == 100000
        assert portfolio.initial_value == 100000
        assert len(portfolio.positions) == 0
        assert len(portfolio.transactions) == 0
    
    def test_add_long_position(self):
        """Test adding a long position."""
        portfolio = Portfolio(initial_cash=100000)
        
        position = portfolio.add_position(
            symbol="AAPL",
            quantity=100,
            price=150.0,
            timestamp=datetime.now(),
            commission=5.0
        )
        
        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.entry_price == 150.0
        assert portfolio.cash == 100000 - (100 * 150) - 5  # 84995
        assert len(portfolio.transactions) == 1
    
    def test_add_position_insufficient_funds(self):
        """Test adding position with insufficient funds."""
        portfolio = Portfolio(initial_cash=1000)
        
        with pytest.raises(ValueError, match="Insufficient cash"):
            portfolio.add_position(
                symbol="AAPL",
                quantity=100,
                price=150.0,
                timestamp=datetime.now()
            )
    
    def test_close_position(self):
        """Test closing a position."""
        portfolio = Portfolio(initial_cash=100000)
        
        # Add position
        portfolio.add_position("AAPL", 100, 150.0, datetime.now(), 5.0)
        
        # Close position
        closing_tx = portfolio.close_position(
            symbol="AAPL",
            price=160.0,
            timestamp=datetime.now(),
            commission=5.0
        )
        
        assert closing_tx is not None
        assert "AAPL" not in portfolio.positions
        assert portfolio.cash > 100000  # Made profit
    
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        portfolio = Portfolio(initial_cash=50000)
        
        # Add positions
        portfolio.add_position("AAPL", 100, 150.0, datetime.now())
        portfolio.add_position("MSFT", 50, 250.0, datetime.now())
        
        # Calculate value with new prices
        market_prices = {
            "AAPL": 160.0,
            "MSFT": 260.0
        }
        
        total_value = portfolio.get_total_value(market_prices)
        positions_value = portfolio.get_positions_value(market_prices)
        
        assert positions_value == (100 * 160) + (50 * 260)  # 29000
        assert total_value == portfolio.cash + positions_value
    
    def test_process_dividend(self):
        """Test dividend processing."""
        portfolio = Portfolio(initial_cash=100000)
        
        # Add position
        portfolio.add_position("AAPL", 100, 150.0, datetime.now())
        initial_cash = portfolio.cash
        
        # Process dividend
        dividend_tx = portfolio.process_dividend(
            symbol="AAPL",
            amount_per_share=0.50,
            timestamp=datetime.now()
        )
        
        assert dividend_tx is not None
        assert dividend_tx.transaction_type == TransactionType.DIVIDEND
        assert portfolio.cash == initial_cash + (100 * 0.50)
    
    def test_process_split(self):
        """Test stock split processing."""
        portfolio = Portfolio(initial_cash=100000)
        
        # Add position
        portfolio.add_position("AAPL", 100, 150.0, datetime.now())
        
        # Process 2:1 split
        portfolio.process_split(
            symbol="AAPL",
            split_ratio=2.0,
            timestamp=datetime.now()
        )
        
        position = portfolio.positions["AAPL"]
        assert position.quantity == 200
        assert position.entry_price == 75.0


class TestPosition:
    """Test Position class."""
    
    def test_position_initialization(self):
        """Test position initialization."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            entry_price=150.0,
            entry_time=datetime.now(),
            commission=5.0
        )
        
        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.entry_price == 150.0
        assert position.cost_basis == (100 * 150) + 5
        assert position.realized_pnl == 0
    
    def test_add_shares_to_position(self):
        """Test adding shares to existing position."""
        position = Position("AAPL", 100, 150.0, datetime.now())
        
        position.add_shares(50, 160.0, 5.0)
        
        assert position.quantity == 150
        assert position.cost_basis == (100 * 150) + (50 * 160) + 5
        assert position.entry_price == position.cost_basis / 150
    
    def test_reduce_position(self):
        """Test reducing a position and calculating realized P&L."""
        position = Position("AAPL", 100, 150.0, datetime.now())
        
        # Sell 50 shares at 160
        position.add_shares(-50, 160.0, 5.0)
        
        assert position.quantity == 50
        assert position.realized_pnl > 0  # Made profit
        assert position.cost_basis == 7500  # Half of original
    
    def test_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        position = Position("AAPL", 100, 150.0, datetime.now())
        
        # Test with price increase
        unrealized = position.get_unrealized_pnl(160.0)
        assert unrealized == 1000  # (160-150) * 100
        
        # Test with price decrease
        unrealized = position.get_unrealized_pnl(140.0)
        assert unrealized == -1000  # (140-150) * 100
    
    def test_stock_split(self):
        """Test applying stock split."""
        position = Position("AAPL", 100, 150.0, datetime.now())
        
        position.apply_split(2.0)
        
        assert position.quantity == 200
        assert position.entry_price == 75.0
        assert position.last_price == 75.0


class TestTransaction:
    """Test Transaction class."""
    
    def test_transaction_creation(self):
        """Test transaction creation."""
        tx = Transaction(
            symbol="AAPL",
            quantity=100,
            price=150.0,
            timestamp=datetime.now(),
            commission=5.0,
            transaction_type=TransactionType.BUY
        )
        
        assert tx.symbol == "AAPL"
        assert tx.quantity == 100
        assert tx.gross_amount == 15000
        assert tx.net_amount == 15005  # Including commission
    
    def test_transaction_serialization(self):
        """Test transaction to/from dict."""
        tx = Transaction(
            symbol="AAPL",
            quantity=100,
            price=150.0,
            timestamp=datetime.now(),
            transaction_type=TransactionType.BUY
        )
        
        # Convert to dict
        tx_dict = tx.to_dict()
        assert isinstance(tx_dict, dict)
        assert tx_dict['symbol'] == "AAPL"
        
        # Create from dict
        tx2 = Transaction.from_dict(tx_dict)
        assert tx2.symbol == tx.symbol
        assert tx2.quantity == tx.quantity