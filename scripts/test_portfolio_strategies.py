"""Test portfolio-based trading strategies with risk management."""

import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.strats.base import BaseStrategy, PositionSide
from src.backtesting.portfolio import Portfolio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EqualWeightPortfolioStrategy(BaseStrategy):
    """Maintain equal weight positions across all symbols."""
    
    def __init__(self, rebalance_days: int = 30, *args, **kwargs):
        super().__init__(
            name="Equal Weight Portfolio",
            description=f"Rebalance to equal weights every {rebalance_days} days",
            *args,
            **kwargs
        )
        self.rebalance_days = rebalance_days
        self.last_rebalance = None
        self.target_weight = None
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events) -> Tuple[bool, Optional[PositionSide], float]:
        """Enter new positions to maintain equal weights."""
        # Initialize on first call
        if self.target_weight is None:
            self.target_weight = 0.95 / len(self.symbols)  # 95% invested, 5% cash
            self.last_rebalance = timestamp
            
        # Check if we should rebalance
        if self.last_rebalance is None or (timestamp - self.last_rebalance).days >= self.rebalance_days:
            self.last_rebalance = timestamp
            # Enter position if not already held
            return True, PositionSide.LONG, self.target_weight
            
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events) -> bool:
        """Only exit for rebalancing."""
        # Don't exit unless rebalancing
        return False


class MomentumPortfolioStrategy(BaseStrategy):
    """Rank and invest in top momentum stocks."""
    
    def __init__(self, lookback: int = 20, top_n: int = 3, rebalance_days: int = 10, *args, **kwargs):
        super().__init__(
            name=f"Momentum Portfolio (Top {top_n})",
            description=f"Hold top {top_n} momentum stocks, rebalance every {rebalance_days} days",
            *args,
            **kwargs
        )
        self.lookback = lookback
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        self.last_rebalance = None
        self.current_holdings = set()
        self.momentum_scores = {}
        
    def calculate_momentum(self, symbol: str, timestamp: datetime, bar_data: Dict) -> float:
        """Calculate momentum score for ranking."""
        # Simulate momentum based on symbol and date
        base_score = hash(symbol) % 100 / 100.0
        date_factor = np.sin(timestamp.toordinal() * 0.1 + base_score * 10)
        return base_score + date_factor * 0.3
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events) -> Tuple[bool, Optional[PositionSide], float]:
        """Enter top momentum positions."""
        # Update momentum scores
        self.momentum_scores[symbol] = self.calculate_momentum(symbol, timestamp, bar_data)
        
        # Check if it's time to rebalance
        if self.last_rebalance is None or (timestamp - self.last_rebalance).days >= self.rebalance_days:
            # Get top N symbols by momentum
            sorted_symbols = sorted(self.momentum_scores.items(), key=lambda x: x[1], reverse=True)
            top_symbols = [s[0] for s in sorted_symbols[:self.top_n]]
            
            if symbol in top_symbols and symbol not in self.current_holdings:
                self.current_holdings.add(symbol)
                position_size = 0.9 / self.top_n  # 90% invested
                logger.info(f"[Momentum Portfolio] Adding {symbol} - Score: {self.momentum_scores[symbol]:.3f}")
                return True, PositionSide.LONG, position_size
                
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events) -> bool:
        """Exit positions no longer in top N."""
        # Update momentum
        self.momentum_scores[position.symbol] = self.calculate_momentum(position.symbol, timestamp, bar_data)
        
        # Rebalance check
        if self.last_rebalance is None or (timestamp - self.last_rebalance).days >= self.rebalance_days:
            self.last_rebalance = timestamp
            
            # Get new top N
            sorted_symbols = sorted(self.momentum_scores.items(), key=lambda x: x[1], reverse=True)
            top_symbols = [s[0] for s in sorted_symbols[:self.top_n]]
            
            if position.symbol not in top_symbols:
                self.current_holdings.discard(position.symbol)
                logger.info(f"[Momentum Portfolio] Removing {position.symbol} - no longer in top {self.top_n}")
                return True
                
        return False


class RiskParityStrategy(BaseStrategy):
    """Allocate based on inverse volatility for risk parity."""
    
    def __init__(self, vol_lookback: int = 20, rebalance_days: int = 20, *args, **kwargs):
        super().__init__(
            name="Risk Parity Portfolio",
            description="Weight positions by inverse volatility",
            *args,
            **kwargs
        )
        self.vol_lookback = vol_lookback
        self.rebalance_days = rebalance_days
        self.last_rebalance = None
        self.volatilities = {}
        self.target_weights = {}
        
    def calculate_volatility(self, symbol: str, timestamp: datetime) -> float:
        """Calculate historical volatility."""
        # Simulate volatility based on symbol characteristics
        base_vol = 0.15 + (hash(symbol) % 20) / 100.0  # 15-35% annualized
        # Add time-varying component
        season_factor = 1 + 0.3 * np.sin(timestamp.month * np.pi / 6)
        return base_vol * season_factor
        
    def calculate_risk_parity_weights(self) -> Dict[str, float]:
        """Calculate risk parity weights based on inverse volatility."""
        if not self.volatilities:
            return {}
            
        # Inverse volatility weighting
        inv_vols = {sym: 1.0 / vol for sym, vol in self.volatilities.items()}
        total_inv_vol = sum(inv_vols.values())
        
        # Normalize to sum to 0.95 (keep 5% cash)
        weights = {sym: 0.95 * inv_vol / total_inv_vol for sym, inv_vol in inv_vols.items()}
        return weights
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events) -> Tuple[bool, Optional[PositionSide], float]:
        """Enter positions based on risk parity weights."""
        # Update volatility
        self.volatilities[symbol] = self.calculate_volatility(symbol, timestamp)
        
        # Rebalance check
        if self.last_rebalance is None or (timestamp - self.last_rebalance).days >= self.rebalance_days:
            self.last_rebalance = timestamp
            self.target_weights = self.calculate_risk_parity_weights()
            
            if symbol in self.target_weights:
                weight = self.target_weights[symbol]
                logger.info(f"[Risk Parity] Entering {symbol} - Vol: {self.volatilities[symbol]:.1%}, Weight: {weight:.1%}")
                return True, PositionSide.LONG, weight
                
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events) -> bool:
        """Exit for rebalancing only."""
        # Update volatility
        self.volatilities[position.symbol] = self.calculate_volatility(position.symbol, timestamp)
        
        # Only exit if significantly overweight (for simplicity, we don't exit in this basic version)
        return False


class SectorRotationStrategy(BaseStrategy):
    """Rotate between sectors based on momentum and seasonality."""
    
    def __init__(self, sectors: Dict[str, List[str]], rotation_days: int = 30, *args, **kwargs):
        super().__init__(
            name="Sector Rotation",
            description="Rotate between sectors based on performance",
            *args,
            **kwargs
        )
        self.sectors = sectors or self._default_sectors()
        self.rotation_days = rotation_days
        self.last_rotation = None
        self.current_sector = None
        self.sector_scores = {}
        
    def _default_sectors(self) -> Dict[str, List[str]]:
        """Default sector mapping."""
        return {
            'tech': ['AAPL', 'MSFT', 'GOOGL'],
            'consumer': ['AMZN', 'TSLA', 'NKE'],
            'finance': ['JPM', 'BAC', 'GS'],
            'healthcare': ['JNJ', 'PFE', 'UNH']
        }
        
    def calculate_sector_score(self, sector: str, timestamp: datetime) -> float:
        """Calculate sector momentum/strength score."""
        # Simulate sector rotation based on seasonality
        sector_phase = {
            'tech': 0,
            'consumer': np.pi/2,
            'finance': np.pi,
            'healthcare': 3*np.pi/2
        }
        
        phase = sector_phase.get(sector, 0)
        # Annual cycle with monthly granularity
        score = np.sin(timestamp.month * np.pi / 6 + phase) * 0.5 + 0.5
        
        # Add some trend
        trend = (timestamp.toordinal() % 365) / 365.0
        score += trend * 0.2
        
        return score
        
    def get_symbol_sector(self, symbol: str) -> Optional[str]:
        """Get sector for a symbol."""
        for sector, symbols in self.sectors.items():
            if symbol in symbols:
                return sector
        return None
        
    def should_enter(self, symbol, timestamp, bar_data, signals, events) -> Tuple[bool, Optional[PositionSide], float]:
        """Enter positions in the best performing sector."""
        symbol_sector = self.get_symbol_sector(symbol)
        if not symbol_sector:
            return False, None, 0
            
        # Update sector scores
        for sector in self.sectors:
            self.sector_scores[sector] = self.calculate_sector_score(sector, timestamp)
            
        # Check for rotation
        if self.last_rotation is None or (timestamp - self.last_rotation).days >= self.rotation_days:
            self.last_rotation = timestamp
            
            # Find best sector
            best_sector = max(self.sector_scores.items(), key=lambda x: x[1])[0]
            
            if best_sector != self.current_sector:
                self.current_sector = best_sector
                logger.info(f"[Sector Rotation] Rotating to {best_sector} sector - Score: {self.sector_scores[best_sector]:.3f}")
                
            # Enter if symbol is in current sector
            if symbol_sector == self.current_sector:
                # Equal weight within sector
                sector_symbols = self.sectors[self.current_sector]
                weight = 0.9 / len(sector_symbols)
                logger.info(f"[Sector Rotation] Entering {symbol} in {symbol_sector} sector")
                return True, PositionSide.LONG, weight
                
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events) -> bool:
        """Exit positions not in current sector."""
        symbol_sector = self.get_symbol_sector(position.symbol)
        
        # Update scores and check rotation
        for sector in self.sectors:
            self.sector_scores[sector] = self.calculate_sector_score(sector, timestamp)
            
        if self.last_rotation is None or (timestamp - self.last_rotation).days >= self.rotation_days:
            best_sector = max(self.sector_scores.items(), key=lambda x: x[1])[0]
            
            if symbol_sector != best_sector:
                logger.info(f"[Sector Rotation] Exiting {position.symbol} - rotating out of {symbol_sector}")
                return True
                
        return False


class DynamicHedgeStrategy(BaseStrategy):
    """Dynamically hedge portfolio based on market conditions."""
    
    def __init__(self, hedge_symbol: str = 'TLT', vol_threshold: float = 0.25, *args, **kwargs):
        super().__init__(
            name="Dynamic Hedge Portfolio",
            description="Increase hedges when volatility rises",
            *args,
            **kwargs
        )
        self.hedge_symbol = hedge_symbol
        self.vol_threshold = vol_threshold
        self.market_vol = 0.15  # Starting volatility
        self.hedge_ratio = 0.0
        self.equity_symbols = []
        
    def calculate_market_volatility(self, timestamp: datetime) -> float:
        """Calculate overall market volatility."""
        # Simulate market volatility cycles
        base_vol = 0.15
        # Volatility clusters and mean reverts
        cycle = np.sin(timestamp.toordinal() * 0.05) * 0.1
        spike = 0.2 if timestamp.month in [3, 9, 10] else 0  # Volatile months
        
        return base_vol + cycle + spike
        
    def calculate_hedge_ratio(self, market_vol: float) -> float:
        """Calculate hedge ratio based on volatility."""
        if market_vol < 0.15:
            return 0.0  # No hedge in calm markets
        elif market_vol < 0.25:
            return 0.1  # 10% hedge
        elif market_vol < 0.35:
            return 0.2  # 20% hedge
        else:
            return 0.3  # 30% hedge in extreme volatility
            
    def should_enter(self, symbol, timestamp, bar_data, signals, events) -> Tuple[bool, Optional[PositionSide], float]:
        """Enter equity or hedge positions based on market conditions."""
        # Update market volatility
        self.market_vol = self.calculate_market_volatility(timestamp)
        new_hedge_ratio = self.calculate_hedge_ratio(self.market_vol)
        
        # If this is the hedge instrument
        if symbol == self.hedge_symbol:
            if new_hedge_ratio > self.hedge_ratio:
                self.hedge_ratio = new_hedge_ratio
                logger.info(f"[Dynamic Hedge] Increasing hedge - Market vol: {self.market_vol:.1%}, Hedge ratio: {new_hedge_ratio:.1%}")
                return True, PositionSide.LONG, new_hedge_ratio
        else:
            # Regular equity position
            if symbol not in self.equity_symbols:
                self.equity_symbols.append(symbol)
                # Allocate remaining after hedge
                equity_allocation = (1 - self.hedge_ratio) * 0.95
                position_size = equity_allocation / len(self.symbols)
                
                if position_size > 0.05:  # Minimum 5% position
                    logger.info(f"[Dynamic Hedge] Entering {symbol} - Position size: {position_size:.1%}")
                    return True, PositionSide.LONG, position_size
                    
        return False, None, 0
        
    def should_exit(self, position, timestamp, bar_data, signals, events) -> bool:
        """Exit to rebalance hedge ratio."""
        # Update market volatility
        self.market_vol = self.calculate_market_volatility(timestamp)
        new_hedge_ratio = self.calculate_hedge_ratio(self.market_vol)
        
        # Exit hedge if ratio decreased significantly
        if position.symbol == self.hedge_symbol and new_hedge_ratio < self.hedge_ratio - 0.05:
            self.hedge_ratio = new_hedge_ratio
            logger.info(f"[Dynamic Hedge] Reducing hedge - Market vol: {self.market_vol:.1%}")
            return True
            
        return False


def run_portfolio_strategy_tests():
    """Run all portfolio strategy tests."""
    
    # Extended test period for portfolio strategies
    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),  # 6 months
        initial_capital=100000,
        commission=0.001,
        slippage=0.0001,
        max_positions=10,  # Allow more positions
        calculate_metrics_every=20
    )
    
    # Diversified symbol universe
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'WMT']
    
    # Test strategies
    strategies = [
        (EqualWeightPortfolioStrategy, "Equal Weight Portfolio", {'rebalance_days': 30}),
        (MomentumPortfolioStrategy, "Momentum Portfolio (Top 3)", {'top_n': 3, 'rebalance_days': 10}),
        (MomentumPortfolioStrategy, "Momentum Portfolio (Top 5)", {'top_n': 5, 'rebalance_days': 10}),
        (RiskParityStrategy, "Risk Parity Portfolio", {'rebalance_days': 20}),
        (SectorRotationStrategy, "Sector Rotation", {'rotation_days': 30}),
        (DynamicHedgeStrategy, "Dynamic Hedge Portfolio", {'hedge_symbol': 'TLT', 'vol_threshold': 0.25}),
    ]
    
    results = []
    
    for strategy_class, name, kwargs in strategies:
        print(f"\n{'='*60}")
        print(f"Testing {name}")
        print(f"{'='*60}")
        
        engine = BacktestEngine(config=config)
        
        try:
            # Create strategy with parameters
            report = engine.run(
                strategy=lambda *args, **kw: strategy_class(**kwargs, *args, **kw),
                symbols=symbols,
                signal_computer=None
            )
            
            # Display results
            print(f"Final Equity: ${report.final_equity:,.2f}")
            print(f"Total Return: {report.total_return:.2%}")
            print(f"Annualized Return: {report.annualized_return:.2%}")
            print(f"Volatility: {report.volatility:.2%}")
            print(f"Sharpe Ratio: {report.sharpe_ratio:.2f}")
            print(f"Max Drawdown: {report.max_drawdown:.2%}")
            print(f"Total Trades: {report.total_trades}")
            
            results.append({
                'strategy': name,
                'return': report.total_return,
                'annual_return': report.annualized_return,
                'volatility': report.volatility,
                'sharpe': report.sharpe_ratio,
                'drawdown': report.max_drawdown,
                'trades': report.total_trades
            })
            
        except Exception as e:
            logger.error(f"Failed to test {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'strategy': name,
                'error': str(e)
            })
    
    # Summary comparison
    print(f"\n{'='*100}")
    print("PORTFOLIO STRATEGY COMPARISON")
    print(f"{'='*100}")
    print(f"{'Strategy':<30} {'Return':>10} {'Annual':>10} {'Vol':>10} {'Sharpe':>10} {'MaxDD':>10} {'Trades':>10}")
    print(f"{'-'*100}")
    
    for r in results:
        if 'error' not in r:
            print(f"{r['strategy']:<30} {r['return']:>9.2%} {r['annual_return']:>9.2%} "
                  f"{r['volatility']:>9.2%} {r['sharpe']:>10.2f} {r['drawdown']:>9.2%} {r['trades']:>10d}")
    
    # Analysis
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best_return = max(valid_results, key=lambda x: x['return'])
        best_sharpe = max(valid_results, key=lambda x: x['sharpe'] if x['sharpe'] else -999)
        lowest_vol = min(valid_results, key=lambda x: x['volatility'])
        
        print(f"\n{'='*100}")
        print("ANALYSIS:")
        print(f"Best Total Return: {best_return['strategy']} ({best_return['return']:.2%})")
        print(f"Best Risk-Adjusted: {best_sharpe['strategy']} (Sharpe: {best_sharpe['sharpe']:.2f})")
        print(f"Lowest Volatility: {lowest_vol['strategy']} ({lowest_vol['volatility']:.2%})")
        print(f"{'='*100}")


if __name__ == "__main__":
    logger.info("Starting portfolio strategy tests...")
    run_portfolio_strategy_tests()
    logger.info("Portfolio strategy tests completed!")