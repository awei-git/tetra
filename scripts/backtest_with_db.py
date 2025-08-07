"""Backtest trading strategies using real data from the database."""

import logging
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.strats.signal_based import SignalBasedStrategy
from src.signals.base.signal_computer import SignalComputer
from src.signals.base.config import SignalConfig
from src.signals.technical.momentum import RSISignal
from src.signals.technical.trend import SMASignal, EMASignal, MACDSignal
from src.signals.technical.volatility import BollingerBandsSignal
from src.strats.base import PositionSide
from src.db.sync_base import get_session
from src.db.models import OHLCVModel
from sqlalchemy import func

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_available_data():
    """Check what data is available in the database."""
    session = get_session()
    try:
        # Get unique symbols
        symbols = session.query(OHLCVModel.symbol).distinct().all()
        symbols = [s[0] for s in symbols]
        
        # Get date range for each symbol
        symbol_info = []
        for symbol in symbols[:10]:  # Check first 10 symbols
            date_range = session.query(
                OHLCVModel.symbol,
                func.min(OHLCVModel.timestamp).label('min_date'),
                func.max(OHLCVModel.timestamp).label('max_date'),
                func.count(OHLCVModel.id).label('count')
            ).filter(OHLCVModel.symbol == symbol).group_by(OHLCVModel.symbol).first()
            
            if date_range:
                symbol_info.append({
                    'symbol': date_range[0],
                    'min_date': date_range[1],
                    'max_date': date_range[2],
                    'count': date_range[3]
                })
        
        return symbols, symbol_info
        
    except Exception as e:
        logger.error(f"Error checking data: {e}")
        return [], []
    finally:
        session.close()


def create_momentum_strategy():
    """Create a momentum-based trading strategy."""
    class MomentumStrategy(SignalBasedStrategy):
        """Momentum strategy using RSI and MACD signals."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.name = "Momentum Strategy"
            self.description = "Trade based on RSI oversold/overbought and MACD crossovers"
            
            # Entry rules - Buy on oversold with bullish MACD
            # Define entry conditions
            def oversold_bullish_entry(signals):
                return (
                    signals.get('rsi', 50) < 30 and
                    signals.get('macd_signal', 0) > 0 and
                    signals.get('close', 0) > signals.get('sma_50', 0)
                )
            
            # Define exit conditions
            def overbought_exit(signals):
                return signals.get('rsi', 50) > 70
            
            def stop_loss_exit(position, signals):
                if hasattr(position, 'unrealized_pnl_pct'):
                    return position.unrealized_pnl_pct < -0.05
                return False
                
            def take_profit_exit(position, signals):
                if hasattr(position, 'unrealized_pnl_pct'):
                    return position.unrealized_pnl_pct > 0.15
                return False
    
    return MomentumStrategy


def create_mean_reversion_strategy():
    """Create a mean reversion trading strategy."""
    class MeanReversionStrategy(SignalBasedStrategy):
        """Mean reversion strategy using Bollinger Bands."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.name = "Mean Reversion Strategy"
            self.description = "Trade based on Bollinger Bands mean reversion"
            
            # Define entry conditions
            def lower_band_entry(signals):
                return (
                    signals.get('close', 0) <= signals.get('bb_lower', 0) and
                    signals.get('rsi', 50) < 40
                )
            
            # Define exit conditions
            def mean_reversion_exit(signals):
                return signals.get('close', 0) >= signals.get('bb_middle', 0)
            
            def stop_loss_exit(position, signals):
                if hasattr(position, 'unrealized_pnl_pct'):
                    return position.unrealized_pnl_pct < -0.03
                return False
    
    return MeanReversionStrategy


def run_backtest_with_real_data():
    """Run backtest using real market data from database."""
    
    # Check available data
    logger.info("Checking available data in database...")
    symbols, symbol_info = check_available_data()
    
    if not symbols:
        logger.error("No data found in database!")
        return None
    
    logger.info(f"Found {len(symbols)} symbols in database")
    for info in symbol_info:
        logger.info(f"  {info['symbol']}: {info['min_date']} to {info['max_date']} ({info['count']} records)")
    
    # Select symbols with good data coverage
    # For this example, let's use some popular stocks if available
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    available_symbols = [s for s in test_symbols if s in symbols]
    
    if not available_symbols:
        # Use first 5 available symbols
        available_symbols = symbols[:5]
    
    logger.info(f"Using symbols: {available_symbols}")
    
    # Configure backtest
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=100000,
        commission=0.001,  # 0.1%
        slippage=0.0001,  # 0.01%
        max_positions=3,
        position_size_limit=0.3,  # Max 30% per position
        max_drawdown_pct=0.20,  # Stop if 20% drawdown
        calculate_metrics_every=20,
        save_trades=True,
        save_positions=True,
        benchmark=None  # Could use SPY if available
    )
    
    # Create signal computer
    signal_config = SignalConfig()
    
    signal_computer = SignalComputer(config=signal_config)
    
    # Add technical signals
    signal_computer.register_signal(RSISignal(period=14))
    signal_computer.register_signal(MACDSignal(fast_period=12, slow_period=26, signal_period=9))
    signal_computer.register_signal(SMASignal(period=20))
    signal_computer.register_signal(SMASignal(period=50))
    signal_computer.register_signal(EMASignal(period=20))
    signal_computer.register_signal(BollingerBandsSignal(period=20, num_std=2))
    
    # Create backtesting engine
    engine = BacktestEngine(config=config)
    
    # Test momentum strategy
    logger.info("Running momentum strategy backtest...")
    try:
        momentum_report = engine.run(
            strategy=create_momentum_strategy(),
            symbols=available_symbols,
            signal_computer=signal_computer
        )
        
        print("\n" + "="*60)
        print("MOMENTUM STRATEGY RESULTS")
        print("="*60)
        print_results(momentum_report)
        
    except Exception as e:
        logger.error(f"Momentum strategy backtest failed: {e}")
        momentum_report = None
    
    # Test mean reversion strategy
    logger.info("\nRunning mean reversion strategy backtest...")
    try:
        # Reset engine for new backtest
        engine = BacktestEngine(config=config)
        
        reversion_report = engine.run(
            strategy=create_mean_reversion_strategy(),
            symbols=available_symbols,
            signal_computer=signal_computer
        )
        
        print("\n" + "="*60)
        print("MEAN REVERSION STRATEGY RESULTS")
        print("="*60)
        print_results(reversion_report)
        
    except Exception as e:
        logger.error(f"Mean reversion strategy backtest failed: {e}")
        reversion_report = None
    
    # Plot comparison if both succeeded
    if momentum_report and reversion_report:
        plot_strategy_comparison(momentum_report, reversion_report)
    
    return momentum_report, reversion_report


def print_results(report):
    """Print backtest results."""
    if not report:
        print("No results available")
        return
        
    print(f"Total Return: {report.total_return:.2%}")
    print(f"Annualized Return: {report.annualized_return:.2%}")
    print(f"Volatility: {report.volatility:.2%}")
    print(f"Sharpe Ratio: {report.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {report.max_drawdown:.2%}")
    print(f"Win Rate: {report.win_rate:.2%}")
    print(f"Total Trades: {report.total_trades}")
    print(f"Profit Factor: {report.profit_factor:.2f}")
    print(f"Final Equity: ${report.final_equity:,.2f}")


def plot_strategy_comparison(report1, report2):
    """Plot comparison of two strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Equity curves
    ax1 = axes[0, 0]
    report1.equity_curve.plot(ax=ax1, label='Momentum', color='blue')
    report2.equity_curve.plot(ax=ax1, label='Mean Reversion', color='green')
    ax1.set_title('Equity Curves Comparison')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Drawdowns
    ax2 = axes[0, 1]
    
    # Calculate drawdowns
    for report, label, color in [(report1, 'Momentum', 'blue'), (report2, 'Mean Reversion', 'green')]:
        running_max = report.equity_curve.expanding().max()
        drawdown = (report.equity_curve - running_max) / running_max * 100
        drawdown.plot(ax=ax2, label=label, color=color, alpha=0.7)
    
    ax2.set_title('Drawdown Comparison')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Returns distribution
    ax3 = axes[1, 0]
    returns1 = report1.returns * 100
    returns2 = report2.returns * 100
    
    ax3.hist([returns1, returns2], bins=30, alpha=0.6, label=['Momentum', 'Mean Reversion'])
    ax3.set_title('Returns Distribution')
    ax3.set_xlabel('Daily Returns (%)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Metrics comparison
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    metrics_text = f"""
    Metric               Momentum    Mean Reversion
    ===============================================
    Total Return:        {report1.total_return:>7.1%}     {report2.total_return:>7.1%}
    Sharpe Ratio:        {report1.sharpe_ratio:>7.2f}     {report2.sharpe_ratio:>7.2f}
    Max Drawdown:        {report1.max_drawdown:>7.1%}     {report2.max_drawdown:>7.1%}
    Win Rate:            {report1.win_rate:>7.1%}     {report2.win_rate:>7.1%}
    Total Trades:        {report1.total_trades:>7d}     {report2.total_trades:>7d}
    """
    
    ax4.text(0.1, 0.5, metrics_text, fontfamily='monospace', fontsize=12,
             verticalalignment='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    try:
        momentum_report, reversion_report = run_backtest_with_real_data()
        
        if momentum_report or reversion_report:
            logger.info("Backtest completed successfully!")
        else:
            logger.error("No successful backtests completed")
            
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()