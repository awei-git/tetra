"""Example backtesting script demonstrating the backtesting engine."""

import logging
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.strategies.signal_based import SignalBasedStrategy
from src.signals.base.signal_computer import SignalComputer
from src.signals.base.config import SignalConfig
from src.signals.technical.momentum import RSISignal, MACDSignal
from src.signals.technical.trend import SMASignal, EMASignal
from src.signals.technical.volume import VolumeSignal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_signal_computer():
    """Create a signal computer with multiple technical indicators."""
    # Create signal configuration
    config = SignalConfig(
        cache_enabled=True,
        parallel_enabled=True,
        max_workers=4
    )
    
    # Initialize signal computer
    signal_computer = SignalComputer(config=config)
    
    # Add technical signals
    signal_computer.add_signal('rsi', RSISignal(period=14))
    signal_computer.add_signal('macd', MACDSignal(fast_period=12, slow_period=26, signal_period=9))
    signal_computer.add_signal('sma_20', SMASignal(period=20))
    signal_computer.add_signal('sma_50', SMASignal(period=50))
    signal_computer.add_signal('ema_20', EMASignal(period=20))
    signal_computer.add_signal('volume', VolumeSignal(period=20))
    
    return signal_computer


def create_simple_strategy():
    """Create a simple momentum strategy."""
    class SimpleMomentumStrategy(SignalBasedStrategy):
        """Simple momentum strategy based on RSI and trend signals."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.name = "Simple Momentum"
            self.description = "Buy when RSI < 30 and price > SMA20, sell when RSI > 70"
            
            # Entry rules
            self.add_entry_rule(
                name="oversold_above_trend",
                condition=lambda signals: (
                    signals.get('rsi', 50) < 30 and
                    signals.get('close', 0) > signals.get('sma_20', 0)
                ),
                action='buy',
                size=0.1  # 10% position size
            )
            
            # Exit rules
            self.add_exit_rule(
                name="overbought",
                condition=lambda signals: signals.get('rsi', 50) > 70,
                action='sell'
            )
            
            # Stop loss
            self.add_exit_rule(
                name="stop_loss",
                condition=lambda position, signals: (
                    position.unrealized_pnl_pct < -0.05  # 5% stop loss
                ),
                action='sell'
            )
    
    return SimpleMomentumStrategy


def run_backtest():
    """Run a simple backtest example."""
    # Configure backtest
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1),
        initial_capital=100000,
        commission=0.001,  # 0.1%
        slippage=0.0001,  # 0.01%
        max_positions=5,
        benchmark="SPY",
        calculate_metrics_every=20,
        save_trades=True,
        save_positions=True
    )
    
    # Create backtesting engine
    engine = BacktestEngine(config=config)
    
    # Create signal computer
    signal_computer = create_signal_computer()
    
    # Define symbols to trade
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Run backtest
    logger.info("Starting backtest...")
    report = engine.run(
        strategy=create_simple_strategy(),
        symbols=symbols,
        signal_computer=signal_computer
    )
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Total Return: {report.total_return:.2%}")
    print(f"Annualized Return: {report.annualized_return:.2%}")
    print(f"Sharpe Ratio: {report.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {report.max_drawdown:.2%}")
    print(f"Win Rate: {report.win_rate:.2%}")
    print(f"Total Trades: {report.total_trades}")
    print(f"Profit Factor: {report.profit_factor:.2f}")
    
    # Plot results
    plot_results(report)
    
    return report


def plot_results(report):
    """Plot backtest results."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Equity curve
    ax1 = axes[0]
    report.equity_curve.plot(ax=ax1, label='Strategy')
    ax1.set_title('Equity Curve')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    ax2 = axes[1]
    running_max = report.equity_curve.expanding().max()
    drawdown = (report.equity_curve - running_max) / running_max * 100
    drawdown.plot(ax=ax2, color='red', alpha=0.7)
    ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    
    # Returns distribution
    ax3 = axes[2]
    returns = report.returns * 100
    returns.hist(ax=ax3, bins=50, alpha=0.7, color='blue')
    ax3.set_title('Returns Distribution')
    ax3.set_xlabel('Daily Returns (%)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150)
    plt.show()
    
    # Trade analysis
    if report.trades:
        plot_trade_analysis(report.trades)


def plot_trade_analysis(trades):
    """Plot trade analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Convert trades to DataFrame for analysis
    trade_data = []
    for trade in trades:
        pnl = (trade.exit_price - trade.entry_price) * trade.quantity
        if trade.side.value == 'short':
            pnl = -pnl
        pnl -= trade.commission + trade.slippage
        
        trade_data.append({
            'symbol': trade.symbol,
            'pnl': pnl,
            'pnl_pct': pnl / (trade.entry_price * trade.quantity) * 100,
            'duration': (trade.exit_time - trade.entry_time).days,
            'side': trade.side.value
        })
    
    df = pd.DataFrame(trade_data)
    
    # P&L by symbol
    ax1 = axes[0, 0]
    symbol_pnl = df.groupby('symbol')['pnl'].sum().sort_values()
    symbol_pnl.plot(kind='bar', ax=ax1)
    ax1.set_title('P&L by Symbol')
    ax1.set_ylabel('P&L ($)')
    ax1.grid(True, alpha=0.3)
    
    # Win/Loss distribution
    ax2 = axes[0, 1]
    wins = df[df['pnl'] > 0]['pnl']
    losses = df[df['pnl'] <= 0]['pnl']
    ax2.hist([wins, losses], bins=20, label=['Wins', 'Losses'], alpha=0.7)
    ax2.set_title('Win/Loss Distribution')
    ax2.set_xlabel('P&L ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Trade duration vs P&L
    ax3 = axes[1, 0]
    ax3.scatter(df['duration'], df['pnl_pct'], alpha=0.6)
    ax3.set_title('Trade Duration vs P&L')
    ax3.set_xlabel('Duration (days)')
    ax3.set_ylabel('P&L (%)')
    ax3.grid(True, alpha=0.3)
    
    # Cumulative P&L
    ax4 = axes[1, 1]
    cumulative_pnl = df['pnl'].cumsum()
    ax4.plot(cumulative_pnl.values)
    ax4.set_title('Cumulative P&L')
    ax4.set_xlabel('Trade Number')
    ax4.set_ylabel('Cumulative P&L ($)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trade_analysis.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    try:
        report = run_backtest()
        
        # Save detailed report
        report_dict = report.to_dict()
        
        # Save to JSON
        import json
        with open('backtest_report.json', 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info("Backtest completed successfully!")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise