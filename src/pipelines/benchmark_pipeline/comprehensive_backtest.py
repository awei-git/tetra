"""Comprehensive strategy backtesting across multiple time windows and scenarios."""

import asyncio
from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

from src.simulators.historical import HistoricalSimulator
from src.simulators.portfolio import Portfolio
from src.strats.benchmark import get_all_benchmarks
from src.simulators.historical.event_periods import EVENT_PERIODS
from src.utils.logging import logger


@dataclass
class TimeWindow:
    """Definition of a testing time window."""
    name: str
    duration_days: int
    
    def get_dates(self, end_date: date) -> Tuple[date, date]:
        """Get start and end dates for window ending at end_date."""
        start_date = end_date - timedelta(days=self.duration_days)
        return start_date, end_date


@dataclass
class BacktestResult:
    """Result of a single backtest."""
    strategy_name: str
    symbol: str
    window: str
    scenario: str
    start_date: date
    end_date: date
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    success: bool
    error: Optional[str] = None


class ComprehensiveBacktester:
    """Run comprehensive backtests across multiple dimensions."""
    
    # Define time windows
    WINDOWS = [
        TimeWindow("2_weeks", 14),
        TimeWindow("3_months", 90),
        TimeWindow("1_year", 365),
    ]
    
    # Define test symbols for different scenarios
    SCENARIO_SYMBOLS = {
        "crisis": ["SPY", "QQQ", "IWM"],
        "bull": ["SPY", "QQQ", "NVDA", "AAPL"],
        "sector": ["XLF", "XLK", "XLE", "XLV"],
        "crypto": ["BTC-USD", "ETH-USD"],
    }
    
    def __init__(self):
        self.results: List[BacktestResult] = []
        self.simulator = HistoricalSimulator()
        
    async def run_single_backtest(
        self,
        strategy_name: str,
        symbol: str,
        start_date: date,
        end_date: date,
        window_name: str,
        scenario_name: str
    ) -> BacktestResult:
        """Run a single backtest."""
        try:
            # Create fresh portfolio and strategy
            portfolio = Portfolio(initial_cash=100000.0)
            strategy = get_all_benchmarks()[strategy_name]
            strategy.set_symbols([symbol])
            
            # Run simulation
            result = await self.simulator.run_simulation(
                portfolio=portfolio,
                start_date=start_date,
                end_date=end_date,
                strategy=strategy
            )
            
            return BacktestResult(
                strategy_name=strategy_name,
                symbol=symbol,
                window=window_name,
                scenario=scenario_name,
                start_date=start_date,
                end_date=end_date,
                total_return=result.total_return,
                annual_return=result.annual_return,
                sharpe_ratio=result.sharpe_ratio,
                max_drawdown=result.max_drawdown,
                win_rate=result.win_rate,
                total_trades=result.total_trades,
                success=True
            )
            
        except Exception as e:
            return BacktestResult(
                strategy_name=strategy_name,
                symbol=symbol,
                window=window_name,
                scenario=scenario_name,
                start_date=start_date,
                end_date=end_date,
                total_return=0.0,
                annual_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                success=False,
                error=str(e)
            )
    
    async def test_event_scenarios(self):
        """Test strategies during specific market events."""
        strategies = list(get_all_benchmarks().keys())
        
        # Select key events (mix of bull and bear)
        key_events = [
            "covid_crash", "post_covid_rally", "financial_crisis", "recovery_2009",
            "dotcom_crash", "trump_rally", "fed_taper_2022", "ai_boom_2023",
            "gme_squeeze", "tech_boom_2024", "svb_collapse", "qe_rally_2013"
        ]
        
        tasks = []
        for event_name in key_events:
            if event_name not in EVENT_PERIODS:
                continue
                
            event = EVENT_PERIODS[event_name]
            
            # Determine appropriate symbol
            if event.affected_symbols:
                symbols = event.affected_symbols[:2]  # Test first 2 affected
            else:
                symbols = ["SPY", "QQQ"]  # Default to major indices
            
            # Test each strategy on each symbol
            for strategy_name in strategies[:5]:  # Test top 5 strategies to save time
                for symbol in symbols:
                    task = self.run_single_backtest(
                        strategy_name=strategy_name,
                        symbol=symbol,
                        start_date=event.start_date,
                        end_date=event.end_date,
                        window_name="event",
                        scenario_name=event_name
                    )
                    tasks.append(task)
        
        # Run in batches
        batch_size = 10
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            results = await asyncio.gather(*batch)
            self.results.extend(results)
            logger.info(f"Completed batch {i//batch_size + 1}/{len(tasks)//batch_size + 1}")
    
    async def test_rolling_windows(self):
        """Test strategies with rolling windows over the last 10 years."""
        strategies = list(get_all_benchmarks().keys())[:5]  # Top 5 strategies
        
        # Define rolling test dates (quarterly for 10 years)
        end_dates = []
        current_date = date.today()
        for quarters_back in range(0, 40, 2):  # Every 6 months for 10 years
            test_date = current_date - timedelta(days=quarters_back * 90)
            if test_date.year >= 2015:  # Ensure we have data
                end_dates.append(test_date)
        
        tasks = []
        for end_date in end_dates:
            for window in self.WINDOWS:
                start_date, _ = window.get_dates(end_date)
                
                # Test on major indices
                for symbol in ["SPY", "QQQ", "IWM"]:
                    for strategy_name in strategies:
                        task = self.run_single_backtest(
                            strategy_name=strategy_name,
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date,
                            window_name=window.name,
                            scenario_name=f"rolling_{end_date.year}"
                        )
                        tasks.append(task)
        
        # Run in batches
        batch_size = 10
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            results = await asyncio.gather(*batch)
            self.results.extend(results)
            logger.info(f"Completed rolling window batch {i//batch_size + 1}")
    
    def analyze_results(self) -> pd.DataFrame:
        """Analyze results and create performance statistics."""
        if not self.results:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([r.__dict__ for r in self.results if r.success])
        
        if df.empty:
            return df
        
        # Calculate performance ranges by strategy
        strategy_stats = df.groupby('strategy_name').agg({
            'total_return': ['mean', 'std', 'min', 'max', 'count'],
            'sharpe_ratio': ['mean', 'min', 'max'],
            'max_drawdown': ['mean', 'min', 'max'],
            'win_rate': 'mean',
            'total_trades': 'mean'
        }).round(4)
        
        # Calculate performance by window
        window_stats = df.groupby(['strategy_name', 'window']).agg({
            'total_return': ['mean', 'std'],
            'sharpe_ratio': 'mean'
        }).round(4)
        
        # Calculate performance in different market conditions
        bull_scenarios = ['post_covid_rally', 'trump_rally', 'ai_boom_2023', 
                         'qe_rally_2013', 'recovery_2009', 'tech_boom_2024']
        bear_scenarios = ['covid_crash', 'financial_crisis', 'dotcom_crash', 
                         'fed_taper_2022', 'svb_collapse']
        
        df['market_condition'] = df['scenario'].apply(
            lambda x: 'bull' if x in bull_scenarios else ('bear' if x in bear_scenarios else 'mixed')
        )
        
        condition_stats = df.groupby(['strategy_name', 'market_condition']).agg({
            'total_return': 'mean',
            'sharpe_ratio': 'mean'
        }).round(4)
        
        return {
            'overall_stats': strategy_stats,
            'window_stats': window_stats,
            'condition_stats': condition_stats,
            'raw_results': df
        }
    
    def generate_report(self, stats: Dict) -> str:
        """Generate comprehensive performance report."""
        if not stats or 'overall_stats' not in stats:
            return "No results to report."
        
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE STRATEGY PERFORMANCE REPORT")
        report.append("="*80)
        
        # Overall performance ranges
        report.append("\n1. OVERALL PERFORMANCE RANGES")
        report.append("-"*40)
        
        overall = stats['overall_stats']
        for strategy in overall.index:
            data = overall.loc[strategy]
            report.append(f"\n{strategy}:")
            report.append(f"  Return: {data[('total_return', 'mean')]*100:.1f}% "
                         f"(range: {data[('total_return', 'min')]*100:.1f}% to "
                         f"{data[('total_return', 'max')]*100:.1f}%)")
            report.append(f"  Sharpe: {data[('sharpe_ratio', 'mean')]:.2f} "
                         f"(range: {data[('sharpe_ratio', 'min')]:.2f} to "
                         f"{data[('sharpe_ratio', 'max')]:.2f})")
            report.append(f"  Max DD: {data[('max_drawdown', 'mean')]*100:.1f}% "
                         f"(worst: {data[('max_drawdown', 'min')]*100:.1f}%)")
            report.append(f"  Avg Trades: {data[('total_trades', 'mean')]:.0f}")
            report.append(f"  Tests Run: {data[('total_return', 'count')]:.0f}")
        
        # Performance by time window
        report.append("\n\n2. PERFORMANCE BY TIME WINDOW")
        report.append("-"*40)
        
        window_stats = stats['window_stats']
        available_windows = window_stats.index.get_level_values('window').unique()
        for window in available_windows:
            report.append(f"\n{window.replace('_', ' ').title()}:")
            window_data = window_stats.xs(window, level='window')
            top_3 = window_data[('total_return', 'mean')].nlargest(3)
            for i, (strategy, ret) in enumerate(top_3.items(), 1):
                report.append(f"  {i}. {strategy}: {ret*100:.1f}% return")
        
        # Performance in different market conditions
        report.append("\n\n3. PERFORMANCE BY MARKET CONDITION")
        report.append("-"*40)
        
        condition_stats = stats['condition_stats']
        for condition in ['bull', 'bear']:
            report.append(f"\n{condition.title()} Markets:")
            if condition in condition_stats.index.get_level_values('market_condition'):
                cond_data = condition_stats.xs(condition, level='market_condition')
                top_3 = cond_data['total_return'].nlargest(3)
                for i, (strategy, ret) in enumerate(top_3.items(), 1):
                    sharpe = cond_data.loc[strategy, 'sharpe_ratio']
                    report.append(f"  {i}. {strategy}: {ret*100:.1f}% return, {sharpe:.2f} Sharpe")
        
        # Best overall strategies
        report.append("\n\n4. TOP STRATEGIES BY RISK-ADJUSTED RETURN")
        report.append("-"*40)
        
        # Calculate risk-adjusted score (Sharpe * (1 - max_dd))
        overall['risk_adjusted_score'] = (
            overall[('sharpe_ratio', 'mean')] * 
            (1 + overall[('max_drawdown', 'mean')])  # max_dd is negative
        )
        
        top_strategies = overall['risk_adjusted_score'].nlargest(5)
        for i, (strategy, score) in enumerate(top_strategies.items(), 1):
            ret = overall.loc[strategy, ('total_return', 'mean')]
            sharpe = overall.loc[strategy, ('sharpe_ratio', 'mean')]
            dd = overall.loc[strategy, ('max_drawdown', 'mean')]
            report.append(f"  {i}. {strategy}:")
            report.append(f"     Score: {score:.3f}, Return: {ret*100:.1f}%, "
                         f"Sharpe: {sharpe:.2f}, Max DD: {dd*100:.1f}%")
        
        return "\n".join(report)


async def run_comprehensive_backtest():
    """Run comprehensive backtesting and generate report."""
    logger.info("Starting comprehensive backtest...")
    
    tester = ComprehensiveBacktester()
    
    # Run event scenarios
    logger.info("Testing event scenarios...")
    await tester.test_event_scenarios()
    
    # Run rolling windows
    logger.info("Testing rolling windows...")
    await tester.test_rolling_windows()
    
    # Analyze results
    logger.info("Analyzing results...")
    stats = tester.analyze_results()
    
    # Generate report
    report = tester.generate_report(stats)
    
    # Save results
    if 'raw_results' in stats:
        stats['raw_results'].to_csv('/tmp/comprehensive_backtest_results.csv', index=False)
        logger.info("Raw results saved to /tmp/comprehensive_backtest_results.csv")
    
    return report


if __name__ == "__main__":
    import asyncio
    report = asyncio.run(run_comprehensive_backtest())
    print(report)