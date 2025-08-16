"""Step 5: Generate rankings and reports for strategy assessment."""

import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
import numpy as np

from src.pipelines.base import PipelineStep, PipelineContext

logger = logging.getLogger(__name__)


class RankingGenerationStep(PipelineStep):
    """Generate strategy rankings and assessment reports."""
    
    def __init__(self):
        super().__init__("RankingGeneration")
        self.output_dir = Path('data/assessment')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def execute(self, context: PipelineContext) -> None:
        """Generate rankings and reports."""
        logger.info("Generating strategy rankings and reports")
        
        # Get comprehensive metrics from context
        comprehensive_metrics = context.data.get('comprehensive_metrics', {})
        if not comprehensive_metrics:
            logger.warning("No comprehensive metrics to rank")
            return
        
        # Generate overall rankings
        overall_rankings = self._generate_overall_rankings(comprehensive_metrics)
        
        # Generate category-specific rankings
        category_rankings = self._generate_category_rankings(
            comprehensive_metrics,
            context.data.get('strategies', [])
        )
        
        # Generate regime-specific rankings
        regime_rankings = self._generate_regime_rankings(comprehensive_metrics)
        
        # Generate risk-adjusted rankings
        risk_adjusted_rankings = self._generate_risk_adjusted_rankings(comprehensive_metrics)
        
        # Combine all rankings
        all_rankings = {
            'overall': overall_rankings,
            'by_category': category_rankings,
            'by_regime': regime_rankings,
            'risk_adjusted': risk_adjusted_rankings,
            'timestamp': datetime.now().isoformat(),
            'total_strategies': len(comprehensive_metrics)
        }
        
        # Store in context
        context.data['rankings'] = all_rankings
        
        # Generate and save reports
        await self._generate_reports(all_rankings, comprehensive_metrics, context)
        
        # Update summary with top strategy
        if overall_rankings:
            top_strategy = overall_rankings[0]
            summary = context.data.get('summary', {})
            summary.update({
                'top_strategy': top_strategy['name'],
                'best_return': top_strategy.get('total_return', 0),
                'best_sharpe': top_strategy.get('sharpe_ratio', 0)
            })
            context.data['summary'] = summary
        
        logger.info(f"Generated rankings for {len(comprehensive_metrics)} strategies")
    
    def _generate_overall_rankings(self, metrics: Dict[str, Dict]) -> List[Dict]:
        """Generate overall strategy rankings."""
        rankings = []
        
        for strategy_name, strategy_metrics in metrics.items():
            ranking_entry = {
                'name': strategy_name,
                'ranking_score': strategy_metrics.get('ranking_score', 0),
                'total_return': strategy_metrics.get('total_return', 0),
                'annualized_return': strategy_metrics.get('annualized_return', 0),
                'sharpe_ratio': strategy_metrics.get('sharpe_ratio', 0),
                'sortino_ratio': strategy_metrics.get('sortino_ratio', 0),
                'max_drawdown': strategy_metrics.get('max_drawdown', 0),
                'win_rate': strategy_metrics.get('win_rate', 0),
                'profit_factor': strategy_metrics.get('profit_factor', 0),
                'total_trades': strategy_metrics.get('total_trades', 0),
                'consistency_score': strategy_metrics.get('consistency_score', 0)
            }
            rankings.append(ranking_entry)
        
        # Sort by ranking score
        rankings.sort(key=lambda x: x['ranking_score'], reverse=True)
        
        # Add rank numbers
        for i, entry in enumerate(rankings):
            entry['rank'] = i + 1
        
        return rankings
    
    def _generate_category_rankings(
        self, 
        metrics: Dict[str, Dict], 
        strategies: List[Dict]
    ) -> Dict[str, List]:
        """Generate rankings by strategy category."""
        # Group strategies by category
        categories = {}
        for strategy in strategies:
            category = strategy.get('category', 'unknown')
            if category not in categories:
                categories[category] = []
            
            strategy_name = strategy['name']
            if strategy_name in metrics:
                categories[category].append({
                    'name': strategy_name,
                    'metrics': metrics[strategy_name]
                })
        
        # Rank within each category
        category_rankings = {}
        for category, strategy_list in categories.items():
            ranked = sorted(
                strategy_list,
                key=lambda x: x['metrics'].get('ranking_score', 0),
                reverse=True
            )
            
            category_rankings[category] = [
                {
                    'rank': i + 1,
                    'name': s['name'],
                    'ranking_score': s['metrics'].get('ranking_score', 0),
                    'sharpe_ratio': s['metrics'].get('sharpe_ratio', 0),
                    'total_return': s['metrics'].get('total_return', 0)
                }
                for i, s in enumerate(ranked)
            ]
        
        return category_rankings
    
    def _generate_regime_rankings(self, metrics: Dict[str, Dict]) -> Dict[str, List]:
        """Generate rankings by market regime performance."""
        regime_types = ['bull', 'bear', 'high_volatility', 'ranging']
        regime_rankings = {}
        
        for regime in regime_types:
            rankings = []
            
            for strategy_name, strategy_metrics in metrics.items():
                regime_perf = strategy_metrics.get('regime_performance', {})
                regime_return = regime_perf.get(f'{regime}_return', 0)
                regime_sharpe = regime_perf.get(f'{regime}_sharpe', 0)
                
                if regime_return != 0 or regime_sharpe != 0:
                    rankings.append({
                        'name': strategy_name,
                        'return': regime_return,
                        'sharpe': regime_sharpe,
                        'score': regime_return * 100 + regime_sharpe * 30
                    })
            
            # Sort by composite score
            rankings.sort(key=lambda x: x['score'], reverse=True)
            
            # Add ranks
            for i, entry in enumerate(rankings):
                entry['rank'] = i + 1
            
            regime_rankings[regime] = rankings[:10]  # Top 10 for each regime
        
        return regime_rankings
    
    def _generate_risk_adjusted_rankings(self, metrics: Dict[str, Dict]) -> List[Dict]:
        """Generate risk-adjusted performance rankings."""
        rankings = []
        
        for strategy_name, strategy_metrics in metrics.items():
            # Calculate risk-adjusted score
            sharpe = strategy_metrics.get('sharpe_ratio', 0)
            sortino = strategy_metrics.get('sortino_ratio', 0)
            calmar = strategy_metrics.get('calmar_ratio', 0)
            omega = strategy_metrics.get('omega_ratio', 0)
            
            # Weighted risk-adjusted score
            risk_adjusted_score = (
                sharpe * 0.3 +
                sortino * 0.3 +
                calmar * 0.2 +
                omega * 0.2
            )
            
            rankings.append({
                'name': strategy_name,
                'risk_adjusted_score': risk_adjusted_score,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'omega_ratio': omega,
                'max_drawdown': strategy_metrics.get('max_drawdown', 0),
                'var_95': strategy_metrics.get('var_95', 0),
                'cvar_95': strategy_metrics.get('cvar_95', 0)
            })
        
        # Sort by risk-adjusted score
        rankings.sort(key=lambda x: x['risk_adjusted_score'], reverse=True)
        
        # Add ranks
        for i, entry in enumerate(rankings):
            entry['rank'] = i + 1
        
        return rankings
    
    async def _generate_reports(
        self, 
        rankings: Dict, 
        metrics: Dict, 
        context: PipelineContext
    ) -> None:
        """Generate and save assessment reports."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Save rankings JSON
        rankings_file = self.output_dir / f'rankings_{timestamp}.json'
        with open(rankings_file, 'w') as f:
            json.dump(rankings, f, indent=2, default=str)
        logger.info(f"Saved rankings to {rankings_file}")
        
        # 2. Save comprehensive metrics
        metrics_file = self.output_dir / f'comprehensive_metrics_{timestamp}.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Saved comprehensive metrics to {metrics_file}")
        
        # 3. Generate summary report
        summary_report = self._generate_summary_report(rankings, metrics, context)
        summary_file = self.output_dir / f'assessment_summary_{timestamp}.txt'
        with open(summary_file, 'w') as f:
            f.write(summary_report)
        logger.info(f"Saved summary report to {summary_file}")
        
        # 4. Save latest summary for easy access
        latest_summary_file = self.output_dir / 'assessment_pipeline_summary.json'
        latest_summary = {
            'timestamp': datetime.now().isoformat(),
            'top_strategies': rankings['overall'][:5] if rankings.get('overall') else [],
            'best_by_category': {
                cat: strategies[0] if strategies else None
                for cat, strategies in rankings.get('by_category', {}).items()
            },
            'best_by_regime': {
                regime: strategies[0] if strategies else None
                for regime, strategies in rankings.get('by_regime', {}).items()
            },
            'statistics': {
                'total_strategies': len(metrics),
                'total_backtests': context.data.get('backtest_summary', {}).get('total_backtests', 0),
                'successful_backtests': context.data.get('backtest_summary', {}).get('successful_backtests', 0),
                'failed_backtests': context.data.get('backtest_summary', {}).get('failed_backtests', 0)
            }
        }
        
        with open(latest_summary_file, 'w') as f:
            json.dump(latest_summary, f, indent=2, default=str)
        logger.info(f"Saved latest summary to {latest_summary_file}")
    
    def _generate_summary_report(
        self, 
        rankings: Dict, 
        metrics: Dict, 
        context: PipelineContext
    ) -> str:
        """Generate human-readable summary report."""
        report = []
        report.append("=" * 80)
        report.append("ASSESSMENT PIPELINE REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        
        # Executive Summary
        report.append("\n=== EXECUTIVE SUMMARY ===")
        summary = context.data.get('backtest_summary', {})
        report.append(f"Total Strategies Evaluated: {len(metrics)}")
        report.append(f"Total Backtests Run: {summary.get('total_backtests', 0):,}")
        report.append(f"Successful Backtests: {summary.get('successful_backtests', 0):,}")
        report.append(f"Failed Backtests: {summary.get('failed_backtests', 0):,}")
        
        # Top Strategies
        report.append("\n=== TOP 5 STRATEGIES (Overall) ===")
        overall = rankings.get('overall', [])[:5]
        for i, strategy in enumerate(overall, 1):
            report.append(f"\n{i}. {strategy['name']} (Score: {strategy['ranking_score']:.2f})")
            report.append(f"   - Sharpe Ratio: {strategy['sharpe_ratio']:.2f}")
            report.append(f"   - Total Return: {strategy['total_return']:.1%}")
            report.append(f"   - Max Drawdown: {strategy['max_drawdown']:.1%}")
            report.append(f"   - Win Rate: {strategy['win_rate']:.1%}")
        
        # Best by Category
        report.append("\n=== BEST STRATEGIES BY CATEGORY ===")
        for category, category_strategies in rankings.get('by_category', {}).items():
            if category_strategies:
                best = category_strategies[0]
                report.append(f"{category}: {best['name']} (Score: {best['ranking_score']:.2f})")
        
        # Best by Regime
        report.append("\n=== BEST STRATEGIES BY MARKET REGIME ===")
        for regime, regime_strategies in rankings.get('by_regime', {}).items():
            if regime_strategies:
                best = regime_strategies[0]
                report.append(f"{regime}: {best['name']} (Return: {best['return']:.1%}, Sharpe: {best['sharpe']:.2f})")
        
        # Risk Analysis
        report.append("\n=== RISK ANALYSIS ===")
        risk_rankings = rankings.get('risk_adjusted', [])[:3]
        report.append("Top 3 Risk-Adjusted Strategies:")
        for i, strategy in enumerate(risk_rankings, 1):
            report.append(f"{i}. {strategy['name']}")
            report.append(f"   - Risk-Adjusted Score: {strategy['risk_adjusted_score']:.2f}")
            report.append(f"   - Max Drawdown: {strategy['max_drawdown']:.1%}")
            report.append(f"   - VaR (95%): {strategy['var_95']:.1%}")
        
        # Recommendations
        report.append("\n=== RECOMMENDATIONS ===")
        if overall:
            top_strategy = overall[0]
            report.append(f"1. Primary Strategy: {top_strategy['name']}")
            report.append(f"   - Deploy with 40% allocation")
            report.append(f"   - Expected Annual Return: {metrics[top_strategy['name']].get('annualized_return', 0):.1%}")
            
            if len(overall) > 1:
                second_strategy = overall[1]
                report.append(f"\n2. Secondary Strategy: {second_strategy['name']}")
                report.append(f"   - Deploy with 30% allocation")
                report.append(f"   - Provides diversification benefit")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)