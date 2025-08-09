"""Strategy ranking step for benchmark pipeline."""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from src.pipelines.base import PipelineStep, PipelineContext
from src.utils.logging import logger


class RankingStep(PipelineStep[Dict[str, Any]]):
    """Rank strategies based on multiple criteria and create composite scores."""
    
    def __init__(self):
        super().__init__(
            name="Ranking",
            description="Rank strategies by performance metrics"
        )
    
    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Rank strategies based on multiple criteria."""
        enhanced_results = context.data.get("enhanced_results", {})
        
        if not enhanced_results:
            return {"status": "failed", "error": "No results to rank"}
        
        # Filter only successful strategies
        successful_strategies = {
            name: results for name, results in enhanced_results.items()
            if results.get("status") == "success"
        }
        
        if not successful_strategies:
            return {"status": "failed", "error": "No successful strategies to rank"}
        
        logger.info(f"Ranking {len(successful_strategies)} successful strategies")
        
        # Create DataFrame for easier ranking
        df = pd.DataFrame(successful_strategies).T
        
        # Calculate rankings for each metric
        rankings = {}
        
        # Primary metrics (higher is better)
        rankings["sharpe"] = self._rank_metric(df, "sharpe_ratio", ascending=False)
        rankings["return"] = self._rank_metric(df, "total_return", ascending=False)
        rankings["sortino"] = self._rank_metric(df, "sortino_ratio", ascending=False)
        rankings["calmar"] = self._rank_metric(df, "calmar_ratio", ascending=False)
        
        # Risk metrics (lower is better)
        rankings["drawdown"] = self._rank_metric(df, "max_drawdown", ascending=True)
        rankings["volatility"] = self._rank_metric(df, "volatility", ascending=True)
        
        # Consistency metrics (higher is better)
        rankings["consistency"] = self._rank_metric(df, "consistency_score", ascending=False)
        rankings["win_rate"] = self._rank_metric(df, "win_rate", ascending=False)
        
        # Calculate composite scores
        composite_scores = self._calculate_composite_scores(df, rankings)
        
        # Create final rankings
        final_rankings = []
        
        for strategy_name in successful_strategies:
            ranking_data = {
                "strategy_name": strategy_name,
                "rank_by_sharpe": rankings["sharpe"].get(strategy_name, None),
                "rank_by_return": rankings["return"].get(strategy_name, None),
                "rank_by_sortino": rankings["sortino"].get(strategy_name, None),
                "rank_by_calmar": rankings["calmar"].get(strategy_name, None),
                "rank_by_drawdown": rankings["drawdown"].get(strategy_name, None),
                "rank_by_consistency": rankings["consistency"].get(strategy_name, None),
                "composite_score": composite_scores.get(strategy_name, 0),
                "category": context.data.get("strategy_categories", {}).get(strategy_name, "unknown")
            }
            
            # Add key metrics for easy access
            metrics = successful_strategies[strategy_name]
            ranking_data.update({
                "total_return": metrics.get("total_return"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "max_drawdown": metrics.get("max_drawdown"),
                "total_trades": metrics.get("total_trades"),
                "win_rate": metrics.get("win_rate")
            })
            
            final_rankings.append(ranking_data)
        
        # Sort by composite score
        final_rankings.sort(key=lambda x: x["composite_score"], reverse=True)
        
        # Add overall rank
        for i, ranking in enumerate(final_rankings):
            ranking["overall_rank"] = i + 1
        
        # Store rankings
        context.data["rankings"] = final_rankings
        
        # Identify top performers by category
        top_by_category = self._get_top_by_category(final_rankings)
        context.data["top_by_category"] = top_by_category
        
        # Identify best for different market conditions
        market_condition_rankings = self._rank_by_market_conditions(df, successful_strategies)
        context.data["market_condition_rankings"] = market_condition_rankings
        
        result = {
            "status": "success",
            "strategies_ranked": len(final_rankings),
            "top_strategy": final_rankings[0]["strategy_name"] if final_rankings else None,
            "top_5_strategies": [r["strategy_name"] for r in final_rankings[:5]],
            "categories_ranked": len(set(r["category"] for r in final_rankings))
        }
        
        logger.info(f"Ranking complete. Top strategy: {result['top_strategy']}")
        return result
    
    def _rank_metric(self, df: pd.DataFrame, metric: str, ascending: bool = True) -> Dict[str, int]:
        """Rank strategies by a specific metric."""
        if metric not in df.columns:
            return {}
        
        # Handle None/NaN values
        valid_df = df[df[metric].notna()]
        
        if valid_df.empty:
            return {}
        
        # Rank (1 is best)
        if ascending:
            ranks = valid_df[metric].rank(method='min').astype(int)
        else:
            ranks = valid_df[metric].rank(method='min', ascending=False).astype(int)
        
        return ranks.to_dict()
    
    def _calculate_composite_scores(self, df: pd.DataFrame, rankings: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate composite scores based on multiple rankings."""
        composite_scores = {}
        
        # Weights for different metrics
        weights = {
            "sharpe": 0.25,
            "return": 0.20,
            "sortino": 0.15,
            "calmar": 0.10,
            "drawdown": 0.10,
            "consistency": 0.10,
            "win_rate": 0.10
        }
        
        for strategy_name in df.index:
            score = 0
            weight_sum = 0
            
            for metric, weight in weights.items():
                if metric in rankings and strategy_name in rankings[metric]:
                    rank = rankings[metric][strategy_name]
                    # Convert rank to score (higher is better)
                    # Best rank (1) gets highest score
                    metric_score = (len(df) - rank + 1) / len(df) * 100
                    score += metric_score * weight
                    weight_sum += weight
            
            # Normalize by actual weights used
            if weight_sum > 0:
                composite_scores[strategy_name] = round(score / weight_sum, 2)
            else:
                composite_scores[strategy_name] = 0
        
        return composite_scores
    
    def _get_top_by_category(self, rankings: List[Dict]) -> Dict[str, List[str]]:
        """Get top strategies by category."""
        category_strategies = {}
        
        for ranking in rankings:
            category = ranking["category"]
            if category not in category_strategies:
                category_strategies[category] = []
            category_strategies[category].append(ranking["strategy_name"])
        
        # Keep top 3 per category
        top_by_category = {}
        for category, strategies in category_strategies.items():
            top_by_category[category] = strategies[:3]
        
        return top_by_category
    
    def _rank_by_market_conditions(self, df: pd.DataFrame, strategies: Dict) -> Dict[str, List[Tuple[str, float]]]:
        """Rank strategies for different market conditions."""
        market_rankings = {}
        
        # High volatility markets - prefer low volatility strategies
        if "volatility" in df.columns:
            # Convert volatility to numeric, handling any non-numeric values
            df["volatility_numeric"] = pd.to_numeric(df["volatility"], errors='coerce')
            valid_vol = df[df["volatility_numeric"].notna()]
            
            if not valid_vol.empty:
                low_vol = valid_vol.nsmallest(5, "volatility_numeric")
                market_rankings["high_volatility_markets"] = [
                    (name, round(row["sharpe_ratio"], 2)) 
                    for name, row in low_vol.iterrows()
                    if pd.notna(row["sharpe_ratio"])
                ]
        
        # Bear markets - prefer strategies with low drawdown
        if "max_drawdown" in df.columns:
            # Convert max_drawdown to numeric, handling any non-numeric values
            df["max_drawdown_numeric"] = pd.to_numeric(df["max_drawdown"], errors='coerce')
            valid_dd = df[df["max_drawdown_numeric"].notna()]
            
            if not valid_dd.empty:
                low_dd = valid_dd.nsmallest(5, "max_drawdown_numeric")
                market_rankings["bear_markets"] = [
                    (name, round(abs(row["max_drawdown_numeric"]), 2))
                    for name, row in low_dd.iterrows()
                ]
        
        # Trending markets - prefer trend following strategies
        trend_strategies = [
            (name, round(strategies[name]["sharpe_ratio"], 2))
            for name in df.index
            if "trend" in name.lower() and pd.notna(strategies[name].get("sharpe_ratio"))
        ]
        if trend_strategies:
            market_rankings["trending_markets"] = sorted(
                trend_strategies, key=lambda x: x[1], reverse=True
            )[:5]
        
        # Range-bound markets - prefer mean reversion strategies
        mr_strategies = [
            (name, round(strategies[name]["win_rate"], 2))
            for name in df.index
            if "mean_reversion" in name.lower() or "pairs" in name.lower()
            and pd.notna(strategies[name].get("win_rate"))
        ]
        if mr_strategies:
            market_rankings["range_bound_markets"] = sorted(
                mr_strategies, key=lambda x: x[1], reverse=True
            )[:5]
        
        return market_rankings