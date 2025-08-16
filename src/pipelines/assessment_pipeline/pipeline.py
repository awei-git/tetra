"""Assessment Pipeline - Stage 4 of Tetra data processing.

This pipeline evaluates all trading strategies across all scenarios and symbols,
calculating comprehensive performance metrics and ranking strategies.

IMPORTANT: This pipeline ONLY runs backtests. It does NOT implement strategies.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass, asdict
import asyncpg
import os

from src.pipelines.base import Pipeline, PipelineStep, PipelineContext
from src.pipelines.assessment_pipeline.steps import (
    DataGatheringStep,
    StrategyLoadingStep,
    BacktestExecutionStep,
    PerformanceCalculationStep,
    RankingGenerationStep,
    DatabaseStorageStep
)

logger = logging.getLogger(__name__)


class AssessmentPipeline(Pipeline):
    """Pipeline for assessing trading strategies across all scenarios."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("AssessmentPipeline", "Pipeline for assessing trading strategies")
        self.config = config or {}
        self.results_dir = Path('data/assessments')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure pipeline steps
        self._configure_steps()
    
    def _configure_steps(self):
        """Configure pipeline steps."""
        # Step 1: Gather all data (scenarios, symbols, metrics)
        self.add_step(DataGatheringStep())
        
        # Step 2: Load strategy implementations
        self.add_step(StrategyLoadingStep())
        
        # Step 3: Execute backtests for all combinations
        self.add_step(BacktestExecutionStep())
        
        # Step 4: Calculate performance metrics
        self.add_step(PerformanceCalculationStep())
        
        # Step 5: Generate rankings and reports
        self.add_step(RankingGenerationStep())
        
        # Step 6: Store results in database
        self.add_step(DatabaseStorageStep())
        
        logger.info(f"Configured {len(self.steps)} steps for assessment pipeline")
    
    async def setup(self) -> PipelineContext:
        """Setup pipeline context."""
        context = PipelineContext()
        context.data['config'] = self.config
        context.data['results_dir'] = str(self.results_dir)
        return context


async def main():
    """Run the assessment pipeline."""
    pipeline = AssessmentPipeline()
    context = PipelineContext()
    
    # Run pipeline
    context = await pipeline.run()
    
    # Print summary
    if context.data.get('summary'):
        summary = context.data.get('summary')
        print("\n" + "=" * 80)
        print("ASSESSMENT PIPELINE RESULTS")
        print("=" * 80)
        print(f"\nTotal backtests run: {summary.get('total_backtests', 0):,}")
        print(f"Successful: {summary.get('successful_backtests', 0):,}")
        print(f"Failed: {summary.get('failed_backtests', 0):,}")
        print(f"\nTop strategy: {summary.get('top_strategy', 'N/A')}")
        print(f"Best return: {summary.get('best_return', 0):.1%}")
        print(f"Best Sharpe: {summary.get('best_sharpe', 0):.2f}")
    else:
        print("\nNo summary available")


if __name__ == "__main__":
    asyncio.run(main())