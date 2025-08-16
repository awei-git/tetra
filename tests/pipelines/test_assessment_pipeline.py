"""Tests for the assessment pipeline."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.pipelines.assessment_pipeline.pipeline import AssessmentPipeline
from src.pipelines.assessment_pipeline.steps import (
    DataGatheringStep,
    StrategyLoadingStep,
    BacktestExecutionStep,
    PerformanceCalculationStep,
    RankingGenerationStep,
    DatabaseStorageStep
)
from src.pipelines.assessment_pipeline.steps.backtest_execution import BacktestResult
from src.pipelines.base import PipelineContext


@pytest.fixture
def pipeline_context():
    """Create a test pipeline context."""
    return PipelineContext()


@pytest.fixture
def sample_scenarios():
    """Create sample scenario data."""
    return [
        {
            'name': 'historical_2020',
            'type': 'historical',
            'start_date': '2020-01-01',
            'end_date': '2020-12-31',
            'description': 'Historical 2020 scenario'
        },
        {
            'name': 'bull_market',
            'type': 'bull',
            'start_date': '2021-01-01',
            'end_date': '2021-12-31',
            'description': 'Bull market scenario'
        }
    ]


@pytest.fixture
def sample_symbols():
    """Create sample symbol list."""
    return ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ']


@pytest.fixture
def sample_strategies():
    """Create sample strategy configurations."""
    return [
        {
            'name': 'golden_cross',
            'category': 'trend_following',
            'config': {
                'category': 'trend_following',
                'description': 'Golden Cross Strategy',
                'parameters': {
                    'fast_ma': 50,
                    'slow_ma': 200
                }
            }
        },
        {
            'name': 'buy_and_hold',
            'category': 'passive',
            'config': {
                'category': 'passive',
                'description': 'Buy and Hold Strategy',
                'parameters': {
                    'investment_amount': 10000
                }
            }
        }
    ]


@pytest.fixture
def sample_metrics_data():
    """Create sample metrics data."""
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    data = {}
    
    for symbol in ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ']:
        df = pd.DataFrame({
            'timestamp': dates,
            'symbol': symbol,
            'close': np.random.randn(252).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, 252),
            'sma_20': np.random.randn(252).cumsum() + 100,
            'sma_50': np.random.randn(252).cumsum() + 100,
            'rsi': np.random.uniform(30, 70, 252),
            'macd': np.random.randn(252),
            'signal': np.random.randn(252)
        })
        data[symbol] = df
    
    return pd.concat(data.values(), ignore_index=True)


@pytest.fixture
def sample_backtest_results():
    """Create sample backtest results."""
    results = []
    
    for strategy in ['golden_cross', 'buy_and_hold']:
        for symbol in ['AAPL', 'MSFT']:
            for scenario in ['historical_2020', 'bull_market']:
                result = BacktestResult(
                    strategy_name=strategy,
                    symbol=symbol,
                    scenario_name=scenario,
                    total_return=np.random.uniform(-0.2, 0.5),
                    annualized_return=np.random.uniform(-0.1, 0.3),
                    volatility=np.random.uniform(0.15, 0.35),
                    sharpe_ratio=np.random.uniform(-0.5, 2.5),
                    max_drawdown=np.random.uniform(-0.3, -0.05),
                    win_rate=np.random.uniform(0.4, 0.7),
                    profit_factor=np.random.uniform(0.8, 2.5),
                    total_trades=np.random.randint(10, 100),
                    equity_curve=[100000 * (1 + np.random.uniform(-0.01, 0.01)) for _ in range(252)],
                    trade_log=[],
                    metadata={'test': True}
                )
                results.append(result)
    
    return results


class TestAssessmentPipeline:
    """Test the main assessment pipeline."""
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = AssessmentPipeline()
        assert pipeline.name == "AssessmentPipeline"
        assert pipeline.results_dir.exists()
    
    @pytest.mark.asyncio
    async def test_pipeline_setup(self, pipeline_context):
        """Test pipeline setup with all steps."""
        pipeline = AssessmentPipeline()
        pipeline.setup(pipeline_context)
        
        assert len(pipeline.steps) == 6
        assert isinstance(pipeline.steps[0], DataGatheringStep)
        assert isinstance(pipeline.steps[1], StrategyLoadingStep)
        assert isinstance(pipeline.steps[2], BacktestExecutionStep)
        assert isinstance(pipeline.steps[3], PerformanceCalculationStep)
        assert isinstance(pipeline.steps[4], RankingGenerationStep)
        assert isinstance(pipeline.steps[5], DatabaseStorageStep)


class TestDataGatheringStep:
    """Test the data gathering step."""
    
    @pytest.mark.asyncio
    async def test_execute_with_mock_data(
        self, 
        pipeline_context, 
        sample_scenarios,
        sample_symbols,
        sample_strategies,
        sample_metrics_data,
        tmp_path
    ):
        """Test data gathering with mock data."""
        step = DataGatheringStep()
        
        # Create mock scenario and metrics files
        scenarios_dir = tmp_path / 'data' / 'scenarios'
        scenarios_dir.mkdir(parents=True)
        
        metadata = {
            scenario['name']: scenario 
            for scenario in sample_scenarios
        }
        
        with open(scenarios_dir / 'scenario_metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        # Create mock metrics files
        metrics_dir = tmp_path / 'data' / 'metrics'
        metrics_dir.mkdir(parents=True)
        
        for scenario in sample_scenarios:
            metrics_file = metrics_dir / f"{scenario['name']}_metrics.parquet"
            sample_metrics_data.to_parquet(metrics_file)
        
        # Patch paths
        with patch.object(Path, 'cwd', return_value=tmp_path):
            with patch('src.pipelines.assessment_pipeline.steps.data_gathering.Path') as mock_path:
                mock_path.side_effect = lambda x: tmp_path / x if isinstance(x, str) else x
                
                # Mock MarketUniverse
                with patch('src.pipelines.assessment_pipeline.steps.data_gathering.MarketUniverse') as mock_universe:
                    mock_universe.return_value.get_symbols.return_value = sample_symbols
                    
                    # Mock DEFAULT_STRATEGIES
                    with patch('src.pipelines.assessment_pipeline.steps.data_gathering.DEFAULT_STRATEGIES', sample_strategies):
                        await step.execute(pipeline_context)
        
        # Verify context was populated
        assert pipeline_context.get('scenarios') is not None
        assert pipeline_context.get('symbols') is not None
        assert pipeline_context.get('strategy_configs') is not None
        assert pipeline_context.get('metrics_data') is not None
        assert pipeline_context.get('total_combinations') > 0


class TestBacktestExecutionStep:
    """Test the backtest execution step."""
    
    @pytest.mark.asyncio
    async def test_execute_backtests(
        self,
        pipeline_context,
        sample_strategies,
        sample_symbols,
        sample_scenarios,
        sample_metrics_data
    ):
        """Test backtest execution."""
        step = BacktestExecutionStep(parallel_workers=2)
        
        # Setup context
        pipeline_context.set('strategies', [
            {
                'name': s['name'],
                'instance': Mock(),  # Mock strategy instance
                'config': s['config'],
                'category': s['category']
            }
            for s in sample_strategies
        ])
        pipeline_context.set('symbols', sample_symbols[:2])  # Use fewer symbols for test
        pipeline_context.set('scenarios', sample_scenarios)
        pipeline_context.set('metrics_data', {
            scenario['name']: sample_metrics_data
            for scenario in sample_scenarios
        })
        
        # Mock the simulator
        with patch('src.pipelines.assessment_pipeline.steps.backtest_execution.HistoricalSimulator') as mock_sim:
            mock_portfolio = Mock()
            mock_portfolio.get_equity_curve.return_value = [100000, 105000, 110000]
            mock_portfolio.get_trades.return_value = [
                {'pnl': 1000}, {'pnl': -500}, {'pnl': 2000}
            ]
            
            mock_sim.return_value.run = AsyncMock(return_value=mock_portfolio)
            
            await step.execute(pipeline_context)
        
        # Verify results
        results = pipeline_context.get('backtest_results')
        assert results is not None
        assert len(results) > 0
        
        summary = pipeline_context.get('backtest_summary')
        assert summary is not None
        assert summary['total_backtests'] > 0


class TestPerformanceCalculationStep:
    """Test the performance calculation step."""
    
    @pytest.mark.asyncio
    async def test_calculate_comprehensive_metrics(
        self,
        pipeline_context,
        sample_backtest_results
    ):
        """Test comprehensive metric calculation."""
        step = PerformanceCalculationStep()
        
        # Setup context
        pipeline_context.set('backtest_results', sample_backtest_results)
        pipeline_context.set('symbols', ['AAPL', 'MSFT'])
        
        await step.execute(pipeline_context)
        
        # Verify metrics
        metrics = pipeline_context.get('comprehensive_metrics')
        assert metrics is not None
        assert len(metrics) > 0
        
        # Check for expected metric keys
        for strategy_name, strategy_metrics in metrics.items():
            assert 'total_return' in strategy_metrics
            assert 'sharpe_ratio' in strategy_metrics
            assert 'max_drawdown' in strategy_metrics
            assert 'win_rate' in strategy_metrics
            assert 'ranking_score' in strategy_metrics


class TestRankingGenerationStep:
    """Test the ranking generation step."""
    
    @pytest.mark.asyncio
    async def test_generate_rankings(
        self,
        pipeline_context,
        tmp_path
    ):
        """Test ranking generation."""
        step = RankingGenerationStep()
        step.output_dir = tmp_path / 'assessment'
        step.output_dir.mkdir(parents=True)
        
        # Setup mock comprehensive metrics
        comprehensive_metrics = {
            'strategy_1': {
                'total_return': 0.25,
                'sharpe_ratio': 1.5,
                'max_drawdown': -0.15,
                'win_rate': 0.60,
                'ranking_score': 150
            },
            'strategy_2': {
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.20,
                'win_rate': 0.55,
                'ranking_score': 120
            }
        }
        
        pipeline_context.set('comprehensive_metrics', comprehensive_metrics)
        pipeline_context.set('strategies', [
            {'name': 'strategy_1', 'category': 'trend_following'},
            {'name': 'strategy_2', 'category': 'mean_reversion'}
        ])
        
        await step.execute(pipeline_context)
        
        # Verify rankings
        rankings = pipeline_context.get('rankings')
        assert rankings is not None
        assert 'overall' in rankings
        assert 'by_category' in rankings
        assert len(rankings['overall']) == 2
        assert rankings['overall'][0]['name'] == 'strategy_1'  # Higher score
        
        # Check if reports were saved
        assert (step.output_dir / 'assessment_pipeline_summary.json').exists()


class TestDatabaseStorageStep:
    """Test the database storage step."""
    
    @pytest.mark.asyncio
    async def test_store_results(
        self,
        pipeline_context,
        sample_backtest_results
    ):
        """Test storing results in database."""
        step = DatabaseStorageStep()
        
        # Setup context
        pipeline_context.set('backtest_results', sample_backtest_results)
        pipeline_context.set('comprehensive_metrics', {
            'golden_cross': {
                'total_return': 0.25,
                'sharpe_ratio': 1.5,
                'ranking_score': 150
            },
            'buy_and_hold': {
                'total_return': 0.20,
                'sharpe_ratio': 1.2,
                'ranking_score': 120
            }
        })
        pipeline_context.set('rankings', {
            'overall': [
                {'name': 'golden_cross', 'rank': 1, 'ranking_score': 150},
                {'name': 'buy_and_hold', 'rank': 2, 'ranking_score': 120}
            ]
        })
        pipeline_context.set('strategies', [
            {'name': 'golden_cross', 'category': 'trend_following'},
            {'name': 'buy_and_hold', 'category': 'passive'}
        ])
        
        # Mock database connection
        with patch('src.pipelines.assessment_pipeline.steps.database_storage.asyncpg.connect') as mock_connect:
            mock_conn = AsyncMock()
            mock_connect.return_value = mock_conn
            
            await step.execute(pipeline_context)
            
            # Verify database operations were called
            mock_connect.assert_called_once()
            assert mock_conn.execute.called
            assert mock_conn.close.called


@pytest.mark.asyncio
async def test_full_pipeline_integration():
    """Test full pipeline integration."""
    pipeline = AssessmentPipeline()
    context = PipelineContext()
    
    # This would be an end-to-end test with real data
    # For now, we'll just verify the pipeline can be set up
    pipeline.setup(context)
    
    assert len(pipeline.steps) == 6
    assert all(step is not None for step in pipeline.steps)