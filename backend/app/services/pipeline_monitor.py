"""Pipeline monitoring service for tracking execution status and metrics."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import asyncpg
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class PipelineStatus:
    """Pipeline execution status."""
    name: str
    status: str  # running, success, failed, idle
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    duration_seconds: Optional[int]
    records_processed: Optional[int]
    error_message: Optional[str] = None


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics."""
    name: str
    avg_duration_seconds: float
    success_rate: float
    total_runs: int
    successful_runs: int
    failed_runs: int
    avg_records_processed: float


class PipelineMonitor:
    """Service for monitoring pipeline executions and status."""
    
    PIPELINES = {
        'data': {
            'name': 'Data Pipeline',
            'schedule': '19:00',  # 7 PM daily
            'log_pattern': '/tmp/tetra_data_pipeline_*.log'
        },
        'scenarios': {
            'name': 'Scenarios Pipeline', 
            'schedule': '19:30',  # 7:30 PM daily
            'log_pattern': '/tmp/tetra_scenarios_pipeline_*.log'
        },
        'metrics': {
            'name': 'Metrics Pipeline',
            'schedule': '20:00',  # 8 PM daily
            'log_pattern': '/tmp/tetra_metrics_pipeline_*.log'
        },
        'assessment': {
            'name': 'Assessment Pipeline',
            'schedule': '20:30',  # 8:30 PM daily
            'log_pattern': '/tmp/tetra_assessment_pipeline_*.log'
        }
    }
    
    def __init__(self, db_connection: asyncpg.Connection):
        self.db = db_connection
        self._cache = {}
        self._cache_expiry = {}
        
    async def get_pipeline_status(self, pipeline_name: Optional[str] = None) -> Dict[str, PipelineStatus]:
        """Get current status of pipelines."""
        pipelines = [pipeline_name] if pipeline_name else self.PIPELINES.keys()
        statuses = {}
        
        for name in pipelines:
            if name not in self.PIPELINES:
                continue
                
            # Check cache
            if self._is_cache_valid(f"status_{name}"):
                statuses[name] = self._cache[f"status_{name}"]
                continue
            
            # Get latest run from database
            query = """
                SELECT 
                    pipeline_name,
                    start_time,
                    end_time,
                    status,
                    records_processed,
                    error_message,
                    execution_time_seconds
                FROM assessment.pipeline_runs
                WHERE pipeline_name = $1
                ORDER BY start_time DESC
                LIMIT 1
            """
            
            row = await self.db.fetchrow(query, name)
            
            if row:
                # Calculate next run based on schedule
                schedule_time = self.PIPELINES[name]['schedule']
                next_run = self._calculate_next_run(row['start_time'], schedule_time)
                
                status = PipelineStatus(
                    name=name,
                    status=row['status'],
                    last_run=row['start_time'],
                    next_run=next_run,
                    duration_seconds=row['execution_time_seconds'],
                    records_processed=row['records_processed'],
                    error_message=row['error_message']
                )
            else:
                # No runs yet
                status = PipelineStatus(
                    name=name,
                    status='idle',
                    last_run=None,
                    next_run=self._calculate_next_run(datetime.now(), self.PIPELINES[name]['schedule']),
                    duration_seconds=None,
                    records_processed=None
                )
            
            statuses[name] = status
            self._cache_result(f"status_{name}", status, ttl_seconds=60)
        
        return statuses
    
    async def get_pipeline_metrics(self, pipeline_name: Optional[str] = None, days: int = 30) -> Dict[str, PipelineMetrics]:
        """Get performance metrics for pipelines."""
        pipelines = [pipeline_name] if pipeline_name else self.PIPELINES.keys()
        metrics = {}
        
        since_date = datetime.now() - timedelta(days=days)
        
        for name in pipelines:
            if name not in self.PIPELINES:
                continue
            
            # Check cache
            cache_key = f"metrics_{name}_{days}"
            if self._is_cache_valid(cache_key):
                metrics[name] = self._cache[cache_key]
                continue
            
            # Get metrics from database
            query = """
                SELECT 
                    COUNT(*) as total_runs,
                    COUNT(CASE WHEN status = 'success' THEN 1 END) as successful_runs,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_runs,
                    AVG(execution_time_seconds) as avg_duration,
                    AVG(records_processed) as avg_records
                FROM assessment.pipeline_runs
                WHERE pipeline_name = $1
                AND start_time >= $2
            """
            
            row = await self.db.fetchrow(query, name, since_date)
            
            if row and row['total_runs'] > 0:
                metric = PipelineMetrics(
                    name=name,
                    avg_duration_seconds=float(row['avg_duration'] or 0),
                    success_rate=float(row['successful_runs']) / float(row['total_runs']),
                    total_runs=row['total_runs'],
                    successful_runs=row['successful_runs'],
                    failed_runs=row['failed_runs'],
                    avg_records_processed=float(row['avg_records'] or 0)
                )
            else:
                metric = PipelineMetrics(
                    name=name,
                    avg_duration_seconds=0,
                    success_rate=0,
                    total_runs=0,
                    successful_runs=0,
                    failed_runs=0,
                    avg_records_processed=0
                )
            
            metrics[name] = metric
            self._cache_result(cache_key, metric, ttl_seconds=300)
        
        return metrics
    
    async def get_execution_history(self, pipeline_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history for a pipeline."""
        query = """
            SELECT 
                pipeline_name,
                start_time,
                end_time,
                status,
                records_processed,
                error_message,
                execution_time_seconds
            FROM assessment.pipeline_runs
            WHERE pipeline_name = $1
            ORDER BY start_time DESC
            LIMIT $2
        """
        
        rows = await self.db.fetch(query, pipeline_name, limit)
        
        history = []
        for row in rows:
            history.append({
                'pipeline_name': row['pipeline_name'],
                'start_time': row['start_time'].isoformat() if row['start_time'] else None,
                'end_time': row['end_time'].isoformat() if row['end_time'] else None,
                'status': row['status'],
                'records_processed': row['records_processed'],
                'error_message': row['error_message'],
                'duration_seconds': row['execution_time_seconds']
            })
        
        return history
    
    async def record_pipeline_start(self, pipeline_name: str) -> int:
        """Record the start of a pipeline execution."""
        query = """
            INSERT INTO assessment.pipeline_runs (pipeline_name, start_time, status)
            VALUES ($1, $2, $3)
            RETURNING id
        """
        
        run_id = await self.db.fetchval(query, pipeline_name, datetime.now(), 'running')
        logger.info(f"Started pipeline run {run_id} for {pipeline_name}")
        return run_id
    
    async def record_pipeline_end(self, run_id: int, status: str, records_processed: int = None, error_message: str = None):
        """Record the end of a pipeline execution."""
        end_time = datetime.now()
        
        # Get start time to calculate duration
        start_time = await self.db.fetchval(
            "SELECT start_time FROM assessment.pipeline_runs WHERE id = $1",
            run_id
        )
        
        if start_time:
            duration = int((end_time - start_time).total_seconds())
        else:
            duration = None
        
        query = """
            UPDATE assessment.pipeline_runs
            SET end_time = $2,
                status = $3,
                records_processed = $4,
                error_message = $5,
                execution_time_seconds = $6
            WHERE id = $1
        """
        
        await self.db.execute(
            query,
            run_id,
            end_time,
            status,
            records_processed,
            error_message,
            duration
        )
        
        logger.info(f"Completed pipeline run {run_id} with status {status}")
    
    async def get_latest_assessment_results(self) -> Dict[str, Any]:
        """Get the latest assessment pipeline results summary."""
        # Get latest run date
        latest_run = await self.db.fetchval("""
            SELECT MAX(run_date) FROM assessment.backtest_results
        """)
        
        if not latest_run:
            return {'has_data': False}
        
        # Get top strategies
        top_strategies = await self.db.fetch("""
            SELECT 
                strategy_name,
                category,
                overall_rank,
                avg_score,
                avg_return,
                avg_sharpe
            FROM assessment.strategy_rankings
            WHERE run_date = $1
            ORDER BY overall_rank ASC
            LIMIT 5
        """, latest_run)
        
        # Get scenario performance
        scenarios = await self.db.fetch("""
            SELECT 
                scenario_name,
                scenario_type,
                top_strategy,
                top_return,
                avg_return
            FROM assessment.scenario_performance
            WHERE run_date = $1
            ORDER BY top_return DESC
            LIMIT 5
        """, latest_run)
        
        return {
            'has_data': True,
            'run_date': latest_run.isoformat(),
            'top_strategies': [dict(row) for row in top_strategies],
            'top_scenarios': [dict(row) for row in scenarios]
        }
    
    async def check_pipeline_health(self) -> Dict[str, Any]:
        """Check overall health of the pipeline system."""
        health = {
            'healthy': True,
            'issues': [],
            'warnings': []
        }
        
        # Check if any pipeline is stuck running
        stuck_query = """
            SELECT pipeline_name, start_time
            FROM assessment.pipeline_runs
            WHERE status = 'running'
            AND start_time < $1
        """
        stuck_threshold = datetime.now() - timedelta(hours=2)
        stuck_pipelines = await self.db.fetch(stuck_query, stuck_threshold)
        
        if stuck_pipelines:
            health['healthy'] = False
            for row in stuck_pipelines:
                health['issues'].append(
                    f"Pipeline '{row['pipeline_name']}' stuck running since {row['start_time']}"
                )
        
        # Check for high failure rates
        failure_query = """
            SELECT 
                pipeline_name,
                COUNT(*) as total,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
            FROM assessment.pipeline_runs
            WHERE start_time >= $1
            GROUP BY pipeline_name
        """
        week_ago = datetime.now() - timedelta(days=7)
        failure_stats = await self.db.fetch(failure_query, week_ago)
        
        for row in failure_stats:
            if row['total'] > 0:
                failure_rate = row['failed'] / row['total']
                if failure_rate > 0.5:
                    health['healthy'] = False
                    health['issues'].append(
                        f"Pipeline '{row['pipeline_name']}' has {failure_rate:.0%} failure rate"
                    )
                elif failure_rate > 0.2:
                    health['warnings'].append(
                        f"Pipeline '{row['pipeline_name']}' has {failure_rate:.0%} failure rate"
                    )
        
        # Check for missing runs
        for name, config in self.PIPELINES.items():
            status = await self.get_pipeline_status(name)
            if name in status:
                if status[name].last_run:
                    hours_since_run = (datetime.now() - status[name].last_run).total_seconds() / 3600
                    if hours_since_run > 48:
                        health['warnings'].append(
                            f"Pipeline '{name}' hasn't run in {hours_since_run:.0f} hours"
                        )
                else:
                    health['warnings'].append(f"Pipeline '{name}' has never run")
        
        return health
    
    def _calculate_next_run(self, last_run: datetime, schedule_time: str) -> datetime:
        """Calculate next scheduled run time."""
        hour, minute = map(int, schedule_time.split(':'))
        
        # Get today's scheduled time
        today_schedule = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # If today's schedule hasn't passed yet, that's the next run
        if datetime.now() < today_schedule:
            return today_schedule
        
        # Otherwise, it's tomorrow
        return today_schedule + timedelta(days=1)
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached result is still valid."""
        if key not in self._cache:
            return False
        if key not in self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry[key]
    
    def _cache_result(self, key: str, value: Any, ttl_seconds: int = 60):
        """Cache a result with TTL."""
        self._cache[key] = value
        self._cache_expiry[key] = datetime.now() + timedelta(seconds=ttl_seconds)