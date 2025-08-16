# Pipeline Orchestration Documentation

## Overview

The Pipeline Orchestration system manages the execution, scheduling, and coordination of the four-stage pipeline architecture in the Tetra platform. It ensures pipelines run in the correct order, handles dependencies, manages resources, and provides monitoring and error recovery capabilities.

## Pipeline Dependencies

### Execution Order and Dependencies

```mermaid
graph LR
    A[Data Pipeline] --> B[Scenarios Pipeline]
    B --> C[Metrics Pipeline]
    C --> D[Assessment Pipeline]
    
    A --> |Raw Data| B
    B --> |Scenario Definitions| C
    C --> |Pre-calculated Metrics| D
    D --> |Performance Reports| E[Trading Decisions]
```

### Dependency Rules

```yaml
pipeline_dependencies:
  data_pipeline:
    depends_on: []  # No dependencies
    provides: ["market_data", "economic_data", "events", "news"]
    frequency: "daily"
    schedule: "0 19 * * *"  # 7 PM daily
    
  scenarios_pipeline:
    depends_on: ["data_pipeline"]
    requires: ["market_data"]
    provides: ["scenario_definitions", "scenario_timeseries"]
    frequency: "weekly"
    schedule: "0 20 * * 0"  # 8 PM Sunday
    
  metrics_pipeline:
    depends_on: ["scenarios_pipeline"]
    requires: ["scenario_definitions", "market_data"]
    provides: ["technical_indicators", "statistical_metrics", "ml_features"]
    frequency: "weekly"
    schedule: "0 21 * * 0"  # 9 PM Sunday
    
  assessment_pipeline:
    depends_on: ["metrics_pipeline"]
    requires: ["scenario_definitions", "calculated_metrics"]
    provides: ["backtest_results", "strategy_rankings", "reports"]
    frequency: "weekly"
    schedule: "0 22 * * 0"  # 10 PM Sunday
```

## Orchestration Architecture

### Directory Structure
```
bin/
├── pipelines/
│   ├── orchestrator.sh        # Main orchestration script
│   ├── run_full_pipeline.sh   # Run all stages
│   ├── run_daily.sh          # Daily data update
│   ├── run_weekly.sh         # Weekly full cycle
│   └── run_backfill.sh       # Historical backfill
├── scripts/
│   └── pipeline_monitor.py    # Monitoring daemon
└── config/
    ├── orchestration.yaml     # Orchestration config
    └── launchd/               # macOS scheduling
        ├── com.tetra.orchestrator.plist
        ├── com.tetra.daily.plist
        └── com.tetra.weekly.plist
```

## Scheduling Configuration

### macOS launchd Configuration

#### Daily Data Pipeline (com.tetra.daily.plist)
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" 
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.tetra.daily</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>/Users/angwei/Repos/tetra/bin/pipelines/run_daily.sh</string>
    </array>
    
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>19</integer>  <!-- 7 PM -->
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    
    <key>StandardOutPath</key>
    <string>/tmp/tetra_daily.log</string>
    
    <key>StandardErrorPath</key>
    <string>/tmp/tetra_daily_error.log</string>
    
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
```

#### Weekly Full Pipeline (com.tetra.weekly.plist)
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" 
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.tetra.weekly</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>/Users/angwei/Repos/tetra/bin/pipelines/run_weekly.sh</string>
    </array>
    
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>0</integer>  <!-- Sunday -->
        <key>Hour</key>
        <integer>20</integer>  <!-- 8 PM -->
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    
    <key>StandardOutPath</key>
    <string>/tmp/tetra_weekly.log</string>
    
    <key>StandardErrorPath</key>
    <string>/tmp/tetra_weekly_error.log</string>
</dict>
</plist>
```

### Installation and Management

```bash
# Install scheduled jobs
launchctl load ~/Library/LaunchAgents/com.tetra.daily.plist
launchctl load ~/Library/LaunchAgents/com.tetra.weekly.plist

# Check status
launchctl list | grep tetra

# Manual trigger
launchctl start com.tetra.daily
launchctl start com.tetra.weekly

# Disable/Enable
launchctl unload ~/Library/LaunchAgents/com.tetra.daily.plist
launchctl load ~/Library/LaunchAgents/com.tetra.daily.plist

# View logs
tail -f /tmp/tetra_daily.log
tail -f /tmp/tetra_weekly.log
```

## Orchestration Scripts

### Main Orchestrator (bin/pipelines/orchestrator.sh)

```bash
#!/bin/bash
# Main pipeline orchestrator with dependency management

set -e  # Exit on error

# Configuration
TETRA_HOME="/Users/angwei/Repos/tetra"
LOG_DIR="/tmp/tetra_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/orchestrator_$TIMESTAMP.log"

# Python environment
source "$TETRA_HOME/.venv/bin/activate"
cd "$TETRA_HOME"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check pipeline status
check_pipeline_status() {
    local pipeline=$1
    python -m src.orchestration.status_checker --pipeline "$pipeline"
    return $?
}

# Run pipeline with retry
run_pipeline() {
    local pipeline=$1
    local max_retries=3
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        log "Running $pipeline (attempt $((retry + 1)))"
        
        if python -m "src.pipelines.${pipeline}.main" "$@"; then
            log "$pipeline completed successfully"
            return 0
        else
            log "$pipeline failed, retrying..."
            retry=$((retry + 1))
            sleep 60  # Wait before retry
        fi
    done
    
    log "ERROR: $pipeline failed after $max_retries attempts"
    return 1
}

# Main orchestration logic
main() {
    local mode=${1:-"full"}  # full, daily, weekly, backfill
    
    log "Starting orchestration in $mode mode"
    
    case $mode in
        daily)
            run_pipeline "data_pipeline" --mode daily
            ;;
            
        weekly)
            # Run full pipeline chain
            run_pipeline "data_pipeline" --mode daily || exit 1
            run_pipeline "scenarios_pipeline" --type all || exit 1
            run_pipeline "metrics_pipeline" --scenario all || exit 1
            run_pipeline "assessment_pipeline" --strategies all || exit 1
            ;;
            
        full)
            # Run everything with backfill
            run_pipeline "data_pipeline" --mode backfill --days 30 || exit 1
            run_pipeline "scenarios_pipeline" --type all || exit 1
            run_pipeline "metrics_pipeline" --scenario all --force || exit 1
            run_pipeline "assessment_pipeline" --strategies all || exit 1
            ;;
            
        backfill)
            # Historical backfill
            local start_date=${2:-"2020-01-01"}
            local end_date=${3:-$(date +%Y-%m-%d)}
            
            run_pipeline "data_pipeline" \
                --mode backfill \
                --start-date "$start_date" \
                --end-date "$end_date" || exit 1
            ;;
            
        *)
            log "Unknown mode: $mode"
            exit 1
            ;;
    esac
    
    log "Orchestration completed"
}

# Execute
main "$@"
```

### Run Full Pipeline (bin/pipelines/run_full_pipeline.sh)

```bash
#!/bin/bash
# Run complete pipeline chain with validation

set -e

TETRA_HOME="/Users/angwei/Repos/tetra"
source "$TETRA_HOME/.venv/bin/activate"

echo "Starting full pipeline execution..."

# Step 1: Data Pipeline
echo "Step 1/4: Running Data Pipeline..."
python -m src.pipelines.data_pipeline.main --mode daily
if [ $? -ne 0 ]; then
    echo "Data Pipeline failed"
    exit 1
fi

# Step 2: Scenarios Pipeline
echo "Step 2/4: Running Scenarios Pipeline..."
python -m src.pipelines.scenarios_pipeline.main --type all
if [ $? -ne 0 ]; then
    echo "Scenarios Pipeline failed"
    exit 1
fi

# Step 3: Metrics Pipeline
echo "Step 3/4: Running Metrics Pipeline..."
python -m src.pipelines.metrics_pipeline.main --scenario all
if [ $? -ne 0 ]; then
    echo "Metrics Pipeline failed"
    exit 1
fi

# Step 4: Assessment Pipeline
echo "Step 4/4: Running Assessment Pipeline..."
python -m src.pipelines.assessment_pipeline.main --strategies all
if [ $? -ne 0 ]; then
    echo "Assessment Pipeline failed"
    exit 1
fi

echo "Full pipeline execution completed successfully!"

# Send notification (optional)
osascript -e 'display notification "Pipeline completed" with title "Tetra"'
```

## Python Orchestration Module

### Orchestrator Class (src/orchestration/orchestrator.py)

```python
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PipelineTask:
    """Definition of a pipeline task"""
    name: str
    module: str
    dependencies: List[str]
    config: Dict
    status: str = "pending"
    
class PipelineOrchestrator:
    """Main orchestration controller"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.tasks = self.create_task_graph()
        self.executor = ProcessPoolExecutor(max_workers=4)
        
    def create_task_graph(self) -> Dict[str, PipelineTask]:
        """Create dependency graph of pipeline tasks"""
        tasks = {}
        
        for pipeline_name, pipeline_config in self.config['pipelines'].items():
            tasks[pipeline_name] = PipelineTask(
                name=pipeline_name,
                module=pipeline_config['module'],
                dependencies=pipeline_config.get('dependencies', []),
                config=pipeline_config.get('config', {})
            )
        
        return tasks
    
    async def run(self, mode: str = "full"):
        """Run pipelines based on mode"""
        
        # Determine which pipelines to run
        pipelines_to_run = self.get_pipelines_for_mode(mode)
        
        # Execute in dependency order
        while pipelines_to_run:
            # Find ready tasks (dependencies satisfied)
            ready_tasks = [
                task for task in pipelines_to_run
                if all(
                    self.tasks[dep].status == "completed"
                    for dep in task.dependencies
                )
            ]
            
            if not ready_tasks:
                raise RuntimeError("Circular dependency detected")
            
            # Run ready tasks in parallel
            await self.run_parallel_tasks(ready_tasks)
            
            # Remove completed tasks
            pipelines_to_run = [
                task for task in pipelines_to_run
                if task.status != "completed"
            ]
    
    async def run_parallel_tasks(self, tasks: List[PipelineTask]):
        """Run multiple tasks in parallel"""
        futures = []
        
        for task in tasks:
            logger.info(f"Starting {task.name}")
            task.status = "running"
            
            future = self.executor.submit(
                self.run_pipeline,
                task
            )
            futures.append((task, future))
        
        # Wait for completion
        for task, future in futures:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, future.result
                )
                task.status = "completed"
                logger.info(f"Completed {task.name}")
            except Exception as e:
                task.status = "failed"
                logger.error(f"Failed {task.name}: {e}")
                raise
    
    def run_pipeline(self, task: PipelineTask):
        """Execute a single pipeline"""
        module = importlib.import_module(task.module)
        return module.main(task.config)
```

## State Management

### Pipeline State Tracking

```python
# Database schema for state tracking
CREATE TABLE orchestration.pipeline_state (
    pipeline_name VARCHAR(50) PRIMARY KEY,
    last_run_timestamp TIMESTAMP,
    last_run_status VARCHAR(20),  -- 'success', 'failed', 'partial'
    last_run_duration_seconds INTEGER,
    next_scheduled_run TIMESTAMP,
    current_status VARCHAR(20),  -- 'idle', 'running', 'scheduled'
    metadata JSONB
);

CREATE TABLE orchestration.pipeline_runs (
    run_id SERIAL PRIMARY KEY,
    pipeline_name VARCHAR(50),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    status VARCHAR(20),
    records_processed INTEGER,
    error_message TEXT,
    log_file_path TEXT,
    metadata JSONB
);
```

### State Manager

```python
class PipelineStateManager:
    """Manages pipeline execution state"""
    
    async def get_pipeline_state(self, pipeline_name: str) -> Dict:
        """Get current state of a pipeline"""
        async with get_db_session() as session:
            result = await session.execute(
                select(PipelineState).where(
                    PipelineState.pipeline_name == pipeline_name
                )
            )
            return result.scalar_one_or_none()
    
    async def update_pipeline_state(
        self,
        pipeline_name: str,
        status: str,
        metadata: Optional[Dict] = None
    ):
        """Update pipeline state"""
        async with get_db_session() as session:
            state = await self.get_pipeline_state(pipeline_name)
            
            if not state:
                state = PipelineState(pipeline_name=pipeline_name)
                session.add(state)
            
            state.last_run_timestamp = datetime.now()
            state.last_run_status = status
            state.current_status = "idle"
            
            if metadata:
                state.metadata = metadata
            
            await session.commit()
    
    async def can_run_pipeline(self, pipeline_name: str) -> bool:
        """Check if pipeline can run based on dependencies"""
        config = self.get_pipeline_config(pipeline_name)
        
        for dependency in config.get('dependencies', []):
            dep_state = await self.get_pipeline_state(dependency)
            
            if not dep_state or dep_state.last_run_status != 'success':
                return False
            
            # Check if dependency data is stale
            if self.is_data_stale(dep_state, config):
                return False
        
        return True
```

## Error Handling and Recovery

### Retry Strategy

```python
class RetryStrategy:
    """Configurable retry strategy for pipeline failures"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: int = 60,
        backoff_factor: float = 2.0,
        max_delay: int = 3600
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
    
    async def execute_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic"""
        delay = self.initial_delay
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay} seconds..."
                )
                
                await asyncio.sleep(delay)
                delay = min(delay * self.backoff_factor, self.max_delay)
```

### Recovery Procedures

```yaml
recovery_procedures:
  data_pipeline:
    failure_actions:
      - notify: ["email", "slack"]
      - retry_count: 3
      - fallback: "use_cached_data"
      - escalation: "page_oncall"
    
  scenarios_pipeline:
    failure_actions:
      - notify: ["email"]
      - retry_count: 2
      - fallback: "use_previous_scenarios"
    
  metrics_pipeline:
    failure_actions:
      - notify: ["email", "slack"]
      - retry_count: 3
      - partial_recovery: true  # Continue with completed metrics
    
  assessment_pipeline:
    failure_actions:
      - notify: ["email", "slack", "sms"]
      - retry_count: 1
      - checkpoint_recovery: true  # Resume from last checkpoint
```

## Monitoring and Alerting

### Health Checks

```python
class PipelineHealthMonitor:
    """Monitor pipeline health and performance"""
    
    async def health_check(self) -> Dict:
        """Comprehensive health check"""
        health_status = {
            'timestamp': datetime.now(),
            'pipelines': {},
            'system': {}
        }
        
        # Check each pipeline
        for pipeline_name in self.pipelines:
            health_status['pipelines'][pipeline_name] = {
                'last_run': await self.get_last_run_time(pipeline_name),
                'status': await self.get_pipeline_status(pipeline_name),
                'data_freshness': await self.check_data_freshness(pipeline_name),
                'error_rate': await self.get_error_rate(pipeline_name),
                'avg_duration': await self.get_avg_duration(pipeline_name)
            }
        
        # System checks
        health_status['system'] = {
            'database': await self.check_database_connection(),
            'disk_space': self.check_disk_space(),
            'memory': self.check_memory_usage(),
            'api_endpoints': await self.check_api_endpoints()
        }
        
        return health_status
    
    async def check_data_freshness(self, pipeline_name: str) -> str:
        """Check if pipeline data is fresh"""
        state = await self.get_pipeline_state(pipeline_name)
        
        if not state:
            return "unknown"
        
        age = datetime.now() - state.last_run_timestamp
        max_age = self.config[pipeline_name].get('max_data_age_hours', 24)
        
        if age.total_seconds() / 3600 > max_age:
            return "stale"
        return "fresh"
```

### Alerting Configuration

```yaml
alerting:
  channels:
    email:
      enabled: true
      recipients: ["ops@tetra.com"]
      smtp_server: "smtp.gmail.com"
      
    slack:
      enabled: true
      webhook_url: "${SLACK_WEBHOOK_URL}"
      channel: "#pipeline-alerts"
      
    pagerduty:
      enabled: false
      api_key: "${PAGERDUTY_API_KEY}"
      service_id: "pipeline-service"
  
  rules:
    - name: "Pipeline Failure"
      condition: "pipeline_status == 'failed'"
      severity: "high"
      channels: ["email", "slack"]
      
    - name: "Data Staleness"
      condition: "data_age_hours > 48"
      severity: "medium"
      channels: ["email"]
      
    - name: "High Error Rate"
      condition: "error_rate > 0.1"
      severity: "high"
      channels: ["email", "slack", "pagerduty"]
      
    - name: "Long Running Pipeline"
      condition: "duration_minutes > 120"
      severity: "low"
      channels: ["slack"]
```

## Performance Optimization

### Resource Management

```yaml
resource_management:
  # CPU allocation
  cpu:
    data_pipeline: 4
    scenarios_pipeline: 8
    metrics_pipeline: 16
    assessment_pipeline: 16
  
  # Memory limits
  memory:
    data_pipeline: "8GB"
    scenarios_pipeline: "16GB"
    metrics_pipeline: "32GB"
    assessment_pipeline: "32GB"
  
  # Parallel workers
  parallelism:
    data_pipeline: 4
    scenarios_pipeline: 8
    metrics_pipeline: 16
    assessment_pipeline: 16
  
  # Database connections
  db_connections:
    pool_size: 20
    max_overflow: 10
    pool_timeout: 30
```

### Caching Strategy

```python
class PipelineCache:
    """Caching for pipeline intermediate results"""
    
    def __init__(self, cache_dir: str = "/tmp/tetra_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def cache_key(self, pipeline: str, params: Dict) -> str:
        """Generate cache key"""
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{pipeline}:{params_str}".encode()).hexdigest()
    
    async def get(self, pipeline: str, params: Dict) -> Optional[Any]:
        """Get cached result"""
        key = self.cache_key(pipeline, params)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(
                cache_file.stat().st_mtime
            )
            
            # Check cache validity
            max_age = self.get_max_cache_age(pipeline)
            if age.total_seconds() < max_age:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        return None
    
    async def set(self, pipeline: str, params: Dict, result: Any):
        """Cache result"""
        key = self.cache_key(pipeline, params)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
```

## CLI Tools

### Pipeline Control Commands

```bash
# Start orchestrator
tetra-orchestrator start

# Stop orchestrator
tetra-orchestrator stop

# Check status
tetra-orchestrator status

# Run specific pipeline
tetra-orchestrator run --pipeline data_pipeline --mode daily

# Run full chain
tetra-orchestrator run --mode full

# Dry run (no execution)
tetra-orchestrator run --mode full --dry-run

# Force run (ignore dependencies)
tetra-orchestrator run --pipeline metrics_pipeline --force

# Show dependency graph
tetra-orchestrator graph

# Show schedule
tetra-orchestrator schedule

# View logs
tetra-orchestrator logs --pipeline data_pipeline --tail 100
```

## Best Practices

1. **Idempotency**: Ensure pipelines can be safely re-run
2. **Checkpointing**: Save progress for long-running pipelines
3. **Monitoring**: Track all pipeline executions and metrics
4. **Documentation**: Document dependencies and requirements
5. **Testing**: Test orchestration logic separately
6. **Versioning**: Version pipeline configurations
7. **Rollback**: Plan for rollback scenarios

## Troubleshooting

### Common Issues

1. **Dependency deadlock**
   - Check for circular dependencies
   - Review dependency graph
   - Use `--force` flag carefully

2. **Resource exhaustion**
   - Monitor memory and CPU usage
   - Adjust parallelism settings
   - Implement resource limits

3. **Schedule conflicts**
   - Review cron expressions
   - Check for overlapping schedules
   - Implement mutex locks

4. **State corruption**
   - Clear state database
   - Reset pipeline status
   - Rebuild from checkpoints

## Future Enhancements

- **Kubernetes Integration**: Deploy on K8s with Airflow/Prefect
- **Event-Driven Triggers**: React to market events in real-time
- **Dynamic Scheduling**: Adjust schedule based on market hours
- **Multi-Environment Support**: Dev/staging/production pipelines
- **Pipeline Versioning**: Blue-green deployments for pipelines
- **Cost Optimization**: Spot instances for batch processing