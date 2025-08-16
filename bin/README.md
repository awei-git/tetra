# Bin Directory - Operational Scripts

This directory contains production operational scripts for the Tetra platform, organized by function.

## Directory Structure

```
bin/
├── pipelines/          # Pipeline execution scripts
│   ├── data.sh        # Run data ingestion pipeline
│   ├── ml.sh          # Run ML training pipeline
│   └── benchmark.sh   # Run benchmark analysis pipeline
├── services/          # Service management scripts
│   ├── launch_services.sh  # Start backend and frontend
│   └── stop_services.sh    # Stop all services
├── database/          # Database management scripts
│   └── create_migration.sh # Create Alembic migrations
└── setup_scheduled_tasks.sh # Configure launchd jobs
```

## Scripts

### Pipeline Scripts (`pipelines/`)

#### `data.sh`
Runs the data ingestion pipeline to fetch market data, economic indicators, events, and news.
```bash
./bin/pipelines/data.sh              # Daily update (default)
./bin/pipelines/data.sh backfill     # Historical backfill
```

#### `ml.sh`
Runs the ML training pipeline to train trading models.
```bash
./bin/pipelines/ml.sh                # Comprehensive training (default)
./bin/pipelines/ml.sh quick          # Quick training
./bin/pipelines/ml.sh test           # Test mode with minimal data
```

#### `benchmark.sh`
Runs benchmark analysis to evaluate trading strategies.
```bash
./bin/pipelines/benchmark.sh         # Daily analysis (default)
./bin/pipelines/benchmark.sh daily   # Daily analysis (2020-present)
./bin/pipelines/benchmark.sh weekly  # Weekly full analysis (2015-present)
```

### Service Scripts (`services/`)

#### `launch_services.sh`
Starts the backend API and frontend UI services.
```bash
./bin/services/launch_services.sh
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

#### `stop_services.sh`
Stops all running Tetra services.
```bash
./bin/services/stop_services.sh
```

### Database Scripts (`database/`)

#### `create_migration.sh`
Helper script to create Alembic database migrations.
```bash
./bin/database/create_migration.sh "migration_description"
```

### Setup Script

#### `setup_scheduled_tasks.sh`
Installs all launchd scheduled tasks for automated pipeline execution.
```bash
./bin/setup_scheduled_tasks.sh
```

## Schedule

The system runs on the following automated schedule:

| Time     | Day | Task                          | Script                    |
|----------|-----|-------------------------------|---------------------------|
| 5:00 AM  | Daily | Launch services             | `services/launch_services.sh` |
| 5:00 AM  | Saturday | Train ML models          | `pipelines/ml.sh`         |
| 7:00 PM  | Daily | Update market data          | `pipelines/data.sh`       |
| 8:00 PM  | Daily | Run daily benchmark         | `pipelines/benchmark.sh daily` |
| 8:00 AM  | Saturday | Run weekly benchmark     | `pipelines/benchmark.sh weekly` |

## Manual Execution

### Quick Commands
```bash
# Start services
./bin/services/launch_services.sh

# Update data
./bin/pipelines/data.sh

# Train ML models (quick mode)
./bin/pipelines/ml.sh quick

# Run benchmark analysis
./bin/pipelines/benchmark.sh
```

### Via launchctl
```bash
launchctl start com.tetra.launch-services
launchctl start com.tetra.data-pipeline
launchctl start com.tetra.ml-training
launchctl start com.tetra.benchmark-pipeline-daily
launchctl start com.tetra.benchmark-pipeline-weekly
```

## Logs

### Service Logs
- `/tmp/tetra-backend.log` - Backend service output
- `/tmp/tetra-frontend.log` - Frontend service output

### Pipeline Logs
- `/tmp/tetra-data-pipeline-*.log` - Data pipeline execution
- `/tmp/tetra-ml-pipeline-*.log` - ML training execution
- `/tmp/tetra-benchmark-*.log` - Benchmark analysis execution

### Scheduled Task Logs
- `/tmp/tetra-launch-services.log` - Service launch logs
- `/tmp/tetra-data-pipeline.log` - Data pipeline scheduler
- `/tmp/tetra-ml-training-stdout.log` - ML training scheduler
- `/tmp/tetra-benchmark-daily.log` - Daily benchmark scheduler
- `/tmp/tetra-benchmark-weekly.log` - Weekly benchmark scheduler

## Uninstall Scheduled Tasks

To remove all scheduled tasks:
```bash
for job in launch-services data-pipeline ml-training benchmark-pipeline-daily benchmark-pipeline-weekly; do
  launchctl unload ~/Library/LaunchAgents/com.tetra.$job.plist
  rm -f ~/Library/LaunchAgents/com.tetra.$job.plist
done
```

## Best Practices

1. **Always check logs** after running pipelines manually
2. **Use test mode** for ML pipeline when testing changes
3. **Monitor disk space** - ML models and data can be large
4. **Run services check** after system restart
5. **Update scheduled tasks** after changing script paths