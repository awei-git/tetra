# Bin Directory - Operational Scripts

This directory contains production operational scripts for the Tetra platform.

## Scripts

### Service Management
- `launch_services.sh` - Launch backend and frontend services (5:00 AM daily)
- `setup_scheduled_tasks.sh` - Setup all launchd scheduled tasks

### Pipeline Management
- `run_data_pipeline.sh` - Run data ingestion pipeline (7:00 PM daily)
- `run_benchmark_pipeline.sh` - Run benchmark pipeline (8:00 PM daily)

### Database Management
- `create_migration.sh` - Helper script to create Alembic migrations

## Schedule

The system runs on the following schedule:

| Time     | Task                          | Script                    |
|----------|-------------------------------|---------------------------|
| 5:00 AM  | Launch services               | `launch_services.sh`      |
| 7:00 PM  | Update market data            | `run_data_pipeline.sh`    |
| 8:00 PM  | Run benchmark tests           | `run_benchmark_pipeline.sh`|

## Setup

To install all scheduled tasks:
```bash
./bin/setup_scheduled_tasks.sh
```

## Manual Execution

Run services manually:
```bash
# Launch backend and frontend
./bin/launch_services.sh

# Run data pipeline
./bin/run_data_pipeline.sh

# Run benchmark pipeline
./bin/run_benchmark_pipeline.sh
```

Trigger via launchd:
```bash
launchctl start com.tetra.launch-services
launchctl start com.tetra.data-pipeline
launchctl start com.tetra.benchmark-pipeline
```

## Logs

Service logs:
- `/tmp/tetra-backend.log` - Backend service output
- `/tmp/tetra-frontend.log` - Frontend service output

Pipeline logs:
- `/tmp/tetra_data_pipeline_YYYYMMDD_HHMMSS.log` - Data pipeline logs
- `/tmp/tetra_benchmark_pipeline_YYYYMMDD_HHMMSS.log` - Benchmark pipeline logs

Launchd logs:
- `/tmp/tetra-launch-services.log` - Service launch logs
- `/tmp/tetra-data-pipeline.log` - Data pipeline scheduler logs
- `/tmp/tetra-benchmark-pipeline.log` - Benchmark pipeline scheduler logs

## Uninstall

To remove all scheduled tasks:
```bash
for job in launch-services data-pipeline benchmark-pipeline; do
  launchctl unload ~/Library/LaunchAgents/com.tetra.$job.plist
done
```