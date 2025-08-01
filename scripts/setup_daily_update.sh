#!/bin/bash
# Setup script for daily data pipeline updates

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create necessary directories
mkdir -p "$PROJECT_ROOT/logs/data_pipeline"

echo "Setting up daily data pipeline..."
echo "==============================="
echo ""
echo "Option 1: Using cron (simple)"
echo "Add this line to your crontab (crontab -e):"
echo "0 20 * * * cd $PROJECT_ROOT && python3 -m src.pipelines.data_pipeline.main --mode=daily >> logs/data_pipeline/daily_\$(date +\%Y\%m\%d).log 2>&1"
echo ""
echo "Option 2: Using systemd timer (recommended for servers)"
echo "See: scripts/systemd/tetra-data-pipeline.service and tetra-data-pipeline.timer"
echo ""
echo "Option 3: Using Airflow (recommended for production)"
echo "Deploy the DAG from: airflow/dags/data_pipeline_dag.py"
echo ""
echo "To test the pipeline now:"
echo "python -m src.pipelines.data_pipeline.main --mode=daily"