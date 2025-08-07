#!/bin/bash
# Run Tetra data pipeline
# Scheduled to run daily at 7:00 PM

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set up logging
LOG_FILE="/tmp/tetra_data_pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date)] Starting data pipeline..." | tee -a "$LOG_FILE"

# Activate virtual environment
cd "$PROJECT_ROOT"
source .venv/bin/activate

# Run data pipeline
echo "[$(date)] Running data ingestion pipeline..." | tee -a "$LOG_FILE"
python -m src.pipelines.data_pipeline.main --mode=daily 2>&1 | tee -a "$LOG_FILE"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "[$(date)] Data pipeline completed successfully" | tee -a "$LOG_FILE"
    exit 0
else
    echo "[$(date)] Data pipeline failed" | tee -a "$LOG_FILE"
    exit 1
fi