#!/bin/bash
# Run Tetra benchmark pipeline
# Scheduled to run daily at 8:00 PM

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set up logging
LOG_FILE="/tmp/tetra_benchmark_pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date)] Starting benchmark pipeline..." | tee -a "$LOG_FILE"

# Activate virtual environment
cd "$PROJECT_ROOT"
source .venv/bin/activate

# Run benchmark pipeline
echo "[$(date)] Running benchmark pipeline..." | tee -a "$LOG_FILE"
python -m src.pipelines.benchmark_pipeline.main --mode=daily 2>&1 | tee -a "$LOG_FILE"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "[$(date)] Benchmark pipeline completed successfully" | tee -a "$LOG_FILE"
    exit 0
else
    echo "[$(date)] Benchmark pipeline failed" | tee -a "$LOG_FILE"
    exit 1
fi