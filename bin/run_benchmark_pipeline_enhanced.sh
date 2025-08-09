#!/bin/bash
# Enhanced Tetra benchmark pipeline with full analysis support
# Runs daily quick analysis and weekly full analysis

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Determine mode based on day of week (0 = Sunday)
DAY_OF_WEEK=$(date +%w)
if [ "$DAY_OF_WEEK" -eq 0 ]; then
    # Sunday - run full analysis
    MODE="full"
    START_DATE=$(date -v-5y +%Y-%m-%d)  # 5 years ago
    END_DATE=$(date +%Y-%m-%d)          # Today
    LOG_PREFIX="full_analysis"
else
    # Weekday - run daily analysis
    MODE="daily"
    LOG_PREFIX="daily"
fi

# Set up logging
LOG_FILE="/tmp/tetra_benchmark_pipeline_${LOG_PREFIX}_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date)] Starting benchmark pipeline in $MODE mode..." | tee -a "$LOG_FILE"

# Activate virtual environment
cd "$PROJECT_ROOT"
source .venv/bin/activate

# Run benchmark pipeline
if [ "$MODE" = "full" ]; then
    echo "[$(date)] Running FULL benchmark analysis from $START_DATE to $END_DATE..." | tee -a "$LOG_FILE"
    echo "[$(date)] This will take several hours to complete..." | tee -a "$LOG_FILE"
    
    # Run with backfill mode for full analysis
    python -m src.pipelines.benchmark_pipeline.main \
        --mode=backfill \
        --start-date="$START_DATE" \
        --end-date="$END_DATE" \
        --universe=all \
        --parallel=8 \
        2>&1 | tee -a "$LOG_FILE"
else
    echo "[$(date)] Running daily benchmark analysis..." | tee -a "$LOG_FILE"
    
    # Run with daily mode for quick analysis
    python -m src.pipelines.benchmark_pipeline.main \
        --mode=daily \
        2>&1 | tee -a "$LOG_FILE"
fi

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "[$(date)] Benchmark pipeline completed successfully" | tee -a "$LOG_FILE"
    
    # Send notification for full analysis completion
    if [ "$MODE" = "full" ]; then
        echo "[$(date)] Full analysis complete! Check results at: $LOG_FILE" | tee -a "$LOG_FILE"
    fi
    
    exit 0
else
    echo "[$(date)] Benchmark pipeline failed" | tee -a "$LOG_FILE"
    exit 1
fi