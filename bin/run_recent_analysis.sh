#!/bin/bash
# Run benchmark analysis on recent data (last 2 years)
# This focuses on strategies that work in current market conditions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default to last 2 years if not specified
START_DATE=${1:-$(date -v-2y +%Y-%m-%d)}
END_DATE=${2:-$(date +%Y-%m-%d)}

# Set up logging
LOG_FILE="/tmp/tetra_recent_analysis_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Tetra Recent Market Analysis (2 Years)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Start Date: $START_DATE" | tee -a "$LOG_FILE"
echo "End Date: $END_DATE" | tee -a "$LOG_FILE"
echo "Log File: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Activate virtual environment
cd "$PROJECT_ROOT"
source .venv/bin/activate

# Run analysis
echo "[$(date)] Starting recent market analysis..." | tee -a "$LOG_FILE"
echo "[$(date)] Focusing on strategies that work in current conditions" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python -m src.pipelines.benchmark_pipeline.main \
    --mode=backfill \
    --start-date="$START_DATE" \
    --end-date="$END_DATE" \
    --universe=all \
    --parallel=8 \
    2>&1 | tee -a "$LOG_FILE"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "[$(date)] Recent analysis completed successfully!" | tee -a "$LOG_FILE"
    echo "Results saved to: $LOG_FILE" | tee -a "$LOG_FILE"
    
    # Show summary of recent performers
    echo "" | tee -a "$LOG_FILE"
    echo "Top performing strategies in recent market:" | tee -a "$LOG_FILE"
    grep "Completed backtest" "$LOG_FILE" | grep -v "Return=0.00%" | sort -k8 -nr | head -10 | tee -a "$LOG_FILE"
    
    exit 0
else
    echo "[$(date)] Recent analysis failed" | tee -a "$LOG_FILE"
    exit 1
fi