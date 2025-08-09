#!/bin/bash
# Run full benchmark analysis manually
# This analyzes all strategies across all symbols for the specified period

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default to last year if not specified
START_DATE=${1:-$(date -v-1y +%Y-%m-%d)}
END_DATE=${2:-$(date +%Y-%m-%d)}

# Set up logging
LOG_FILE="/tmp/tetra_full_analysis_manual_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Tetra Full Benchmark Analysis" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Start Date: $START_DATE" | tee -a "$LOG_FILE"
echo "End Date: $END_DATE" | tee -a "$LOG_FILE"
echo "Log File: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Activate virtual environment
cd "$PROJECT_ROOT"
source .venv/bin/activate

# Run full analysis
echo "[$(date)] Starting full benchmark analysis..." | tee -a "$LOG_FILE"
echo "[$(date)] This will analyze all strategies across all symbols" | tee -a "$LOG_FILE"
echo "[$(date)] Expected duration: 2-4 hours" | tee -a "$LOG_FILE"
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
    echo "[$(date)] Full analysis completed successfully!" | tee -a "$LOG_FILE"
    echo "Results saved to: $LOG_FILE" | tee -a "$LOG_FILE"
    
    # Show summary
    echo "" | tee -a "$LOG_FILE"
    echo "Top performing strategies:" | tee -a "$LOG_FILE"
    grep "Completed backtest" "$LOG_FILE" | grep -v "Return=0.00%" | sort -k8 -nr | head -10 | tee -a "$LOG_FILE"
    
    exit 0
else
    echo "[$(date)] Full analysis failed" | tee -a "$LOG_FILE"
    exit 1
fi