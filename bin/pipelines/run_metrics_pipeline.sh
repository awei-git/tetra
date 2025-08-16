#!/bin/bash
#
# Run the Metrics Pipeline - Stage 3 of Tetra data processing
# This calculates all technical indicators and statistical metrics for each scenario
#

set -e  # Exit on error

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Change to project root
cd "$PROJECT_ROOT"

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export TETRA_LOG_LEVEL="${TETRA_LOG_LEVEL:-INFO}"

# Create log directory
LOG_DIR="/tmp"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/tetra_metrics_pipeline_${TIMESTAMP}.log"

echo "=========================================="
echo "METRICS PIPELINE - STAGE 3"
echo "=========================================="
echo "Start time: $(date)"
echo "Project root: $PROJECT_ROOT"
echo "Log file: $LOG_FILE"
echo "Log level: $TETRA_LOG_LEVEL"
echo ""

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first."
    echo "Run: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Install/update dependencies if needed
echo "Checking dependencies..."
uv sync --all-extras

echo ""
echo "Starting Metrics Pipeline..."
echo "Calculating indicators for all 131 scenarios..."
echo ""

# Run the metrics pipeline
uv run python -m src.pipelines.metrics_pipeline.main \
    --parallel 8 \
    --log-level "${TETRA_LOG_LEVEL}" \
    2>&1 | tee "$LOG_FILE"

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "METRICS PIPELINE COMPLETED SUCCESSFULLY"
    
    # Show storage usage
    if [ -d "data/metrics" ]; then
        echo ""
        echo "Storage usage:"
        du -sh data/metrics/
        echo "Number of metric files:"
        ls -1 data/metrics/*.parquet 2>/dev/null | wc -l
    fi
else
    echo "METRICS PIPELINE FAILED"
    echo "Check log file: $LOG_FILE"
fi
echo "=========================================="
echo "End time: $(date)"
echo ""

exit $EXIT_CODE