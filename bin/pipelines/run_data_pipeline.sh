#!/bin/bash

# Data Pipeline Runner
# Downloads market data, economic indicators, and news
# Runs daily at 7:00 PM

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Create log directory if it doesn't exist
LOG_DIR="/tmp"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/tetra_data_pipeline_${TIMESTAMP}.log"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}       Tetra Data Pipeline${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Starting at: $(date)"
echo "Log file: ${LOG_FILE}"
echo ""

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

# Function to handle errors
handle_error() {
    log_message "${RED}ERROR: Pipeline failed at step: $1${NC}"
    log_message "Check log file for details: ${LOG_FILE}"
    exit 1
}

# Change to project directory
cd "${PROJECT_ROOT}"

# Check if PostgreSQL is running
log_message "Checking database connection..."
if ! docker ps | grep -q tetra-postgres; then
    log_message "${YELLOW}PostgreSQL container not running. Starting...${NC}"
    docker-compose up -d postgres >> "${LOG_FILE}" 2>&1 || handle_error "Starting PostgreSQL"
    sleep 5  # Wait for database to be ready
fi

# Set environment variables
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export TETRA_LOG_LEVEL="${TETRA_LOG_LEVEL:-INFO}"

# Determine mode based on arguments or day of week
PIPELINE_MODE="${1:-daily}"
if [ "$PIPELINE_MODE" = "auto" ]; then
    DAY_OF_WEEK=$(date +%u)
    if [ "$DAY_OF_WEEK" -eq 6 ] || [ "$DAY_OF_WEEK" -eq 7 ]; then
        PIPELINE_MODE="full"
        log_message "Weekend detected - running in FULL mode"
    else
        PIPELINE_MODE="daily"
        log_message "Weekday detected - running in DAILY mode"
    fi
else
    log_message "Running in ${PIPELINE_MODE} mode"
fi

# Run data pipeline
log_message "${GREEN}Starting Data Pipeline...${NC}"
log_message "========================================" 
log_message "Fetching market data, economic indicators, and news..."

# Run with uv
uv run python -m src.pipelines.data_pipeline.main \
    --mode "${PIPELINE_MODE}" \
    2>&1 | tee -a "${LOG_FILE}"

PIPELINE_EXIT_CODE=${PIPESTATUS[0]}

if [ $PIPELINE_EXIT_CODE -eq 0 ]; then
    log_message "${GREEN}✓ Data pipeline completed successfully${NC}"
    
    # Get summary statistics
    log_message ""
    log_message "Pipeline Summary:"
    log_message "----------------------------------------"
    
    # Extract metrics from log (if available)
    if grep -q "symbols_processed" "${LOG_FILE}"; then
        SYMBOLS_COUNT=$(grep "symbols_processed" "${LOG_FILE}" | tail -1 | grep -oE '[0-9]+' | tail -1)
        log_message "Symbols processed: ${SYMBOLS_COUNT:-0}"
    fi
    
    if grep -q "records_inserted" "${LOG_FILE}"; then
        RECORDS_COUNT=$(grep "records_inserted" "${LOG_FILE}" | tail -1 | grep -oE '[0-9]+' | tail -1)
        log_message "Records inserted: ${RECORDS_COUNT:-0}"
    fi
    
    log_message "----------------------------------------"
    
    # Clean up old log files (keep last 7 days)
    log_message "Cleaning up old log files..."
    find "${LOG_DIR}" -name "tetra_data_pipeline_*.log" -mtime +7 -delete 2>/dev/null || true
    
else
    handle_error "Data pipeline execution"
fi

log_message ""
log_message "${GREEN}========================================${NC}"
log_message "${GREEN}Data Pipeline Complete${NC}"
log_message "Ended at: $(date)"
log_message "${GREEN}========================================${NC}"

exit 0