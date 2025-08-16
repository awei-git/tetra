#!/bin/bash

# Scenarios Pipeline Runner
# Generates market scenarios for strategy testing
# Runs daily at 7:30 PM (30 minutes after data pipeline)

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
LOG_FILE="${LOG_DIR}/tetra_scenarios_pipeline_${TIMESTAMP}.log"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}     Tetra Scenarios Pipeline${NC}"
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

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    log_message "${YELLOW}Virtual environment not found. Creating...${NC}"
    uv venv || handle_error "Creating virtual environment"
    log_message "${GREEN}Virtual environment created${NC}"
fi

# Activate virtual environment
log_message "Activating virtual environment..."
source .venv/bin/activate || handle_error "Activating virtual environment"

# Install/update dependencies if needed
log_message "Checking dependencies..."
uv sync --all-extras >> "${LOG_FILE}" 2>&1 || handle_error "Installing dependencies"

# Check if PostgreSQL is running
log_message "Checking database connection..."
if ! docker ps | grep -q tetra-postgres; then
    log_message "${YELLOW}PostgreSQL container not running. Starting...${NC}"
    docker-compose up -d postgres >> "${LOG_FILE}" 2>&1 || handle_error "Starting PostgreSQL"
    sleep 5  # Wait for database to be ready
fi

# Run scenarios pipeline
log_message "${GREEN}Starting Scenarios Pipeline...${NC}"
log_message "========================================" 

# Set environment variables
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export TETRA_LOG_LEVEL="${TETRA_LOG_LEVEL:-INFO}"

# Run the pipeline
log_message "Executing scenarios generation..."
log_message "Generating all scenario types with 100 stochastic scenarios..."

# Run with uv
uv run python -m src.pipelines.scenarios_pipeline.main \
    --type all \
    --include-full-cycles \
    --num-scenarios 100 \
    --severity severe \
    --save-timeseries \
    --parallel \
    >> "${LOG_FILE}" 2>&1

PIPELINE_EXIT_CODE=$?

if [ $PIPELINE_EXIT_CODE -eq 0 ]; then
    log_message "${GREEN}✓ Scenarios pipeline completed successfully${NC}"
    
    # Get summary statistics
    log_message ""
    log_message "Pipeline Summary:"
    log_message "----------------------------------------"
    
    # Extract metrics from log (if available)
    if grep -q "scenarios_generated" "${LOG_FILE}"; then
        SCENARIOS_COUNT=$(grep "scenarios_generated" "${LOG_FILE}" | tail -1 | grep -oE '[0-9]+' | tail -1)
        log_message "Total scenarios generated: ${SCENARIOS_COUNT:-0}"
    fi
    
    if grep -q "historical_scenarios" "${LOG_FILE}"; then
        HISTORICAL_COUNT=$(grep "historical_scenarios" "${LOG_FILE}" | tail -1 | grep -oE '[0-9]+' | tail -1)
        log_message "Historical scenarios: ${HISTORICAL_COUNT:-0}"
    fi
    
    if grep -q "stress_scenarios" "${LOG_FILE}"; then
        STRESS_COUNT=$(grep "stress_scenarios" "${LOG_FILE}" | tail -1 | grep -oE '[0-9]+' | tail -1)
        log_message "Stress test scenarios: ${STRESS_COUNT:-0}"
    fi
    
    if grep -q "stochastic_scenarios" "${LOG_FILE}"; then
        STOCHASTIC_COUNT=$(grep "stochastic_scenarios" "${LOG_FILE}" | tail -1 | grep -oE '[0-9]+' | tail -1)
        log_message "Stochastic scenarios: ${STOCHASTIC_COUNT:-0}"
    fi
    
    log_message "----------------------------------------"
    
    # Optional: Clean up old log files (keep last 7 days)
    log_message "Cleaning up old log files..."
    find "${LOG_DIR}" -name "tetra_scenarios_pipeline_*.log" -mtime +7 -delete 2>/dev/null || true
    
else
    handle_error "Scenarios pipeline execution"
fi

# Deactivate virtual environment
deactivate

log_message ""
log_message "${GREEN}========================================${NC}"
log_message "${GREEN}Scenarios Pipeline Complete${NC}"
log_message "Ended at: $(date)"
log_message "${GREEN}========================================${NC}"

# Optional: Send notification (uncomment and configure as needed)
# osascript -e 'display notification "Scenarios pipeline completed" with title "Tetra"' 2>/dev/null || true

exit 0