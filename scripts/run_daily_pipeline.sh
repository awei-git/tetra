#!/bin/bash
# Daily data pipeline script
# Can be run directly or installed as a cron job

set -e  # Exit on error

# Handle setup mode
if [ "$1" = "--setup" ]; then
    SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    
    echo "Setting up daily data pipeline..."
    echo "==============================="
    echo ""
    
    # Make sure script is executable
    chmod +x "$SCRIPT_PATH"
    
    # The cron job to add (5 AM daily)
    CRON_JOB="0 5 * * * $SCRIPT_PATH"
    
    # Check if cron job already exists
    if crontab -l 2>/dev/null | grep -q "run_daily_pipeline.sh"; then
        echo "✓ Cron job already exists:"
        crontab -l | grep "run_daily_pipeline.sh"
    else
        # Add the cron job
        (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
        echo "✓ Added cron job to run at 5 AM daily:"
        echo "  $CRON_JOB"
    fi
    
    echo ""
    echo "✓ Daily pipeline setup complete!"
    echo ""
    echo "The pipeline will run automatically at 5 AM every day."
    echo "Logs will be saved to: $(dirname "$SCRIPT_PATH")/../logs/data_pipeline/"
    echo ""
    echo "To test the pipeline now:"
    echo "  $SCRIPT_PATH"
    echo ""
    echo "To view your cron jobs:"
    echo "  crontab -l"
    echo ""
    echo "To remove the cron job:"
    echo "  crontab -e  # then delete the line with run_daily_pipeline.sh"
    exit 0
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -d "$PROJECT_ROOT/../dev" ]; then
    source "$PROJECT_ROOT/../dev/bin/activate"
elif [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/logs/data_pipeline"

# Set log file with date
LOG_FILE="$PROJECT_ROOT/logs/data_pipeline/daily_$(date +%Y%m%d_%H%M%S).log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Start pipeline
log "========================================="
log "Starting daily data pipeline"
log "Python: $(which python)"
log "Working directory: $(pwd)"
log "========================================="

# Run the daily pipeline
python -m src.pipelines.data_pipeline.main --mode=daily 2>&1 | tee -a "$LOG_FILE"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log "✅ Daily pipeline completed successfully"
else
    log "❌ Daily pipeline failed with exit code: ${PIPESTATUS[0]}"
    # You could add email notification here
fi

# Run a quick backfill for the last 2 days to catch any gaps
log "Running 2-day backfill to ensure no gaps..."
python -m src.pipelines.data_pipeline.main --mode=backfill --days=2 2>&1 | tee -a "$LOG_FILE"

log "========================================="
log "All pipeline tasks completed"
log "========================================="

# Cleanup old logs (keep last 30 days)
find "$PROJECT_ROOT/logs/data_pipeline" -name "daily_*.log" -mtime +30 -delete

# Optional: Send summary email or notification
# python -m src.utils.notifications --pipeline-complete --log="$LOG_FILE"