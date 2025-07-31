#!/bin/bash
# Cron wrapper for daily backfill

# Set up environment
export PATH="/usr/local/bin:/usr/bin:/bin"
cd "$(dirname "$0")/.."

# Log file
LOG_DIR="$PWD/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/backfill_$(date +%Y%m%d_%H%M%S).log"

echo "Starting daily backfill at $(date)" >> "$LOG_FILE"

# Run the backfill script
# Using python3 directly, adjust if using virtual environment
python3 scripts/daily_backfill.py --scheduled --quiet >> "$LOG_FILE" 2>&1

echo "Backfill completed at $(date)" >> "$LOG_FILE"

# Keep only last 7 days of logs
find "$LOG_DIR" -name "backfill_*.log" -mtime +7 -delete
