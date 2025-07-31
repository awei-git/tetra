#!/bin/bash
# Setup cron job for daily backfill

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create a wrapper script that activates the environment and runs the backfill
cat > "$PROJECT_ROOT/scripts/cron_backfill.sh" << 'EOF'
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
EOF

chmod +x "$PROJECT_ROOT/scripts/cron_backfill.sh"

echo "Cron wrapper script created at: $PROJECT_ROOT/scripts/cron_backfill.sh"
echo ""
echo "To add to crontab, run: crontab -e"
echo "Then add one of these lines:"
echo ""
echo "# Run daily at 6 AM (after market close)"
echo "0 6 * * * $PROJECT_ROOT/scripts/cron_backfill.sh"
echo ""
echo "# Run every weekday at 6 PM (after market close)"
echo "0 18 * * 1-5 $PROJECT_ROOT/scripts/cron_backfill.sh"
echo ""
echo "# Run every 6 hours"
echo "0 */6 * * * $PROJECT_ROOT/scripts/cron_backfill.sh"
echo ""
echo "To view cron logs:"
echo "tail -f $PROJECT_ROOT/logs/backfill_*.log"