#!/bin/bash
# Setup script to add WebGUI restart to crontab

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
RESTART_SCRIPT="$SCRIPT_DIR/restart_webgui.sh"

echo "Setting up WebGUI daily restart at 6 AM..."
echo "========================================="

# The cron job to add (6 AM daily)
CRON_JOB="0 6 * * * $RESTART_SCRIPT"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "restart_webgui.sh"; then
    echo "✓ WebGUI restart cron job already exists:"
    crontab -l | grep "restart_webgui.sh"
else
    # Add the cron job
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    echo "✓ Added cron job to restart WebGUI at 6 AM daily:"
    echo "  $CRON_JOB"
fi

echo ""
echo "To view your cron jobs:"
echo "  crontab -l"
echo ""
echo "To remove the cron job:"
echo "  crontab -e  # then delete the line with restart_webgui.sh"