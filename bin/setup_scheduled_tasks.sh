#!/bin/bash
# Setup script for all Tetra scheduled tasks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Setting up Tetra scheduled tasks..."

# Array of plist files to install
PLISTS=(
    "com.tetra.launch-services.plist"
    "com.tetra.data-pipeline.plist"
    "com.tetra.benchmark-pipeline.plist"
)

# Unload any existing jobs
echo "Unloading existing jobs..."
for plist in "${PLISTS[@]}"; do
    launchctl unload "$HOME/Library/LaunchAgents/$plist" 2>/dev/null || true
done

# Also unload old nightly pipeline job if exists
launchctl unload "$HOME/Library/LaunchAgents/com.tetra.nightly-pipelines.plist" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/com.tetra.nightly-pipelines.plist"

# Install new jobs
echo "Installing scheduled tasks..."
for plist in "${PLISTS[@]}"; do
    echo "  Installing $plist..."
    cp "$PROJECT_ROOT/config/launchd/$plist" "$HOME/Library/LaunchAgents/"
    launchctl load "$HOME/Library/LaunchAgents/$plist"
done

# Check status
echo ""
echo "Checking job status..."
for label in "com.tetra.launch-services" "com.tetra.data-pipeline" "com.tetra.benchmark-pipeline"; do
    launchctl list | grep "$label" || echo "  $label: not loaded"
done

echo ""
echo "Setup complete!"
echo ""
echo "Schedule:"
echo "  - 5:00 AM: Launch backend and frontend services"
echo "  - 7:00 PM: Run data pipeline"
echo "  - 8:00 PM: Run benchmark pipeline"
echo ""
echo "Manual commands:"
echo "  - Launch services: launchctl start com.tetra.launch-services"
echo "  - Data pipeline:   launchctl start com.tetra.data-pipeline"
echo "  - Benchmark:       launchctl start com.tetra.benchmark-pipeline"
echo ""
echo "Log locations:"
echo "  - /tmp/tetra-launch-services.log"
echo "  - /tmp/tetra-data-pipeline.log"
echo "  - /tmp/tetra-benchmark-pipeline.log"
echo ""
echo "To uninstall all jobs:"
echo "  for job in launch-services data-pipeline benchmark-pipeline; do"
echo "    launchctl unload ~/Library/LaunchAgents/com.tetra.\$job.plist"
echo "  done"