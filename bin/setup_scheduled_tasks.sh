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
    "com.tetra.scenarios-pipeline.plist"
    "com.tetra.metrics-pipeline.plist"
    "com.tetra.assessment-pipeline.plist"
)

# Unload any existing jobs
echo "Unloading existing jobs..."
for plist in "${PLISTS[@]}"; do
    launchctl unload "$HOME/Library/LaunchAgents/$plist" 2>/dev/null || true
done

# Also unload old/deprecated jobs that may exist
OLD_JOBS=(
    "com.tetra.nightly-pipelines.plist"
    "com.tetra.ml-training.plist"
    "com.tetra.ml-pipeline.plist"
    "com.tetra.benchmark-pipeline.plist"
    "com.tetra.benchmark-pipeline-daily.plist"
    "com.tetra.benchmark-pipeline-weekly.plist"
)
for old_job in "${OLD_JOBS[@]}"; do
    launchctl unload "$HOME/Library/LaunchAgents/$old_job" 2>/dev/null || true
    rm -f "$HOME/Library/LaunchAgents/$old_job"
done

# Install new jobs
echo "Installing scheduled tasks..."
for plist in "${PLISTS[@]}"; do
    if [ -f "$PROJECT_ROOT/config/launchd/$plist" ]; then
        echo "  Installing $plist..."
        cp "$PROJECT_ROOT/config/launchd/$plist" "$HOME/Library/LaunchAgents/"
        launchctl load "$HOME/Library/LaunchAgents/$plist"
    else
        echo "  Warning: $plist not found in config/launchd/"
    fi
done

# Check status
echo ""
echo "Checking job status..."
for label in "com.tetra.launch-services" "com.tetra.data-pipeline" "com.tetra.scenarios-pipeline" "com.tetra.metrics-pipeline" "com.tetra.assessment-pipeline"; do
    if launchctl list | grep -q "$label"; then
        echo "  ✓ $label: loaded"
    else
        echo "  ✗ $label: not loaded"
    fi
done

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "SCHEDULE:"
echo "  5:00 AM  - Launch backend and frontend services"
echo "  7:00 PM  - Run data pipeline (market data, economic indicators, news)"
echo "  7:30 PM  - Run scenarios pipeline (generate 131 market scenarios)"
echo "  8:00 PM  - Run metrics pipeline (calculate indicators for all scenarios)"
echo "  9:00 PM  - Run assessment pipeline (backtest all strategies)"
echo ""
echo "MANUAL COMMANDS:"
echo "  Start jobs:"
echo "    launchctl start com.tetra.launch-services"
echo "    launchctl start com.tetra.data-pipeline"
echo "    launchctl start com.tetra.scenarios-pipeline"
echo "    launchctl start com.tetra.metrics-pipeline"
echo "    launchctl start com.tetra.assessment-pipeline"
echo ""
echo "  Run scripts directly:"
echo "    ./bin/launch_services.sh"
echo "    ./bin/pipelines/run_data_pipeline.sh"
echo "    ./bin/pipelines/run_scenarios_pipeline.sh"
echo "    ./bin/pipelines/run_metrics_pipeline.sh"
echo "    ./bin/pipelines/run_assessment_pipeline.sh"
echo ""
echo "LOG LOCATIONS:"
echo "  Services:"
echo "    /tmp/tetra-backend.log"
echo "    /tmp/tetra-frontend.log"
echo ""
echo "  Pipelines:"
echo "    /tmp/tetra_data_pipeline_*.log"
echo "    /tmp/tetra_scenarios_pipeline_*.log"
echo "    /tmp/tetra_metrics_pipeline_*.log"
echo "    /tmp/tetra_assessment_pipeline_*.log"
echo ""
echo "  LaunchD output:"
echo "    /tmp/tetra-launch-services.out"
echo "    /tmp/tetra-data-pipeline.out"
echo "    /tmp/tetra-scenarios-pipeline.out"
echo "    /tmp/tetra-metrics-pipeline.out"
echo ""
echo "TO UNINSTALL ALL JOBS:"
echo "  for job in launch-services data-pipeline scenarios-pipeline metrics-pipeline; do"
echo "    launchctl unload ~/Library/LaunchAgents/com.tetra.\$job.plist 2>/dev/null"
echo "    rm -f ~/Library/LaunchAgents/com.tetra.\$job.plist"
echo "  done"
echo ""
echo "TO VIEW JOB DETAILS:"
echo "  launchctl list | grep com.tetra"
echo ""
echo "TO CHECK NEXT RUN TIME:"
echo "  for job in launch-services data-pipeline scenarios-pipeline metrics-pipeline; do"
echo "    echo -n \"com.tetra.\$job: \""
echo "    launchctl print gui/\$(id -u)/com.tetra.\$job 2>/dev/null | grep -A1 'next run time' | tail -1 || echo 'not scheduled'"
echo "  done"