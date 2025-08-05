#!/bin/bash

echo "Cleaning up test launchd jobs..."

# Unload and remove test jobs
for job in "750pm" "8pm" "810pm"; do
    plist="$HOME/Library/LaunchAgents/com.tetra.daily-test-${job}.plist"
    if [ -f "$plist" ]; then
        echo "Removing test job: $job"
        launchctl unload "$plist" 2>/dev/null
        rm "$plist"
    fi
done

# Clean up test logs
rm -f /Users/angwei/Repos/tetra/logs/test_*pm_*.log

echo "Cleanup complete!"
echo ""
echo "Remaining launchd jobs:"
launchctl list | grep tetra