#!/bin/bash

echo "=== Monitoring Daily Update Test Runs ==="
echo "Current time: $(date)"
echo ""

# Check each test log
for time in "750pm" "8pm" "810pm"; do
    out_log="/Users/angwei/Repos/tetra/logs/test_${time}_out.log"
    err_log="/Users/angwei/Repos/tetra/logs/test_${time}_err.log"
    
    echo "--- Test run at $time ---"
    
    if [ -f "$out_log" ]; then
        echo "Output log exists. Last 10 lines:"
        tail -10 "$out_log" | grep -E "(Starting daily|Target date|Ingested|complete|failed)"
    else
        echo "Output log not found yet"
    fi
    
    if [ -f "$err_log" ] && [ -s "$err_log" ]; then
        echo "ERROR LOG:"
        tail -10 "$err_log"
    fi
    
    echo ""
done

# Also check the main pipeline log
echo "--- Main pipeline log (last run) ---"
tail -20 /Users/angwei/Repos/tetra/logs/launchd_pipeline_out.log | grep -E "(Starting daily|Target date|failed|complete)"