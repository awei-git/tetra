#!/bin/bash
# Tetra pre-market morning briefing
# Runs ~6:30 AM ET weekdays — lightweight pipeline before market open
set -euo pipefail

TETRA_DIR="/Users/angwei/Library/Mobile Documents/com~apple~CloudDocs/MtJoy/Tetra"
cd "$TETRA_DIR"

# Ensure Docker DB is running
docker start tetra-db 2>/dev/null || true
sleep 2

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="$TETRA_DIR:$TETRA_DIR/src"

echo "$(date): Starting Tetra pre-market briefing"

# Run pre-market report (includes lightweight ingest + LLM commentary + Mira push)
python scripts/run_premarket_report.py --llm-provider deepseek 2>&1 || echo "WARN: premarket report had errors"

echo "$(date): Pre-market briefing complete"
