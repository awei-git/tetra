#!/bin/bash
# Stop Tetra backend and frontend services

echo "[$(date)] Stopping Tetra services..."

# Kill backend processes
echo "Stopping backend..."
pkill -f "uvicorn backend.app.main:app" || true
pkill -f "uvicorn app.main:app" || true

# Kill frontend processes  
echo "Stopping frontend..."
pkill -f "npm.*run dev.*frontend" || true
pkill -f "next dev" || true

# Wait a moment
sleep 1

# Check if any processes are still running
REMAINING=$(ps aux | grep -E "(uvicorn|next dev)" | grep -v grep | wc -l)

if [ $REMAINING -eq 0 ]; then
    echo "[$(date)] All Tetra services stopped successfully"
else
    echo "[$(date)] Warning: $REMAINING processes may still be running"
    echo "Running processes:"
    ps aux | grep -E "(uvicorn|next dev)" | grep -v grep
fi