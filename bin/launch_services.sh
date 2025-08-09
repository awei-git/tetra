#!/bin/bash
# Launch Tetra backend and frontend services
# Can be run manually or scheduled

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "[$(date)] Starting Tetra services..."

# Function to check if process is running
is_running() {
    pgrep -f "$1" > /dev/null 2>&1
}

# Kill existing processes
echo "Stopping any existing services..."
pkill -f "uvicorn backend.app.main:app" || true
pkill -f "npm.*run dev.*frontend" || true
pkill -f "next dev" || true

# Wait for processes to stop
sleep 2

# Start backend
echo "Starting backend service..."
cd "$PROJECT_ROOT/backend"

# Check if venv exists in backend directory, otherwise use project root venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "Error: No virtual environment found"
    echo "Checked:"
    echo "  - $PWD/venv/bin/activate"
    echo "  - $PROJECT_ROOT/.venv/bin/activate"
    echo "  - $PROJECT_ROOT/venv/bin/activate"
    exit 1
fi

nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/tetra-backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Backend started successfully"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Backend failed to start. Check logs at /tmp/tetra-backend.log"
        tail -20 /tmp/tetra-backend.log
        exit 1
    fi
    sleep 1
done

# Start frontend
echo "Starting frontend service..."
cd "$PROJECT_ROOT/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

nohup npm run dev > /tmp/tetra-frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to start
echo "Waiting for frontend to start..."
for i in {1..30}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "Frontend started successfully"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Frontend failed to start. Check logs at /tmp/tetra-frontend.log"
        tail -20 /tmp/tetra-frontend.log
        exit 1
    fi
    sleep 1
done

echo ""
echo "[$(date)] Tetra services launched successfully!"
echo "================================================"
echo "Backend:  http://localhost:8000 (PID: $BACKEND_PID)"
echo "Frontend: http://localhost:3000 (PID: $FRONTEND_PID)"
echo ""
echo "Logs:"
echo "  Backend:  /tmp/tetra-backend.log"
echo "  Frontend: /tmp/tetra-frontend.log"
echo ""
echo "To stop services, run: pkill -f 'uvicorn|next dev'"
echo "================================================"