#!/bin/bash
# Launch WebGUI in browser

# Check if frontend is running
if ! pgrep -f "vite.*frontend" > /dev/null; then
    echo "Frontend is not running. Starting WebGUI..."
    # Start both backend and frontend
    "$(dirname "$0")/restart_webgui.sh"
    echo "Waiting for services to start..."
    sleep 8
fi

# Open in default browser
echo "Opening WebGUI in browser..."
open http://localhost:5173