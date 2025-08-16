#!/usr/bin/env python3
"""
Clear frontend cache and force reload with fresh data.
"""

import subprocess
import time
import json

def clear_and_restart():
    """Clear cache and restart frontend."""
    
    print("🧹 Clearing frontend cache and restarting...")
    
    # Kill frontend process
    print("Stopping frontend...")
    subprocess.run(["lsof", "-ti:3000"], capture_output=True, text=True)
    subprocess.run(["lsof", "-ti:3000", "|", "xargs", "kill", "-9"], shell=True, capture_output=True)
    time.sleep(2)
    
    # Clear any node cache
    print("Clearing node cache...")
    subprocess.run(["rm", "-rf", "/Users/angwei/Repos/tetra/frontend/node_modules/.vite"], capture_output=True)
    subprocess.run(["rm", "-rf", "/Users/angwei/Repos/tetra/frontend/dist"], capture_output=True)
    
    # Verify API is returning real data
    print("\n✅ Verifying API data...")
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:8000/api/strategies/trades"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data.get('success') and data.get('data', {}).get('trades'):
                trades = data['data']['trades']
                print(f"API returning {len(trades)} trades")
                if trades:
                    first_trade = trades[0]
                    print(f"Sample trade: {first_trade['strategy']} - {first_trade['symbol']} @ ${first_trade['current_price']:.2f}")
                    print(f"Score: {first_trade['score']:.1f}")
            else:
                print("⚠️ API not returning trade data")
        else:
            print("⚠️ API check failed")
    except Exception as e:
        print(f"⚠️ API check failed: {e}")
    
    # Restart frontend with clean build
    print("\n🚀 Starting frontend with clean build...")
    subprocess.Popen(
        ["npm", "run", "dev"],
        cwd="/Users/angwei/Repos/tetra/frontend",
        stdout=open("/tmp/tetra-frontend.log", "w"),
        stderr=subprocess.STDOUT
    )
    
    time.sleep(5)
    
    print("\n✅ Frontend restarted at http://localhost:3000/trades")
    print("\n📋 To see real data:")
    print("1. Open http://localhost:3000/trades in an incognito/private window")
    print("2. Or press Cmd+Shift+R (Mac) / Ctrl+Shift+R (PC) for hard refresh")
    print("3. Check developer console for any errors")
    
    # Show current data sample
    print("\n📊 Current database trades:")
    subprocess.run([
        "psql", 
        "postgresql://tetra_user:tetra_password@localhost:5432/tetra",
        "-c",
        "SELECT strategy_name, symbol, current_price, composite_score FROM strategies.strategy_trades ORDER BY rank LIMIT 5"
    ])

if __name__ == "__main__":
    clear_and_restart()