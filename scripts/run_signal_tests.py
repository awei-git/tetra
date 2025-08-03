#!/usr/bin/env python3
"""Run signal module tests."""

import subprocess
import sys

def run_tests():
    """Run signal tests with pytest."""
    print("Running signal module tests...")
    print("=" * 60)
    
    # Run tests
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_signals/",
        "-v",  # Verbose
        "--tb=short",  # Short traceback
        "-x",  # Stop on first failure
    ]
    
    result = subprocess.run(cmd, cwd="/Users/angwei/Repos/tetra")
    
    return result.returncode

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)