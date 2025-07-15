#!/bin/bash

# Script to run MATLAB tests for GSN functions
# Usage: ./run_tests.sh [test_file_name]

# Set MATLAB path (adjust if your MATLAB is in a different location)
MATLAB_PATH="/Applications/MATLAB_R2022b.app/bin/matlab"

# Check if MATLAB exists
if [ ! -f "$MATLAB_PATH" ]; then
    echo "MATLAB not found at $MATLAB_PATH"
    echo "Please update the MATLAB_PATH variable in this script"
    exit 1
fi

# Change to the tests directory
cd "$(dirname "$0")"

# Run tests
if [ $# -eq 0 ]; then
    # Run all tests
    echo "Running all tests..."
    "$MATLAB_PATH" -nosplash -nodesktop -r "try; runtests('.'); catch ME; disp('Error running tests:'); disp(ME.message); end; exit"
else
    # Run specific test
    TEST_NAME="$1"
    echo "Running test: $TEST_NAME"
    "$MATLAB_PATH" -nosplash -nodesktop -r "try; runtests('$TEST_NAME'); catch ME; disp('Error running tests:'); disp(ME.message); end; exit"
fi
