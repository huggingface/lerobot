#!/bin/bash

# Usage: ./run_detached.sh python my_script.py [args...]

timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="run_${timestamp}.log"

echo "Running: $@"
echo "Logging to: $logfile"

nohup "$@" > "$logfile" 2>&1 &

echo "Detached process started."
echo "To view log: tail -f $logfile"
