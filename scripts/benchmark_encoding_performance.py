#!/usr/bin/env python

"""
Simple script to demonstrate encoding performance benchmarking.

This script shows how to measure and compare synchronous vs asynchronous
video encoding performance in the LeRobot recording pipeline.
"""

import subprocess
import sys
from pathlib import Path


def run_benchmark_command(cmd_args):
    """Run a benchmark command and return the output."""
    cmd = [sys.executable, "scripts/run_recording_benchmark.py"] + cmd_args
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("Error:", result.stderr)
    
    return result.returncode == 0


def main():
    """Run encoding performance benchmarks."""
    print("=" * 80)
    print("LEROBOT ENCODING PERFORMANCE BENCHMARKING")
    print("=" * 80)
    
    print("\nThis script demonstrates how to measure video encoding performance")
    print("in the LeRobot recording pipeline.\n")
    
    # Example 1: Compare sync vs async encoding
    print("1. COMPARING SYNCHRONOUS VS ASYNCHRONOUS ENCODING")
    print("=" * 60)
    print("This will run both sync and async benchmarks and show the difference.")
    
    success = run_benchmark_command([
        "--compare",
        "--episodes", "2",
        "--episode-time", "10",
        "--fps", "30"
    ])
    
    if not success:
        print("Benchmark failed!")
        return
    
    print("\n" + "=" * 80)
    print("2. SYNCHRONOUS ENCODING BENCHMARK")
    print("=" * 60)
    print("This shows the performance with traditional blocking encoding.")
    
    run_benchmark_command([
        "--mode", "synthetic",
        "--episodes", "2",
        "--episode-time", "10",
        "--fps", "30"
    ])
    
    print("\n" + "=" * 80)
    print("3. ASYNCHRONOUS ENCODING BENCHMARK")
    print("=" * 60)
    print("This shows the performance with non-blocking async encoding.")
    
    run_benchmark_command([
        "--mode", "synthetic",
        "--async-encoding",
        "--episodes", "2",
        "--episode-time", "10",
        "--fps", "30",
        "--video-encoding-workers", "2"
    ])
    
    print("\n" + "=" * 80)
    print("BENCHMARKING COMPLETE!")
    print("=" * 80)
    
    print("\nKey takeaways:")
    print("• Synchronous encoding blocks the recording thread")
    print("• Asynchronous encoding happens in background")
    print("• Async encoding provides significant speedup")
    print("• The improvement depends on encoding time vs other operations")
    
    print("\nTo use async encoding in real recording:")
    print("python -m lerobot.record --dataset.async_video_encoding=true")


if __name__ == "__main__":
    main() 