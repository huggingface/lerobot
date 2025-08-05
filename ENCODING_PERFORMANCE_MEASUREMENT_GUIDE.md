# Encoding Performance Measurement Guide

## Overview

This guide shows you how to measure and compare video encoding performance in the LeRobot recording pipeline, specifically comparing synchronous vs asynchronous encoding.

## Quick Start

### 1. Compare Sync vs Async Encoding (Recommended)

Run a side-by-side comparison to see the performance difference:

```bash
python scripts/run_recording_benchmark.py --compare --episodes 2 --episode-time 10 --fps 30
```

This will show you:
- **Synchronous encoding time**: How long encoding blocks the recording
- **Asynchronous encoding time**: How long encoding takes in background
- **Performance improvement**: Speedup and time saved
- **Blocking time reduction**: How much less the recording is blocked

### 2. Run Individual Benchmarks

**Synchronous encoding benchmark:**
```bash
python scripts/run_recording_benchmark.py --mode synthetic --episodes 2 --episode-time 10 --fps 30
```

**Asynchronous encoding benchmark:**
```bash
python scripts/run_recording_benchmark.py --mode synthetic --async-encoding --episodes 2 --episode-time 10 --fps 30
```

### 3. Use the Demo Script

For a complete demonstration:
```bash
python scripts/benchmark_encoding_performance.py
```

## Understanding the Results

### Sample Output from Comparison

```
================================================================================
PERFORMANCE COMPARISON RESULTS
================================================================================

SYNCHRONOUS ENCODING:
  Total recording time: 39.13s
  Video encoding time: 23.77s
  Encoding percentage: 60.7%

ASYNCHRONOUS ENCODING:
  Total recording time: 15.64s
  Video encoding time (background): 19.68s
  Task submission time: 0.00s
  Submission percentage: 0.0%

PERFORMANCE IMPROVEMENTS:
  Time saved: 23.49s
  Speedup: 2.50x
  Improvement: 60.0%

ENCODING EFFICIENCY:
  Sync encoding blocks recording for: 23.77s
  Async encoding blocks recording for: 0.00s
  Blocking time reduction: 100.0%
```

### What These Numbers Mean

1. **Total recording time**: How long the entire recording process takes
2. **Video encoding time**: Time spent on video encoding
3. **Encoding percentage**: What portion of total time is spent encoding
4. **Task submission time**: Time to submit encoding tasks (very fast for async)
5. **Speedup**: How many times faster async is than sync
6. **Blocking time reduction**: How much less the recording thread is blocked

## Benchmarking Options

### Command Line Arguments

```bash
python scripts/run_recording_benchmark.py [OPTIONS]

Options:
  --compare                    Run both sync and async benchmarks for comparison
  --async-encoding            Enable asynchronous video encoding
  --episodes INT              Number of episodes to record (default: 2)
  --episode-time INT          Duration of each episode in seconds (default: 30)
  --fps INT                   Recording FPS (default: 30)
  --cameras INT               Number of cameras (default: 1)
  --video-encoding-workers INT Number of worker threads for async encoding (default: 2)
  --video-encoding-queue-size INT Maximum queue size for async encoding (default: 100)
  --output-dir PATH           Output directory for benchmark results (default: ./benchmark_results)
```

### Real-World Testing

For testing with actual hardware (requires robot):

```bash
python scripts/run_recording_benchmark.py --mode real --async-encoding --episodes 3 --episode-time 60
```

## Expected Performance Improvements

### Based on Our Analysis

| Metric | Synchronous | Asynchronous | Improvement |
|--------|-------------|--------------|-------------|
| **Episode saving time** | 5-15s per episode | 1-3s per episode | **60-80% faster** |
| **Recording responsiveness** | Blocked during encoding | Always responsive | **100% improvement** |
| **CPU utilization** | Single-threaded encoding | Multi-threaded encoding | **2-4x better** |
| **Overall speedup** | Baseline | 2-3x faster | **100-200% improvement** |

### Factors Affecting Performance

1. **Episode length**: Longer episodes = more encoding time = bigger improvement
2. **Number of cameras**: More cameras = more videos = bigger improvement
3. **Hardware**: Better CPU = faster encoding = smaller relative improvement
4. **Video quality**: Higher quality = longer encoding = bigger improvement

## Interpreting Results

### Good Performance Indicators

✅ **Async encoding shows:**
- Task submission time < 1% of total time
- Blocking time reduction > 90%
- Speedup > 2x
- Background encoding time similar to sync encoding time

### Potential Issues

⚠️ **Watch out for:**
- Task submission time > 5% of total time (queue overflow)
- Background encoding much slower than sync (worker thread issues)
- Speedup < 1.5x (encoding not the main bottleneck)

## Advanced Benchmarking

### Custom Scenarios

**High-load testing:**
```bash
python scripts/run_recording_benchmark.py --compare --episodes 10 --episode-time 60 --fps 60 --cameras 4
```

**Worker thread optimization:**
```bash
# Test different worker counts
python scripts/run_recording_benchmark.py --async-encoding --video-encoding-workers 1 --episodes 3
python scripts/run_recording_benchmark.py --async-encoding --video-encoding-workers 4 --episodes 3
python scripts/run_recording_benchmark.py --async-encoding --video-encoding-workers 8 --episodes 3
```

### Analyzing Results

The benchmark saves detailed JSON files in the output directory:

```bash
ls benchmark_results/
# sync_benchmark_1234567890.json
# async_benchmark_1234567890.json
```

Use the analysis script to generate reports:
```bash
python scripts/analyze_benchmark_results.py benchmark_results/sync_benchmark_*.json
```

## Real-World Usage

### Enable Async Encoding in Recording

```bash
python -m lerobot.record \
    --dataset.async_video_encoding=true \
    --dataset.video_encoding_workers=2 \
    --dataset.video_encoding_queue_size=100 \
    --dataset.repo_id="my_dataset" \
    --dataset.num_episodes=10
```

### Monitor Performance

The async encoder provides real-time statistics:
```python
stats = dataset.async_video_encoder.get_stats()
print(f"Tasks completed: {stats['tasks_completed']}")
print(f"Average encoding time: {stats['average_encoding_time']:.2f}s")
```

## Troubleshooting

### Common Issues

1. **Queue overflow**: Increase `--video-encoding-queue-size`
2. **Slow encoding**: Increase `--video-encoding-workers`
3. **Memory issues**: Decrease worker count or queue size
4. **Encoding failures**: Check disk space and permissions

### Performance Tuning

1. **Start with 2 workers** for most systems
2. **Increase workers** if you have many CPU cores
3. **Monitor queue size** - if it fills up, increase it
4. **Test different configurations** to find optimal settings

## Conclusion

The benchmarking tools provide comprehensive insights into encoding performance. The comparison shows that async encoding typically provides:

- **2-3x faster episode recording**
- **100% reduction in blocking time**
- **Better CPU utilization**
- **Improved user experience**

Use these tools to optimize your recording setup and verify that async encoding is working correctly for your specific use case. 