# Camera System Optimization Tracker

## Current Performance Baseline

- **3 Cameras (2 RealSense + Kinect)**: ~18-20ms parallel read time
- **Hardware Capture Time**: ~30ms (30fps cameras = 33.3ms theoretical limit)
- **CPU Usage**: 200-300% (multi-core usage)
- **Depth Processing Overhead**: <0.5ms with LUT optimization (previously 5ms)

## Key Insights

1. **30ms capture time is NORMAL** - This is the hardware frame interval for 30fps cameras (1000ms/30fps = 33.3ms)
2. **Parallel reading is working** - We're reading all cameras in ~20ms instead of 3×30ms = 90ms sequential
3. **CPU usage is high** - 200-300% indicates heavy multi-threading overhead

## Optimization Approaches to Test

### ✅ Completed Optimizations

- [x] **LUT-based Depth Colorization** - Reduced from 5ms to <0.5ms
- [x] **Parallel Camera Reading** - Reduced from 45ms to 20ms for 3 cameras
- [x] **Improved Logging** - Now shows HW wait vs SW overhead separately

### 🔄 In Progress

- [ ] **Fix Logging to Show True Metrics**
  - Separate hardware frame wait time from software processing
  - Show per-camera bottlenecks clearly
  - Track depth processing overhead separately

### 📋 Threading & Concurrency Optimizations

#### 1. **Persistent Thread Pool** ⭐ High Priority

- [ ] Keep ThreadPoolExecutor alive between reads
- [ ] Pre-allocate threads at startup
- **Expected Impact**: Save 2-5ms per read cycle
- **Implementation**: Already partially done, needs testing

#### 2. **Increase Thread Pool Size**

- [ ] Test with 2x camera count threads (6 threads for 3 cameras)
- [ ] Test with fixed 8-16 threads regardless of camera count
- **Expected Impact**: May reduce wait time by 1-3ms
- **Risk**: May increase CPU usage

#### 3. **Async/Await with asyncio**

- [ ] Replace ThreadPoolExecutor with asyncio event loop
- [ ] Use asyncio.gather() for concurrent reads
- **Expected Impact**: Lower overhead, save 2-4ms
- **Complexity**: High - requires refactoring camera classes

#### 4. **Thread Pinning & Priority**

- [ ] Pin camera threads to specific CPU cores
- [ ] Set thread priority to REALTIME
- **Expected Impact**: More consistent timing, reduce jitter
- **Platform**: Windows-specific implementation needed

### 🎯 Camera-Specific Optimizations

#### 5. **Frame Buffering Strategy**

- [ ] Implement double/triple buffering in camera classes
- [ ] Pre-allocate numpy arrays for zero-copy transfers
- **Expected Impact**: Save 1-2ms per frame
- **Memory Cost**: 3x frame memory usage

#### 6. **Skip Frame Processing**

- [ ] Add option to get raw frames without colorization
- [ ] Defer depth colorization to display thread
- **Expected Impact**: Save processing time for non-display uses
- **Trade-off**: Moves processing to different thread

#### 7. **Camera Read Prioritization**

- [ ] Read fastest cameras first
- [ ] Start slowest camera reads earlier
- **Expected Impact**: Better parallelization, save 2-3ms
- **Complexity**: Requires profiling each camera

### 🚀 System-Level Optimizations

#### 8. **Process Affinity**

- [ ] Set process CPU affinity to high-performance cores
- [ ] Disable CPU frequency scaling during operation
- **Expected Impact**: More consistent performance
- **Platform**: Requires admin privileges

#### 9. **Memory Pool Allocation**

- [ ] Pre-allocate memory pools for frame data
- [ ] Use memory-mapped buffers for zero-copy
- **Expected Impact**: Reduce allocation overhead, save 1-2ms
- **Complexity**: Medium

#### 10. **Native Extensions**

- [ ] Implement critical paths in Cython
- [ ] Use numba JIT for hot loops
- **Expected Impact**: 10-20% overall speedup
- **Complexity**: High

### 📊 Profiling & Analysis Tools

#### 11. **Detailed Profiling**

- [ ] Use cProfile to identify bottlenecks
- [ ] Add perf counters for each operation
- [ ] Create flame graphs for visualization
- **Purpose**: Identify unexpected bottlenecks

#### 12. **Hardware Monitoring**

- [ ] Monitor USB bandwidth usage
- [ ] Check for thermal throttling
- [ ] Verify camera firmware settings
- **Purpose**: Ensure hardware is performing optimally

### 🔧 Configuration Optimizations

#### 13. **Camera Resolution/FPS Trade-offs**

- [ ] Test with 15fps instead of 30fps (double the time budget)
- [ ] Reduce resolution for non-critical cameras
- [ ] Use hardware decimation on RealSense
- **Impact**: Dramatic - 15fps gives 66ms budget vs 33ms

#### 14. **Selective Depth Streaming**

- [ ] Only enable depth on cameras that need it
- [ ] Alternate depth capture between cameras
- [ ] Use lower resolution for depth vs color
- **Impact**: Reduce USB bandwidth and processing

#### 15. **Pipeline Selection**

- [ ] Force all cameras to use same pipeline type
- [ ] Test OpenGL vs CUDA vs OpenCL performance
- [ ] Optimize pipeline parameters
- **Impact**: May save 5-10ms on Kinect

## Testing Methodology

For each optimization:

1. **Baseline**: Record current performance (FPS, latency, CPU)
2. **Implement**: Make single change
3. **Measure**: Run for 5 minutes, collect stats
4. **Compare**: Document improvement/regression
5. **Decision**: Keep if >5% improvement or revert

## Performance Targets

- **Goal**: 60fps system operation (16.7ms budget)
- **Stretch Goal**: 90fps system operation (11.1ms budget)
- **Minimum**: Stable 30fps with <150% CPU usage

## Quick Test Commands

```bash
# Test current performance
./teleop_simple.sh  # Choose option 8 for full system test

# Monitor detailed timing
python -c "from lerobot.cameras.parallel_camera_reader import ParallelCameraReader; print('Parallel reader ready')"

# Check USB bandwidth
usb-devices | grep -A 5 "RealSense\|Kinect"
```

## Notes

- The 30ms "issue" is actually expected behavior for 30fps cameras
- Focus should be on reducing software overhead and CPU usage
- Consider that 60fps operation requires either 60fps cameras or frame interpolation
