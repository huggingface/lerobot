# Depth Camera Performance Benchmark

## Questions

What is the optimal configuration for depth camera systems in robotics applications?

- How does depth processing impact overall frame rates?
- What are the USB bandwidth limitations when using multiple RealSense cameras?
- Which threading strategy provides the best performance: sequential vs parallel camera reads?
- What timeout values optimize responsiveness vs reliability?
- How much overhead do depth colorization and visualization (Rerun) add?
- What's the performance difference between RGB-only and RGB+Depth modes?

## Background

Depth cameras are increasingly important in robotics for:
- 3D scene understanding and manipulation planning  
- Obstacle detection and navigation
- Object segmentation and pose estimation
- Dataset collection with spatial awareness

However, depth cameras introduce significant complexity:
- **Hardware bottlenecks**: USB 3.0 bandwidth limits with multiple cameras
- **Processing overhead**: Depth stream decoding, alignment, and colorization
- **Threading challenges**: Background frame collection vs main thread access
- **Latency vs accuracy trade-offs**: Timeout optimization for real-time performance

This benchmark quantifies these trade-offs to guide optimal depth camera integration.

## Hardware Requirements

**Tested Cameras:**
- Intel RealSense D405 (primary target)
- Intel RealSense D435i (compatible)
- Intel RealSense D455 (compatible)

**System Requirements:**
- USB 3.0+ ports (dedicated controllers recommended for multiple cameras)
- Modern CPU with sufficient threading capacity
- 8GB+ RAM for multiple camera streams
- Windows 10/11 or Ubuntu 20.04+ with librealsense2

**Network Requirements:**
- For Rerun visualization: Local network access for web viewer

## Variables

### Camera Configuration
| parameter       | values                                    |
| --------------- | ----------------------------------------- |
| **camera_count** | `1`, `2`, `3`, `4`                       |
| **resolution**   | `640x480`, `848x480`, `1280x720`        |
| **fps**          | `15`, `30`, `60`                         |
| **use_depth**    | `false` (RGB-only), `true` (RGB+Depth)  |

### Threading Strategy  
| parameter         | values                                    |
| ----------------- | ----------------------------------------- |
| **read_strategy** | `sequential`, `parallel`, `camera_manager` |
| **timeout_ms**    | `50`, `100`, `150`, `200`, `300`, `500`  |

### Processing Pipeline
| parameter              | values                                |
| ---------------------- | ------------------------------------- |
| **depth_colorization** | `disabled`, `turbo`, `jet`, `plasma` |
| **rerun_logging**      | `disabled`, `rgb_only`, `full`       |

### Hardware Scenarios
Different USB configurations and camera placements that affect bandwidth:

- **single_controller**: All cameras on same USB 3.0 controller
- **multi_controller**: Cameras distributed across USB controllers  
- **usb2_fallback**: Mixed USB 3.0 and USB 2.0 connections
- **extended_cables**: Long USB cables (potential signal degradation)

## Metrics

**Frame Rate (higher is better)**
`avg_fps` measures the sustained frame rate during camera operations. Target thresholds:
- ✅ **25+ Hz**: Excellent for real-time robotics
- ⚠️ **15-25 Hz**: Acceptable for most applications  
- ❌ **<15 Hz**: Too slow for responsive robotics

**Frame Rate Stability (lower is better)**
`fps_std_dev` measures frame rate consistency. Lower variance indicates more predictable performance.

**Success Rate (higher is better)**
`success_rate` is the percentage of successful frame reads without timeouts or errors.

**USB Bandwidth Utilization**  
`bandwidth_usage` estimates USB 3.0 bandwidth consumption:
- Single D405 RGB+Depth @ 848x480x30fps ≈ 200-300 MB/s
- USB 3.0 theoretical limit: 5 Gbps (625 MB/s practical)
- Multiple cameras can saturate bandwidth

**Memory Usage (lower is better)**
`peak_memory_mb` tracks maximum memory consumption during testing.

**Processing Overhead (lower is better)**
`processing_time_ms` measures additional latency from depth processing:
- Raw frame reads (baseline)
- + Depth colorization overhead
- + Rerun visualization overhead  

**Latency (lower is better)**
`frame_to_frame_latency_ms` measures time from hardware capture to application access.

## Benchmark Methodology

The benchmark systematically tests depth camera performance across different configurations:

**Camera Setup:** For each camera configuration, the benchmark:
1. Connects cameras with proper USB distribution
2. Validates camera capabilities (RGB, depth streams)
3. Establishes background reading threads
4. Performs 3-second warmup period

**Performance Testing:** Each test scenario runs for a configurable duration (default 10 seconds):
1. **Individual Camera Tests**: Baseline performance per camera
2. **Multi-Camera Tests**: USB bandwidth limits and scaling
3. **Processing Overhead**: Impact of depth colorization and logging
4. **Threading Strategy**: Sequential vs parallel vs CameraManager
5. **Timeout Optimization**: Reliability vs responsiveness trade-offs

**Data Collection:** For each configuration:
- Record frame rate, latency, success rate over time
- Monitor USB bandwidth usage and memory consumption  
- Capture processing overhead at each pipeline stage
- Log errors, timeouts, and failure modes

**Hardware Detection:** The benchmark automatically:
- Detects available RealSense cameras by serial number
- Identifies USB controller distribution
- Validates camera capabilities and supported resolutions
- Adapts test matrix to available hardware

## Hardware-Specific Testing

Different hardware setups require different test approaches:

**Development Setup (1-2 cameras):**
- Focus on processing overhead and timeout optimization
- Test RGB vs RGB+Depth performance differences
- Validate threading strategies

**Production Setup (3+ cameras):**
- Emphasize USB bandwidth analysis
- Test controller distribution strategies  
- Validate real-world robot integration scenarios

**Mixed Hardware:**
- Test compatibility across different RealSense models
- Validate graceful degradation with USB 2.0 fallback
- Test extended cable scenarios

## Results Interpretation

**Performance Targets by Use Case:**

*Real-time Teleoperation:*
- Target: 25+ Hz with <100ms latency
- Depth processing acceptable if <5Hz impact
- High success rate (>95%) critical

*Dataset Collection:*
- Target: 15+ Hz sustained over long periods
- Depth quality prioritized over raw speed
- Storage bandwidth may be limiting factor

*Offline Processing:*
- Target: Maximum quality depth data
- Frame rate less critical than data fidelity
- Can tolerate higher processing overhead

## Example Configurations

**High-Performance RGB+Depth (2 cameras):**
```bash
python benchmarks/depth_cameras/run_depth_benchmark.py \
    --camera-count 2 \
    --resolution 848x480 \
    --fps 30 \
    --use-depth true \
    --read-strategy camera_manager \
    --timeout-ms 150 \
    --depth-colorization turbo \
    --test-duration 10 \
    --output-dir outputs/depth_benchmark
```

**USB Bandwidth Analysis (4 cameras):**
```bash  
python benchmarks/depth_cameras/run_depth_benchmark.py \
    --camera-count 4 \
    --resolution 640x480 \
    --fps 30 \
    --use-depth true \
    --read-strategy parallel \
    --timeout-ms 100 200 300 \
    --bandwidth-analysis \
    --output-dir outputs/bandwidth_test
```

**Processing Overhead Analysis:**
```bash
python benchmarks/depth_cameras/run_depth_benchmark.py \
    --camera-count 1 \
    --resolution 848x480 \
    --fps 30 \
    --use-depth true \
    --depth-colorization disabled turbo jet plasma \
    --rerun-logging disabled rgb_only full \
    --processing-analysis \
    --output-dir outputs/processing_test
```

## Installation

**RealSense SDK:**
```bash
# Ubuntu 20.04+
sudo apt-get update && sudo apt-get install -y \
    librealsense2-dkms \
    librealsense2-utils \
    librealsense2-dev

# Or install from source for latest features
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) && sudo make install
```

**Python Dependencies:**
```bash
pip install pyrealsense2 rerun-sdk numpy opencv-python
```

**Hardware Setup:**
1. Connect RealSense cameras to dedicated USB 3.0 ports
2. Verify camera detection: `rs-enumerate-devices`  
3. Test individual cameras: `realsense-viewer`
4. For multiple cameras, distribute across USB controllers

## Hardware Validation

Before running benchmarks, validate your setup:

**Check Camera Detection:**
```bash
python -c "
import pyrealsense2 as rs
ctx = rs.context()
devices = ctx.query_devices()
print(f'Found {len(devices)} RealSense cameras')
for i, dev in enumerate(devices):
    print(f'  {i}: {dev.get_info(rs.camera_info.serial_number)}')"
```

**USB Bandwidth Test:**
```bash
# Single camera bandwidth test
rs-enumerate-devices -c
# Look for USB 3.0 vs 2.0 indication
```

**Performance Baseline:**
```bash
# Quick single-camera test
python benchmarks/depth_cameras/run_depth_benchmark.py \
    --camera-count 1 \
    --quick-test \
    --output-dir outputs/baseline
```

## Troubleshooting

**Common Issues:**

*Cameras not detected:*
- Check USB 3.0 connection (blue connector)
- Install latest librealsense2 drivers
- Verify permissions: `sudo usermod -a -G video $USER`

*Low frame rates:*
- Check USB bandwidth with `rs-enumerate-devices -c`
- Distribute cameras across USB controllers
- Reduce resolution or disable depth if needed

*High frame rate variance:*
- Increase timeout values for USB-constrained setups  
- Check system CPU load and thermal throttling
- Verify adequate power supply to cameras

*Memory issues:*
- Reduce number of concurrent cameras
- Implement frame dropping in application logic
- Monitor system memory with `htop` during tests

## Data Analysis

The benchmark outputs CSV files with detailed performance metrics:

**Primary Results Table:**
- Configuration parameters and measured performance
- Statistical analysis (mean, std dev, percentiles)
- Hardware resource utilization

**Time Series Data:**
- Frame-by-frame performance over test duration
- Useful for identifying performance degradation patterns
- Can reveal USB bandwidth contention timing

**Error Analysis:**
- Timeout patterns and error rates by configuration
- Correlation between errors and system load
- Hardware failure mode analysis

Use the provided analysis scripts to generate:
- Performance vs configuration scatter plots  
- USB bandwidth utilization charts
- Frame rate stability histograms
- Processing overhead breakdown

## Integration with LeRobot

The benchmark results inform optimal camera configurations for specific LeRobot use cases:

**Teleoperation:** Use benchmark results to select camera count, resolution, and timeout values that maintain responsive control while maximizing visual information.

**Dataset Collection:** Choose configurations that sustain target frame rates over long recording sessions without memory leaks or performance degradation.

**Policy Training:** Understand depth processing overhead to budget compute resources between camera systems and ML inference during real-time policy execution.

Results from this benchmark directly inform the default camera configurations in LeRobot robot classes and provide guidance for custom hardware integrations.