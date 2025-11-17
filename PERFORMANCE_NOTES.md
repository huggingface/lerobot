# XLeRobot 30 Hz Optimization Notes

## Current Status
- **Target:** 30 Hz (33ms per loop) with cameras active
- **Current:** 6 Hz (170ms per loop)
- **Challenge:** Hardware bottlenecks limit achievable rate

## Hardware Limitations
- **3 Serial Buses:** 80ms (sequential reads from USB adapters)
- **Camera Capture:** 40ms (USB camera frame capture)
- **Total Hardware Time:** 120ms minimum = **8.3 Hz theoretical maximum**

## Optimization Strategy
To approach 30 Hz with cameras, we need:
1. **Parallel serial bus reads** (threading)
2. **Async camera capture** (already implemented)
3. **Reduce VR WebSocket latency**
4. **Optimize IK computation**

## Realistic Expectations
- **With current hardware:** 12-15 Hz achievable with optimizations
- **For 30 Hz:** Would require hardware changes (single fast bus, MIPI camera)
- **Best case with software only:** ~18-20 Hz with aggressive parallelization

## Implementation
Applying all feasible software optimizations while maintaining camera functionality.

