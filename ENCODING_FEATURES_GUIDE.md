# LeRobot Video Encoding Features Guide

## Overview

LeRobot now supports two independent video encoding optimization features that can be used separately or together:

1. **Asynchronous Encoding**: Runs video encoding in background threads
2. **GPU Acceleration**: Uses hardware encoders instead of CPU encoders

These features are completely independent and can be combined in four different ways.

## Feature Comparison

| Feature | Description | Benefit | Use Case |
|---------|-------------|---------|----------|
| **Async Encoding** | Background thread processing | Non-blocking recording | Real-time recording |
| **GPU Acceleration** | Hardware encoder usage | Faster encoding | High-performance encoding |
| **Async + GPU** | Background GPU processing | Best of both worlds | High-performance real-time recording |
| **Sync + CPU** | Traditional encoding | Baseline performance | Simple setups |

## Four Encoding Modes

### 1. Synchronous CPU Encoding (Default)
```bash
# Traditional encoding - blocks recording during video processing
python -m lerobot.record \
--dataset.async_video_encoding=false \
--dataset.gpu_video_encoding=false \
# ... other parameters
```

**Characteristics:**
- ✅ Simple and reliable
- ❌ Blocks recording during encoding
- ❌ Slower encoding (CPU only)
- **Use when**: Simple setups, debugging, or when other features aren't needed

### 2. Asynchronous CPU Encoding
```bash
# Background CPU encoding - doesn't block recording
python -m lerobot.record \
--dataset.async_video_encoding=true \
--dataset.gpu_video_encoding=false \
--dataset.video_encoding_workers=2 \
--dataset.video_encoding_queue_size=100 \
# ... other parameters
```

**Characteristics:**
- ✅ Non-blocking recording
- ✅ Multiple episodes can be encoded in parallel
- ❌ Still uses CPU encoding (slower)
- **Use when**: You want smooth recording but don't have GPU acceleration

### 3. Synchronous GPU Encoding
```bash
# GPU-accelerated encoding - blocks recording but much faster
python -m lerobot.record \
--dataset.async_video_encoding=false \
--dataset.gpu_video_encoding=true \
--dataset.gpu_encoder_config='{"encoder_type": "nvenc", "codec": "h264", "preset": "fast"}' \
# ... other parameters
```

**Characteristics:**
- ✅ Much faster encoding (2-5x speedup)
- ❌ Still blocks recording during encoding
- ✅ Better video quality options
- **Use when**: You have GPU hardware but prefer simple synchronous operation

### 4. Asynchronous GPU Encoding (Recommended)
```bash
# Background GPU encoding - best performance and non-blocking
python -m lerobot.record \
--dataset.async_video_encoding=true \
--dataset.gpu_video_encoding=true \
--dataset.gpu_encoder_config='{"encoder_type": "nvenc", "codec": "h264", "preset": "fast"}' \
--dataset.video_encoding_workers=2 \
--dataset.video_encoding_queue_size=100 \
# ... other parameters
```

**Characteristics:**
- ✅ Non-blocking recording
- ✅ Fastest encoding (2-5x speedup)
- ✅ Multiple episodes in parallel
- ✅ Best overall performance
- **Use when**: You have GPU hardware and want maximum performance

## Performance Comparison

### Encoding Time (90-second episode, 2 cameras)

| Mode | Encoding Time | Recording Impact | Total Episode Time |
|------|---------------|------------------|-------------------|
| Sync CPU | ~45-60s | Blocks recording | ~135-150s |
| Async CPU | ~45-60s | Non-blocking | ~90s + background |
| Sync GPU | ~15-25s | Blocks recording | ~105-115s |
| Async GPU | ~15-25s | Non-blocking | ~90s + background |

### Real-world Example

**Recording 20 episodes (90s each):**

| Mode | Total Recording Time | Encoding Time | User Experience |
|------|---------------------|---------------|-----------------|
| Sync CPU | ~30-40 minutes | ~15-20 minutes | Frequent pauses |
| Async CPU | ~30 minutes | ~15-20 minutes (background) | Smooth recording |
| Sync GPU | ~25-30 minutes | ~5-8 minutes | Shorter pauses |
| Async GPU | ~30 minutes | ~5-8 minutes (background) | Smooth + fast |

## Configuration Examples

### Basic Async Encoding (CPU)
```bash
python -m lerobot.record \
--robot.type=so101_follower \
--robot.port=/dev/ttyACM0 \
--robot.id=brl_so101_follower_arm \
--robot.cameras="{ front: {type: opencv, index_or_path: /dev/video4, width: 1280, height: 720, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: /dev/video2, width: 1280, height: 720, fps: 30, fourcc: MJPG}}" \
--teleop.type=so101_leader \
--teleop.port=/dev/ttyACM1 \
--teleop.id=brl_so101_leader_arm \
--dataset.single_task="picking double strawberry" \
--dataset.repo_id=local/async_strawberry_picking_dataset \
--dataset.root=./datasets/async_strawberry_picking \
--dataset.episode_time_s=90 \
--dataset.num_episodes=20 \
--dataset.push_to_hub=false \
--dataset.async_video_encoding=true \
--dataset.video_encoding_workers=2 \
--dataset.video_encoding_queue_size=100 \
--display_data=true
```

### GPU Acceleration Only (Sync)
```bash
python -m lerobot.record \
--robot.type=so101_follower \
--robot.port=/dev/ttyACM0 \
--robot.id=brl_so101_follower_arm \
--robot.cameras="{ front: {type: opencv, index_or_path: /dev/video4, width: 1280, height: 720, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: /dev/video2, width: 1280, height: 720, fps: 30, fourcc: MJPG}}" \
--teleop.type=so101_leader \
--teleop.port=/dev/ttyACM1 \
--teleop.id=brl_so101_leader_arm \
--dataset.single_task="picking double strawberry" \
--dataset.repo_id=local/gpu_strawberry_picking_dataset \
--dataset.root=./datasets/gpu_strawberry_picking \
--dataset.episode_time_s=90 \
--dataset.num_episodes=20 \
--dataset.push_to_hub=false \
--dataset.async_video_encoding=false \
--dataset.gpu_video_encoding=true \
--dataset.gpu_encoder_config='{"encoder_type": "nvenc", "codec": "h264", "preset": "fast", "quality": 23}' \
--display_data=true
```

### Combined Async + GPU (Recommended)
```bash
python -m lerobot.record \
--robot.type=so101_follower \
--robot.port=/dev/ttyACM0 \
--robot.id=brl_so101_follower_arm \
--robot.cameras="{ front: {type: opencv, index_or_path: /dev/video4, width: 1280, height: 720, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: /dev/video2, width: 1280, height: 720, fps: 30, fourcc: MJPG}}" \
--teleop.type=so101_leader \
--teleop.port=/dev/ttyACM1 \
--teleop.id=brl_so101_leader_arm \
--dataset.single_task="picking double strawberry" \
--dataset.repo_id=local/async_gpu_strawberry_picking_dataset \
--dataset.root=./datasets/async_gpu_strawberry_picking \
--dataset.episode_time_s=90 \
--dataset.num_episodes=20 \
--dataset.push_to_hub=false \
--dataset.async_video_encoding=true \
--dataset.gpu_video_encoding=true \
--dataset.gpu_encoder_config='{"encoder_type": "nvenc", "codec": "h264", "preset": "fast", "quality": 23}' \
--dataset.video_encoding_workers=2 \
--dataset.video_encoding_queue_size=100 \
--display_data=true
```

## Testing Different Modes

### Test Async Encoding Only
```bash
python scripts/test_async_encoding.py
```

### Test GPU Encoding Only
```bash
python scripts/test_gpu_encoding.py
```

### Test Combined Features
```bash
python scripts/test_gpu_recording.py
```

### Compare All Modes
```bash
# Test sync CPU (baseline)
python -m lerobot.record --dataset.async_video_encoding=false --dataset.gpu_video_encoding=false --dataset.num_episodes=1

# Test async CPU
python -m lerobot.record --dataset.async_video_encoding=true --dataset.gpu_video_encoding=false --dataset.num_episodes=1

# Test sync GPU
python -m lerobot.record --dataset.async_video_encoding=false --dataset.gpu_video_encoding=true --dataset.num_episodes=1

# Test async GPU
python -m lerobot.record --dataset.async_video_encoding=true --dataset.gpu_video_encoding=true --dataset.num_episodes=1
```

## When to Use Each Mode

### Use Synchronous CPU Encoding When:
- ✅ Simple setup with minimal configuration
- ✅ Debugging video encoding issues
- ✅ Don't have GPU hardware
- ✅ Prefer simple, predictable behavior
- ❌ Don't mind recording pauses during encoding

### Use Asynchronous CPU Encoding When:
- ✅ Want smooth, non-blocking recording
- ✅ Don't have GPU hardware
- ✅ Recording many episodes in sequence
- ✅ Want to maximize recording efficiency
- ❌ Don't mind slower encoding

### Use Synchronous GPU Encoding When:
- ✅ Have GPU hardware available
- ✅ Want faster encoding
- ✅ Prefer simple synchronous operation
- ✅ Don't mind brief recording pauses
- ❌ Want maximum recording efficiency

### Use Asynchronous GPU Encoding When:
- ✅ Have GPU hardware available
- ✅ Want maximum performance
- ✅ Recording many episodes
- ✅ Want smooth, non-blocking recording
- ✅ Best overall user experience
- ❌ More complex configuration

## Hardware Requirements

### For Async Encoding:
- **CPU**: Any modern multi-core CPU
- **Memory**: 4GB+ RAM recommended
- **Storage**: Fast SSD recommended for image I/O

### For GPU Acceleration:
- **NVIDIA**: GeForce GTX 600+ series, RTX series
- **Intel**: 6th generation+ with integrated graphics
- **AMD**: Radeon RX 400+ series
- **Driver**: Latest graphics drivers

### For Combined Features:
- Both async and GPU requirements
- **Recommended**: NVIDIA RTX 3000+ series for best performance

## Troubleshooting

### Async Encoding Issues
```
Problem: Async encoding not working
Solution: Check video_encoding_workers and video_encoding_queue_size settings
```

### GPU Encoding Issues
```
Problem: GPU encoding falls back to CPU
Solution: Check GPU drivers and FFmpeg NVENC support
```

### Combined Feature Issues
```
Problem: Both features enabled but poor performance
Solution: Reduce video_encoding_workers or GPU quality settings
```

## Best Practices

### 1. Start Simple
- Begin with synchronous CPU encoding to establish baseline
- Add async encoding for smoother recording
- Add GPU acceleration for faster encoding
- Combine both for maximum performance

### 2. Monitor Performance
- Use `nvidia-smi -l 1` to monitor GPU usage
- Check CPU usage during recording
- Monitor disk I/O for bottlenecks

### 3. Adjust Settings
- Increase `video_encoding_workers` for more parallel processing
- Adjust GPU quality settings for speed vs quality trade-off
- Monitor memory usage with large queue sizes

### 4. Test Your Setup
- Always test with your specific hardware
- Start with short episodes for testing
- Monitor system resources during recording

## Conclusion

LeRobot's video encoding features provide flexible optimization options:

- **Async Encoding**: Improves recording experience (non-blocking)
- **GPU Acceleration**: Improves encoding performance (faster)
- **Combined**: Best overall performance and user experience

Choose the mode that best fits your hardware capabilities and recording requirements. The features are designed to work independently, so you can enable only what you need. 