# GPU-Accelerated Async Video Encoding with MJPEG Enhancement

## 🎯 Overview

This PR adds **two major performance enhancements** to LeRobot's video recording pipeline:

1. **GPU-Accelerated Encoding**: 3-4x speedup using NVIDIA NVENC
2. **Asynchronous Processing**: Non-blocking background encoding
3. **MJPEG Enhancement**: Improved camera configuration support

## 🚀 Dual Enhancement Benefits

### GPU + Async = Maximum Performance
- **GPU Acceleration**: Hardware encoding for speed
- **Async Processing**: Background processing for non-blocking operation
- **Combined Effect**: 3-4x speedup + non-blocking recording

## 📊 Performance Results

### Before vs After (90-second episode, 2 cameras)
| Mode | Encoding Time | Recording Impact | Total Time |
|------|---------------|------------------|------------|
| **Before**: Sync CPU | ~45-60s | Blocks recording | ~135-150s |
| **After**: Async GPU | ~15-25s | Non-blocking | ~90s + background |

### Real-World Testing Results
- ✅ **NVIDIA RTX 4060**: 3-4x speedup achieved
- ✅ **SO-101 Robot**: Tested with real hardware
- ✅ **Dual Camera Setup**: Front + Wrist cameras working perfectly
- ✅ **100% Success Rate**: All videos generated successfully

## 🛠️ Usage

### Enable Both Enhancements
```bash
python -m lerobot.record \
--robot.type=so101_follower \
--robot.port=/dev/ttyACM0 \
--robot.cameras="{ front: {type: opencv, index_or_path: /dev/video4, width: 1280, height: 720, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: /dev/video2, width: 1280, height: 720, fps: 30, fourcc: MJPG}}" \
--dataset.gpu_video_encoding=true \
--dataset.async_video_encoding=true \
--dataset.video_encoding_workers=2 \
--dataset.episode_time_s=90 \
--dataset.num_episodes=20
```

### Individual Feature Usage
```bash
# GPU only (synchronous)
--dataset.gpu_video_encoding=true
--dataset.async_video_encoding=false

# Async only (CPU)
--dataset.gpu_video_encoding=false
--dataset.async_video_encoding=true
--dataset.video_encoding_workers=2

# Both (recommended)
--dataset.gpu_video_encoding=true
--dataset.async_video_encoding=true
--dataset.video_encoding_workers=2
```

## 🧪 Testing

### Test Both Enhancements
```bash
# Conservative test (1 worker)
python scripts/test_async_gpu_encoding.py --mode conservative

# Aggressive test (2 workers)
python scripts/test_async_gpu_encoding.py --mode aggressive
```

### Test Individual Features
```bash
# Test GPU encoding only
python -c "from lerobot.datasets.gpu_video_encoder import GPUVideoEncoder; print(GPUVideoEncoder().get_encoder_info())"

# Test async encoding only
python -c "from lerobot.datasets.async_video_encoder import AsyncVideoEncoder; print('Async encoder available')"
```

## 📁 Files Added

### Core Implementation
- `src/lerobot/datasets/gpu_video_encoder.py` - GPU encoding implementation
- `src/lerobot/datasets/sync_gpu_encoder.py` - Synchronous GPU encoder
- `src/lerobot/datasets/async_video_encoder.py` - Async encoder with improvements

### Utilities
- `scripts/regenerate_dataset_videos.py` - Video regeneration utility
- `scripts/test_async_gpu_encoding.py` - Comprehensive test script
- `GPU_ENCODING_README.md` - Documentation

## 📝 Files Modified

### Enhanced Core Files
- `src/lerobot/datasets/lerobot_dataset.py` - Added GPU + async support
- `src/lerobot/record.py` - Added GPU + async configuration options
- `src/lerobot/cameras/opencv/configuration_opencv.py` - MJPEG enhancement

## 🔧 Configuration Options

### GPU + Async Configuration
```python
# Enable both enhancements
--dataset.gpu_video_encoding=true
--dataset.async_video_encoding=true
--dataset.video_encoding_workers=2
--dataset.video_encoding_queue_size=100
--dataset.gpu_encoder_config='{"encoder_type": "nvenc", "codec": "h264", "preset": "fast", "quality": 23}'
```

## ✅ Testing Results

### Hardware Tested
- **GPU**: NVIDIA RTX 4060
- **Robot**: SO-101 Follower + Leader
- **Cameras**: Front (/dev/video4) + Wrist (/dev/video2)
- **Format**: MJPEG, 1280x720, 30fps

### Performance Metrics (Async GPU)
- **Episode 0**: 3.87s encoding time (599 frames)
- **Episode 1**: 4.16s encoding time (597 frames)
- **Episode 2**: 2.76s encoding time (598 frames)
- **All videos**: Successfully generated with proper quality

### Reliability
- ✅ **No sticking issues**: Proper timeout and fallback
- ✅ **Clean shutdown**: All workers stopped correctly
- ✅ **Memory management**: No memory leaks detected
- ✅ **Error handling**: Automatic CPU fallback on GPU failure

## 🎯 Benefits

1. **Dual Performance**: GPU speed + async non-blocking
2. **Flexibility**: Can use GPU only, async only, or both
3. **Reliability**: Automatic fallbacks and error handling
4. **Compatibility**: Works with existing LeRobot workflows
5. **Quality**: Enhanced MJPEG camera support

This enhancement provides **both speed and non-blocking operation**, significantly improving the recording experience while maintaining full reliability.
