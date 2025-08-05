# GPU-Accelerated Video Encoding Guide

## Overview

LeRobot now supports GPU-accelerated video encoding for significantly faster video processing during dataset recording. This feature leverages hardware encoders to reduce encoding time and improve overall recording performance.

## Supported Hardware Encoders

### NVIDIA NVENC (Recommended)
- **Codecs**: H.264, HEVC (H.265), AV1
- **GPUs**: GeForce GTX 600+ series, RTX series, Quadro series
- **Performance**: 2-5x faster than CPU encoding
- **Quality**: Excellent quality with proper configuration

### Intel Quick Sync Video (QSV)
- **Codecs**: H.264, HEVC
- **CPUs**: Intel 6th generation+ with integrated graphics
- **Performance**: 2-3x faster than CPU encoding
- **Quality**: Good quality for most use cases

### AMD Video Coding Engine (VCE)
- **Codecs**: H.264, HEVC
- **GPUs**: AMD Radeon RX 400+ series
- **Performance**: 2-4x faster than CPU encoding
- **Quality**: Good quality with proper configuration

### Software Fallback
- **Codecs**: H.264 (libx264), HEVC (libx265), AV1 (libsvtav1)
- **Performance**: Baseline CPU performance
- **Quality**: Excellent quality, slower encoding

## Quick Start

### Enable GPU Encoding

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
--dataset.async_video_encoding=true \
--dataset.gpu_video_encoding=true \
--dataset.video_encoding_workers=2 \
--dataset.video_encoding_queue_size=100 \
--display_data=true
```

### Advanced GPU Configuration

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
--dataset.async_video_encoding=true \
--dataset.gpu_video_encoding=true \
--dataset.gpu_encoder_config='{"encoder_type": "nvenc", "codec": "hevc", "preset": "fast", "quality": 20, "bitrate": "10M", "gpu_id": 0}' \
--dataset.video_encoding_workers=2 \
--dataset.video_encoding_queue_size=100 \
--display_data=true
```

## Configuration Options

### GPU Encoder Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder_type` | string | `"auto"` | Encoder type: `"nvenc"`, `"qsv"`, `"vce"`, `"software"`, `"auto"` |
| `codec` | string | `"h264"` | Video codec: `"h264"`, `"hevc"`, `"av1"` |
| `preset` | string | `"fast"` | Encoding preset: `"fast"`, `"medium"`, `"slow"`, `"hq"` |
| `quality` | int | `23` | Quality setting (lower = better, range depends on encoder) |
| `bitrate` | string | `"5M"` | Target bitrate (e.g., `"5M"`, `"10M"`) |
| `gpu_id` | int | `0` | GPU device ID for multi-GPU systems |

### Performance vs Quality Trade-offs

#### Fast Encoding (Recommended for real-time recording)
```json
{
    "encoder_type": "nvenc",
    "codec": "h264",
    "preset": "fast",
    "quality": 25,
    "bitrate": "5M"
}
```

#### High Quality (Recommended for final datasets)
```json
{
    "encoder_type": "nvenc",
    "codec": "hevc",
    "preset": "medium",
    "quality": 20,
    "bitrate": "10M"
}
```

#### Maximum Quality (For archival purposes)
```json
{
    "encoder_type": "nvenc",
    "codec": "hevc",
    "preset": "slow",
    "quality": 18,
    "bitrate": "15M"
}
```

## Performance Benchmarks

### Expected Performance Improvements

| Hardware | Codec | Speedup | Use Case |
|----------|-------|---------|----------|
| NVIDIA RTX 4060 | H.264 | 3-5x | Real-time recording |
| NVIDIA RTX 4060 | HEVC | 2-4x | High-quality recording |
| Intel i7-12700K | H.264 | 2-3x | Integrated graphics |
| AMD RX 6700 XT | H.264 | 2-4x | AMD GPU systems |

### Real-world Performance Example

**90-second episode with 2 cameras (1280x720 @ 30 FPS):**

- **CPU Encoding**: ~45-60 seconds per episode
- **GPU Encoding**: ~15-25 seconds per episode
- **Speedup**: 2-4x faster
- **Recording Impact**: Minimal blocking during encoding

## Hardware Requirements

### NVIDIA GPUs
- **Minimum**: GeForce GTX 600 series
- **Recommended**: GeForce RTX 3000+ series
- **Driver**: NVIDIA driver 418.30+
- **CUDA**: CUDA 10.0+ (for some features)

### Intel CPUs
- **Minimum**: 6th generation Intel Core i3/i5/i7
- **Recommended**: 10th generation+ with UHD Graphics
- **Driver**: Latest Intel graphics driver

### AMD GPUs
- **Minimum**: Radeon RX 400 series
- **Recommended**: Radeon RX 5000+ series
- **Driver**: AMD driver 18.12.1+

## Troubleshooting

### Common Issues

#### 1. GPU Encoder Not Available
```
Error: No suitable encoder found, falling back to software H.264
```

**Solutions:**
- Check GPU driver installation
- Verify FFmpeg was compiled with GPU support
- Try different encoder type: `"auto"`, `"nvenc"`, `"qsv"`

#### 2. GPU Memory Issues
```
Error: GPU memory allocation failed
```

**Solutions:**
- Reduce video resolution or bitrate
- Close other GPU-intensive applications
- Use CPU encoding as fallback

#### 3. Poor Video Quality
```
Issue: Videos look compressed or blocky
```

**Solutions:**
- Increase quality setting (lower number)
- Increase bitrate
- Use HEVC codec instead of H.264
- Try different preset: `"medium"` or `"slow"`

### Testing GPU Encoding

Run the GPU encoding test to verify your setup:

```bash
python scripts/test_gpu_encoding.py
```

This will test:
- NVIDIA NVENC H.264 encoding
- NVIDIA NVENC HEVC encoding
- Auto-encoder selection
- CPU vs GPU performance comparison

### Monitoring GPU Usage

Monitor GPU usage during encoding:

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor GPU encoding processes
watch -n 1 'ps aux | grep ffmpeg'
```

## Best Practices

### 1. Choose the Right Encoder
- **NVIDIA GPUs**: Use `"nvenc"` for best performance
- **Intel CPUs**: Use `"qsv"` for integrated graphics
- **AMD GPUs**: Use `"vce"` for dedicated graphics
- **Unknown/Testing**: Use `"auto"` for automatic selection

### 2. Optimize for Your Use Case
- **Real-time recording**: Use `"fast"` preset with H.264
- **High-quality datasets**: Use `"medium"` preset with HEVC
- **Archival storage**: Use `"slow"` preset with high bitrate

### 3. Monitor Performance
- Start with default settings
- Adjust quality/bitrate based on results
- Test with your specific hardware
- Monitor GPU temperature and usage

### 4. Fallback Strategy
- GPU encoding automatically falls back to CPU if needed
- Monitor logs for fallback messages
- Consider hybrid approach for critical recordings

## Advanced Configuration

### Multi-GPU Systems
```json
{
    "encoder_type": "nvenc",
    "codec": "h264",
    "preset": "fast",
    "quality": 23,
    "gpu_id": 1
}
```

### Custom FFmpeg Parameters
The GPU encoder uses optimized FFmpeg parameters for each encoder type. For custom configurations, you can modify the `gpu_video_encoder.py` file.

### Batch Processing
GPU encoding works seamlessly with async encoding and batch processing:

```bash
# Process multiple episodes in parallel
--dataset.video_encoding_workers=4
--dataset.video_encoding_queue_size=200
```

## Integration with Existing Workflows

### Existing Async Encoding
GPU encoding is fully compatible with existing async encoding workflows. Simply add the GPU configuration:

```bash
# Existing async encoding
--dataset.async_video_encoding=true

# Add GPU acceleration
--dataset.gpu_video_encoding=true
--dataset.gpu_encoder_config='{"encoder_type": "nvenc", "codec": "h264"}'
```

### Benchmarking
Compare performance with the benchmarking tools:

```bash
# Test CPU encoding
python scripts/run_recording_benchmark.py --mode synthetic

# Test GPU encoding
python scripts/run_recording_benchmark.py --mode synthetic --gpu-encoding
```

## Conclusion

GPU-accelerated encoding provides significant performance improvements for video encoding in LeRobot datasets. With proper configuration, you can achieve 2-5x faster encoding while maintaining high video quality.

The system automatically detects available hardware and falls back to CPU encoding when needed, ensuring reliable operation across different hardware configurations.

For best results:
1. Use NVIDIA NVENC for NVIDIA GPUs
2. Start with `"fast"` preset for real-time recording
3. Monitor performance and adjust settings as needed
4. Test with your specific hardware and use case 