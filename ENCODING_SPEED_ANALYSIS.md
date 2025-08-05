# LeRobot Dataset Recording: Encoding Speed Analysis

## Current Data Processing Pipeline

### 1. Recording Flow Overview

The recording process follows this sequence:

1. **Frame Capture**: Robot observations are captured at the specified FPS (default 30)
2. **Frame Processing**: Each frame is processed and added to an episode buffer
3. **Image Writing**: Images are written to disk asynchronously using `AsyncImageWriter`
4. **Episode Saving**: When an episode completes, data is saved to HuggingFace datasets format
5. **Video Encoding**: Images are encoded into video files using FFmpeg
6. **Cleanup**: Temporary image files are removed

### 2. Current Performance Characteristics

#### Image Writing (Already Optimized)
- **AsyncImageWriter**: Uses multiprocessing/threading for parallel image saving
- **Configurable**: `num_image_writer_processes` and `num_image_writer_threads_per_camera`
- **Default**: 0 processes, 4 threads per camera
- **Performance**: This is already well-optimized and shouldn't be a bottleneck

#### Video Encoding (Current Bottleneck)
- **Synchronous**: Video encoding happens in the main thread
- **Per-Episode**: Each episode is encoded immediately after completion
- **FFmpeg-based**: Uses PyAV library with FFmpeg backend
- **Codec**: Default is `libsvtav1` with `yuv420p` pixel format
- **Compression**: CRF=30 (good quality, moderate compression)

### 3. Identified Performance Bottlenecks

#### Primary Issues:
1. **Synchronous Video Encoding**: Blocks the main recording thread
2. **Per-Episode Encoding**: No batching of encoding operations
3. **Sequential Processing**: Episodes are encoded one after another
4. **No GPU Acceleration**: Currently CPU-only encoding

#### Secondary Issues:
1. **Image Format**: PNG files are large and slow to process
2. **File I/O**: Reading many small PNG files is inefficient
3. **Memory Usage**: Loading all frames into memory for encoding

## Optimization Opportunities

### 1. Asynchronous Video Encoding

**Current State**: Video encoding blocks the recording loop
```python
# Current: Blocks recording
self.encode_episode_videos(episode_index)
```

**Proposed Solution**: Move encoding to background thread/process
```python
# Proposed: Non-blocking
self.video_encoding_queue.put((episode_index, video_keys))
```

### 2. Batched Video Encoding

**Current State**: Each episode encoded individually
**Proposed Solution**: Batch multiple episodes for encoding

**Benefits**:
- Better CPU utilization
- Reduced overhead from starting/stopping FFmpeg processes
- More efficient memory usage

### 3. GPU-Accelerated Encoding

**Current State**: CPU-only encoding with PyAV
**Proposed Solution**: Use NVIDIA NVENC or similar hardware encoders

**Implementation Options**:
- `nvidia-ml-py3` for GPU monitoring
- `ffmpeg-python` with NVENC codec
- Direct GPU memory transfer

### 4. Optimized Image Format

**Current State**: PNG files (lossless, large)
**Proposed Solutions**:
- **JPEG**: Faster encoding, smaller files, acceptable quality
- **WebP**: Better compression than JPEG
- **Raw frames**: Skip PNG entirely, encode directly from memory

### 5. Memory-Mapped Video Encoding

**Current State**: Read PNG files from disk
**Proposed Solution**: Stream frames directly from memory buffer

### 6. Parallel Episode Encoding

**Current State**: Sequential episode processing
**Proposed Solution**: Process multiple episodes in parallel

## Implementation Strategy

### Phase 1: Asynchronous Encoding (High Impact, Low Risk)
1. Create `AsyncVideoEncoder` class
2. Use `ThreadPoolExecutor` or `ProcessPoolExecutor`
3. Implement queue-based episode submission
4. Add progress tracking and error handling

### Phase 2: Batching and Optimization (Medium Impact, Medium Risk)
1. Implement episode batching logic
2. Optimize FFmpeg parameters for speed
3. Add memory-efficient frame handling
4. Implement cleanup strategies

### Phase 3: GPU Acceleration (High Impact, High Risk)
1. Add GPU encoding support
2. Implement fallback to CPU encoding
3. Add hardware detection and configuration
4. Benchmark performance improvements

## Configuration Parameters to Add

```python
@dataclass
class DatasetRecordConfig:
    # Existing parameters...
    
    # New encoding optimization parameters
    async_video_encoding: bool = True
    video_encoding_workers: int = 2
    video_encoding_batch_size: int = 5
    use_gpu_encoding: bool = False
    video_encoding_format: str = "jpeg"  # png, jpeg, webp
    video_encoding_quality: int = 85  # for lossy formats
    video_encoding_preset: str = "fast"  # fast, medium, slow
```

## Expected Performance Improvements

### Conservative Estimates:
- **Asynchronous encoding**: 20-30% faster recording
- **Batched encoding**: 15-25% additional improvement
- **JPEG format**: 40-60% faster encoding
- **GPU acceleration**: 2-5x faster encoding (if available)

### Aggressive Estimates:
- **Combined optimizations**: 3-5x faster overall encoding
- **GPU + batching**: 5-10x faster encoding
- **Memory streaming**: Additional 20-30% improvement

## Risk Assessment

### Low Risk:
- Asynchronous encoding (well-tested pattern)
- JPEG format (standard, widely supported)
- Batching logic (similar to existing batch_encoding_size)

### Medium Risk:
- GPU encoding (hardware dependencies)
- Memory streaming (complex memory management)
- Parallel processing (synchronization issues)

### High Risk:
- Changing core encoding pipeline
- Removing PNG intermediate files
- Real-time memory management

## Next Steps

1. **Implement Phase 1**: Asynchronous video encoding
2. **Benchmark current performance**: Measure baseline encoding times
3. **Test with different datasets**: Validate improvements across scenarios
4. **Add configuration options**: Make optimizations configurable
5. **Document performance gains**: Create benchmarks and documentation

## Files to Modify

### Primary Files:
- `src/lerobot/datasets/lerobot_dataset.py`: Main dataset class
- `src/lerobot/datasets/video_utils.py`: Video encoding utilities
- `src/lerobot/record.py`: Recording configuration

### New Files:
- `src/lerobot/datasets/async_video_encoder.py`: Async encoding implementation
- `src/lerobot/datasets/gpu_video_encoder.py`: GPU acceleration
- `tests/test_async_encoding.py`: Unit tests

### Configuration:
- `src/lerobot/configs/`: Add new configuration parameters
- `examples/`: Add examples with optimized settings 