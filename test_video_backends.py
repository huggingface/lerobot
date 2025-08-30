#!/usr/bin/env python

"""
Quick benchmark to test video decoding speed across different backends.
"""

import time
from pathlib import Path
import torch

def test_video_backend(video_path, backend_name, num_frames=10):
    """Test video decoding speed for a specific backend."""
    try:
        from lerobot.datasets.video_utils import decode_video_frames
        
        # Create timestamps for first N frames
        fps = 30  # Assume 30fps, adjust if needed
        timestamps = [i / fps for i in range(num_frames)]
        
        # Time the decoding
        start_time = time.perf_counter()
        frames = decode_video_frames(video_path, timestamps, tolerance_s=1e-4, backend=backend_name)
        decode_time = time.perf_counter() - start_time
        
        frames_decoded = frames.shape[1] if frames.dim() > 1 else frames.shape[0]
        ms_per_frame = (decode_time * 1000) / max(frames_decoded, 1)
        
        print(f"✅ {backend_name:12} | {decode_time*1000:6.1f}ms total | {ms_per_frame:6.1f}ms/frame | {frames_decoded} frames")
        return decode_time, frames_decoded
        
    except Exception as e:
        print(f"❌ {backend_name:12} | ERROR: {str(e)[:50]}...")
        return float('inf'), 0

def main():
    # Find your video files
    video_dir = Path.home() / ".cache/huggingface/lerobot/kenmacken/record-test-2/videos"
    video_files = list(video_dir.rglob("*.mp4"))
    
    if not video_files:
        print("❌ No video files found! Check the path.")
        return
        
    test_video = video_files[0]
    print(f"Testing video: {test_video.name}")
    print(f"File size: {test_video.stat().st_size / 1024 / 1024:.1f} MB")
    print("-" * 60)
    
    backends = ["torchcodec", "pyav", "video_reader"]
    results = {}
    
    for backend in backends:
        decode_time, frames = test_video_backend(test_video, backend)
        results[backend] = (decode_time, frames)
    
    print("-" * 60)
    print("RECOMMENDATION:")
    
    # Find fastest backend
    valid_results = {k: v for k, v in results.items() if v[0] != float('inf')}
    if valid_results:
        fastest = min(valid_results.items(), key=lambda x: x[1][0])
        print(f"🚀 Use '{fastest[0]}' - fastest backend!")
        print(f"   Add to your config: video_backend: \"{fastest[0]}\"")
        
        slowest_time = max(valid_results.values())[0]
        speedup = slowest_time / fastest[1][0]
        print(f"   Speedup vs slowest: {speedup:.1f}x faster")
    else:
        print("❌ No backends worked!")

if __name__ == "__main__":
    main()
