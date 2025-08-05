#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to run recording benchmarks and collect baseline performance measurements.

This script runs a controlled recording session to measure the performance
of different components in the recording pipeline.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.datasets.recording_benchmark import RecordingBenchmark, benchmark_recording
from lerobot.record import record, RecordConfig, DatasetRecordConfig
from lerobot.robots import RobotConfig
from lerobot.teleoperators import TeleoperatorConfig


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('recording_benchmark.log')
        ]
    )


def create_mock_config(
    num_episodes: int = 2,
    episode_time_s: int = 30,
    fps: int = 30,
    num_cameras: int = 1,
    use_video: bool = True,
    image_writer_threads: int = 4,
    image_writer_processes: int = 0,
    video_encoding_batch_size: int = 1,
    async_video_encoding: bool = False,
    video_encoding_workers: int = 2,
    video_encoding_queue_size: int = 100,
) -> RecordConfig:
    """Create a mock recording configuration for benchmarking."""
    
    # Create mock camera configuration
    cameras = {}
    for i in range(num_cameras):
        cameras[f"camera_{i}"] = {
            "type": "opencv",
            "camera_index": i,
            "width": 640,
            "height": 480,
            "fps": fps
        }
    
    # Create robot config (mock - will need real robot for actual testing)
    robot_config = RobotConfig(
        type="mock_robot",  # This would need to be a real robot type
        cameras=cameras,
        id="benchmark_robot"
    )
    
    # Create dataset config
    dataset_config = DatasetRecordConfig(
        repo_id="benchmark/test_recording",
        single_task="Benchmark recording performance",
        fps=fps,
        episode_time_s=episode_time_s,
        num_episodes=num_episodes,
        video=use_video,
        push_to_hub=False,  # Don't push during benchmarking
        num_image_writer_threads_per_camera=image_writer_threads,
        num_image_writer_processes=image_writer_processes,
        video_encoding_batch_size=video_encoding_batch_size,
        async_video_encoding=async_video_encoding,
        video_encoding_workers=video_encoding_workers,
        video_encoding_queue_size=video_encoding_queue_size,
        root="./benchmark_data"  # Local storage for benchmarking
    )
    
    # Create teleop config (mock keyboard teleop)
    teleop_config = TeleoperatorConfig(
        type="keyboard",
        id="benchmark_teleop"
    )
    
    return RecordConfig(
        robot=robot_config,
        dataset=dataset_config,
        teleop=teleop_config,
        display_data=False,
        play_sounds=False
    )


@benchmark_recording
def run_benchmark_recording(cfg: RecordConfig, benchmark: RecordingBenchmark = None) -> None:
    """Run a benchmark recording session."""
    logging.info("Starting benchmark recording session")
    
    # Run the recording with benchmarking
    try:
        dataset = record(cfg)
        logging.info(f"Recording completed successfully. Dataset: {dataset}")
    except Exception as e:
        logging.error(f"Recording failed: {e}")
        raise


def run_synthetic_benchmark(
    num_episodes: int = 2,
    episode_time_s: int = 30,
    fps: int = 30,
    num_cameras: int = 1,
    output_dir: str = "./benchmark_results",
    async_encoding: bool = False,
    video_encoding_workers: int = 2
) -> None:
    """Run a synthetic benchmark without requiring real hardware."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    with RecordingBenchmark() as benchmark:
        logging.info("Running synthetic benchmark")
        
        # Simulate episodes
        for episode in range(num_episodes):
            benchmark.start_episode(episode)
            
            # Simulate frames
            num_frames = episode_time_s * fps
            for frame in range(num_frames):
                # Simulate frame capture time (1-5ms)
                import time
                import random
                
                # Frame capture simulation
                capture_time = random.uniform(0.001, 0.005)
                time.sleep(capture_time)
                benchmark.frame_capture.add_measurement(capture_time)
                
                # Frame processing simulation (2-8ms)
                processing_time = random.uniform(0.002, 0.008)
                time.sleep(processing_time)
                benchmark.frame_processing.add_measurement(processing_time)
                
                # Image writing simulation (5-15ms)
                writing_time = random.uniform(0.005, 0.015)
                time.sleep(writing_time)
                benchmark.image_writing.add_measurement(writing_time)
                
                benchmark.increment_frame_count()
                
                if frame % 100 == 0:
                    benchmark.log_stats()
            
            # Simulate episode saving (1-3 seconds)
            saving_time = random.uniform(1.0, 3.0)
            time.sleep(saving_time)
            benchmark.episode_saving.add_measurement(saving_time)
            
            # Simulate video encoding (5-15 seconds)
            if async_encoding:
                # For async encoding, the encoding happens in background
                # So we only measure the time to submit the task (very fast)
                submission_time = random.uniform(0.001, 0.005)  # 1-5ms
                time.sleep(submission_time)
                benchmark.video_encoding.add_measurement(submission_time)
                
                # Track the actual encoding time separately (simulated)
                actual_encoding_time = random.uniform(5.0, 15.0)
                # In real async encoding, this would happen in background
                # For simulation, we just track it
                if not hasattr(benchmark, 'async_encoding_times'):
                    benchmark.async_encoding_times = []
                benchmark.async_encoding_times.append(actual_encoding_time)
            else:
                # Synchronous encoding blocks the main thread
                encoding_time = random.uniform(5.0, 15.0)
                time.sleep(encoding_time)
                benchmark.video_encoding.add_measurement(encoding_time)
            
            benchmark.end_episode(episode)
        
        # Save results
        benchmark.save_results(output_path / f"synthetic_benchmark_{int(time.time())}.json")
        # Add async encoding summary if applicable
        if async_encoding and hasattr(benchmark, 'async_encoding_times'):
            total_async_time = sum(benchmark.async_encoding_times)
            print(f"\nASYNC ENCODING SUMMARY:")
            print(f"Total async encoding time (background): {total_async_time:.2f}s")
            print(f"Average async encoding time per episode: {total_async_time/len(benchmark.async_encoding_times):.2f}s")
        
        benchmark.print_summary()


def run_comparison_benchmark(
    num_episodes: int = 2,
    episode_time_s: int = 30,
    fps: int = 30,
    num_cameras: int = 1,
    output_dir: str = "./benchmark_results",
    video_encoding_workers: int = 2
) -> None:
    """Run both sync and async benchmarks for comparison."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("RECORDING ENCODING PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Run synchronous benchmark
    print("\n1. RUNNING SYNCHRONOUS ENCODING BENCHMARK")
    print("-" * 50)
    sync_start = time.time()
    
    with RecordingBenchmark() as sync_benchmark:
        for episode in range(num_episodes):
            sync_benchmark.start_episode(episode)
            
            # Simulate frames
            num_frames = episode_time_s * fps
            for frame in range(num_frames):
                import random
                
                # Frame capture simulation
                capture_time = random.uniform(0.001, 0.005)
                time.sleep(capture_time)
                sync_benchmark.frame_capture.add_measurement(capture_time)
                
                # Frame processing simulation
                processing_time = random.uniform(0.002, 0.008)
                time.sleep(processing_time)
                sync_benchmark.frame_processing.add_measurement(processing_time)
                
                # Image writing simulation
                writing_time = random.uniform(0.005, 0.015)
                time.sleep(writing_time)
                sync_benchmark.image_writing.add_measurement(writing_time)
                
                sync_benchmark.increment_frame_count()
            
            # Synchronous episode saving
            saving_time = random.uniform(1.0, 3.0)
            time.sleep(saving_time)
            sync_benchmark.episode_saving.add_measurement(saving_time)
            
            # Synchronous video encoding (blocks main thread)
            encoding_time = random.uniform(5.0, 15.0)
            time.sleep(encoding_time)
            sync_benchmark.video_encoding.add_measurement(encoding_time)
            
            sync_benchmark.end_episode(episode)
    
    sync_total_time = time.time() - sync_start
    sync_encoding_time = sync_benchmark.video_encoding.total_time
    
    # Run asynchronous benchmark
    print("\n2. RUNNING ASYNCHRONOUS ENCODING BENCHMARK")
    print("-" * 50)
    async_start = time.time()
    
    with RecordingBenchmark() as async_benchmark:
        async_benchmark.async_encoding_times = []
        
        for episode in range(num_episodes):
            async_benchmark.start_episode(episode)
            
            # Simulate frames
            num_frames = episode_time_s * fps
            for frame in range(num_frames):
                import random
                
                # Frame capture simulation
                capture_time = random.uniform(0.001, 0.005)
                time.sleep(capture_time)
                async_benchmark.frame_capture.add_measurement(capture_time)
                
                # Frame processing simulation
                processing_time = random.uniform(0.002, 0.008)
                time.sleep(processing_time)
                async_benchmark.frame_processing.add_measurement(processing_time)
                
                # Image writing simulation
                writing_time = random.uniform(0.005, 0.015)
                time.sleep(writing_time)
                async_benchmark.image_writing.add_measurement(writing_time)
                
                async_benchmark.increment_frame_count()
            
            # Asynchronous episode saving (encoding happens in background)
            saving_time = random.uniform(1.0, 3.0)
            time.sleep(saving_time)
            async_benchmark.episode_saving.add_measurement(saving_time)
            
            # Asynchronous video encoding (non-blocking)
            submission_time = random.uniform(0.001, 0.005)
            time.sleep(submission_time)
            async_benchmark.video_encoding.add_measurement(submission_time)
            
            # Track actual encoding time (simulated background processing)
            actual_encoding_time = random.uniform(5.0, 15.0)
            async_benchmark.async_encoding_times.append(actual_encoding_time)
            
            async_benchmark.end_episode(episode)
    
    async_total_time = time.time() - async_start
    async_encoding_time = sum(async_benchmark.async_encoding_times)
    
    # Save results
    sync_benchmark.save_results(output_path / f"sync_benchmark_{int(time.time())}.json")
    async_benchmark.save_results(output_path / f"async_benchmark_{int(time.time())}.json")
    
    # Print comparison
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"\nSYNCHRONOUS ENCODING:")
    print(f"  Total recording time: {sync_total_time:.2f}s")
    print(f"  Video encoding time: {sync_encoding_time:.2f}s")
    print(f"  Encoding percentage: {(sync_encoding_time/sync_total_time)*100:.1f}%")
    
    print(f"\nASYNCHRONOUS ENCODING:")
    print(f"  Total recording time: {async_total_time:.2f}s")
    print(f"  Video encoding time (background): {async_encoding_time:.2f}s")
    print(f"  Task submission time: {async_benchmark.video_encoding.total_time:.2f}s")
    print(f"  Submission percentage: {(async_benchmark.video_encoding.total_time/async_total_time)*100:.1f}%")
    
    print(f"\nPERFORMANCE IMPROVEMENTS:")
    time_saved = sync_total_time - async_total_time
    speedup = sync_total_time / async_total_time
    print(f"  Time saved: {time_saved:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Improvement: {((sync_total_time - async_total_time) / sync_total_time) * 100:.1f}%")
    
    print(f"\nENCODING EFFICIENCY:")
    print(f"  Sync encoding blocks recording for: {sync_encoding_time:.2f}s")
    print(f"  Async encoding blocks recording for: {async_benchmark.video_encoding.total_time:.2f}s")
    print(f"  Blocking time reduction: {((sync_encoding_time - async_benchmark.video_encoding.total_time) / sync_encoding_time) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Run recording performance benchmarks")
    parser.add_argument(
        "--mode",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Benchmark mode: synthetic (no hardware) or real (requires robot)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="Number of episodes to record"
    )
    parser.add_argument(
        "--episode-time",
        type=int,
        default=30,
        help="Duration of each episode in seconds"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording FPS"
    )
    parser.add_argument(
        "--cameras",
        type=int,
        default=1,
        help="Number of cameras"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for benchmark results"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="Number of image writer threads per camera"
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=0,
        help="Number of image writer processes"
    )
    parser.add_argument(
        "--video-encoding-batch-size",
        type=int,
        default=1,
        help="Video encoding batch size"
    )
    parser.add_argument(
        "--async-encoding",
        action="store_true",
        help="Enable asynchronous video encoding"
    )
    parser.add_argument(
        "--video-encoding-workers",
        type=int,
        default=2,
        help="Number of worker threads for async video encoding"
    )
    parser.add_argument(
        "--video-encoding-queue-size",
        type=int,
        default=100,
        help="Maximum queue size for async video encoding"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both sync and async benchmarks for comparison"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    logging.info(f"Starting recording benchmark in {args.mode} mode")
    logging.info(f"Configuration: {args.episodes} episodes, {args.episode_time}s each, {args.fps} FPS")
    
    if args.compare:
        # Run comparison benchmark
        run_comparison_benchmark(
            num_episodes=args.episodes,
            episode_time_s=args.episode_time,
            fps=args.fps,
            num_cameras=args.cameras,
            output_dir=args.output_dir,
            video_encoding_workers=args.video_encoding_workers
        )
    elif args.mode == "synthetic":
        run_synthetic_benchmark(
            num_episodes=args.episodes,
            episode_time_s=args.episode_time,
            fps=args.fps,
            num_cameras=args.cameras,
            output_dir=args.output_dir,
            async_encoding=args.async_encoding,
            video_encoding_workers=args.video_encoding_workers
        )
    else:
        # Real recording mode
        cfg = create_mock_config(
            num_episodes=args.episodes,
            episode_time_s=args.episode_time,
            fps=args.fps,
            num_cameras=args.cameras,
            image_writer_threads=args.image_writer_threads,
            image_writer_processes=args.image_writer_processes,
            video_encoding_batch_size=args.video_encoding_batch_size,
            async_video_encoding=args.async_encoding,
            video_encoding_workers=args.video_encoding_workers,
            video_encoding_queue_size=args.video_encoding_queue_size
        )
        
        run_benchmark_recording(cfg)
    
    logging.info("Benchmark completed")


if __name__ == "__main__":
    main() 