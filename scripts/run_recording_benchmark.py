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
    output_dir: str = "./benchmark_results"
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
            encoding_time = random.uniform(5.0, 15.0)
            time.sleep(encoding_time)
            benchmark.video_encoding.add_measurement(encoding_time)
            
            benchmark.end_episode(episode)
        
        # Save results
        benchmark.save_results(output_path / f"synthetic_benchmark_{int(time.time())}.json")
        benchmark.print_summary()


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
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    logging.info(f"Starting recording benchmark in {args.mode} mode")
    logging.info(f"Configuration: {args.episodes} episodes, {args.episode_time}s each, {args.fps} FPS")
    
    if args.mode == "synthetic":
        run_synthetic_benchmark(
            num_episodes=args.episodes,
            episode_time_s=args.episode_time,
            fps=args.fps,
            num_cameras=args.cameras,
            output_dir=args.output_dir
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
            video_encoding_batch_size=args.video_encoding_batch_size
        )
        
        run_benchmark_recording(cfg)
    
    logging.info("Benchmark completed")


if __name__ == "__main__":
    main() 