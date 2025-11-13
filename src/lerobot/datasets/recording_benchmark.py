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
Recording performance benchmarking utilities.

This module provides tools to measure and analyze the performance of different
steps in the dataset recording process, including frame capture, image writing,
episode saving, and video encoding.
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TimingStats:
    """Statistics for timing measurements."""

    count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    mean_time: float = 0.0
    std_time: float = 0.0
    times: list[float] = field(default_factory=list)

    def add_measurement(self, duration: float) -> None:
        """Add a timing measurement."""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.times.append(duration)

        # Update mean and std
        self.mean_time = self.total_time / self.count
        if self.count > 1:
            self.std_time = np.std(self.times)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "count": self.count,
            "total_time": self.total_time,
            "min_time": self.min_time if self.min_time != float("inf") else 0.0,
            "max_time": self.max_time,
            "mean_time": self.mean_time,
            "std_time": self.std_time,
            "times": self.times,
        }


@dataclass
class RecordingBenchmark:
    """Benchmarking context manager for recording operations."""

    # Timing categories
    frame_capture: TimingStats = field(default_factory=TimingStats)
    frame_processing: TimingStats = field(default_factory=TimingStats)
    image_writing: TimingStats = field(default_factory=TimingStats)
    episode_saving: TimingStats = field(default_factory=TimingStats)
    video_encoding: TimingStats = field(default_factory=TimingStats)
    total_recording: TimingStats = field(default_factory=TimingStats)

    # Episode tracking
    current_episode: int = 0
    episode_start_time: float | None = None
    episode_stats: dict[int, dict[str, Any]] = field(default_factory=dict)

    # Configuration
    save_detailed_times: bool = True
    log_interval: int = 100  # Log stats every N frames

    def __enter__(self):
        """Start benchmarking."""
        self.start_time = time.perf_counter()
        logging.info("Recording benchmark started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End benchmarking and save results."""
        self.end_time = time.perf_counter()
        total_duration = self.end_time - self.start_time
        self.total_recording.add_measurement(total_duration)

        logging.info(f"Recording benchmark completed in {total_duration:.2f}s")
        self.save_results()

    def start_episode(self, episode_index: int) -> None:
        """Start timing a new episode."""
        self.current_episode = episode_index
        self.episode_start_time = time.perf_counter()
        self.episode_stats[episode_index] = {
            "start_time": self.episode_start_time,
            "frame_count": 0,
            "timings": defaultdict(list),
        }
        logging.info(f"Started episode {episode_index}")

    def end_episode(self, episode_index: int) -> None:
        """End timing for current episode."""
        if self.episode_start_time is not None:
            episode_duration = time.perf_counter() - self.episode_start_time
            self.episode_stats[episode_index]["end_time"] = time.perf_counter()
            self.episode_stats[episode_index]["duration"] = episode_duration
            self.episode_stats[episode_index]["fps"] = (
                self.episode_stats[episode_index]["frame_count"] / episode_duration
            )
            logging.info(f"Episode {episode_index} completed in {episode_duration:.2f}s")

    def time_frame_capture(self, func):
        """Decorator to time frame capture operations."""

        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time

            self.frame_capture.add_measurement(duration)
            if self.save_detailed_times:
                self.episode_stats[self.current_episode]["timings"]["frame_capture"].append(duration)

            return result

        return wrapper

    def time_frame_processing(self, func):
        """Decorator to time frame processing operations."""

        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time

            self.frame_processing.add_measurement(duration)
            if self.save_detailed_times:
                self.episode_stats[self.current_episode]["timings"]["frame_processing"].append(duration)

            return result

        return wrapper

    def time_image_writing(self, func):
        """Decorator to time image writing operations."""

        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time

            self.image_writing.add_measurement(duration)
            if self.save_detailed_times:
                self.episode_stats[self.current_episode]["timings"]["image_writing"].append(duration)

            return result

        return wrapper

    def time_episode_saving(self, func):
        """Decorator to time episode saving operations."""

        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time

            self.episode_saving.add_measurement(duration)
            if self.save_detailed_times:
                self.episode_stats[self.current_episode]["timings"]["episode_saving"].append(duration)

            return result

        return wrapper

    def time_video_encoding(self, func):
        """Decorator to time video encoding operations."""

        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time

            self.video_encoding.add_measurement(duration)
            if self.save_detailed_times:
                self.episode_stats[self.current_episode]["timings"]["video_encoding"].append(duration)

            return result

        return wrapper

    def increment_frame_count(self) -> None:
        """Increment frame count for current episode."""
        self.episode_stats[self.current_episode]["frame_count"] += 1

    def log_stats(self, force: bool = False) -> None:
        """Log current statistics."""
        frame_count = self.episode_stats[self.current_episode]["frame_count"]

        if force or frame_count % self.log_interval == 0:
            logging.info(f"Episode {self.current_episode}, Frame {frame_count}:")
            logging.info(
                f"  Frame capture: {self.frame_capture.mean_time * 1000:.2f}ms ± {self.frame_capture.std_time * 1000:.2f}ms"
            )
            logging.info(
                f"  Frame processing: {self.frame_processing.mean_time * 1000:.2f}ms ± {self.frame_processing.std_time * 1000:.2f}ms"
            )
            logging.info(
                f"  Image writing: {self.image_writing.mean_time * 1000:.2f}ms ± {self.image_writing.std_time * 1000:.2f}ms"
            )

    def save_results(self, output_path: Path | None = None) -> None:
        """Save benchmark results to JSON file."""
        if output_path is None:
            output_path = Path(f"recording_benchmark_{int(time.time())}.json")

        total_frames = sum(ep["frame_count"] for ep in self.episode_stats.values())
        total_duration = self.total_recording.total_time
        average_fps = total_frames / total_duration if total_duration > 0 else 0.0

        results = {
            "summary": {
                "total_episodes": len(self.episode_stats),
                "total_frames": total_frames,
                "total_duration": total_duration,
                "average_fps": average_fps,
            },
            "timing_stats": {
                "frame_capture": self.frame_capture.to_dict(),
                "frame_processing": self.frame_processing.to_dict(),
                "image_writing": self.image_writing.to_dict(),
                "episode_saving": self.episode_saving.to_dict(),
                "video_encoding": self.video_encoding.to_dict(),
                "total_recording": self.total_recording.to_dict(),
            },
            "episode_stats": self.episode_stats,
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logging.info(f"Benchmark results saved to {output_path}")

    def print_summary(self) -> None:
        """Print a summary of benchmark results."""
        print("\n" + "=" * 60)
        print("RECORDING BENCHMARK SUMMARY")
        print("=" * 60)

        total_frames = sum(ep["frame_count"] for ep in self.episode_stats.values())
        total_duration = self.total_recording.total_time

        print(f"Total Episodes: {len(self.episode_stats)}")
        print(f"Total Frames: {total_frames}")
        print(f"Total Duration: {total_duration:.2f}s")
        print(
            f"Average FPS: {total_frames / total_duration:.2f}" if total_duration > 0 else "Average FPS: 0.00"
        )
        print()

        print("TIMING BREAKDOWN:")
        print("-" * 40)
        categories = [
            ("Frame Capture", self.frame_capture),
            ("Frame Processing", self.frame_processing),
            ("Image Writing", self.image_writing),
            ("Episode Saving", self.episode_saving),
            ("Video Encoding", self.video_encoding),
        ]

        for name, stats in categories:
            if stats.count > 0:
                percentage = (stats.total_time / total_duration) * 100 if total_duration > 0 else 0.0
                print(
                    f"{name:20} {stats.mean_time * 1000:8.2f}ms ± {stats.std_time * 1000:6.2f}ms ({percentage:5.1f}%)"
                )

        print()
        print("BOTTLENECK ANALYSIS:")
        print("-" * 40)
        categories_with_data = [(name, stats) for name, stats in categories if stats.count > 0]
        if categories_with_data:
            max_time = max(stats.total_time for _, stats in categories_with_data)
            for name, stats in categories_with_data:
                percentage = (stats.total_time / max_time) * 100
                print(f"{name:20} {stats.total_time:8.2f}s ({percentage:5.1f}% of slowest)")
        else:
            print("No timing data available")


class BenchmarkDecorator:
    """Decorator class for easy benchmarking of functions."""

    def __init__(self, benchmark: RecordingBenchmark, category: str):
        self.benchmark = benchmark
        self.category = category

    def __call__(self, func):
        if self.category == "frame_capture":
            return self.benchmark.time_frame_capture(func)
        elif self.category == "frame_processing":
            return self.benchmark.time_frame_processing(func)
        elif self.category == "image_writing":
            return self.benchmark.time_image_writing(func)
        elif self.category == "episode_saving":
            return self.benchmark.time_episode_saving(func)
        elif self.category == "video_encoding":
            return self.benchmark.time_video_encoding(func)
        else:
            raise ValueError(f"Unknown category: {self.category}")


def benchmark_recording(func):
    """Decorator to benchmark a recording function."""

    def wrapper(*args, **kwargs):
        with RecordingBenchmark() as benchmark:
            # Store benchmark in kwargs for access within the function
            kwargs["benchmark"] = benchmark
            result = func(*args, **kwargs)
            benchmark.print_summary()
            return result

    return wrapper
