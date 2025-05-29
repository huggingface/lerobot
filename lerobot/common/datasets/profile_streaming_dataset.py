#!/usr/bin/env python
"""
Script to profile the StreamingLeRobotDataset iteration speed.
Run with: python -m lerobot.common.datasets.profile_streaming_dataset
"""

import argparse
import time

import numpy as np
from line_profiler import LineProfiler
from tqdm import tqdm

from lerobot.common.datasets.streaming_dataset import StreamingLeRobotDataset


def timing_stats(times):
    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "median": np.median(times),
    }


def measure_iteration_times(dataset, num_samples=100, num_runs=5, warmup_iters=2):
    """
    Measure individual iteration times and compute statistics.

    Args:
        dataset: The dataset to iterate over.
        num_samples (int): Number of samples to iterate through per run.
        num_runs (int): Number of timing runs to perform.
        warmup_iters (int): Number of warmup iterations before timing.

    Returns:
        dict: Statistics including mean, std, min, max times per sample.
    """
    print(f"Measuring iteration times over {num_runs} runs of {num_samples} samples each...")
    print(f"Using {warmup_iters} warmup iterations per run")

    all_sample_times = []
    run_times = []

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}...")
        run_start = time.time()

        # Warmup phase
        print(f"  Performing {warmup_iters} warmup iterations...")
        warmup_iterator = iter(dataset)
        warmup_times = []
        try:
            for _ in range(warmup_iters):
                start_time = time.time()
                next(warmup_iterator)
                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000
                warmup_times.append(elapsed_ms)
        except StopIteration:
            print("  Warning: Iterator exhausted during warmup")

        # timing phase
        iterator = iter(dataset)
        sample_times = []

        for _ in tqdm(range(num_samples)):
            start_time = time.time()
            next(iterator)
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            sample_times.append(elapsed_ms)

        run_end = time.time()
        run_times.append(run_end - run_start)
        all_sample_times.extend(sample_times)

    # Compute statistics
    sample_times_array = np.array(all_sample_times)
    warmup_times_array = np.array(warmup_times)
    run_times_array = np.array(run_times)

    stats = {
        "sample_times_ms": timing_stats(sample_times_array),
        "warmup_times_ms": timing_stats(warmup_times_array),
        "run_times_s": timing_stats(run_times_array),
        "samples_per_second": num_samples / np.mean(run_times_array),
    }

    return stats


def print_timing_stats(stats):
    """Print timing statistics in a readable format."""
    print("\n" + "=" * 60)
    print("TIMING STATISTICS")
    print("=" * 60)

    warmup_stats = stats["warmup_times_ms"]
    print("Warmup timing (ms):")
    print(f"  Mean: {warmup_stats['mean']:.2f} ± {warmup_stats['std']:.2f}")
    print(f"  Median: {warmup_stats['median']:.2f}")
    print(f"  Range: [{warmup_stats['min']:.2f}, {warmup_stats['max']:.2f}]")

    sample_stats = stats["sample_times_ms"]
    print("\nPer-sample timing (ms):")
    print(f"  Mean: {sample_stats['mean']:.2f} ± {sample_stats['std']:.2f}")
    print(f"  Median: {sample_stats['median']:.2f}")
    print(f"  Range: [{sample_stats['min']:.2f}, {sample_stats['max']:.2f}]")

    run_stats = stats["run_times_s"]
    print("\nPer-run timing (seconds):")
    print(f"  Mean: {run_stats['mean']:.2f} ± {run_stats['std']:.2f}")
    print(f"  Range: [{run_stats['min']:.2f}, {run_stats['max']:.2f}]")

    print("\nThroughput:")
    print(f"  Samples/second: {stats['samples_per_second']:.2f}")
    print("=" * 60)


def _time_iterations(dataset, num_samples, num_runs, warmup_iters, stats_file_path):
    # Measure iteration times with statistics
    timing_stats = measure_iteration_times(dataset, num_samples, num_runs, warmup_iters)
    print_timing_stats(timing_stats)

    # Save results to a file
    with open(stats_file_path, "w") as f:
        f.write("TIMING STATISTICS\n")
        f.write("=" * 60 + "\n")
        warmup_stats = timing_stats["warmup_times_ms"]
        f.write("Warmup timing (ms):\n")
        f.write(f"  Mean: {warmup_stats['mean']:.2f} ± {warmup_stats['std']:.2f}\n")
        f.write(f"  Median: {warmup_stats['median']:.2f}\n")
        f.write(f"  Range: [{warmup_stats['min']:.2f}, {warmup_stats['max']:.2f}]\n\n")

        sample_stats = timing_stats["sample_times_ms"]
        f.write("Per-sample timing (ms):\n")
        f.write(f"  Mean: {sample_stats['mean']:.2f} ± {sample_stats['std']:.2f}\n")
        f.write(f"  Median: {sample_stats['median']:.2f}\n")
        f.write(f"  Range: [{sample_stats['min']:.2f}, {sample_stats['max']:.2f}]\n\n")

        run_stats = timing_stats["run_times_s"]
        f.write("Per-run timing (seconds):\n")
        f.write(f"  Mean: {run_stats['mean']:.2f} ± {run_stats['std']:.2f}\n")
        f.write(f"  Range: [{run_stats['min']:.2f}, {run_stats['max']:.2f}]\n\n")

        throughput_stats = timing_stats["samples_per_second"]
        f.write("Throughput:\n")
        f.write(f"  Samples/second: {throughput_stats:.2f}\n")
        f.write("=" * 60 + "\n\n")

        f.write("DETAILED LINE PROFILING RESULTS\n")
        f.write("=" * 60 + "\n")

    print(f"\nDetailed profiling results saved to {stats_file_path}")


def _profile_iteration(dataset, num_samples, stats_file_path):
    # Create a line profiler instance for detailed profiling
    profiler = LineProfiler()

    # Add functions to profile
    profiler.add_function(dataset.__iter__)
    profiler.add_function(dataset.make_frame)
    profiler.add_function(dataset._make_backtrackable_dataset)

    # Profile the iteration

    # Define the function to profile
    def iterate_dataset(ds, n):
        # Iterating without warmup for line profiling
        iterator = iter(ds)
        start_time = time.time()
        for _ in range(n):
            next(iterator)
        end_time = time.time()
        return end_time - start_time

    # Add the function to the profiler
    profiler.add_function(iterate_dataset)

    # Run the profiled function
    profiler.runcall(iterate_dataset, dataset, num_samples)

    with open(stats_file_path, "a") as f:
        profiler.print_stats(stream=f)


def _analyze_randomness(dataset, num_samples, stats_file_path):
    """
    Analyze the randomness of dataset iteration by checking correlation between
    iteration index and frame index.

    Args:
        dataset: The dataset to analyze.
        num_samples: Number of samples to use for analysis.
        stats_file_path: Path to save the analysis results.
    """
    print("\nAnalyzing randomness of dataset iteration...")

    # Collect iteration index and frame index pairs
    points = []
    iterator = iter(dataset)

    for i in tqdm(range(num_samples)):
        try:
            frame = next(iterator)
            points.append(np.array([i, frame["frame_index"]]))
        except (StopIteration, KeyError) as e:
            if isinstance(e, StopIteration):
                print(f"  Warning: Iterator exhausted after {i} samples")
            else:
                print(f"  Warning: frame_index not found in sample {i}")
            break

    # Compute correlation between iteration index and frame index
    points_array = np.array(points)
    correlation = np.corrcoef(points_array[:, 0], points_array[:, 1])[0, 1]

    # Save results to file
    with open(stats_file_path, "a") as f:
        f.write("\nRANDOMNESS ANALYSIS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Correlation between iteration index and frame index: {correlation:.4f}\n")
        f.write("(Correlation close to 0 indicates more random access pattern)\n")
        f.write("(Correlation close to 1 indicates sequential access pattern)\n")
        f.write("=" * 60 + "\n")

    print(f"Correlation between iteration index and frame index: {correlation:.4f}")
    print("(Correlation close to 0 indicates more random access pattern)")
    print("(Correlation close to 1 indicates sequential access pattern)")


def profile_dataset(
    repo_id, num_samples=100, buffer_size=1000, max_num_shards=16, seed=42, num_runs=3, warmup_iters=10
):
    """
    Profile the streaming dataset iteration speed.

    Args:
        repo_id (str): HuggingFace repository ID for the dataset.
        num_samples (int): Number of samples to iterate through.
        buffer_size (int): Buffer size for the dataset.
        max_num_shards (int): Number of shards to use.
        seed (int): Random seed for reproducibility.
        num_runs (int): Number of timing runs to perform.
        warmup_iters (int): Number of warmup iterations before timing.
    """
    stats_file_path = "streaming_dataset_profile.txt"

    print(f"Creating dataset from {repo_id} with buffer_size={buffer_size}, max_num_shards={max_num_shards}")
    camera_key = "observation.images.cam_right_wrist"
    fps = 50

    delta_timestamps = {
        # loads 4 images: 1 second before current frame, 500 ms before, 200 ms before, and current frame
        camera_key: [-1, -0.5, -0.20, 0],
        # loads 6 state vectors: 1.5 seconds before, 1 second before, ... 200 ms, 100 ms, and current frame
        "observation.state": [-1.5, -1, -0.5, -0.20, -0.10, 0],
        # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
        "action": [t / fps for t in range(64)],
    }

    dataset = StreamingLeRobotDataset(
        repo_id=repo_id,
        buffer_size=buffer_size,
        max_num_shards=max_num_shards,
        seed=seed,
        delta_timestamps=delta_timestamps,
    )

    _time_iterations(dataset, num_samples, num_runs, warmup_iters, stats_file_path)
    _profile_iteration(dataset, num_samples, stats_file_path)
    _analyze_randomness(dataset, num_samples, stats_file_path)


def main():
    parser = argparse.ArgumentParser(description="Profile StreamingLeRobotDataset iteration speed")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/aloha_mobile_cabinet",
        help="HuggingFace repository ID for the dataset",
    )
    parser.add_argument("--num-samples", type=int, default=2_000, help="Number of samples to iterate through")
    parser.add_argument("--buffer-size", type=int, default=1000, help="Buffer size for the dataset")
    parser.add_argument("--max-num-shards", type=int, default=1, help="Number of shards to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--num-runs", type=int, default=10, help="Number of timing runs to perform for statistics"
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=1, help="Number of warmup iterations before timing"
    )

    args = parser.parse_args()

    profile_dataset(
        repo_id=args.repo_id,
        num_samples=args.num_samples,
        buffer_size=args.buffer_size,
        max_num_shards=args.max_num_shards,
        seed=args.seed,
        num_runs=args.num_runs,
        warmup_iters=args.warmup_iters,
    )


if __name__ == "__main__":
    main()
