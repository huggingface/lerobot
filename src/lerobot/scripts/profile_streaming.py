import argparse
import datetime
import os
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset


def profile_throughput_indexed(
    dataset: LeRobotDataset, num_samples: int, warmup_iters: int = 3
) -> np.ndarray:
    """Measure per-item access time on an indexable LeRobotDataset.

    Accesses dataset[i % len(dataset)] for ``num_samples`` iterations, with an initial warmup.
    """
    next_times = np.zeros(num_samples)
    total = len(dataset)

    # warmup
    for k in range(warmup_iters):
        _ = dataset[k % total]

    for j in tqdm(range(num_samples), desc="Profiling dataset throughput", unit="item"):
        start_time = time.perf_counter()
        _ = dataset[j % total]
        end_time = time.perf_counter()
        next_times[j] = end_time - start_time

    return next_times


def profile_throughput(
    dataset: StreamingLeRobotDataset, num_samples: int, warmup_iters: int = 3
) -> np.ndarray:
    """Measure ``.next()`` call latency on a streaming dataset.

    Performs a configurable warmup. This does not numerically "normalize" times; it simply
    avoids including initialization overhead in the timing window.
    """
    next_times = np.zeros(num_samples)
    iter_dataset = iter(dataset)

    # warmup
    for _ in range(warmup_iters):
        _ = next(iter_dataset)

    for j in tqdm(range(num_samples), desc="Profiling throughput", unit="call"):
        start_time = time.perf_counter()
        _sample = next(iter_dataset)
        end_time = time.perf_counter()
        next_times[j] = end_time - start_time

    return next_times


def profile_init(dataset_factory: Callable[[], StreamingLeRobotDataset], num_runs: int) -> np.ndarray:
    """Measure time-to-first-sample by re-instantiating the dataset ``num_runs`` times.

    Using a factory avoids unsafe ``deepcopy`` of objects that may own threads or file handles.
    """
    init_times = np.zeros(num_runs)
    for i in tqdm(range(num_runs), desc="Profiling init", unit="run"):
        fresh_dataset = dataset_factory()
        iter_dataset = iter(fresh_dataset)
        start_time = time.perf_counter()
        _ = next(iter_dataset)
        end_time = time.perf_counter()
        init_times[i] = end_time - start_time

    return init_times


def profile_randomness(dataset: StreamingLeRobotDataset, num_samples: int) -> float:
    """Measure how random the sample order is via correlation.

    Returns a Pearson correlation between retrieved frame index and iteration index.
    - ~0: random order
    - ~+1: strictly increasing (in-order)
    - ~-1: strictly decreasing (reverse order)
    """
    frame_indices = np.zeros(num_samples, dtype=float)
    iter_indices = np.arange(num_samples, dtype=float)

    iter_dataset = iter(dataset)

    for i in tqdm(range(num_samples), desc="Profiling randomness", unit="sample"):
        sample = next(iter_dataset)
        if "index" in sample:
            frame_idx_value = sample["index"]
        elif "frame_index" in sample:
            frame_idx_value = sample["frame_index"]
        else:
            raise KeyError("Sample is missing 'index' (or 'frame_index') required to compute randomness.")
        frame_indices[i] = float(frame_idx_value)

    # Guard against degenerate cases
    if num_samples < 2 or np.std(frame_indices) == 0 or np.std(iter_indices) == 0:
        return np.nan, None

    correlation = float(np.corrcoef(frame_indices, iter_indices)[0, 1])
    return correlation


def profile_streaming_dataset(
    repo_id: str,
    delta_timestamps: dict[str, list[float]] | None = None,
    num_samples: int = 100,
    warmup_iters: int = 10,
    buffer_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run init, throughput, and randomness profiles on a StreamingLeRobotDataset."""

    def dataset_factory() -> StreamingLeRobotDataset:
        return StreamingLeRobotDataset(repo_id, delta_timestamps=delta_timestamps, buffer_size=buffer_size)

    # Measure init by repeated instantiation
    init_times = profile_init(dataset_factory, num_runs=warmup_iters)

    # Throughput and randomness on a single fresh dataset instance
    dataset = dataset_factory()
    next_times = profile_throughput(dataset, num_samples=num_samples, warmup_iters=warmup_iters)
    correlation = profile_randomness(dataset, num_samples=num_samples)

    return init_times, next_times, correlation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile StreamingLeRobotDataset performance metrics.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/svla_so101_pickplace",
        help="Dataset repo_id to profile.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to measure for throughput/randomness.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Number of iterations for init and throughput warmup.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1000,
        help="Buffer size for the streaming dataset.",
    )
    parser.add_argument(
        "--with-delta-timestamps",
        action="store_true",
        help="Profile with delta timestamps.",
    )
    parser.add_argument(
        "--compare-with-local",
        action="store_true",
        help="Also profile local LeRobotDataset throughput for comparison.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("outputs", "benchmarks"),
        help="Directory to write CSVs/PNGs to.",
    )

    args = parser.parse_args()

    delta_timestamps = (
        None
        if not args.with_delta_timestamps
        else {
            "observation.state": [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0],
            "action": [
                -0.1,
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
            ],
        }
    )

    init_times, next_times, correlation = profile_streaming_dataset(
        repo_id=args.repo_id,
        delta_timestamps=delta_timestamps,
        num_samples=args.num_samples,
        warmup_iters=args.warmup_iters,
        buffer_size=args.buffer_size,
    )

    os.makedirs(args.outdir, exist_ok=True)

    repo_id_str = args.repo_id.replace("/", "-")
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    name_suffix = f"{repo_id_str}_buf{args.buffer_size}_{date_str}"

    # Visualization disabled by default; figures are not created or saved.

    init_df = pd.DataFrame({"init_times": init_times})
    next_df = pd.DataFrame({"next_times": next_times})
    correlation_df = pd.DataFrame({"correlation": [correlation]})

    init_df.to_csv(os.path.join(args.outdir, f"init_times_{name_suffix}.csv"), index=False)
    next_df.to_csv(os.path.join(args.outdir, f"next_times_{name_suffix}.csv"), index=False)
    correlation_df.to_csv(os.path.join(args.outdir, f"correlation_{name_suffix}.csv"), index=False)

    if args.compare_with_local:
        # Profile local non-streaming dataset throughput for comparison
        local_ds = LeRobotDataset(args.repo_id, delta_timestamps=delta_timestamps)
        local_next_times = profile_throughput_indexed(
            local_ds, num_samples=args.num_samples, warmup_iters=args.warmup_iters
        )
        local_df = pd.DataFrame({"next_times": local_next_times})
        local_df.to_csv(
            os.path.join(args.outdir, f"next_times_local_{repo_id_str}_{date_str}.csv"),
            index=False,
        )
