#!/usr/bin/env python
"""Benchmark the timestamp drift produced by the *actual* codebase recording path.

Unlike the simulation in ``tests/datasets/test_video_drift.py``
(``test_round6_accumulates_drift_but_actual_duration_does_not``), this script does not
re-implement any arithmetic. It records episodes through the real
``LeRobotDataset.create / add_frame / save_episode / finalize`` pipeline (PNG -> mp4
encoding + ``concatenate_video_files``), then measures how far the ``from_timestamp`` /
``to_timestamp`` values stored in the episode metadata drift from the PTS actually
decoded from the concatenated video file.

Drift sources exercised here:
- float accumulation of ``to_timestamp = from_timestamp + ep_duration``
- per-episode ``get_video_duration_in_s`` vs the frame's real PTS after concatenation

Run:
    python benchmarks/video/benchmark_video_drift.py
    python benchmarks/video/benchmark_video_drift.py --fps 30 --num-episodes 500
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import av
import numpy as np

from lerobot.datasets.io_utils import load_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import get_video_duration_in_s

VIDEO_KEY = "observation.images.laptop"


def _decode_all_frame_pts(video_path: Path | str) -> list[float]:
    """Return the PTS (seconds) of every frame in decode order, in a single pass."""
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        return [float(frame.pts * time_base) for frame in container.decode(stream)]


def _record_dataset(
    root: Path,
    fps: int,
    frames_per_episode: list[int],
    streaming: bool,
) -> LeRobotDataset:
    features = {
        VIDEO_KEY: {"dtype": "video", "shape": (3, 64, 96), "names": ["channels", "height", "width"]},
        "state": {"dtype": "float32", "shape": (2,), "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id="benchmark/video_drift",
        fps=fps,
        features=features,
        root=root,
        streaming_encoding=streaming,
        # Force every episode into a single concatenated video file.
        video_files_size_in_mb=10_000,
    )
    rng = np.random.RandomState(0)
    for n_frames in frames_per_episode:
        for _ in range(n_frames):
            dataset.add_frame(
                {
                    VIDEO_KEY: rng.randint(0, 256, (64, 96, 3), dtype=np.uint8),
                    "state": rng.randn(2).astype(np.float32),
                    "task": "benchmark",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return dataset


def _measure_drift(dataset: LeRobotDataset, fps: int, frames_per_episode: list[int]) -> dict:
    episodes = load_episodes(dataset.root)
    num_episodes = len(frames_per_episode)

    chunk_idx = episodes[0][f"videos/{VIDEO_KEY}/chunk_index"]
    file_idx = episodes[0][f"videos/{VIDEO_KEY}/file_index"]
    video_path = dataset.root / dataset.meta.video_path.format(
        video_key=VIDEO_KEY, chunk_index=chunk_idx, file_index=file_idx
    )

    actual_pts = _decode_all_frame_pts(video_path)
    actual_duration = get_video_duration_in_s(video_path)

    boundary_drifts_s: list[float] = []
    cumulative = 0
    single_file = True
    for ep_idx in range(num_episodes):
        # If episodes spilled into multiple files, boundary indexing no longer holds.
        if (
            episodes[ep_idx][f"videos/{VIDEO_KEY}/chunk_index"] != chunk_idx
            or episodes[ep_idx][f"videos/{VIDEO_KEY}/file_index"] != file_idx
        ):
            single_file = False
            break

        if cumulative > 0:
            from_ts = float(episodes[ep_idx][f"videos/{VIDEO_KEY}/from_timestamp"])
            boundary_drifts_s.append(abs(from_ts - actual_pts[cumulative]))
        cumulative += frames_per_episode[ep_idx]

    last_to_ts = float(episodes[num_episodes - 1][f"videos/{VIDEO_KEY}/to_timestamp"])
    duration_drift_s = abs(last_to_ts - actual_duration)

    drifts = np.array(boundary_drifts_s) if boundary_drifts_s else np.array([0.0])
    half_frame_s = 0.5 / fps
    return {
        "num_episodes": num_episodes,
        "num_boundaries": len(boundary_drifts_s),
        "single_file": single_file,
        "total_frames": cumulative,
        "max_drift_s": float(drifts.max()),
        "mean_drift_s": float(drifts.mean()),
        "p99_drift_s": float(np.percentile(drifts, 99)),
        "max_drift_frames": float(drifts.max() * fps),
        "duration_drift_s": duration_drift_s,
        "half_frame_s": half_frame_s,
        "exceeds_half_frame": bool(drifts.max() >= half_frame_s),
    }


def run_config(fps: int, num_episodes: int, min_frames: int, max_frames: int, seed: int, streaming: bool):
    rng = np.random.RandomState(seed)
    frames_per_episode = rng.randint(min_frames, max_frames + 1, size=num_episodes).tolist()
    tmp = Path(tempfile.mkdtemp(prefix="lerobot_drift_bench_"))
    try:
        dataset = _record_dataset(tmp / "dataset", fps, frames_per_episode, streaming)
        return _measure_drift(dataset, fps, frames_per_episode)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _print_report(label: str, r: dict) -> None:
    note = "" if r["single_file"] else "  (truncated: episodes spilled to multiple files)"
    print(f"\n=== {label}{note} ===")
    print(f"  episodes / boundaries : {r['num_episodes']} / {r['num_boundaries']}")
    print(f"  total frames          : {r['total_frames']}")
    print(f"  max boundary drift    : {r['max_drift_s']:.3e} s  ({r['max_drift_frames']:.4f} frames)")
    print(f"  mean boundary drift   : {r['mean_drift_s']:.3e} s")
    print(f"  p99 boundary drift    : {r['p99_drift_s']:.3e} s")
    print(f"  total-duration drift  : {r['duration_drift_s']:.3e} s")
    print(f"  half-frame threshold  : {r['half_frame_s']:.3e} s")
    print(f"  exceeds half-frame    : {'YES <-- FAIL' if r['exceeds_half_frame'] else 'no'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=int, default=None, help="Override fps (default: sweep presets).")
    parser.add_argument("--num-episodes", type=int, default=None, help="Override episode count.")
    parser.add_argument("--min-frames", type=int, default=7)
    parser.add_argument("--max-frames", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming", action="store_true", help="Use the streaming encoder path.")
    args = parser.parse_args()

    if args.fps is not None or args.num_episodes is not None:
        fps = args.fps or 30
        num_episodes = args.num_episodes or 50
        configs = [(fps, num_episodes)]
    else:
        configs = [(30, 50), (30, 200), (60, 200), (50, 200)]

    for fps, num_episodes in configs:
        r = run_config(fps, num_episodes, args.min_frames, args.max_frames, args.seed, args.streaming)
        label = f"fps={fps}, episodes={num_episodes}, streaming={args.streaming}"
        _print_report(label, r)


if __name__ == "__main__":
    main()
