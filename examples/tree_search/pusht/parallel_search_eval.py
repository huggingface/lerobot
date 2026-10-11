#!/usr/bin/env python
"""Run PushT tree-search episodes in parallel by launching `search_eval.py`.

This is intentionally an external wrapper. It does not import or modify the
core evaluator. Each episode runs in its own subprocess with its own env, policy
instance, RNG, and output directory. The wrapper collates per-episode
`eval_info.json` files into a single aggregate file.

Example:

```bash
uv run python examples/tree_search/pusht/parallel_search_eval.py \
  --policy.path=aadarshram/act_pusht \
  --policy.device=cuda \
  --policy.use_amp=false \
  --episodes=10 \
  --episode-workers=2 \
  --seed=0 \
  --depth=3 \
  --num-candidates=40 \
  --chunk-size=3 \
  --execute-steps=10 \
  --render-videos=30 \
  --dump_frames=true \
  --plot_policy_trace \
  --dump_search_images=true \
  --video_overlay=false
```
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("pusht_parallel_search")


@dataclass(frozen=True)
class ParallelConfig:
    episodes: int
    episode_workers: int
    seed: int | None
    output_dir: Path
    render_videos: int
    dump_frames: bool
    plot_policy_trace: bool
    dump_search_images: bool
    video_overlay: bool
    script: Path


@dataclass(frozen=True)
class EpisodeJob:
    episode_index: int
    seed: int | None
    output_dir: Path
    render_videos: int


@dataclass
class EpisodeRun:
    episode_index: int
    seed: int | None
    output_dir: str
    returncode: int
    stdout_path: str
    stderr_path: str
    metrics: dict[str, Any] | None


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def parse_args() -> tuple[ParallelConfig, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch independent PushT search_eval.py episodes in parallel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", "--eval.n_episodes", dest="episodes", type=int, default=10)
    parser.add_argument("--episode-workers", "--episode_workers", dest="episode_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=Path("outputs/tree_search/pusht_parallel"),
    )
    parser.add_argument(
        "--render-videos",
        "--render_videos",
        dest="render_videos",
        type=int,
        default=1,
        help="Total number of episode videos to render across all workers.",
    )
    parser.add_argument(
        "--dump-frames",
        "--dump_frames",
        dest="dump_frames",
        type=str_to_bool,
        default=False,
        help="Forward --dump-frames to each child evaluator.",
    )
    parser.add_argument(
        "--plot-policy-trace",
        "--plot_policy_trace",
        dest="plot_policy_trace",
        action="store_true",
        help="Forward --plot-policy-trace to each child evaluator.",
    )
    parser.add_argument(
        "--dump-search-images",
        "--dump_search_images",
        dest="dump_search_images",
        type=str_to_bool,
        default=False,
        help="Forward --dump-search-images to each child evaluator.",
    )
    parser.add_argument(
        "--video-overlay",
        "--video_overlay",
        dest="video_overlay",
        type=str_to_bool,
        default=True,
        help="Forward --video-overlay to each child evaluator.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).with_name("search_eval.py"),
        help="Path to the single-process PushT evaluator.",
    )

    cfg_args, search_args = parser.parse_known_args()
    if cfg_args.episodes <= 0:
        parser.error("--episodes must be positive.")
    if cfg_args.episode_workers <= 0:
        parser.error("--episode-workers must be positive.")
    if cfg_args.render_videos < 0:
        parser.error("--render-videos must be non-negative.")

    return ParallelConfig(**vars(cfg_args)), search_args


def episode_seed(base_seed: int | None, episode_index: int) -> int | None:
    return None if base_seed is None else base_seed + episode_index


def run_episode_subprocess(
    *,
    job: EpisodeJob,
    cfg: ParallelConfig,
    search_args: list[str],
) -> EpisodeRun:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = job.output_dir / "stdout.log"
    stderr_path = job.output_dir / "stderr.log"

    cmd = [
        sys.executable,
        str(cfg.script),
        *search_args,
        "--episodes=1",
        f"--output-dir={job.output_dir}",
        f"--render-videos={job.render_videos}",
        f"--dump-frames={str(cfg.dump_frames).lower()}",
        f"--dump-search-images={str(cfg.dump_search_images).lower()}",
        f"--video-overlay={str(cfg.video_overlay).lower()}",
    ]
    if cfg.plot_policy_trace:
        cmd.append("--plot-policy-trace")
    if job.seed is not None:
        cmd.append(f"--seed={job.seed}")

    started_at = time.time()
    LOGGER.info(
        "episode=%s seed=%s started render_videos=%s output_dir=%s",
        job.episode_index,
        job.seed,
        job.render_videos,
        job.output_dir,
    )
    with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
        result = subprocess.run(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            check=False,
        )

    metrics = None
    metrics_path = job.output_dir / "eval_info.json"
    if metrics_path.exists():
        with metrics_path.open() as f:
            metrics = json.load(f)

    LOGGER.info(
        "episode=%s finished returncode=%s elapsed_s=%.1f",
        job.episode_index,
        result.returncode,
        time.time() - started_at,
    )
    return EpisodeRun(
        episode_index=job.episode_index,
        seed=job.seed,
        output_dir=str(job.output_dir),
        returncode=result.returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        metrics=metrics,
    )


def collate_runs(cfg: ParallelConfig, runs: list[EpisodeRun], elapsed_s: float) -> dict[str, Any]:
    failed = [run for run in runs if run.returncode != 0 or run.metrics is None]
    successful = [run for run in runs if run.returncode == 0 and run.metrics is not None]

    per_episode: list[dict[str, Any]] = []
    video_paths: list[str] = []
    for run in sorted(successful, key=lambda item: item.episode_index):
        assert run.metrics is not None
        child_per_episode = run.metrics.get("per_episode", [])
        item = dict(child_per_episode[0]) if child_per_episode else {}
        item["episode_index"] = run.episode_index
        item["seed"] = run.seed
        item["output_dir"] = run.output_dir
        per_episode.append(item)

        child_videos = run.metrics.get("aggregated", {}).get("video_paths", [])
        video_paths.extend(str(path) for path in child_videos)

    sum_rewards = [float(item.get("sum_reward", 0.0)) for item in per_episode]
    max_rewards = [float(item.get("max_reward", 0.0)) for item in per_episode]
    successes = [bool(item.get("success", False)) for item in per_episode]
    alternative_selection_count = sum(int(item.get("alternative_selection_count", 0)) for item in per_episode)
    selection_count = sum(int(item.get("selection_count", 0)) for item in per_episode)

    aggregate = {
        "avg_sum_reward": float(np.mean(sum_rewards)) if sum_rewards else 0.0,
        "avg_max_reward": float(np.mean(max_rewards)) if max_rewards else 0.0,
        "pc_success": float(np.mean(successes) * 100.0) if successes else 0.0,
        "asr": (
            float(alternative_selection_count / selection_count * 100.0) if selection_count else 0.0
        ),
        "alternative_selection_count": alternative_selection_count,
        "selection_count": selection_count,
        "n_episodes": len(per_episode),
        "eval_s": elapsed_s,
        "eval_ep_s": elapsed_s / len(per_episode) if per_episode else 0.0,
        "video_paths": video_paths,
    }

    return {
        "config": {**asdict(cfg), "output_dir": str(cfg.output_dir), "script": str(cfg.script)},
        "aggregated": aggregate,
        "per_episode": per_episode,
        "runs": [asdict(run) for run in sorted(runs, key=lambda item: item.episode_index)],
        "failed_runs": [asdict(run) for run in sorted(failed, key=lambda item: item.episode_index)],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    cfg, search_args = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        EpisodeJob(
            episode_index=episode_index,
            seed=episode_seed(cfg.seed, episode_index),
            output_dir=cfg.output_dir / "episodes" / f"episode_{episode_index:03d}",
            render_videos=1 if episode_index < cfg.render_videos else 0,
        )
        for episode_index in range(cfg.episodes)
    ]

    LOGGER.info(
        "Launching %s episodes with %s workers. Extra search args: %s",
        cfg.episodes,
        cfg.episode_workers,
        search_args,
    )
    started_at = time.time()
    runs: list[EpisodeRun] = []
    with cf.ThreadPoolExecutor(max_workers=cfg.episode_workers) as executor:
        futures = [
            executor.submit(run_episode_subprocess, job=job, cfg=cfg, search_args=search_args)
            for job in jobs
        ]
        for future in cf.as_completed(futures):
            runs.append(future.result())

    payload = collate_runs(cfg, runs, elapsed_s=time.time() - started_at)
    metrics_path = cfg.output_dir / "eval_info.json"
    with metrics_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload["aggregated"], indent=2))
    LOGGER.info("Saved collated metrics to %s", metrics_path)

    failed_runs = payload["failed_runs"]
    if failed_runs:
        LOGGER.error("%s episode subprocess(es) failed. See per-episode stderr.log files.", len(failed_runs))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
