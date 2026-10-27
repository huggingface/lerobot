#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Generate an annotated SO-101 pick-and-place dataset in simulation.

Shards are generated in parallel, one process and one LeRobotDataset each, then merged:

    python -m examples.so101_datagen.generate \\
        --output_dir=outputs/so101_datagen --repo_id=<hf_user>/so101_datagen \\
        --num_shards=32 --episodes_per_shard=50 --num_workers=14

Rerunning the same command resumes: finished shards are kept, interrupted ones start over.
"""

import json
import logging
import multiprocessing
import time
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat

from lerobot.annotations.steerable_pipeline.executor import Executor
from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.utils.utils import init_logging

from .assets import fetch_environments, load_meshes
from .config import GenerationConfig
from .so101 import menagerie_dir
from .worker import EPISODES_FILE, REPORT_FILE, EncodingConfig, ShardJob, resolve_encoding, run_shard


@dataclass
class GenerateConfig:
    repo_id: str = "local/so101_datagen"
    output_dir: Path = Path("outputs/so101_datagen")
    num_shards: int = 4
    episodes_per_shard: int = 10
    # Worker processes; each renders, simulates and encodes one shard at a time.
    num_workers: int = 4
    seed: int = 0
    merge: bool = True
    push_to_hub: bool = False
    private: bool = True
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)


@parser.wrap()
def generate(cfg: GenerateConfig) -> None:
    init_logging()
    logging.info(pformat(cfg))
    cfg.encoding = resolve_encoding(cfg.encoding, cfg.generation.height, cfg.generation.width)
    logging.info(
        f"Encoding with {cfg.encoding.vcodec} (crf {cfg.encoding.crf}, preset {cfg.encoding.preset})"
    )
    shards_dir = cfg.output_dir / "shards"
    jobs = [
        ShardJob(
            index=i,
            root=shards_dir / f"shard_{i:04d}",
            repo_id=f"{cfg.repo_id}_shard_{i:04d}",
            episodes=cfg.episodes_per_shard,
            seed=cfg.seed,
            generation=cfg.generation,
            encoding=cfg.encoding,
        )
        for i in range(cfg.num_shards)
    ]
    started = time.perf_counter()
    reports = []
    # Fetched once here, rather than by every worker at the same time.
    menagerie_dir()
    fetch_environments()
    if cfg.generation.objects.mesh_prob > 0:
        load_meshes()
    # Spawn: every worker builds its own GL context and MuJoCo state from scratch.
    with multiprocessing.get_context("spawn").Pool(cfg.num_workers, maxtasksperchild=1) as pool:
        for report in pool.imap_unordered(run_shard, jobs):
            reports.append(report)
            logging.info(
                f"shard {report['shard']} done ({len(reports)}/{len(jobs)}): {report['episodes']} episodes, "
                f"{report['frames'] / report['seconds']:.0f} frames/s, failures {report['failures']}"
            )
    elapsed = time.perf_counter() - started
    summary = summarize(reports, elapsed)
    logging.info(pformat(summary))
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / REPORT_FILE).write_text(json.dumps(summary, indent=2))

    if not cfg.merge:
        return
    merged = cfg.output_dir / "merged"
    aggregate_datasets(
        repo_ids=[job.repo_id for job in jobs],
        aggr_repo_id=cfg.repo_id,
        roots=[job.root for job in jobs],
        aggr_root=merged,
    )
    # The merge keeps the language columns but not the annotation tools listed in info.json.
    Executor._ensure_annotation_metadata_in_info(merged)
    merge_sidecars(jobs, merged)
    logging.info(f"Merged {summary['episodes']} episodes into {merged}")
    if cfg.push_to_hub:
        LeRobotDataset(cfg.repo_id, root=merged).push_to_hub(private=cfg.private)


def merge_sidecars(jobs: list[ShardJob], merged: Path) -> None:
    """Concatenate the shards' per-episode generation records, renumbering episodes as the merge does."""
    offset = 0
    lines = []
    for job in jobs:
        entries = [json.loads(line) for line in (job.root / EPISODES_FILE).read_text().splitlines()]
        for entry in entries:
            lines.append(
                json.dumps({**entry, "episode_index": entry["episode_index"] + offset, "shard": job.index})
            )
        offset += len(entries)
    path = merged / EPISODES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in lines))


def summarize(reports: list[dict], elapsed: float) -> dict:
    episodes = sum(r["episodes"] for r in reports)
    frames = sum(r["frames"] for r in reports)
    attempts = sum(r["attempts"] for r in reports)
    failures: dict[str, int] = {}
    timings: dict[str, float] = {}
    tasks: dict[str, int] = {}
    for r in reports:
        for key, value in r["failures"].items():
            failures[key] = failures.get(key, 0) + value
        for key, value in r["timings"].items():
            timings[key] = timings.get(key, 0.0) + value
        for key, value in r["tasks"].items():
            tasks[key] = tasks.get(key, 0) + value
    total = sum(timings.values()) or 1.0
    return {
        "episodes": episodes,
        "frames": frames,
        "wall_seconds": round(elapsed, 1),
        "frames_per_second": round(frames / elapsed, 1),
        "episodes_per_hour": round(episodes / elapsed * 3600),
        "success_rate": round(episodes / max(attempts, 1), 3),
        "failures": failures,
        "tasks": tasks,
        # Share of worker time per stage.
        "time_split": {key: round(value / total, 3) for key, value in sorted(timings.items())},
    }


if __name__ == "__main__":
    generate()
