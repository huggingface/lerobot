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

"""One shard: a LeRobotDataset of its own, filled by one process, then annotated.

LeRobotDataset has a single writer, so every worker process writes its own shard and ``generate.py``
merges them at the end. A finished shard leaves a ``generation_report.json``, so a rerun skips it.
"""

import json
import logging
import os
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Headless rendering, decided before MuJoCo is imported.
os.environ.setdefault("MUJOCO_GL", "egl")

from lerobot.configs.video import RGBEncoderConfig  # noqa: E402
from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STR  # noqa: E402
from lerobot.utils.feature_utils import hw_to_dataset_features  # noqa: E402

from .annotations import (  # noqa: E402
    EpisodeRows,
    persistent_rows,
    vqa_frames,
    vqa_pair,
    write_language_columns,
)
from .assets import load_meshes  # noqa: E402
from .config import GenerationConfig  # noqa: E402
from .episode import Episode, EpisodeError, prepare, simulate  # noqa: E402
from .processors import (  # noqa: E402
    make_command_pipeline,
    make_recording_pipelines,
    start_episode,
    to_dict,
    to_vector,
)
from .render import CAMERAS, StateRenderer  # noqa: E402
from .so101 import JOINT_NAMES  # noqa: E402
from .tasks import TASKS  # noqa: E402

ROBOT_TYPE = "so_follower"
REPORT_FILE = "generation_report.json"
EPISODES_FILE = Path("meta") / "generation" / "episodes.jsonl"
# Give up on a shard whose episodes keep failing, rather than loop forever.
MAX_ATTEMPTS_PER_EPISODE = 20


@dataclass
class EncodingConfig:
    vcodec: str = "libsvtav1"
    crf: int = 30
    # Codec-specific speed preset, e.g. 12 for libsvtav1, "veryfast" for h264, "p1" for h264_nvenc.
    preset: int | str | None = None
    # Threads per camera encoder; a worker runs one encoder per camera.
    encoder_threads: int = 2
    # Smaller video files keep the per-episode append cheap.
    video_files_size_in_mb: int = 100


@dataclass
class ShardJob:
    index: int
    root: Path
    repo_id: str
    episodes: int
    seed: int
    generation: GenerationConfig
    encoding: EncodingConfig = field(default_factory=EncodingConfig)


def dataset_features(height: int, width: int) -> dict:
    joints = dict.fromkeys(JOINT_NAMES, float)
    cameras = dict.fromkeys(CAMERAS, (height, width, 3))
    return {
        **hw_to_dataset_features(joints, ACTION),
        **hw_to_dataset_features({**joints, **cameras}, OBS_STR),
    }


def episode_seed(base: int, shard: int, attempt: int) -> int:
    """A seed unique to (run seed, shard, attempt), so shards never repeat each other's episodes."""
    return int(np.random.SeedSequence([base, shard, attempt]).generate_state(1, dtype=np.uint64)[0] >> 1)


def wait_for_encoder(dataset: LeRobotDataset) -> None:
    """Block while a camera's encoder queue is full: the streaming encoder drops frames otherwise.

    TODO: needs a blocking mode in ``StreamingVideoEncoder``; this reads its private queues.
    """
    encoder = dataset.writer._streaming_encoder if dataset.writer is not None else None
    if encoder is None:
        return
    while any(q.full() for q in encoder._frame_queues.values()):
        time.sleep(0.001)


def run_shard(job: ShardJob) -> dict:
    report_path = job.root / REPORT_FILE
    if report_path.is_file():
        return json.loads(report_path.read_text())
    if job.root.exists():
        # A shard interrupted mid-way: start it over.
        shutil.rmtree(job.root)
    logging.basicConfig(level=logging.INFO, format=f"[shard {job.index}] %(message)s", force=True)
    cfg = job.generation
    meshes = load_meshes() if cfg.objects.mesh_prob > 0 else {}
    dataset = LeRobotDataset.create(
        repo_id=job.repo_id,
        fps=cfg.fps,
        features=dataset_features(cfg.height, cfg.width),
        root=job.root,
        robot_type=ROBOT_TYPE,
        use_videos=True,
        streaming_encoding=True,
        encoder_threads=job.encoding.encoder_threads,
        rgb_encoder=RGBEncoderConfig(
            vcodec=job.encoding.vcodec, crf=job.encoding.crf, preset=job.encoding.preset
        ),
        video_files_size_in_mb=job.encoding.video_files_size_in_mb,
    )
    command_pipeline = make_command_pipeline(cfg.noise, cfg.fps)
    observation_pipeline, action_pipeline, calibration = make_recording_pipelines(cfg.calibration)
    camera_keys = tuple(f"{OBS_STR}.images.{name}" for name in CAMERAS)

    failures: Counter = Counter()
    timings: dict[str, float] = defaultdict(float)
    tasks: Counter = Counter()
    rows: dict[int, EpisodeRows] = {}
    sidecar = []
    frames = 0
    attempt = 0
    started = time.perf_counter()
    while dataset.num_episodes < job.episodes:
        if attempt >= MAX_ATTEMPTS_PER_EPISODE * job.episodes:
            raise RuntimeError(f"Shard {job.index}: too many failed episodes ({dict(failures)})")
        seed = episode_seed(job.seed, job.index, attempt)
        attempt += 1
        rng = np.random.default_rng([seed, 1])
        t0 = time.perf_counter()
        try:
            episode = prepare(seed, cfg, meshes)
            t1 = time.perf_counter()
            start_episode(rng, command_pipeline)
            simulate(
                episode,
                TASKS[episode.task],
                cfg.fps,
                command_filter=lambda q: to_vector(command_pipeline((to_dict(q), {}))),
            )
        except EpisodeError as e:
            failures[e.reason] += 1
            continue
        t2 = time.perf_counter()
        timings["prepare"] += t1 - t0
        timings["simulate"] += t2 - t1

        calibration.start_episode(rng)
        start_episode(rng, observation_pipeline, action_pipeline)
        index = dataset.num_episodes
        rows[index] = EpisodeRows(persistent=persistent_rows(episode.instance, episode.plan, cfg.annotation))
        record_episode(
            dataset,
            episode,
            observation_pipeline,
            action_pipeline,
            rows[index],
            camera_keys,
            cfg,
            rng,
            timings,
        )
        t3 = time.perf_counter()
        dataset.save_episode()
        timings["save"] += time.perf_counter() - t3
        frames += len(episode.commands)
        tasks[episode.task] += 1
        sidecar.append(
            {
                "episode_index": index,
                "seed": seed,
                "task": episode.task,
                "instruction": episode.instance.instruction,
                "roles": episode.instance.roles,
                "start_mode": episode.start_mode,
                "start": episode.start.tolist(),
                "calibration": calibration.as_dict(),
                "scene": asdict(episode.scene),
            }
        )
        if dataset.num_episodes % 10 == 0:
            elapsed = time.perf_counter() - started
            logging.info(f"{dataset.num_episodes}/{job.episodes} episodes, {frames / elapsed:.0f} frames/s")

    t4 = time.perf_counter()
    dataset.finalize()
    write_language_columns(job.root, rows, camera_keys)
    timings["finalize"] += time.perf_counter() - t4
    episodes_path = job.root / EPISODES_FILE
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_path.write_text("".join(json.dumps(entry) + "\n" for entry in sidecar))
    report = {
        "shard": job.index,
        "episodes": dataset.num_episodes,
        "frames": frames,
        "attempts": attempt,
        "failures": dict(failures),
        "tasks": dict(tasks),
        "seconds": time.perf_counter() - started,
        "timings": dict(timings),
    }
    report_path.write_text(json.dumps(report, indent=2))
    return report


def record_episode(
    dataset: LeRobotDataset,
    episode: Episode,
    observation_pipeline,
    action_pipeline,
    rows: EpisodeRows,
    camera_keys: tuple[str, ...],
    cfg: GenerationConfig,
    rng: np.random.Generator,
    timings: dict[str, float],
) -> None:
    """Pass 2: render the recorded states, map joints through the robot-side processors, add frames."""
    renderer = StateRenderer(episode.model, cfg.height, cfg.width)
    vqa_at = set(vqa_frames(len(episode.commands), cfg.fps, cfg.annotation, rng))
    info = episode.info
    try:
        for t in range(len(episode.commands)):
            t0 = time.perf_counter()
            renderer.set_state(episode.qpos[t])
            frame = {f"{OBS_STR}.images.{name}": renderer.rgb(name) for name in CAMERAS}
            t1 = time.perf_counter()
            if t in vqa_at:
                positions = {
                    name: renderer.data.xpos[body].copy() for name, body in info.object_bodies.items()
                }
                for name, key in zip(CAMERAS, camera_keys, strict=True):
                    visible = renderer.visible(name, info, cfg.annotation.min_visible_pixels)
                    pair = vqa_pair(key, visible, episode.scene, positions, cfg.annotation, rng)
                    if pair is not None:
                        rows.events.extend((t, r) for r in pair)
            t2 = time.perf_counter()
            state = observation_pipeline(to_dict(episode.arm.from_qpos(episode.qpos[t])))
            action = action_pipeline((to_dict(episode.commands[t]), state))
            frame[f"{OBS_STR}.state"] = to_vector(state).astype(np.float32)
            frame[ACTION] = to_vector(action).astype(np.float32)
            frame["task"] = episode.instance.instruction
            wait_for_encoder(dataset)
            t3 = time.perf_counter()
            dataset.add_frame(frame)
            t4 = time.perf_counter()
            timings["render"] += t1 - t0
            timings["annotate"] += t2 - t1
            timings["encoder_wait"] += t3 - t2
            timings["add_frame"] += t4 - t3
    finally:
        renderer.close()
