#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Docker eval worker — runs inside a benchmark container.

Runs gym episodes for a sharded subset of the configured env's tasks, calling
a remote HTTP policy inference server (running on the host GPU) for action chunks.

Usage (normally invoked by docker_runtime.run_eval_in_docker, not directly):
    lerobot-eval-worker \\
        --env.type=libero_plus \\
        --server_address=host.docker.internal:50051 \\
        --n_episodes=5 \\
        --seed=1000 \\
        --instance_id=0 \\
        --instance_count=2 \\
        --output_path=/results/worker_0.json
"""

from __future__ import annotations

import json
import logging
import pickle  # nosec B403 — internal serialisation only
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import numpy as np

from lerobot import envs  # noqa: F401 — registers all env subclasses
from lerobot.envs.configs import EnvConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass
class EvalWorkerConfig:
    env: EnvConfig
    # Address of the policy inference HTTP server on the host.
    server_address: str = "host.docker.internal:50051"
    # Number of episodes to run per task.
    n_episodes: int = 1
    # Starting random seed; episode i of a task uses seed + i.
    seed: int = 0
    # 0-indexed shard id for this worker.
    instance_id: int = 0
    # Total number of shards (workers).
    instance_count: int = 1
    # Path (inside the container) to write the JSON per-task results.
    output_path: Path = field(default_factory=lambda: Path("/results/worker.json"))
    # Timeout in seconds for each HTTP request to the policy server.
    server_timeout: float = 120.0


def _call_server(server_address: str, obs_t: dict, timeout: float) -> np.ndarray:
    """POST pickled obs to /predict_chunk, return numpy chunk (T, action_dim)."""
    body = pickle.dumps({"obs_t": obs_t})  # nosec B301
    req = urllib.request.Request(
        f"http://{server_address}/predict_chunk",
        data=body,
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return pickle.loads(resp.read())  # nosec B301


def run_worker(cfg: EvalWorkerConfig) -> dict:
    """Run cfg.n_episodes episodes per assigned task. Returns per-task results dict."""
    # Build envs: {task_group: {task_id: vec_env}}
    envs_dict = make_env(cfg.env, n_envs=1)

    # Flatten to list of (task_group, task_id, env)
    tasks = [
        (task_group, task_id, vec)
        for task_group, group in envs_dict.items()
        for task_id, vec in group.items()
    ]

    # Shard: this worker handles tasks where index % instance_count == instance_id
    if cfg.instance_count > 1:
        total = len(tasks)
        tasks = [t for idx, t in enumerate(tasks) if idx % cfg.instance_count == cfg.instance_id]
        logger.info(
            "Shard %d/%d: %d/%d tasks assigned.",
            cfg.instance_id + 1,
            cfg.instance_count,
            len(tasks),
            total,
        )

    per_task: list[dict] = []

    for task_group, task_id, env in tasks:
        sum_rewards: list[float] = []
        max_rewards: list[float] = []
        successes: list[bool] = []

        for ep_idx in range(cfg.n_episodes):
            obs, _info = env.reset(seed=[cfg.seed + ep_idx])
            obs_t = preprocess_observation(obs)
            obs_t = add_envs_task(env, obs_t)

            action_buffer: list[np.ndarray] = []  # each element: (1, action_dim)
            ep_rewards: list[float] = []
            ep_success = False
            done = np.zeros(1, dtype=bool)

            while not np.all(done):
                if not action_buffer:
                    chunk_np = _call_server(cfg.server_address, obs_t, cfg.server_timeout)
                    # chunk_np: (T, action_dim) — split into per-step slices of shape (1, action_dim)
                    action_buffer = [chunk_np[i : i + 1] for i in range(chunk_np.shape[0])]

                action_np = action_buffer.pop(0)  # (1, action_dim)
                obs, reward, terminated, truncated, info = env.step(action_np)

                done = terminated | truncated | done
                ep_rewards.append(float(np.mean(reward)))

                if "final_info" in info:
                    final_info = info["final_info"]
                    if isinstance(final_info, dict) and "is_success" in final_info:
                        ep_success = bool(final_info["is_success"][0])

                if not np.all(done):
                    obs_t = preprocess_observation(obs)
                    obs_t = add_envs_task(env, obs_t)

            sum_rewards.append(float(np.sum(ep_rewards)))
            max_rewards.append(float(np.max(ep_rewards)) if ep_rewards else 0.0)
            successes.append(ep_success)
            logger.info(
                "Task %s[%d] ep %d/%d — success=%s",
                task_group,
                task_id,
                ep_idx + 1,
                cfg.n_episodes,
                ep_success,
            )

        per_task.append(
            {
                "task_group": task_group,
                "task_id": task_id,
                "metrics": {
                    "sum_rewards": sum_rewards,
                    "max_rewards": max_rewards,
                    "successes": successes,
                    "video_paths": [],
                },
            }
        )
        env.close()

    return {"per_task": per_task}


def worker_main(cfg: EvalWorkerConfig) -> None:
    results = run_worker(cfg)
    output = Path(cfg.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    logger.info("Worker %d wrote results to %s", cfg.instance_id, output)


def main() -> None:
    init_logging()
    cfg = draccus.parse(config_class=EvalWorkerConfig)
    worker_main(cfg)


if __name__ == "__main__":
    main()
