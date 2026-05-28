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

"""Collect simulation rollouts from a PI0.5 policy and save a LeRobotDataset.

This script records raw observations/actions/tasks. RLT VLA embeddings should be
computed later as a sidecar cache from the saved dataset, because they are large
and tied to the exact frozen VLA checkpoint.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import trange

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, DONE, REWARD, TRUNCATED
from lerobot.utils.random_utils import set_seed


def _feature_from_tensor(key: str, value: torch.Tensor, *, use_videos: bool) -> dict[str, Any]:
    value = value.detach().cpu()
    if value.ndim > 0 and value.shape[0] == 1:
        value = value.squeeze(0)

    if key.startswith("observation.image"):
        return {
            "dtype": "video" if use_videos else "image",
            "shape": tuple(value.shape),
            "names": ["channels", "height", "width"],
        }

    return {
        "dtype": "float32",
        "shape": tuple(value.shape),
        "names": [f"{key}.{i}" for i in range(int(np.prod(value.shape)))],
    }


def _make_dataset_features(
    observation: dict[str, Any], action: torch.Tensor, *, use_videos: bool
) -> dict[str, dict[str, Any]]:
    features = {}
    for key, value in observation.items():
        if key == "task" or not isinstance(value, torch.Tensor):
            continue
        features[key] = _feature_from_tensor(key, value, use_videos=use_videos)

    features[ACTION] = _feature_from_tensor(ACTION, action, use_videos=use_videos)
    features[REWARD] = {"dtype": "float32", "shape": (1,), "names": ["reward"]}
    features[DONE] = {"dtype": "bool", "shape": (1,), "names": ["done"]}
    features[TRUNCATED] = {"dtype": "bool", "shape": (1,), "names": ["truncated"]}
    return features


def _squeeze_batch(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().cpu()
    return value.squeeze(0) if value.ndim > 0 and value.shape[0] == 1 else value


def _frame_from_observation(
    observation: dict[str, Any],
    action: torch.Tensor,
    reward: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
) -> dict[str, Any]:
    task = observation.get("task", [""])
    if isinstance(task, list):
        task = task[0]

    done = np.asarray(terminated, dtype=np.bool_) | np.asarray(truncated, dtype=np.bool_)
    frame: dict[str, Any] = {
        "task": task,
        ACTION: _squeeze_batch(action),
        REWARD: np.asarray([float(np.asarray(reward).reshape(-1)[0])], dtype=np.float32),
        DONE: np.asarray([bool(done.reshape(-1)[0])], dtype=np.bool_),
        TRUNCATED: np.asarray([bool(np.asarray(truncated).reshape(-1)[0])], dtype=np.bool_),
    }
    for key, value in observation.items():
        if key == "task" or not isinstance(value, torch.Tensor):
            continue
        frame[key] = _squeeze_batch(value)
    return frame


def _first_env(envs):
    suite_name = next(iter(envs))
    task_id = next(iter(envs[suite_name]))
    return envs[suite_name][task_id]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True, help="PI0.5 policy checkpoint or Hub repo.")
    parser.add_argument("--repo-id", required=True, help="Repo id recorded in the local LeRobotDataset.")
    parser.add_argument("--root", required=True, help="Local output directory for the dataset.")
    parser.add_argument("--env-type", default="libero", choices=["libero", "pusht", "aloha"])
    parser.add_argument("--env-task", default="libero_10")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    root = Path(args.root)
    if root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(root)

    env_cfg = make_env_config(args.env_type, task=args.env_task)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = _first_env(envs)

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = Path(args.policy_path)
    policy_cfg.device = args.device

    policy = make_policy(policy_cfg, env_cfg=env_cfg)
    policy.eval()

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy_cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
    )

    dataset = None
    try:
        for ep_idx in range(args.episodes):
            policy.reset()
            raw_observation, _ = env.reset(seed=[args.seed + ep_idx])
            max_steps = args.max_steps or env.call("_max_episode_steps")[0]

            for _step in trange(max_steps, desc=f"episode {ep_idx}", leave=False):
                observation = preprocess_observation(raw_observation)
                observation = add_envs_task(env, observation)
                observation = env_preprocessor(observation)

                policy_input = preprocessor(observation)
                with torch.inference_mode():
                    policy_action = policy.select_action(policy_input)
                policy_action = postprocessor(policy_action)

                env_action = env_postprocessor({ACTION: policy_action})[ACTION]
                next_raw_observation, reward, terminated, truncated, _info = env.step(
                    env_action.detach().cpu().numpy()
                )

                if dataset is None:
                    features = _make_dataset_features(
                        observation, policy_action, use_videos=args.use_videos
                    )
                    dataset = LeRobotDataset.create(
                        repo_id=args.repo_id,
                        fps=env_cfg.fps,
                        features=features,
                        root=root,
                        robot_type=args.env_type,
                        use_videos=args.use_videos,
                    )

                dataset.add_frame(
                    _frame_from_observation(observation, policy_action, reward, terminated, truncated)
                )

                raw_observation = next_raw_observation
                if bool((terminated | truncated)[0]):
                    break

            if dataset is not None:
                dataset.save_episode()

        if dataset is not None:
            dataset.finalize()
            print(f"Saved {dataset.meta.total_episodes} episodes to {dataset.root}")
    finally:
        close_envs(envs)


if __name__ == "__main__":
    main()
