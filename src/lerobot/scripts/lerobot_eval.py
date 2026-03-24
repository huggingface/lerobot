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
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import concurrent.futures as cf
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any, TypedDict

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig

from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.lazy_vec_env import LazyVectorEnv
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    close_envs,
    preprocess_observation,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, DONE, OBS_STR, REWARD
from lerobot.utils.hf_eval_results import (
    build_eval_results_rows,
    default_eval_date,
    upload_eval_results_yaml,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Reset the policy and environments.
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)
    while not np.all(done) and step < max_steps:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        # Infer "task" from attributes of environments.
        # TODO: works with SyncVectorEnv but not AsyncVectorEnv
        observation = add_envs_task(env, observation)

        # Apply environment-specific preprocessing (e.g., LiberoProcessorStep for LIBERO)
        observation = env_preprocessor(observation)

        observation = preprocessor(observation)
        with torch.inference_mode():
            action = policy.select_action(observation)
        action = postprocessor(action)

        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition[ACTION]

        # Convert to CPU / numpy.
        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available if none of the envs finished.
        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                    "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                )
            successes = final_info["is_success"].tolist()
        else:
            successes = [False] * env.num_envs

        # Keep track of which environments are done so far.
        # Mark the episode as done if we reach the maximum step limit.
        # This ensures that the rollout always terminates cleanly at `max_steps`,
        # and allows logging/saving (e.g., videos) to be triggered consistently.
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        exc = ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )
        try:
            from peft import PeftModel

            if not isinstance(policy, PeftModel):
                raise exc
        except ImportError:
            raise exc from None

    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data[OBS_STR]:
            ep_dict[key] = rollout_data[OBS_STR][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


def _serializable_config(obj: Any) -> Any:
    """Recursively convert a config dict so it is JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _serializable_config(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable_config(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def push_eval_to_hub(
    repo_id: str,
    output_dir: Path,
    info: dict,
    env_type: str,
    env_task: str | None,
    benchmark_dataset_id: str,
    source_url: str | None = None,
    notes: str | None = None,
) -> str:
    """Upload eval artifacts and `.eval_results` rows to the Hub.

    Args:
        repo_id: HF model repo (e.g. "user/my_policy").
        output_dir: Local directory containing eval_info.json and videos/.
        info: The eval results dict (as returned by eval_policy_all).
        env_type: Environment type string (e.g. "libero_plus", "pusht").
        env_task: The env task string from eval config.
        benchmark_dataset_id: HF dataset id of the consolidated benchmark dataset.
        source_url: Optional source URL for `.eval_results` attribution.
        notes: Optional setup notes to include in `.eval_results`.

    Returns:
        URL of the last Hub commit.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)

    commit_url = ""

    # 1. Upload eval_info.json
    eval_json_path = output_dir / "eval_info.json"
    if eval_json_path.exists():
        commit_url = api.upload_file(
            path_or_fileobj=str(eval_json_path),
            path_in_repo=f"eval/{env_type}/eval_info.json",
            repo_id=repo_id,
            commit_message=f"Upload eval results for {env_type}",
        )

    # 2. Upload eval_config.json (policy, env, and eval settings used)
    eval_config_path = output_dir / "eval_config.json"
    if eval_config_path.exists():
        api.upload_file(
            path_or_fileobj=str(eval_config_path),
            path_in_repo=f"eval/{env_type}/eval_config.json",
            repo_id=repo_id,
            commit_message=f"Upload eval config for {env_type}",
        )

    # 3. Upload rollout videos
    videos_dir = output_dir / "videos"
    if videos_dir.is_dir():
        api.upload_folder(
            folder_path=str(videos_dir),
            path_in_repo=f"eval/{env_type}/videos",
            repo_id=repo_id,
            commit_message=f"Upload eval rollout videos for {env_type}",
        )

    # 4. Upload HF-native `.eval_results` rows (canonical leaderboard surface).
    rows = build_eval_results_rows(
        info=info,
        env_type=env_type,
        env_task=env_task,
        benchmark_dataset_id=benchmark_dataset_id,
        source_url=source_url,
        notes=notes,
        eval_date=default_eval_date(),
    )
    commit_url = upload_eval_results_yaml(
        api=api,
        repo_id=repo_id,
        rows=rows,
        env_type=env_type,
        env_task=env_task,
        output_dir=output_dir,
    )

    logging.info(f"Eval results pushed to https://huggingface.co/{repo_id}")
    return commit_url


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))
    # Multi-instance orchestration only applies to local runtime.
    # For docker runtime, instance_count controls the number of env containers
    # spawned directly by run_eval_in_docker — no extra lerobot-eval processes needed.
    if cfg.eval.runtime == "local" and cfg.eval.instance_count > 1 and cfg.eval.instance_id == 0:
        _orchestrate_multi_instance_eval(cfg)
    else:
        _run_eval_worker(cfg)


def _maybe_add_libero_plus_perturbation(info: dict, cfg: EvalPipelineConfig) -> None:
    if cfg.env.type != "libero_plus":
        return
    try:
        from lerobot.envs.libero import aggregate_by_perturbation, build_perturbation_index

        suite_names = [s.strip() for s in cfg.env.task.split(",") if s.strip()]
        suite_indices = {s: build_perturbation_index(s) for s in suite_names}
        perturbation_results = aggregate_by_perturbation(info["per_task"], suite_indices)
        info["perturbation_results"] = perturbation_results
        print("\n=== Perturbation Results ===")
        for dim, stats in perturbation_results.items():
            print(f"  {dim}: {stats['pc_success']:.1f}% ({stats['n_episodes']} episodes)")
    except Exception as exc:
        # Never fail a finished long-running eval on post-processing.
        print(f"WARNING: Failed to compute LIBERO-Plus perturbation breakdown: {exc}")
        print("Continuing with per-suite + overall metrics only.")


def _save_eval_outputs(cfg: EvalPipelineConfig, info: dict) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    eval_cfg_dict = _serializable_config(asdict(cfg))
    with open(output_dir / "eval_config.json", "w") as f:
        json.dump(eval_cfg_dict, f, indent=2)


def _maybe_push_eval_outputs(cfg: EvalPipelineConfig, info: dict) -> None:
    if not cfg.push_to_hub:
        return
    repo_id = str(cfg.policy.pretrained_path)
    try:
        push_eval_to_hub(
            repo_id=repo_id,
            output_dir=Path(cfg.output_dir),
            info=info,
            env_type=cfg.env.type,
            env_task=cfg.env.task,
            benchmark_dataset_id=cfg.benchmark_dataset_id,
            source_url=cfg.eval_result_source_url,
            notes=cfg.eval_result_notes,
        )
    except Exception as exc:
        logging.warning("Failed to push eval artifacts/results to Hub: %s", exc)


def _run_eval_worker(cfg: EvalPipelineConfig) -> dict:
    logging.info(pformat(asdict(cfg)))

    if cfg.eval.runtime in ("docker", "multiprocess"):
        from lerobot.envs.docker_runtime import run_eval_in_docker, run_eval_multiprocess

        if cfg.eval.runtime == "docker":
            run_eval_in_docker(cfg)
        else:
            run_eval_multiprocess(cfg)
        output_dir = Path(cfg.output_dir)
        with open(output_dir / "eval_info.json") as f:
            return json.load(f)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )

    policy.eval()

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    try:
        with (
            torch.no_grad(),
            torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
        ):
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=cfg.eval.n_episodes,
                max_episodes_rendered=10,
                videos_dir=Path(cfg.output_dir) / "videos",
                start_seed=cfg.seed,
                max_parallel_tasks=cfg.env.max_parallel_tasks,
                instance_count=cfg.eval.instance_count,
                instance_id=cfg.eval.instance_id,
            )
            print("Overall Aggregated Metrics:")
            print(info["overall"])

            for key, val in info.get("per_group", {}).items():
                print(f"\nAggregated Metrics for {key}:")
                print(val)

            _maybe_add_libero_plus_perturbation(info, cfg)
    finally:
        close_envs(envs)

    _save_eval_outputs(cfg, info)
    _maybe_push_eval_outputs(cfg, info)

    logging.info("End of eval")
    return info


def _orchestrate_multi_instance_eval(cfg: EvalPipelineConfig) -> None:
    start_t = time.time()
    root_output_dir = Path(cfg.output_dir)
    instances_root = root_output_dir / "instances"
    instances_root.mkdir(parents=True, exist_ok=True)

    n_instances = cfg.eval.instance_count
    logging.info(f"Launching multi-instance eval with {n_instances} workers.")

    # Spawn workers for shard 1..N-1, run shard 0 in-process.
    child_procs: list[tuple[int, subprocess.Popen]] = []
    argv = [
        arg
        for arg in sys.argv[1:]
        if not arg.startswith("--eval.instance_id=")
        and not arg.startswith("--output_dir=")
        and not arg.startswith("--push_to_hub=")
    ]
    for i in range(1, n_instances):
        child_output_dir = instances_root / str(i)
        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_eval",
            *argv,
            f"--eval.instance_id={i}",
            f"--output_dir={child_output_dir}",
            "--push_to_hub=false",
        ]
        logging.info("Starting eval worker %s/%s", i + 1, n_instances)
        child_procs.append((i, subprocess.Popen(cmd)))

    cfg0 = deepcopy(cfg)
    cfg0.eval.instance_id = 0
    cfg0.push_to_hub = False
    cfg0.output_dir = instances_root / "0"
    _run_eval_worker(cfg0)

    failed = []
    for idx, proc in child_procs:
        rc = proc.wait()
        if rc != 0:
            failed.append((idx, rc))
    if failed:
        raise RuntimeError(f"Multi-instance eval failed for workers: {failed}")

    partial_infos: list[dict] = []
    for i in range(n_instances):
        info_path = instances_root / str(i) / "eval_info.json"
        with open(info_path) as f:
            partial_infos.append(json.load(f))

    merged_per_task = []
    for info in partial_infos:
        merged_per_task.extend(info.get("per_task", []))
    merged_per_task.sort(key=lambda x: (x["task_group"], x["task_id"]))

    # Merge videos from each shard into final output dir.
    merged_videos_dir = root_output_dir / "videos"
    for i in range(n_instances):
        shard_dir = instances_root / str(i)
        shard_videos = shard_dir / "videos"
        if shard_videos.is_dir():
            shutil.copytree(shard_videos, merged_videos_dir, dirs_exist_ok=True)
            old_prefix = str(shard_videos)
            new_prefix = str(merged_videos_dir)
            for entry in merged_per_task:
                paths = entry.get("metrics", {}).get("video_paths", [])
                entry["metrics"]["video_paths"] = [
                    p.replace(old_prefix, new_prefix, 1) if p.startswith(old_prefix) else p for p in paths
                ]

    merged_info = _aggregate_eval_from_per_task(merged_per_task, total_eval_s=time.time() - start_t)
    _maybe_add_libero_plus_perturbation(merged_info, cfg)
    print("Overall Aggregated Metrics:")
    print(merged_info["overall"])

    _save_eval_outputs(cfg, merged_info)
    _maybe_push_eval_outputs(cfg, merged_info)
    logging.info("End of eval")


# ---- typed payload returned by one task eval ----
class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]


ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths")


def _aggregate_eval_from_per_task(per_task_infos: list[dict], total_eval_s: float) -> dict:
    """Aggregate eval metrics from per-task payloads."""
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}

    def _append(group: str, key: str, value: Any):
        if value is None:
            return
        if isinstance(value, list):
            group_acc[group][key].extend(value)
            overall[key].extend(value)
        else:
            group_acc[group][key].append(value)
            overall[key].append(value)

    for entry in per_task_infos:
        group = entry["task_group"]
        metrics = entry["metrics"]
        _append(group, "sum_rewards", metrics.get("sum_rewards"))
        _append(group, "max_rewards", metrics.get("max_rewards"))
        _append(group, "successes", metrics.get("successes"))
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)

    def _agg_from_list(xs: list[float]) -> float:
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    groups_aggregated = {}
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }

    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": total_eval_s,
        "eval_ep_s": total_eval_s / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
    }

    return {"per_task": per_task_infos, "per_group": groups_aggregated, "overall": overall_agg}


def _eval_task_batch(
    batch: list[tuple[str, int, LazyVectorEnv]],
    policy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    start_seed: int | None,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
) -> list[tuple[str, int, TaskMetrics]]:
    """Evaluate N tasks in a single batched rollout for GPU efficiency.

    Each task contributes one sub-env to a combined SyncVectorEnv so the policy
    processes all N observations in one forward pass per step.
    """
    all_fns: list[Callable] = []
    task_slices: list[tuple[str, int, int, int]] = []
    offset = 0
    for task_group, task_id, lazy_env in batch:
        fns = lazy_env.factory_fns
        if not fns:
            continue
        start = offset
        offset += len(fns)
        all_fns.extend(fns)
        task_slices.append((task_group, task_id, start, offset))

    if not all_fns:
        return []

    env_cls = batch[0][2].env_cls
    combined_env = env_cls(all_fns)

    try:
        seeds = None
        if start_seed is not None:
            seeds = list(range(start_seed, start_seed + combined_env.num_envs))

        ep_frames: list[np.ndarray] = []

        def render_frame(env: gym.vector.VectorEnv):
            if max_episodes_rendered <= 0:
                return
            n = min(max_episodes_rendered, env.num_envs)
            if isinstance(env, gym.vector.SyncVectorEnv):
                ep_frames.append(np.stack([env.envs[i].render() for i in range(n)]))
            elif isinstance(env, gym.vector.AsyncVectorEnv):
                ep_frames.append(np.stack(env.call("render")[:n]))

        rollout_data = rollout(
            env=combined_env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=seeds,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
        )

        n_steps = rollout_data["done"].shape[1]
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")

        video_paths_per_task: dict[tuple[str, int], list[str]] = defaultdict(list)
        if max_episodes_rendered > 0 and ep_frames and videos_dir:
            stacked = np.stack(ep_frames, axis=1)  # (batch, time, h, w, c)
            rendered = 0
            threads = []
            for tg, tid, start_i, end_i in task_slices:
                if rendered >= max_episodes_rendered:
                    break
                task_dir = videos_dir / f"{tg}_{tid}"
                task_dir.mkdir(parents=True, exist_ok=True)
                for env_idx in range(start_i, end_i):
                    if rendered >= max_episodes_rendered:
                        break
                    episode_index = env_idx - start_i
                    video_path = task_dir / f"eval_episode_{episode_index}.mp4"
                    video_paths_per_task[(tg, tid)].append(str(video_path))
                    di = done_indices[env_idx].item()
                    thread = threading.Thread(
                        target=write_video,
                        args=(
                            str(video_path),
                            stacked[env_idx, : di + 1],
                            combined_env.unwrapped.metadata["render_fps"],
                        ),
                    )
                    thread.start()
                    threads.append(thread)
                    rendered += 1
            for t in threads:
                t.join()

        results: list[tuple[str, int, TaskMetrics]] = []
        for tg, tid, start_i, end_i in task_slices:
            results.append(
                (
                    tg,
                    tid,
                    TaskMetrics(
                        sum_rewards=batch_sum_rewards[start_i:end_i].tolist(),
                        max_rewards=batch_max_rewards[start_i:end_i].tolist(),
                        successes=batch_successes[start_i:end_i].tolist(),
                        video_paths=video_paths_per_task.get((tg, tid), []),
                    ),
                )
            )
        return results
    finally:
        combined_env.close()


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
) -> TaskMetrics:
    """Evaluates one task_id of one suite using the provided vec env."""

    task_videos_dir = videos_dir

    task_result = eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    per_episode = task_result["per_episode"]
    return TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
    )


def run_one(
    task_group: str,
    task_id: int,
    env: Any,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
):
    """
    Run eval_one for a single (task_group, task_id, env).
    Returns (task_group, task_id, task_metrics_dict).
    This function is intentionally module-level to make it easy to test.
    """
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    # Call the existing eval_one (assumed to return TaskMetrics-like dict)
    metrics = eval_one(
        env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )
    # ensure we always provide video_paths key to simplify accumulation
    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
    return task_group, task_id, metrics


def eval_policy_all(
    envs: dict[str, dict[int, Any]],
    policy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
    instance_count: int = 1,
    instance_id: int = 0,
) -> dict:
    """
    Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}}.
    This implementation flattens tasks, runs them sequentially or via ThreadPoolExecutor,
    accumulates per-group and overall statistics, and returns the same aggregate metrics
    schema as the single-env evaluator (avg_sum_reward / avg_max_reward / pc_success / timings)
    plus per-task infos.
    """
    start_t = time.time()

    # Flatten envs into list of (task_group, task_id, env)
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]
    if instance_count > 1:
        total_tasks = len(tasks)
        tasks = [task for idx, task in enumerate(tasks) if idx % instance_count == instance_id]
        logging.info(
            f"Instance shard {instance_id + 1}/{instance_count}: {len(tasks)}/{total_tasks} tasks assigned."
        )

    # accumulators: track metrics at both per-group level and across all groups
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []

    # small inline helper to accumulate one task's metrics into accumulators
    def _accumulate_to(group: str, metrics: dict):
        # metrics expected to contain 'sum_rewards', 'max_rewards', 'successes', optionally 'video_paths'
        # but eval_one may store per-episode lists; we assume metrics uses scalars averaged per task as before.
        # To be robust, accept scalars or lists.
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        # video_paths is list-like
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)

    # Choose runner (sequential vs threaded)
    task_runner = partial(
        run_one,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    all_lazy = all(isinstance(env, LazyVectorEnv) for _, _, env in tasks)
    single_factory_per_task = all(
        not isinstance(env, LazyVectorEnv) or env.num_factory_fns == 1 for _, _, env in tasks
    )
    can_batch = max_parallel_tasks > 1 and all_lazy and single_factory_per_task and n_episodes == 1

    if can_batch:
        # Multi-task batched path: combine N tasks into one SyncVectorEnv per chunk
        # so the policy processes all N observations in a single forward pass per step.
        chunk_size = max_parallel_tasks
        logging.info(f"Task scheduler mode: batched_lazy (chunk_size={chunk_size})")
        n_chunks = (len(tasks) + chunk_size - 1) // chunk_size
        rendered_so_far = 0
        for chunk_idx in range(n_chunks):
            chunk = tasks[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
            render_budget = max(0, max_episodes_rendered - rendered_so_far)
            logging.info(
                f"Batch {chunk_idx + 1}/{n_chunks}: evaluating {len(chunk)} tasks "
                f"({chunk_idx * chunk_size + 1}–{chunk_idx * chunk_size + len(chunk)}/{len(tasks)})"
            )
            batch_results = _eval_task_batch(
                chunk,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                start_seed=start_seed,
                max_episodes_rendered=render_budget,
                videos_dir=videos_dir,
            )
            for tg, tid, metrics in batch_results:
                _accumulate_to(tg, metrics)
                per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
                rendered_so_far += len(metrics.get("video_paths", []))

            if overall["successes"]:
                sr = np.nanmean(overall["successes"]) * 100
                logging.info(f"  running success rate: {sr:.1f}%")
    elif max_parallel_tasks <= 1:
        logging.info("Task scheduler mode: sequential")
        for task_group, task_id, env in tasks:
            try:
                tg, tid, metrics = task_runner(task_group, task_id, env)
                _accumulate_to(tg, metrics)
                per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
            finally:
                env.close()
    else:
        # Threaded fallback for cases where batched lazy mode cannot be used.
        if all_lazy and n_episodes != 1:
            logging.info("Task scheduler mode: threaded (lazy batching disabled because n_episodes != 1)")
        elif all_lazy and not single_factory_per_task:
            logging.info("Task scheduler mode: threaded (lazy batching disabled because eval.batch_size > 1)")
        else:
            logging.info(f"Task scheduler mode: threaded (max_workers={max_parallel_tasks})")
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            fut2meta: dict[cf.Future, tuple[str, int, Any]] = {}
            for task_group, task_id, env in tasks:
                fut = executor.submit(task_runner, task_group, task_id, env)
                fut2meta[fut] = (task_group, task_id, env)
            for fut in cf.as_completed(fut2meta):
                tg, tid, env = fut2meta[fut]
                try:
                    _, _, metrics = fut.result()
                    _accumulate_to(tg, metrics)
                    per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
                finally:
                    env.close()

    return _aggregate_eval_from_per_task(per_task_infos, total_eval_s=time.time() - start_t)


def main():
    init_logging()
    register_third_party_plugins()
    eval_main()


if __name__ == "__main__":
    main()
