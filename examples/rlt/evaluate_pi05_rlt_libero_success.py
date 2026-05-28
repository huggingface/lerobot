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

"""Evaluate LIBERO success rate for PI0.5 alone or PI0.5 + RLT.

The script runs vectorized LIBERO rollouts and records per-episode success.
Failed episodes are recorded as videos by default.

Modes:
  - vla: execute PI0.5 reference action chunks directly.
  - rlt: execute action chunks refined by an RLT policy checkpoint.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

import torch
from tqdm import tqdm, trange

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.rlt.modeling_rlt import RLTPolicy
from lerobot.policies.rlt.vla_adapter import (
    OBS_REFERENCE_ACTION,
    OBS_RLT_STATE,
    OBS_VLA_EMBEDDINGS,
    PI05PrefixRLTAdapter,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed


def _parse_task_ids(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return [0]
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _as_array(value: Any, *, dtype, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.size == 1 and size != 1:
        array = np.repeat(array, size)
    if array.size != size:
        raise ValueError(f"Expected {size} values, got shape {np.asarray(value).shape}.")
    return array


def _info_array(info: Any, keys: tuple[str, ...], size: int) -> np.ndarray | None:
    if not isinstance(info, dict):
        return None

    for key in keys:
        if key in info:
            return _as_array(info[key], dtype=np.float32, size=size)

    final_info = info.get("final_info")
    if isinstance(final_info, dict):
        for key in keys:
            if key in final_info:
                return _as_array(final_info[key], dtype=np.float32, size=size)

    if isinstance(final_info, (list, tuple, np.ndarray)):
        values = np.zeros(size, dtype=np.float32)
        found = False
        for idx, item in enumerate(final_info):
            if idx >= size or not isinstance(item, dict):
                continue
            for key in keys:
                if key in item:
                    values[idx] = float(item[key])
                    found = True
                    break
        if found:
            return values

    return None


def _maybe_latest_checkpoint(path: Path) -> Path:
    if path.is_dir() and (path / "config.json").exists():
        return path
    checkpoints = sorted(
        p
        for p in path.glob("checkpoint-*")
        if p.is_dir() and not p.name.endswith(".tmp") and (p / "config.json").exists()
    )
    if not checkpoints:
        raise FileNotFoundError(f"No valid checkpoint-* directories with config.json found under {path}")
    return checkpoints[-1]


def _metadata_path(path: Path) -> Path | None:
    candidates = []
    if path.is_dir():
        candidates.extend([path / "online_training.json", path.parent / "online_training.json"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_online_metadata(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    metadata_path = _metadata_path(Path(path))
    if metadata_path is None:
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _torch_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _env_tasks(env, n_envs: int) -> list[str]:
    for attr in ("task_description", "task"):
        try:
            task_result = env.call(attr)
        except Exception:
            continue
        if isinstance(task_result, tuple):
            task_result = list(task_result)
        elif isinstance(task_result, np.ndarray):
            task_result = task_result.tolist()
        elif isinstance(task_result, str):
            task_result = [task_result] * n_envs
        if not isinstance(task_result, list):
            continue
        if len(task_result) == 1 and n_envs != 1:
            task_result = task_result * n_envs
        if len(task_result) != n_envs:
            raise ValueError(f"Expected {n_envs} task strings from env.{attr}, got {len(task_result)}.")
        return [str(item) for item in task_result]
    return ["" for _ in range(n_envs)]


def _make_pi05_policy(
    *,
    policy_path: str,
    env_cfg,
    device: str,
    dtype: str | None,
    num_inference_steps: int | None,
    tokenizer_path: str | None,
    disable_compile: bool,
) -> tuple[PI05Policy, Any, Any, Any]:
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = Path(policy_path)
    policy_cfg.device = device
    if dtype is not None and hasattr(policy_cfg, "dtype"):
        policy_cfg.dtype = dtype
    if num_inference_steps is not None and hasattr(policy_cfg, "num_inference_steps"):
        policy_cfg.num_inference_steps = num_inference_steps
    if tokenizer_path is not None and hasattr(policy_cfg, "tokenizer_name"):
        policy_cfg.tokenizer_name = tokenizer_path
    if disable_compile and hasattr(policy_cfg, "compile_model"):
        policy_cfg.compile_model = False

    policy = make_policy(policy_cfg, env_cfg=env_cfg)
    if not isinstance(policy, PI05Policy):
        raise TypeError(f"Expected PI05Policy, got {type(policy).__name__}.")
    policy.eval()
    for param in policy.parameters():
        param.requires_grad_(False)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
    )
    return policy, policy_cfg, preprocessor, postprocessor


def _load_rlt_policy(path: str, device: str, dtype: str | None) -> tuple[RLTPolicy, Path]:
    rlt_path = _maybe_latest_checkpoint(Path(path))
    rlt_cfg = PreTrainedConfig.from_pretrained(rlt_path, local_files_only=True)
    rlt_cfg.device = device
    policy = RLTPolicy.from_pretrained(rlt_path, config=rlt_cfg, local_files_only=True, strict=True)
    torch_dtype = _torch_dtype(dtype)
    if torch_dtype is None:
        policy.to(device)
    else:
        policy.to(device=device, dtype=torch_dtype)
    policy.eval()
    return policy, rlt_path


def _prepare_pi05_input(
    raw_observation,
    env,
    env_preprocessor,
    pi05_preprocessor,
    task_descriptions: list[str] | None,
) -> dict[str, torch.Tensor]:
    observation = preprocess_observation(raw_observation)
    if task_descriptions is None:
        observation = add_envs_task(env, observation)
    else:
        observation["task"] = task_descriptions
    observation = env_preprocessor(observation)
    return pi05_preprocessor(observation)


@torch.no_grad()
def _rlt_features(
    *,
    raw_observation,
    env,
    env_preprocessor,
    pi05_preprocessor,
    adapter: PI05PrefixRLTAdapter,
    rlt_device: str,
    include_proprio: bool,
    task_descriptions: list[str] | None,
) -> dict[str, torch.Tensor]:
    pi05_input = _prepare_pi05_input(
        raw_observation,
        env,
        env_preprocessor,
        pi05_preprocessor,
        task_descriptions,
    )
    batch = adapter(pi05_input)
    out = {
        OBS_VLA_EMBEDDINGS: batch[OBS_VLA_EMBEDDINGS].detach().to(rlt_device),
        OBS_REFERENCE_ACTION: batch[OBS_REFERENCE_ACTION].detach().to(rlt_device),
    }
    if include_proprio and "observation.state" in pi05_input:
        out["observation.state"] = pi05_input["observation.state"].detach().to(rlt_device)
    return out


@torch.no_grad()
def _vla_action_chunk(
    *,
    raw_observation,
    env,
    env_preprocessor,
    pi05_preprocessor,
    pi05_policy: PI05Policy,
    task_descriptions: list[str] | None,
) -> torch.Tensor:
    pi05_input = _prepare_pi05_input(
        raw_observation,
        env,
        env_preprocessor,
        pi05_preprocessor,
        task_descriptions,
    )
    return pi05_policy.predict_action_chunk(pi05_input).detach()


@torch.no_grad()
def _attach_rlt_state(features: dict[str, torch.Tensor], rlt_policy: RLTPolicy) -> dict[str, torch.Tensor]:
    encoder_param = next(rlt_policy.rl_token_encoder.parameters())
    vla_embeddings = features[OBS_VLA_EMBEDDINGS].to(
        device=encoder_param.device,
        dtype=encoder_param.dtype,
    )
    features[OBS_RLT_STATE] = rlt_policy.rl_token_encoder(vla_embeddings).detach()
    return features


def _to_policy_tensor(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if tensor.is_floating_point():
        return tensor.to(device=device, dtype=dtype)
    return tensor.to(device=device)


def _policy_batch_from_features(
    features: dict[str, torch.Tensor],
    include_proprio: bool,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    batch = {
        OBS_REFERENCE_ACTION: _to_policy_tensor(features[OBS_REFERENCE_ACTION], device=device, dtype=dtype),
    }
    if OBS_RLT_STATE in features:
        batch[OBS_RLT_STATE] = _to_policy_tensor(features[OBS_RLT_STATE], device=device, dtype=dtype)
    else:
        batch[OBS_VLA_EMBEDDINGS] = _to_policy_tensor(features[OBS_VLA_EMBEDDINGS], device=device, dtype=dtype)
    if include_proprio and "observation.state" in features:
        batch["observation.state"] = _to_policy_tensor(
            features["observation.state"],
            device=device,
            dtype=dtype,
        )
    return batch


def _postprocess_action_step(
    action_step_norm: torch.Tensor,
    *,
    pi05_postprocessor,
    env_postprocessor,
    clip_normalized_action: bool,
) -> torch.Tensor:
    if clip_normalized_action:
        action_step_norm = action_step_norm.clamp(-1.0, 1.0)
    action_cpu = pi05_postprocessor(action_step_norm.detach().cpu())
    return env_postprocessor({ACTION: action_cpu})[ACTION]


def _reset_finished_envs(env, done_mask: np.ndarray, seed_base: int | None = None):
    options = {"reset_mask": done_mask.astype(bool)}
    seeds = None
    if seed_base is not None:
        seeds = [seed_base + env_idx for env_idx in range(done_mask.size)]
    try:
        return env.reset(seed=seeds, options=options)
    except TypeError:
        return env.reset(seed=seeds)


def _env_action_dim(env) -> int | None:
    space = getattr(env, "single_action_space", None)
    if space is None:
        space = getattr(env, "action_space", None)
    shape = getattr(space, "shape", None)
    if not shape:
        return None
    if len(shape) == 1:
        return int(shape[0])
    if len(shape) >= 2:
        return int(shape[-1])
    return None


def _write_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item) + "\n")


def _video_image_batch(raw_observation: Any, video_key: str, n_envs: int) -> np.ndarray:
    if not isinstance(raw_observation, dict):
        raise TypeError(
            f"Expected dict observation for video recording, got {type(raw_observation).__name__}."
        )

    pixels = raw_observation.get("pixels")
    if isinstance(pixels, dict):
        if video_key not in pixels:
            raise KeyError(f"Video key '{video_key}' not found in observation pixels: {sorted(pixels)}.")
        images = pixels[video_key]
    else:
        flat_keys = (
            video_key,
            f"pixels.{video_key}",
            f"pixels/{video_key}",
            f"observation.images.{video_key}",
        )
        for key in flat_keys:
            if key in raw_observation:
                images = raw_observation[key]
                break
        else:
            raise KeyError(f"Video key '{video_key}' not found in observation.")

    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    array = np.asarray(images)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4:
        raise ValueError(f"Expected video frames with 3 or 4 dims, got shape {array.shape}.")
    if array.shape[0] != n_envs:
        raise ValueError(f"Expected {n_envs} video frames, got shape {array.shape}.")

    if array.shape[-1] not in (1, 3, 4) and array.shape[1] in (1, 3, 4):
        array = np.moveaxis(array, 1, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]
    elif array.shape[-1] != 3:
        raise ValueError(f"Expected video frames with 1, 3, or 4 channels, got shape {array.shape}.")

    if array.dtype != np.uint8:
        is_unit_float = (
            np.issubdtype(array.dtype, np.floating)
            and array.size
            and array.min() >= 0.0
            and array.max() <= 1.0
        )
        if is_unit_float:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)

    # Match LiberoEnv.render(), which flips the raw simulator camera for visualization.
    array = array[:, ::-1, ::-1, :]
    return np.ascontiguousarray(array)


def _append_video_frames(
    episode_frames: list[list[np.ndarray]],
    raw_observation: Any,
    *,
    video_key: str,
    mask: np.ndarray | None = None,
) -> None:
    n_envs = len(episode_frames)
    frames = _video_image_batch(raw_observation, video_key=video_key, n_envs=n_envs)
    if mask is None:
        indices = range(n_envs)
    else:
        indices = np.flatnonzero(mask.astype(bool))
    for env_idx in indices:
        episode_frames[int(env_idx)].append(frames[int(env_idx)].copy())


def _write_episode_video(
    *,
    frames: list[np.ndarray],
    video_dir: Path,
    mode: str,
    suite_name: str,
    task_id: int,
    episode: int,
    env_idx: int,
    success: bool,
    fps: int,
) -> Path | None:
    if not frames:
        return None
    video_dir.mkdir(parents=True, exist_ok=True)
    status = "success" if success else "failure"
    video_path = video_dir / (
        f"{mode}_{suite_name}_task{task_id:02d}_episode{episode:04d}_env{env_idx}_{status}.mp4"
    )
    write_video(str(video_path), np.stack(frames, axis=0), fps=fps)
    return video_path


def _advance_step_progress(
    step_pbar: tqdm,
    n_steps: int,
    *,
    total_env_steps: int,
    completed: int,
    episodes: int,
    active_step: int,
    max_steps: int,
) -> None:
    if step_pbar.total is None:
        step_pbar.update(n_steps)
    else:
        remaining = int(step_pbar.total - step_pbar.n)
        if remaining > 0:
            step_pbar.update(min(n_steps, remaining))
    step_pbar.set_postfix(
        {
            "completed": f"{completed}/{episodes}",
            "active_step": f"{active_step}/{max_steps}",
            "env_steps": total_env_steps,
        }
    )


def _evaluate_one_task(
    *,
    args,
    mode: str,
    suite_name: str,
    task_id: int,
    pi05_cfg,
    pi05_preprocessor,
    pi05_postprocessor,
    pi05_policy: PI05Policy,
    rlt_policy: RLTPolicy | None,
    chunk_size: int,
    execute_steps: int,
    action_dim: int,
    metrics_path: Path,
) -> dict[str, Any]:
    env_cfg = make_env_config(args.env_type, task=suite_name, task_ids=[task_id])
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, pi05_cfg)
    envs = make_env(env_cfg, n_envs=args.n_envs, use_async_envs=args.use_async_envs)
    env = envs[suite_name][task_id]
    n_envs = int(getattr(env, "num_envs", args.n_envs))
    env_action_dim = _env_action_dim(env)
    if env_action_dim is not None and int(env_action_dim) != int(action_dim):
        raise ValueError(
            f"Environment action_dim {env_action_dim} != policy action_dim {action_dim}. "
            "Use a PI0.5/RLT checkpoint whose action dimension matches this arm-gripper environment."
        )
    task_descriptions = _env_tasks(env, n_envs)
    task_description = task_descriptions[0] if task_descriptions else ""

    include_proprio = bool(rlt_policy is not None and rlt_policy._proprioception_dim > 0)
    adapter = PI05PrefixRLTAdapter(pi05_policy, rlt_chunk_size=chunk_size) if mode == "rlt" else None
    max_steps = args.max_steps_per_episode or env.call("_max_episode_steps")[0]

    completed = 0
    success_count = 0
    total_env_steps = 0
    completed_rewards: list[float] = []
    completed_steps: list[int] = []

    episode_reward = np.zeros(n_envs, dtype=np.float32)
    episode_steps = np.zeros(n_envs, dtype=np.int64)
    episode_success = np.zeros(n_envs, dtype=bool)
    record_episode_videos = args.record_all_videos or not args.no_record_failure_videos
    record_failure_videos = not args.no_record_failure_videos
    video_dir = Path(args.video_dir) if args.video_dir else Path(args.output_dir) / "videos"
    failure_video_dir = (
        Path(args.failure_video_dir) if args.failure_video_dir else Path(args.output_dir) / "failure_videos"
    )
    video_paths: list[str] = []
    failure_video_paths: list[str] = []
    episode_frames: list[list[np.ndarray]] = [[] for _ in range(n_envs)]

    seeds = [args.seed + task_id * 100000 + env_idx for env_idx in range(n_envs)]
    raw_obs, _ = env.reset(seed=seeds)
    if record_episode_videos:
        _append_video_frames(episode_frames, raw_obs, video_key=args.failure_video_key)

    pbar = trange(args.episodes, desc=f"{suite_name}:{task_id}", leave=False)
    step_pbar = tqdm(
        total=args.episodes * max_steps,
        desc=f"{suite_name}:{task_id} steps",
        leave=False,
        unit="step",
    )
    try:
        while completed < args.episodes:
            with torch.inference_mode():
                if mode == "vla":
                    action_chunk = _vla_action_chunk(
                        raw_observation=raw_obs,
                        env=env,
                        env_preprocessor=env_preprocessor,
                        pi05_preprocessor=pi05_preprocessor,
                        pi05_policy=pi05_policy,
                        task_descriptions=task_descriptions,
                    )
                    if action_chunk.shape[1] < chunk_size:
                        raise ValueError(
                            f"PI0.5 action chunk has length {action_chunk.shape[1]}, "
                            f"but evaluation requested {chunk_size}."
                        )
                    action_chunk = action_chunk[:, :chunk_size, :action_dim]
                else:
                    if rlt_policy is None or adapter is None:
                        raise RuntimeError("RLT mode requires rlt_policy.")
                    features = _rlt_features(
                        raw_observation=raw_obs,
                        env=env,
                        env_preprocessor=env_preprocessor,
                        pi05_preprocessor=pi05_preprocessor,
                        adapter=adapter,
                        rlt_device=args.rlt_device,
                        include_proprio=include_proprio,
                        task_descriptions=task_descriptions,
                    )
                    features = _attach_rlt_state(features, rlt_policy)
                    actor_param = next(rlt_policy.actor.parameters())
                    action_chunk = rlt_policy.select_action(
                        _policy_batch_from_features(
                            features,
                            include_proprio=include_proprio,
                            device=actor_param.device,
                            dtype=actor_param.dtype,
                        )
                    )
                    action_chunk = action_chunk.reshape(n_envs, chunk_size, action_dim)

            if not args.no_clip_normalized_action:
                action_chunk = action_chunk.clamp(-1.0, 1.0)

            chunk_reward = np.zeros(n_envs, dtype=np.float32)
            terminated = np.zeros(n_envs, dtype=bool)
            truncated = np.zeros(n_envs, dtype=bool)

            for chunk_step in range(execute_steps):
                action_step = action_chunk[:, chunk_step, :]
                env_action = _postprocess_action_step(
                    action_step,
                    pi05_postprocessor=pi05_postprocessor,
                    env_postprocessor=env_postprocessor,
                    clip_normalized_action=not args.no_clip_normalized_action,
                )
                raw_obs, reward, term, trunc, info = env.step(env_action.detach().cpu().numpy())
                if record_episode_videos:
                    _append_video_frames(episode_frames, raw_obs, video_key=args.failure_video_key)

                reward_array = _as_array(reward, dtype=np.float32, size=n_envs)
                term_array = _as_array(term, dtype=bool, size=n_envs)
                trunc_array = _as_array(trunc, dtype=bool, size=n_envs)
                episode_steps += 1
                timeout_array = episode_steps >= max_steps
                trunc_array |= timeout_array

                success_value = _info_array(info, ("success", "is_success", "task_success"), n_envs)
                if success_value is not None:
                    episode_success |= success_value.astype(bool)

                chunk_reward += reward_array
                episode_reward += reward_array
                total_env_steps += n_envs
                _advance_step_progress(
                    step_pbar,
                    min(n_envs, max(0, args.episodes - completed)),
                    total_env_steps=total_env_steps,
                    completed=completed,
                    episodes=args.episodes,
                    active_step=int(episode_steps.max()),
                    max_steps=max_steps,
                )
                terminated |= term_array
                truncated |= trunc_array

                if np.any(term_array | trunc_array) or completed >= args.episodes:
                    break

            done_mask = terminated | truncated
            if np.any(done_mask):
                for env_idx in np.flatnonzero(done_mask):
                    if completed >= args.episodes:
                        break
                    env_idx = int(env_idx)
                    success = bool(episode_success[env_idx])
                    success_count += int(success)
                    completed += 1
                    completed_rewards.append(float(episode_reward[env_idx]))
                    completed_steps.append(int(episode_steps[env_idx]))
                    episode_video_path = None
                    failure_video_path = None
                    if record_episode_videos and (args.record_all_videos or (record_failure_videos and not success)):
                        episode_video_path = _write_episode_video(
                            frames=episode_frames[env_idx],
                            video_dir=video_dir if args.record_all_videos else failure_video_dir,
                            mode=mode,
                            suite_name=suite_name,
                            task_id=task_id,
                            episode=completed,
                            env_idx=env_idx,
                            success=success,
                            fps=args.failure_video_fps,
                        )
                        if episode_video_path is not None:
                            video_paths.append(str(episode_video_path))
                        if not success and episode_video_path is not None:
                            failure_video_path = episode_video_path
                            failure_video_paths.append(str(failure_video_path))
                    item = {
                        "type": "episode",
                        "mode": mode,
                        "suite": suite_name,
                        "task_id": task_id,
                        "task": task_description,
                        "episode": completed,
                        "env_index": int(env_idx),
                        "success": success,
                        "reward": float(episode_reward[env_idx]),
                        "steps": int(episode_steps[env_idx]),
                        "terminated": bool(terminated[env_idx]),
                        "truncated": bool(truncated[env_idx]),
                        "env_step": int(total_env_steps),
                        "running_success_rate": float(success_count / completed),
                    }
                    if episode_video_path is not None:
                        item["video_path"] = str(episode_video_path)
                    if failure_video_path is not None:
                        item["failure_video_path"] = str(failure_video_path)
                    print(json.dumps(item), flush=True)
                    _write_jsonl(metrics_path, item)
                    pbar.update(1)
                    step_pbar.set_postfix(
                        {
                            "completed": f"{completed}/{args.episodes}",
                            "active_step": f"{int(episode_steps.max())}/{max_steps}",
                            "env_steps": total_env_steps,
                        }
                    )

                if completed >= args.episodes:
                    break

                episode_reward[done_mask] = 0.0
                episode_steps[done_mask] = 0
                episode_success[done_mask] = False
                if record_episode_videos:
                    for env_idx in np.flatnonzero(done_mask):
                        episode_frames[int(env_idx)] = []
                raw_obs, _ = _reset_finished_envs(
                    env,
                    done_mask,
                    seed_base=args.seed + task_id * 100000 + completed * n_envs,
                )
                if record_episode_videos:
                    _append_video_frames(
                        episode_frames,
                        raw_obs,
                        video_key=args.failure_video_key,
                        mask=done_mask,
                    )

        summary = {
            "type": "task_summary",
            "mode": mode,
            "suite": suite_name,
            "task_id": task_id,
            "task": task_description,
            "episodes": completed,
            "success_count": success_count,
            "success_rate": float(success_count / max(1, completed)),
            "mean_reward": float(np.mean(completed_rewards)) if completed_rewards else 0.0,
            "mean_steps": float(np.mean(completed_steps)) if completed_steps else 0.0,
            "total_env_steps": int(total_env_steps),
            "record_all_videos": bool(args.record_all_videos),
            "video_count": len(video_paths),
            "video_paths": video_paths,
            "failure_video_count": len(failure_video_paths),
            "failure_video_paths": failure_video_paths,
        }
        print(json.dumps(summary), flush=True)
        _write_jsonl(metrics_path, summary)
        return summary
    finally:
        step_pbar.close()
        pbar.close()
        close_envs(envs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rlt", "vla"], required=True)
    parser.add_argument("--pi05-policy-path", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--rlt-policy-path", default=None, help="Required for --mode rlt.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-type", default="libero")
    parser.add_argument("--env-task", default=None, help="Defaults to checkpoint metadata or libero_10.")
    parser.add_argument(
        "--env-task-ids",
        default=None,
        help="Comma-separated LIBERO task ids. Defaults to checkpoint metadata for RLT, otherwise 0.",
    )
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--use-async-envs", action="store_true")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes to evaluate per task id.")
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None, help="Defaults to RLT chunk_size or 10 for VLA.")
    parser.add_argument("--execute-steps-per-chunk", type=int, default=None)
    parser.add_argument("--pi05-device", default="cuda")
    parser.add_argument("--rlt-device", default="cuda")
    parser.add_argument("--pi05-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--rlt-dtype",
        default=None,
        choices=["bfloat16", "float16", "float32"],
        help="Optional dtype for the RLT policy. Defaults to the checkpoint dtype.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument(
        "--disable-pi05-compile",
        action="store_true",
        help="Disable torch.compile for PI0.5 inference to avoid Inductor/Triton autotune startup.",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--metrics-file", default=None)
    parser.add_argument(
        "--failure-video-dir",
        default=None,
        help="Directory for failed-episode videos. Defaults to <output-dir>/failure_videos.",
    )
    parser.add_argument(
        "--record-all-videos",
        action="store_true",
        help="Record every completed episode, including successes. Videos default to <output-dir>/videos.",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Directory for --record-all-videos. Defaults to <output-dir>/videos.",
    )
    parser.add_argument("--failure-video-key", default="image", help="Observation pixels key to record.")
    parser.add_argument("--failure-video-fps", type=int, default=20)
    parser.add_argument(
        "--no-record-failure-videos",
        action="store_true",
        help="Disable the default failed-episode video recording.",
    )
    parser.add_argument("--no-clip-normalized-action", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.env_type != "libero":
        raise ValueError("This evaluator currently supports --env-type libero only.")
    if args.mode == "rlt" and args.rlt_policy_path is None:
        raise ValueError("--rlt-policy-path is required when --mode rlt.")
    if args.failure_video_fps <= 0:
        raise ValueError("--failure-video-fps must be positive.")

    set_seed(args.seed)

    metadata = _load_online_metadata(args.rlt_policy_path) if args.mode == "rlt" else {}
    env_task = args.env_task or metadata.get("env_task") or "libero_10"
    task_ids = (
        _parse_task_ids(args.env_task_ids)
        if args.env_task_ids is not None
        else [int(x) for x in metadata.get("env_task_ids", [0])]
    )

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_file) if args.metrics_file else output_dir / "success_metrics.jsonl"

    policy_env_cfg = make_env_config(args.env_type, task=env_task, task_ids=[task_ids[0]])
    pi05_policy, pi05_cfg, pi05_preprocessor, pi05_postprocessor = _make_pi05_policy(
        policy_path=args.pi05_policy_path,
        env_cfg=policy_env_cfg,
        device=args.pi05_device,
        dtype=args.pi05_dtype,
        num_inference_steps=args.num_inference_steps,
        tokenizer_path=args.tokenizer_path,
        disable_compile=args.disable_pi05_compile,
    )

    rlt_policy = None
    rlt_path = None
    if args.mode == "rlt":
        rlt_policy, rlt_path = _load_rlt_policy(args.rlt_policy_path, args.rlt_device, args.rlt_dtype)
        chunk_size = args.chunk_size or int(rlt_policy.config.chunk_size)
        action_dim = int(rlt_policy._action_dim)
    else:
        chunk_size = args.chunk_size or 10
        action_dim = int(pi05_cfg.output_features[ACTION].shape[0])

    execute_steps = args.execute_steps_per_chunk or chunk_size
    execute_steps = min(execute_steps, chunk_size)

    task_summaries = []
    for task_id in task_ids:
        summary = _evaluate_one_task(
            args=args,
            mode=args.mode,
            suite_name=env_task,
            task_id=task_id,
            pi05_cfg=pi05_cfg,
            pi05_preprocessor=pi05_preprocessor,
            pi05_postprocessor=pi05_postprocessor,
            pi05_policy=pi05_policy,
            rlt_policy=rlt_policy,
            chunk_size=chunk_size,
            execute_steps=execute_steps,
            action_dim=action_dim,
            metrics_path=metrics_path,
        )
        task_summaries.append(summary)

    total_episodes = sum(item["episodes"] for item in task_summaries)
    total_successes = sum(item["success_count"] for item in task_summaries)
    summary = {
        "format": "lerobot_pi05_rlt_libero_success_eval_v1",
        "mode": args.mode,
        "pi05_policy_path": args.pi05_policy_path,
        "pi05_dtype": args.pi05_dtype,
        "rlt_policy_path": None if rlt_path is None else str(rlt_path),
        "rlt_dtype": args.rlt_dtype,
        "env_task": env_task,
        "env_task_ids": task_ids,
        "n_envs": args.n_envs,
        "episodes_per_task": args.episodes,
        "chunk_size": chunk_size,
        "execute_steps_per_chunk": execute_steps,
        "record_all_videos": bool(args.record_all_videos),
        "video_dir": None
        if not args.record_all_videos
        else str(Path(args.video_dir) if args.video_dir else output_dir / "videos"),
        "record_failure_videos": not args.no_record_failure_videos,
        "failure_video_dir": None
        if args.no_record_failure_videos
        else str(Path(args.failure_video_dir) if args.failure_video_dir else output_dir / "failure_videos"),
        "failure_video_key": args.failure_video_key,
        "failure_video_fps": args.failure_video_fps,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_success_rate": float(total_successes / max(1, total_episodes)),
        "tasks": task_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"type": "overall_summary", **summary}), flush=True)


if __name__ == "__main__":
    main()
