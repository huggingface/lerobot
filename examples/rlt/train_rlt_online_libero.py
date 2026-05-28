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

"""Single-process online RLT actor-critic training in LIBERO.

This script is intentionally direct: one process owns PI0.5, the RLT policy,
the LIBERO environment, replay buffer, and learner update. It is meant to
validate the online RLT data path before moving to a multi-process actor/learner
setup.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

import torch
from tqdm import trange

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
from lerobot.rl.algorithms.rlt.configuration_rlt import RLTAlgorithmConfig
from lerobot.rl.buffer import ReplayBuffer
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def _first_env(envs):
    suite_name = next(iter(envs))
    task_id = next(iter(envs[suite_name]))
    return envs[suite_name][task_id]


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


def _as_array(value: Any, *, dtype, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.size == 1 and size != 1:
        array = np.repeat(array, size)
    if array.size != size:
        raise ValueError(f"Expected {size} values, got shape {np.asarray(value).shape}.")
    return array


def _info_array(info: Any, keys: tuple[str, ...], size: int) -> np.ndarray | None:
    """Extract vector-env scalar metrics when present."""
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
    return None


def _maybe_latest_checkpoint(path: Path) -> Path:
    if path.is_dir() and (path / "config.json").exists():
        return path
    checkpoints = sorted(p for p in path.glob("checkpoint-*") if p.is_dir() and not p.name.endswith(".tmp"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found under {path}")
    return checkpoints[-1]


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


def _load_rlt_policy(path: str, device: str) -> tuple[RLTPolicy, Path]:
    rlt_path = _maybe_latest_checkpoint(Path(path))
    rlt_cfg = PreTrainedConfig.from_pretrained(rlt_path, local_files_only=True)
    rlt_cfg.device = device
    policy = RLTPolicy.from_pretrained(rlt_path, config=rlt_cfg, local_files_only=True, strict=True)
    policy.to(device)
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
def _attach_rlt_state(features: dict[str, torch.Tensor], rlt_policy: RLTPolicy) -> dict[str, torch.Tensor]:
    encoder_param = next(rlt_policy.rl_token_encoder.parameters())
    vla_embeddings = features[OBS_VLA_EMBEDDINGS].to(
        device=encoder_param.device,
        dtype=encoder_param.dtype,
    )
    features[OBS_RLT_STATE] = rlt_policy.rl_token_encoder(vla_embeddings).detach()
    return features


def _slice_state(
    features: dict[str, torch.Tensor],
    *,
    index: int,
    include_proprio: bool,
) -> dict[str, torch.Tensor]:
    state_key = OBS_RLT_STATE if OBS_RLT_STATE in features else OBS_VLA_EMBEDDINGS
    state = {state_key: features[state_key][index : index + 1]}
    if include_proprio and "observation.state" in features:
        state["observation.state"] = features["observation.state"][index : index + 1]
    return state


def _policy_batch_from_features(features: dict[str, torch.Tensor], include_proprio: bool) -> dict[str, torch.Tensor]:
    batch = {
        OBS_REFERENCE_ACTION: features[OBS_REFERENCE_ACTION],
    }
    if OBS_RLT_STATE in features:
        batch[OBS_RLT_STATE] = features[OBS_RLT_STATE]
    else:
        batch[OBS_VLA_EMBEDDINGS] = features[OBS_VLA_EMBEDDINGS]
    if include_proprio and "observation.state" in features:
        batch["observation.state"] = features["observation.state"]
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
    env_action = env_postprocessor({ACTION: action_cpu})[ACTION]
    return env_action


def _save_checkpoint(
    *,
    output_dir: Path,
    step: int,
    rlt_policy: RLTPolicy,
    algorithm,
    total_env_steps: int,
    episode: int,
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{step:06d}"
    tmp_dir = checkpoint_dir.with_name(f"{checkpoint_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    rlt_policy.save_pretrained(tmp_dir)
    torch.save(
        {
            "step": step,
            "total_env_steps": total_env_steps,
            "episode": episode,
            "critics": algorithm.critics.state_dict(),
            "critic_targets": algorithm.critic_targets.state_dict(),
            "optimizers": {name: opt.state_dict() for name, opt in algorithm.optimizers.items()},
        },
        tmp_dir / "online_state.pt",
    )
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    tmp_dir.rename(checkpoint_dir)


def _try_resume_online_state(checkpoint_dir: Path, algorithm) -> tuple[int, int, int]:
    state_path = checkpoint_dir / "online_state.pt"
    if not state_path.exists():
        return 0, 0, 0
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    algorithm.critics.load_state_dict(state["critics"])
    algorithm.critic_targets.load_state_dict(state["critic_targets"])
    for name, opt_state in state.get("optimizers", {}).items():
        if name in algorithm.optimizers:
            algorithm.optimizers[name].load_state_dict(opt_state)
    return int(state.get("step", 0)), int(state.get("total_env_steps", 0)), int(state.get("episode", 0))


def _write_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item) + "\n")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi05-policy-path", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--rlt-policy-path", required=True, help="Stage-1 RLT checkpoint or online checkpoint.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-type", default="libero")
    parser.add_argument("--env-task", default="libero_10")
    parser.add_argument("--env-task-ids", default="0", help="Comma-separated LIBERO task ids. First id is used.")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel LIBERO envs for rollout.")
    parser.add_argument("--use-async-envs", action="store_true", help="Use AsyncVectorEnv instead of SyncVectorEnv.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None, help="Defaults to the RLT policy chunk_size.")
    parser.add_argument("--execute-steps-per-chunk", type=int, default=None)
    parser.add_argument("--buffer-capacity", type=int, default=10000)
    parser.add_argument("--warmup-transitions", type=int, default=20)
    parser.add_argument("--updates-per-chunk", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pi05-device", default="cuda")
    parser.add_argument("--rlt-device", default="cuda")
    parser.add_argument("--storage-device", default="cpu")
    parser.add_argument("--pi05-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument(
        "--disable-pi05-compile",
        action="store_true",
        help="Disable torch.compile for PI0.5 inference to avoid Inductor/Triton autotune startup.",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--log-freq", type=int, default=1)
    parser.add_argument("--save-freq", type=int, default=50)
    parser.add_argument(
        "--success-window",
        type=int,
        default=1000,
        help="Number of recently completed episodes used for success_rate. Cumulative rate is logged separately.",
    )
    parser.add_argument("--metrics-file", default=None)
    parser.add_argument("--tensorboard-log-dir", default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--no-clip-normalized-action", action="store_true")
    parser.add_argument(
        "--exploration-std",
        type=float,
        default=0.02,
        help="Gaussian exploration std added to the RLT action chunk during rollout. Use 0 for deterministic.",
    )
    parser.add_argument(
        "--exploration-steps",
        type=int,
        default=0,
        help="If >0, linearly decay exploration std within each run/window. Default keeps std constant.",
    )
    parser.add_argument("--resume", action="store_true", help="Load online_state.pt from --rlt-policy-path if present.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.resume:
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(args.metrics_file) if args.metrics_file else output_dir / "online_metrics.jsonl"
    tb_dir = Path(args.tensorboard_log_dir) if args.tensorboard_log_dir else output_dir / "runs"
    writer = None if args.no_tensorboard or SummaryWriter is None else SummaryWriter(log_dir=str(tb_dir))

    task_ids = [int(x.strip()) for x in args.env_task_ids.split(",") if x.strip()]
    env_cfg = make_env_config(args.env_type, task=args.env_task, task_ids=task_ids)
    envs = make_env(env_cfg, n_envs=args.n_envs, use_async_envs=args.use_async_envs)
    env = _first_env(envs)
    n_envs = int(getattr(env, "num_envs", args.n_envs))
    task_descriptions = _env_tasks(env, n_envs)

    rlt_policy, rlt_path = _load_rlt_policy(args.rlt_policy_path, args.rlt_device)
    include_proprio = rlt_policy._proprioception_dim > 0
    chunk_size = args.chunk_size or int(rlt_policy.config.chunk_size)
    execute_steps = args.execute_steps_per_chunk or chunk_size
    execute_steps = min(execute_steps, chunk_size)
    action_dim = rlt_policy._action_dim
    env_action_dim = _env_action_dim(env)
    if env_action_dim is not None and int(env_action_dim) != int(action_dim):
        raise ValueError(
            f"Environment action_dim {env_action_dim} != RLT action_dim {action_dim}. "
            "Use a checkpoint whose action dimension matches this arm-gripper environment."
        )

    pi05_policy, pi05_cfg, pi05_preprocessor, pi05_postprocessor = _make_pi05_policy(
        policy_path=args.pi05_policy_path,
        env_cfg=env_cfg,
        device=args.pi05_device,
        dtype=args.pi05_dtype,
        num_inference_steps=args.num_inference_steps,
        tokenizer_path=args.tokenizer_path,
        disable_compile=args.disable_pi05_compile,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, pi05_cfg)
    adapter = PI05PrefixRLTAdapter(pi05_policy, rlt_chunk_size=chunk_size)

    algorithm_cfg = RLTAlgorithmConfig.from_policy_config(rlt_policy.config)
    algorithm = algorithm_cfg.build_algorithm(rlt_policy)
    algorithm.transition_to_online()

    start_update_step = 0
    total_env_steps = 0
    start_episode = 0
    if args.resume:
        start_update_step, total_env_steps, start_episode = _try_resume_online_state(rlt_path, algorithm)
        print(
            f"Resumed online state from {rlt_path}: update_step={start_update_step}, "
            f"env_steps={total_env_steps}, episode={start_episode}",
            flush=True,
        )

    replay_buffer = ReplayBuffer(
        capacity=args.buffer_capacity,
        device=args.rlt_device,
        storage_device=args.storage_device,
        use_drq=False,
    )

    def batch_iterator():
        while True:
            yield replay_buffer.sample(args.batch_size)

    update_step = start_update_step
    max_steps = args.max_steps_per_episode or env.call("_max_episode_steps")[0]
    seeds = [args.seed + start_episode * n_envs + env_idx for env_idx in range(n_envs)]
    raw_obs, _ = env.reset(seed=seeds)
    current = _attach_rlt_state(
        _rlt_features(
            raw_observation=raw_obs,
            env=env,
            env_preprocessor=env_preprocessor,
            pi05_preprocessor=pi05_preprocessor,
            adapter=adapter,
            rlt_device=args.rlt_device,
            include_proprio=include_proprio,
            task_descriptions=task_descriptions,
        ),
        rlt_policy,
    )
    episode_reward = np.zeros(n_envs, dtype=np.float32)
    episode_steps = np.zeros(n_envs, dtype=np.int64)
    episode_success = np.zeros(n_envs, dtype=bool)
    episode_count = np.zeros(n_envs, dtype=np.int64)
    success_count = np.zeros(n_envs, dtype=np.int64)
    completed_episode_reward = np.zeros(n_envs, dtype=np.float32)
    completed_episode_steps = np.zeros(n_envs, dtype=np.int64)
    completed_success = np.zeros(n_envs, dtype=bool)
    recent_successes: deque[bool] = deque(maxlen=max(1, args.success_window))

    try:
        for ep_idx in range(start_episode, args.episodes):
            pbar = trange(max_steps, desc=f"rollout window {ep_idx}", leave=False)
            window_step = 0
            while window_step < max_steps:
                with torch.no_grad():
                    env_batch_size = current[OBS_REFERENCE_ACTION].shape[0]
                    ref_action_chunk = current[OBS_REFERENCE_ACTION].reshape(
                        env_batch_size, chunk_size, action_dim
                    )
                    action_chunk = rlt_policy.select_action(
                        _policy_batch_from_features(current, include_proprio=include_proprio)
                    )
                    action_chunk = action_chunk.reshape(env_batch_size, chunk_size, action_dim)
                    if args.exploration_steps > 0:
                        exploration_scale = max(0.0, 1.0 - window_step / args.exploration_steps)
                    else:
                        exploration_scale = 1.0
                    exploration_std = args.exploration_std * exploration_scale
                    if exploration_std > 0:
                        action_chunk = action_chunk + torch.randn_like(action_chunk) * exploration_std
                    if not args.no_clip_normalized_action:
                        action_chunk = action_chunk.clamp(-1.0, 1.0)
                    action_delta = action_chunk - ref_action_chunk
                    action_delta_l2 = float(action_delta.norm(dim=-1).mean().item())
                    action_delta_linf = float(action_delta.abs().max().item())

                chunk_reward = np.zeros(env_batch_size, dtype=np.float32)
                terminated = np.zeros(env_batch_size, dtype=bool)
                truncated = np.zeros(env_batch_size, dtype=bool)
                success = np.zeros(env_batch_size, dtype=np.float32)
                for chunk_step in range(execute_steps):
                    action_step = action_chunk[:, chunk_step, :]
                    env_action = _postprocess_action_step(
                        action_step,
                        pi05_postprocessor=pi05_postprocessor,
                        env_postprocessor=env_postprocessor,
                        clip_normalized_action=not args.no_clip_normalized_action,
                    )
                    raw_obs, reward, term, trunc, info = env.step(env_action.detach().cpu().numpy())
                    reward_array = _as_array(reward, dtype=np.float32, size=env_batch_size)
                    term_array = _as_array(term, dtype=bool, size=env_batch_size)
                    trunc_array = _as_array(trunc, dtype=bool, size=env_batch_size)
                    episode_steps += 1
                    timeout_array = episode_steps >= max_steps
                    trunc_array |= timeout_array
                    chunk_reward += reward_array
                    episode_reward += reward_array
                    success_value = _info_array(info, ("success", "is_success", "task_success"), env_batch_size)
                    if success_value is not None:
                        success = np.maximum(success, success_value)
                        episode_success |= success_value.astype(bool)
                    total_env_steps += env_batch_size
                    window_step += 1
                    pbar.update(1)
                    terminated |= term_array
                    truncated |= trunc_array
                    if np.any(term_array | trunc_array) or window_step >= max_steps:
                        break

                next_features = _attach_rlt_state(
                    _rlt_features(
                        raw_observation=raw_obs,
                        env=env,
                        env_preprocessor=env_preprocessor,
                        pi05_preprocessor=pi05_preprocessor,
                        adapter=adapter,
                        rlt_device=args.rlt_device,
                        include_proprio=include_proprio,
                        task_descriptions=task_descriptions,
                    ),
                    rlt_policy,
                )

                for env_idx in range(env_batch_size):
                    replay_buffer.add(
                        state=_slice_state(current, index=env_idx, include_proprio=include_proprio),
                        action=action_chunk[env_idx : env_idx + 1].reshape(1, -1).detach(),
                        reward=float(chunk_reward[env_idx]),
                        next_state=_slice_state(next_features, index=env_idx, include_proprio=include_proprio),
                        done=bool(terminated[env_idx]),
                        truncated=bool(truncated[env_idx]),
                        complementary_info={
                            "reference_action": current[OBS_REFERENCE_ACTION][env_idx : env_idx + 1].detach(),
                            "next_reference_action": next_features[OBS_REFERENCE_ACTION][
                                env_idx : env_idx + 1
                            ].detach(),
                        },
                    )

                done_mask = terminated | truncated
                if np.any(done_mask):
                    done_success = done_mask & episode_success
                    completed_episode_reward[done_mask] = episode_reward[done_mask]
                    completed_episode_steps[done_mask] = episode_steps[done_mask]
                    completed_success[done_mask] = episode_success[done_mask]
                    episode_count += done_mask.astype(np.int64)
                    success_count += done_success.astype(np.int64)
                    recent_successes.extend(episode_success[done_mask].tolist())
                    episode_reward[done_mask] = 0.0
                    episode_steps[done_mask] = 0
                    episode_success[done_mask] = False
                    raw_obs, _ = _reset_finished_envs(
                        env,
                        done_mask,
                        seed_base=args.seed + (ep_idx + 1) * 100000 + int(episode_count.sum()) * n_envs,
                    )
                    next_features = _attach_rlt_state(
                        _rlt_features(
                            raw_observation=raw_obs,
                            env=env,
                            env_preprocessor=env_preprocessor,
                            pi05_preprocessor=pi05_preprocessor,
                            adapter=adapter,
                            rlt_device=args.rlt_device,
                            include_proprio=include_proprio,
                            task_descriptions=task_descriptions,
                        ),
                        rlt_policy,
                    )
                current = next_features

                train_stats = {}
                if len(replay_buffer) >= args.warmup_transitions:
                    for _ in range(args.updates_per_chunk):
                        stats = algorithm.update(batch_iterator())
                        update_step += 1
                        train_stats = stats.to_log_dict()
                        if writer is not None:
                            for key, value in train_stats.items():
                                writer.add_scalar(f"train/{key}", value, update_step)

                if update_step == 0 or update_step % args.log_freq == 0:
                    cumulative_success_rate = float(success_count.sum() / max(1, episode_count.sum()))
                    recent_success_rate = float(np.mean(recent_successes)) if recent_successes else None
                    item = {
                        "episode": ep_idx,
                        "n_envs": env_batch_size,
                        "env_step": total_env_steps,
                        "window_step": window_step,
                        "episode_count": int(episode_count.sum()),
                        "episode_env_step_mean": float(episode_steps.mean()),
                        "update_step": update_step,
                        "buffer_size": len(replay_buffer),
                        "chunk_reward_mean": float(chunk_reward.mean()),
                        "chunk_reward_sum": float(chunk_reward.sum()),
                        "episode_reward_mean": float(episode_reward.mean()),
                        "last_completed_episode_reward_mean": float(completed_episode_reward[done_mask].mean())
                        if np.any(done_mask)
                        else None,
                        "last_completed_episode_steps_mean": float(completed_episode_steps[done_mask].mean())
                        if np.any(done_mask)
                        else None,
                        "last_completed_success_rate": float(completed_success[done_mask].mean())
                        if np.any(done_mask)
                        else None,
                        "terminated_count": int(terminated.sum()),
                        "truncated_count": int(truncated.sum()),
                        "success_rate": recent_success_rate,
                        "recent_success_rate": recent_success_rate,
                        "recent_success_count": int(sum(recent_successes)),
                        "success_window_size": len(recent_successes),
                        "success_window_capacity": recent_successes.maxlen,
                        "cumulative_success_rate": cumulative_success_rate,
                        "chunk_success_rate": float(success.astype(bool).mean()),
                        "exploration_std": exploration_std,
                        "action_delta_l2": action_delta_l2,
                        "action_delta_linf": action_delta_linf,
                        **train_stats,
                    }
                    print(json.dumps(item), flush=True)
                    _write_jsonl(metrics_path, item)
                    if writer is not None:
                        writer.add_scalar("rollout/chunk_reward_mean", float(chunk_reward.mean()), total_env_steps)
                        writer.add_scalar("rollout/episode_reward_mean", float(episode_reward.mean()), total_env_steps)
                        writer.add_scalar("rollout/buffer_size", len(replay_buffer), total_env_steps)
                        writer.add_scalar(
                            "rollout/cumulative_success_rate",
                            cumulative_success_rate,
                            total_env_steps,
                        )
                        if recent_success_rate is not None:
                            writer.add_scalar("rollout/success_rate", recent_success_rate, total_env_steps)
                            writer.add_scalar(
                                "rollout/recent_success_rate", recent_success_rate, total_env_steps
                            )
                        writer.add_scalar("rollout/exploration_std", exploration_std, total_env_steps)
                        writer.add_scalar("action/delta_l2", action_delta_l2, total_env_steps)
                        writer.add_scalar("action/delta_linf", action_delta_linf, total_env_steps)

                if args.save_freq > 0 and update_step > 0 and update_step % args.save_freq == 0:
                    _save_checkpoint(
                        output_dir=output_dir,
                        step=update_step,
                        rlt_policy=rlt_policy,
                        algorithm=algorithm,
                        total_env_steps=total_env_steps,
                        episode=ep_idx,
                    )

            pbar.close()

        _save_checkpoint(
            output_dir=output_dir,
            step=max(update_step, 1),
            rlt_policy=rlt_policy,
            algorithm=algorithm,
            total_env_steps=total_env_steps,
            episode=args.episodes,
        )
        rlt_policy.save_pretrained(output_dir)
        (output_dir / "online_training.json").write_text(
            json.dumps(
                {
                    "format": "lerobot_rlt_online_libero_v1",
                    "pi05_policy_path": args.pi05_policy_path,
                    "rlt_initial_path": str(rlt_path),
                    "env_type": args.env_type,
                    "env_task": args.env_task,
                    "env_task_ids": task_ids,
                    "n_envs": n_envs,
                    "use_async_envs": args.use_async_envs,
                    "continuous_rollout": True,
                    "chunk_size": chunk_size,
                    "execute_steps_per_chunk": execute_steps,
                    "success_window": recent_successes.maxlen,
                    "total_env_steps": total_env_steps,
                    "update_step": update_step,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        if writer is not None:
            writer.close()
        close_envs(envs)


if __name__ == "__main__":
    main()
