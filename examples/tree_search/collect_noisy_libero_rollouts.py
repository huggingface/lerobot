#!/usr/bin/env python
"""Collect paired clean/noisy LIBERO rollout videos for reward-model negatives.

This is a debug-first collector. It starts from official LIBERO init states,
runs the policy clean and with structured action noise from the same init state,
then saves rollout videos plus a JSON manifest. The resulting videos
are meant to be inspected before generating a large negative set.
"""

import json
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from lerobot import envs, policies  # noqa: F401 - registers config subclasses
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

from policy_inference_api import (
    LeRobotActionAPI,
    _extract_success,
    _get_single_base_env,
    _infer_task,
    _render_base_env_frame,
    _select_env,
    _step_base_env_no_reset,
    _to_jsonable,
    make_action_api,
)


@dataclass
class NoisyLiberoRolloutConfig:
    env: envs.EnvConfig
    policy: PreTrainedConfig | None = None
    output_dir: Path = Path("outputs/tree_search/noisy_libero_debug")
    suite: str | None = None
    task_id: int | None = None
    seed: int | None = 1000
    num_pairs: int = 10
    steps: int = 80
    video_fps: int = 30
    init_state_start: int = 0
    init_state_stride: int = 1
    noise_std: float = 0.1
    noise_probability: float = 1.0
    noise_dims: str = "xyz"
    noise_temporal_mode: str = "constant"
    noise_gain_min: float = 0.7
    noise_gain_max: float = 1.3
    action_low: float = -1.0
    action_high: float = 1.0
    save_pair_video: bool = True
    rename_map: dict[str, str] = field(default_factory=dict)
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)
        elif self.policy is None:
            raise ValueError("Provide a policy with `--policy.path=<hub-id-or-local-path>`.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def _set_init_state_id(base_env: gym.Env, init_state_id: int) -> None:
    if not hasattr(base_env, "init_state_id"):
        raise ValueError(f"Environment {type(base_env).__name__} does not expose init_state_id.")
    setattr(base_env, "init_state_id", int(init_state_id))


def _noise_mask(action_dim: int, noise_dims: str) -> np.ndarray:
    mask = np.zeros(action_dim, dtype=bool)
    if noise_dims == "all":
        mask[:] = True
    elif noise_dims == "xyz":
        mask[: min(3, action_dim)] = True
    elif noise_dims == "rotation":
        mask[3 : min(6, action_dim)] = True
    elif noise_dims == "gripper":
        if action_dim:
            mask[-1] = True
    elif noise_dims == "xyz_gripper":
        mask[: min(3, action_dim)] = True
        if action_dim:
            mask[-1] = True
    else:
        raise ValueError("`noise_dims` must be one of: all, xyz, rotation, gripper, xyz_gripper.")
    return mask


def _sample_noise(
    *,
    rng: np.random.Generator,
    action_dim: int,
    mask: np.ndarray,
    std: float,
    probability: float,
) -> np.ndarray:
    noise = np.zeros(action_dim, dtype=np.float32)
    if std <= 0 or probability <= 0 or not mask.any():
        return noise
    if rng.random() > probability:
        return noise
    noise[mask] = rng.normal(0.0, std, size=int(mask.sum())).astype(np.float32)
    return noise


def _sample_gain(
    *,
    rng: np.random.Generator,
    action_dim: int,
    mask: np.ndarray,
    gain_min: float,
    gain_max: float,
    probability: float,
) -> np.ndarray:
    gain = np.ones(action_dim, dtype=np.float32)
    if probability <= 0 or not mask.any():
        return gain
    if rng.random() > probability:
        return gain
    low = min(gain_min, gain_max)
    high = max(gain_min, gain_max)
    gain[mask] = rng.uniform(low, high, size=int(mask.sum())).astype(np.float32)
    return gain


def _reset_env_to_init_state(
    *,
    action_api: LeRobotActionAPI,
    env: gym.vector.VectorEnv,
    base_env: gym.Env,
    init_state_id: int,
    seed: int | None,
) -> dict[str, Any]:
    action_api.reset()
    _set_init_state_id(base_env, init_state_id)
    if seed is None:
        observation, _ = env.reset()
    else:
        observation, _ = env.reset(seed=[seed])
    return dict(observation)


def _run_rollout(
    *,
    action_api: LeRobotActionAPI,
    env: gym.vector.VectorEnv,
    base_env: gym.Env,
    task: str,
    init_state_id: int,
    seed: int | None,
    steps: int,
    noisy: bool,
    rng: np.random.Generator,
    cfg: NoisyLiberoRolloutConfig,
) -> dict[str, Any]:
    observation = _reset_env_to_init_state(
        action_api=action_api,
        env=env,
        base_env=base_env,
        init_state_id=init_state_id,
        seed=seed,
    )

    terminal = False
    success = False
    reward_sum = 0.0
    action_dim: int | None = None
    mask: np.ndarray | None = None
    constant_noise: np.ndarray | None = None
    rollout_gain: np.ndarray | None = None
    frames: list[np.ndarray] = [_render_base_env_frame(base_env)]
    step_records: list[dict[str, Any]] = []

    for step in range(steps):
        policy_action = action_api.select_action(observation, env=env, task=task).astype(np.float32)
        if action_dim is None:
            action_dim = int(policy_action.shape[0])
            mask = _noise_mask(action_dim, cfg.noise_dims)
            if noisy and cfg.noise_temporal_mode in {"chunk", "constant"}:
                constant_noise = _sample_noise(
                    rng=rng,
                    action_dim=action_dim,
                    mask=mask,
                    std=cfg.noise_std,
                    probability=cfg.noise_probability,
                )
            elif noisy and cfg.noise_temporal_mode == "gain":
                rollout_gain = _sample_gain(
                    rng=rng,
                    action_dim=action_dim,
                    mask=mask,
                    gain_min=cfg.noise_gain_min,
                    gain_max=cfg.noise_gain_max,
                    probability=cfg.noise_probability,
                )
        assert mask is not None

        noise = np.zeros_like(policy_action, dtype=np.float32)
        gain = np.ones_like(policy_action, dtype=np.float32)
        if noisy:
            if cfg.noise_temporal_mode == "per_step":
                noise = _sample_noise(
                    rng=rng,
                    action_dim=int(policy_action.shape[0]),
                    mask=mask,
                    std=cfg.noise_std,
                    probability=cfg.noise_probability,
                )
            elif cfg.noise_temporal_mode in {"chunk", "constant"}:
                assert constant_noise is not None
                noise = constant_noise.astype(np.float32, copy=True)
            elif cfg.noise_temporal_mode == "gain":
                assert rollout_gain is not None
                gain = rollout_gain.astype(np.float32, copy=True)
            else:
                raise ValueError("`noise_temporal_mode` must be one of: per_step, chunk, constant, gain.")

        executed_action = np.clip(policy_action * gain + noise, cfg.action_low, cfg.action_high).astype(
            np.float32
        )
        observation, reward, terminated, truncated, info = _step_base_env_no_reset(base_env, executed_action)
        observation = dict(observation)
        success = success or _extract_success(info)
        terminal = bool(terminated or truncated)
        reward_sum += float(reward)
        frames.append(_render_base_env_frame(base_env))
        step_records.append(
            {
                "step": step + 1,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "success": bool(success),
                "policy_action": policy_action.tolist(),
                "executed_action": executed_action.tolist(),
                "noise": noise.tolist(),
                "gain": gain.tolist(),
            }
        )

        if terminal:
            break

    return {
        "frames": frames,
        "success": bool(success),
        "terminal": bool(terminal),
        "reward_sum": float(reward_sum),
        "step_records": step_records,
    }


def _pair_frames(clean_frames: list[np.ndarray], noisy_frames: list[np.ndarray]) -> list[np.ndarray]:
    pair_count = min(len(clean_frames), len(noisy_frames))
    paired: list[np.ndarray] = []
    for clean, noisy in zip(clean_frames[:pair_count], noisy_frames[:pair_count], strict=False):
        height = max(clean.shape[0], noisy.shape[0])
        width = clean.shape[1] + noisy.shape[1]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[: clean.shape[0], : clean.shape[1]] = clean
        canvas[: noisy.shape[0], clean.shape[1] : clean.shape[1] + noisy.shape[1]] = noisy
        paired.append(canvas)
    return paired


def _close_env_quietly(env: gym.vector.VectorEnv) -> None:
    with suppress(Exception):
        env.close()


@parser.wrap()
def main(cfg: NoisyLiberoRolloutConfig) -> None:
    if cfg.num_pairs <= 0:
        raise ValueError("`num_pairs` must be positive.")
    if cfg.steps <= 0:
        raise ValueError("`steps` must be positive.")
    if cfg.video_fps <= 0:
        raise ValueError("`video_fps` must be positive.")
    if cfg.noise_temporal_mode not in {"per_step", "chunk", "constant", "gain"}:
        raise ValueError("`noise_temporal_mode` must be one of: per_step, chunk, constant, gain.")
    if cfg.noise_probability < 0 or cfg.noise_probability > 1:
        raise ValueError("`noise_probability` must be in [0, 1].")

    init_logging()
    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = cfg.output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    action_api, envs_dict = make_action_api(cfg)
    suite, task_id, env = _select_env(envs_dict, suite=cfg.suite, task_id=cfg.task_id)
    if env.num_envs != 1:
        raise ValueError(f"This collector expects a single env, got {env.num_envs}.")
    base_env = _get_single_base_env(env)
    task = _infer_task(env)
    rng = np.random.default_rng(cfg.seed)

    manifest: dict[str, Any] = {
        "suite": suite,
        "task_id": task_id,
        "task": task,
        "policy": str(cfg.policy.pretrained_path) if cfg.policy is not None else None,
        "num_pairs": cfg.num_pairs,
        "steps": cfg.steps,
        "video_fps": cfg.video_fps,
        "init_state_start": cfg.init_state_start,
        "init_state_stride": cfg.init_state_stride,
        "noise_std": cfg.noise_std,
        "noise_probability": cfg.noise_probability,
        "noise_dims": cfg.noise_dims,
        "noise_temporal_mode": cfg.noise_temporal_mode,
        "noise_gain_min": cfg.noise_gain_min,
        "noise_gain_max": cfg.noise_gain_max,
        "pairs": [],
    }

    try:
        for pair_ix in range(cfg.num_pairs):
            init_state_id = cfg.init_state_start + pair_ix * cfg.init_state_stride
            pair_seed = None if cfg.seed is None else int(cfg.seed) + pair_ix
            clean = _run_rollout(
                action_api=action_api,
                env=env,
                base_env=base_env,
                task=task,
                init_state_id=init_state_id,
                seed=pair_seed,
                steps=cfg.steps,
                noisy=False,
                rng=rng,
                cfg=cfg,
            )
            noisy = _run_rollout(
                action_api=action_api,
                env=env,
                base_env=base_env,
                task=task,
                init_state_id=init_state_id,
                seed=pair_seed,
                steps=cfg.steps,
                noisy=True,
                rng=rng,
                cfg=cfg,
            )

            prefix = f"pair_{pair_ix:03d}_init_{init_state_id:04d}"
            clean_path = video_dir / f"{prefix}_clean.mp4"
            noisy_path = video_dir / f"{prefix}_noisy.mp4"
            pair_path = video_dir / f"{prefix}_clean_vs_noisy.mp4"
            write_video(clean_path, clean["frames"], fps=cfg.video_fps)
            write_video(noisy_path, noisy["frames"], fps=cfg.video_fps)
            if cfg.save_pair_video:
                write_video(pair_path, _pair_frames(clean["frames"], noisy["frames"]), fps=cfg.video_fps)

            manifest["pairs"].append(
                {
                    "pair_index": pair_ix,
                    "init_state_id": init_state_id,
                    "seed": pair_seed,
                    "clean_video_path": str(clean_path.relative_to(cfg.output_dir)),
                    "noisy_video_path": str(noisy_path.relative_to(cfg.output_dir)),
                    "pair_video_path": (
                        str(pair_path.relative_to(cfg.output_dir)) if cfg.save_pair_video else None
                    ),
                    "clean": {
                        "frame_count": len(clean["frames"]),
                        "success": clean["success"],
                        "terminal": clean["terminal"],
                        "reward_sum": clean["reward_sum"],
                        "last_step": clean["step_records"][-1] if clean["step_records"] else None,
                    },
                    "noisy": {
                        "frame_count": len(noisy["frames"]),
                        "success": noisy["success"],
                        "terminal": noisy["terminal"],
                        "reward_sum": noisy["reward_sum"],
                        "last_step": noisy["step_records"][-1] if noisy["step_records"] else None,
                    },
                }
            )
            print(
                "[noisy-rollout] "
                f"pair={pair_ix} init_state={init_state_id} "
                f"clean_success={clean['success']} noisy_success={noisy['success']} "
                f"clean_video={clean_path} noisy_video={noisy_path}",
                flush=True,
            )
    finally:
        (cfg.output_dir / "manifest.json").write_text(json.dumps(_to_jsonable(manifest), indent=2))
        print(f"[noisy-rollout] wrote {cfg.output_dir / 'manifest.json'}", flush=True)
        _close_env_quietly(env)


if __name__ == "__main__":
    main()
