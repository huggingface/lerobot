#!/usr/bin/env python
"""Collect paired clean/noisy LIBERO rollout frames for reward-model negatives.

This is a debug-first collector. It starts from official LIBERO init states,
runs the policy clean and with structured action noise from the same init state,
then saves side-by-side frame pairs plus a JSON manifest. The resulting frames
are meant to be inspected before generating a large negative set.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from lerobot import envs, policies  # noqa: F401 - registers config subclasses
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import write_image
from lerobot.envs import close_envs
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
    save_step: int = 60
    init_state_start: int = 0
    init_state_stride: int = 1
    noise_std: float = 0.1
    noise_probability: float = 1.0
    noise_dims: str = "xyz"
    noise_temporal_mode: str = "per_step"
    action_low: float = -1.0
    action_high: float = 1.0
    save_clean_individual: bool = True
    save_noisy_individual: bool = True
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
    save_step: int,
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
    saved_frame = _render_base_env_frame(base_env)
    saved_step = 0
    action_dim: int | None = None
    mask: np.ndarray | None = None
    chunk_noise: np.ndarray | None = None
    step_records: list[dict[str, Any]] = []

    for step in range(steps):
        policy_action = action_api.select_action(observation, env=env, task=task).astype(np.float32)
        if action_dim is None:
            action_dim = int(policy_action.shape[0])
            mask = _noise_mask(action_dim, cfg.noise_dims)
            if cfg.noise_temporal_mode == "chunk":
                chunk_noise = _sample_noise(
                    rng=rng,
                    action_dim=action_dim,
                    mask=mask,
                    std=cfg.noise_std,
                    probability=cfg.noise_probability,
                )
        assert mask is not None

        noise = np.zeros_like(policy_action, dtype=np.float32)
        if noisy:
            if cfg.noise_temporal_mode == "per_step":
                noise = _sample_noise(
                    rng=rng,
                    action_dim=int(policy_action.shape[0]),
                    mask=mask,
                    std=cfg.noise_std,
                    probability=cfg.noise_probability,
                )
            elif cfg.noise_temporal_mode == "chunk":
                assert chunk_noise is not None
                noise = chunk_noise.astype(np.float32, copy=True)
            else:
                raise ValueError("`noise_temporal_mode` must be one of: per_step, chunk.")

        executed_action = np.clip(policy_action + noise, cfg.action_low, cfg.action_high).astype(np.float32)
        observation, reward, terminated, truncated, info = _step_base_env_no_reset(base_env, executed_action)
        observation = dict(observation)
        success = success or _extract_success(info)
        terminal = bool(terminated or truncated)
        reward_sum += float(reward)
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
            }
        )

        if step + 1 == save_step or step == steps - 1 or terminal:
            saved_frame = _render_base_env_frame(base_env)
            saved_step = step + 1
        if terminal:
            break

    return {
        "frame": saved_frame,
        "saved_step": saved_step,
        "success": bool(success),
        "terminal": bool(terminal),
        "reward_sum": float(reward_sum),
        "step_records": step_records,
    }


def _pair_image(clean: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    height = max(clean.shape[0], noisy.shape[0])
    width = clean.shape[1] + noisy.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: clean.shape[0], : clean.shape[1]] = clean
    canvas[: noisy.shape[0], clean.shape[1] : clean.shape[1] + noisy.shape[1]] = noisy
    return canvas


@parser.wrap()
def main(cfg: NoisyLiberoRolloutConfig) -> None:
    if cfg.num_pairs <= 0:
        raise ValueError("`num_pairs` must be positive.")
    if cfg.steps <= 0:
        raise ValueError("`steps` must be positive.")
    if cfg.save_step < 0:
        raise ValueError("`save_step` must be non-negative.")
    if cfg.noise_temporal_mode not in {"per_step", "chunk"}:
        raise ValueError("`noise_temporal_mode` must be one of: per_step, chunk.")
    if cfg.noise_probability < 0 or cfg.noise_probability > 1:
        raise ValueError("`noise_probability` must be in [0, 1].")

    init_logging()
    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = cfg.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    action_api, envs_dict = make_action_api(cfg)
    suite, task_id, env = _select_env(envs_dict, suite=cfg.suite, task_id=cfg.task_id)
    if env.num_envs != 1:
        raise ValueError(f"This collector expects a single env, got {env.num_envs}.")
    base_env = _get_single_base_env(env)
    task = _infer_task(env)
    rng = np.random.default_rng(cfg.seed)
    max_save_step = min(cfg.save_step if cfg.save_step > 0 else cfg.steps, cfg.steps)

    manifest: dict[str, Any] = {
        "suite": suite,
        "task_id": task_id,
        "task": task,
        "policy": str(cfg.policy.pretrained_path) if cfg.policy is not None else None,
        "num_pairs": cfg.num_pairs,
        "steps": cfg.steps,
        "save_step": max_save_step,
        "init_state_start": cfg.init_state_start,
        "init_state_stride": cfg.init_state_stride,
        "noise_std": cfg.noise_std,
        "noise_probability": cfg.noise_probability,
        "noise_dims": cfg.noise_dims,
        "noise_temporal_mode": cfg.noise_temporal_mode,
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
                save_step=max_save_step,
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
                save_step=max_save_step,
                noisy=True,
                rng=rng,
                cfg=cfg,
            )

            prefix = f"pair_{pair_ix:03d}_init_{init_state_id:04d}"
            clean_path = image_dir / f"{prefix}_clean_step_{clean['saved_step']:04d}.png"
            noisy_path = image_dir / f"{prefix}_noisy_step_{noisy['saved_step']:04d}.png"
            pair_path = image_dir / f"{prefix}_clean_vs_noisy.png"
            if cfg.save_clean_individual:
                write_image(clean["frame"], clean_path)
            if cfg.save_noisy_individual:
                write_image(noisy["frame"], noisy_path)
            write_image(_pair_image(clean["frame"], noisy["frame"]), pair_path)

            manifest["pairs"].append(
                {
                    "pair_index": pair_ix,
                    "init_state_id": init_state_id,
                    "seed": pair_seed,
                    "clean_image_path": str(clean_path.relative_to(cfg.output_dir)),
                    "noisy_image_path": str(noisy_path.relative_to(cfg.output_dir)),
                    "pair_image_path": str(pair_path.relative_to(cfg.output_dir)),
                    "clean": {
                        "saved_step": clean["saved_step"],
                        "success": clean["success"],
                        "terminal": clean["terminal"],
                        "reward_sum": clean["reward_sum"],
                        "last_step": clean["step_records"][-1] if clean["step_records"] else None,
                    },
                    "noisy": {
                        "saved_step": noisy["saved_step"],
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
                f"pair_image={pair_path}",
                flush=True,
            )
    finally:
        close_envs(env)

    (cfg.output_dir / "manifest.json").write_text(json.dumps(_to_jsonable(manifest), indent=2))
    print(f"[noisy-rollout] wrote {cfg.output_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
