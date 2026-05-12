#!/usr/bin/env python
"""Collect noisy LIBERO rollouts for reward-model negatives.

By default this writes noisy rollouts as a LeRobotDataset suitable for reward
model training. Debug video export is optional; when enabled, only a side-by-side
clean-vs-noisy video is saved for each init state.
"""

import json
import shutil
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from huggingface_hub import HfApi

from lerobot import envs, policies  # noqa: F401 - registers config subclasses
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.io_utils import write_info
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

from policy_inference_api import (
    LeRobotActionAPI,
    _extract_proprioception,
    _extract_success,
    _get_single_base_env,
    _image_array,
    _infer_task,
    _load_task_language_map,
    _render_base_env_frame,
    _step_base_env_no_reset,
    _translate_task_language,
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
    task_ids: str | None = None
    seed: int | None = 1000
    num_pairs: int = 10
    steps: int = 80
    save_debug_video: bool = False
    video_fps: int = 30
    init_state_start: int = 0
    init_state_stride: int = 1
    noise_std: float = 0.1
    noise_probability: float = 1.0
    noise_dims: str = "xyz"
    randomize_noise_dims: bool = False
    noise_temporal_mode: str = "constant"
    noise_gain_min: float = 0.7
    noise_gain_max: float = 1.3
    dataset_clip_initial_min: int = 50
    dataset_clip_initial_max: int = 80
    action_low: float = -1.0
    action_high: float = 1.0
    save_dataset: bool = True
    dataset_repo_id: str = "local/noisy_libero_rollouts"
    dataset_root: Path | None = None
    dataset_use_videos: bool = True
    dataset_val_fraction: float = 0.2
    dataset_vcodec: str = "libsvtav1"
    dataset_image_writer_threads: int = 4
    dataset_flip_images: bool = True
    overwrite_dataset: bool = False
    push_to_hub: bool = False
    push_private: bool = False
    push_videos: bool = True
    push_branch: str | None = None
    upload_large_folder: bool = False
    task_language_map: Path | None = None
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


def _randomize_noise_mask(*, rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    randomized = np.zeros_like(mask, dtype=bool)
    eligible = np.flatnonzero(mask)
    if eligible.size == 0:
        return randomized
    keep = rng.random(eligible.size) < 0.5
    if not keep.any():
        keep[int(rng.integers(0, eligible.size))] = True
    randomized[eligible[keep]] = True
    return randomized


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


def _dataset_features(*, height: int, width: int, use_videos: bool) -> dict[str, dict[str, Any]]:
    image_dtype = "video" if use_videos else "image"
    return {
        "observation.images.image": {
            "dtype": image_dtype,
            "shape": (3, height, width),
            "names": ["channel", "height", "width"],
        },
        "observation.images.image2": {
            "dtype": image_dtype,
            "shape": (3, height, width),
            "names": ["channel", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": [
                "eef_pos_x",
                "eef_pos_y",
                "eef_pos_z",
                "eef_axisangle_x",
                "eef_axisangle_y",
                "eef_axisangle_z",
                "gripper_qpos_0",
                "gripper_qpos_1",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "rot_x", "rot_y", "rot_z", "gripper"],
        },
        "is_bad_sequence": {"dtype": "bool", "shape": (1,), "names": None},
    }


def _make_dataset(
    cfg: NoisyLiberoRolloutConfig,
    *,
    height: int,
    width: int,
) -> LeRobotDataset | None:
    if not cfg.save_dataset:
        return None
    root = _dataset_root(cfg)
    if root.exists():
        if not cfg.overwrite_dataset:
            raise FileExistsError(
                f"Dataset root already exists: {root}. Use --overwrite_dataset=true or choose a new root."
            )
        shutil.rmtree(root)
    return LeRobotDataset.create(
        repo_id=cfg.dataset_repo_id,
        root=root,
        fps=cfg.video_fps,
        features=_dataset_features(height=height, width=width, use_videos=cfg.dataset_use_videos),
        robot_type="libero",
        use_videos=cfg.dataset_use_videos,
        image_writer_threads=cfg.dataset_image_writer_threads,
        vcodec=cfg.dataset_vcodec,
    )


def _dataset_root(cfg: NoisyLiberoRolloutConfig) -> Path:
    return cfg.dataset_root if cfg.dataset_root is not None else cfg.output_dir / "dataset"


def _validate_output_paths(cfg: NoisyLiberoRolloutConfig) -> None:
    if not cfg.save_dataset:
        return
    root = _dataset_root(cfg)
    if root.exists() and not cfg.overwrite_dataset:
        raise FileExistsError(
            f"Dataset root already exists: {root}. "
            "Pass --overwrite_dataset=true to replace it, or use a new --output_dir/--dataset_root."
        )


def _parse_task_ids(raw: str, *, available_task_ids: list[int]) -> list[int]:
    text = raw.strip()
    if text.lower() == "all":
        return list(available_task_ids)
    if text.startswith("["):
        task_ids = [int(item) for item in json.loads(text)]
    else:
        task_ids = [int(item.strip()) for item in text.split(",") if item.strip()]
    return list(dict.fromkeys(task_ids))


def _select_collection_tasks(
    envs_dict: dict[str, dict[int, gym.vector.VectorEnv]],
    *,
    suite: str | None,
    task_id: int | None,
    task_ids: str | None,
) -> tuple[str, list[int]]:
    if suite is None:
        suite = next(iter(envs_dict))
    if suite not in envs_dict:
        raise ValueError(f"Unknown suite '{suite}'. Available suites: {list(envs_dict)}")

    available_task_ids = sorted(envs_dict[suite])
    if task_ids is not None:
        selected_task_ids = _parse_task_ids(task_ids, available_task_ids=available_task_ids)
    elif task_id is not None:
        selected_task_ids = [int(task_id)]
    else:
        selected_task_ids = available_task_ids

    missing = [task for task in selected_task_ids if task not in envs_dict[suite]]
    if missing:
        raise ValueError(
            f"Unknown task_ids {missing} for suite '{suite}'. "
            f"Available env task ids: {available_task_ids}. "
            "Make sure --env.task_ids includes every requested task."
        )
    return suite, selected_task_ids


def _observation_image(observation: dict[str, Any], key: str, *, flip_hw: bool) -> np.ndarray:
    pixels = observation.get("pixels")
    if not isinstance(pixels, dict):
        raise KeyError("Observation has no `pixels` dictionary.")
    image = _image_array(pixels.get(key), flip_hw=flip_hw)
    if image is None:
        raise KeyError(f"Observation has no image key `pixels.{key}`.")
    return image


def _dataset_frame(
    *,
    observation: dict[str, Any],
    action: np.ndarray,
    task: str,
    is_bad_sequence: bool,
    flip_images: bool,
) -> dict[str, Any]:
    proprioception = _extract_proprioception(observation)
    if proprioception is None:
        raise ValueError("Could not extract observation.state proprioception for dataset frame.")
    return {
        "task": task,
        "observation.images.image": _observation_image(observation, "image", flip_hw=flip_images),
        "observation.images.image2": _observation_image(observation, "image2", flip_hw=flip_images),
        "observation.state": proprioception.astype(np.float32),
        "action": action.astype(np.float32),
        "is_bad_sequence": np.array([is_bad_sequence], dtype=np.bool_),
    }


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
    collect_frames: bool,
    collect_dataset: bool,
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
    dataset_clip_initial = (
        int(rng.integers(cfg.dataset_clip_initial_min, cfg.dataset_clip_initial_max + 1))
        if noisy and collect_dataset and cfg.dataset_clip_initial_max > 0
        else 0
    )
    action_dim: int | None = None
    mask: np.ndarray | None = None
    constant_noise: np.ndarray | None = None
    rollout_gain: np.ndarray | None = None
    frames: list[np.ndarray] = [_render_base_env_frame(base_env)] if collect_frames else []
    dataset_frames: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []

    for step in range(steps):
        policy_action = action_api.select_action(observation, env=env, task=task).astype(np.float32)
        if action_dim is None:
            action_dim = int(policy_action.shape[0])
            mask = _noise_mask(action_dim, cfg.noise_dims)
            if noisy and cfg.randomize_noise_dims:
                mask = _randomize_noise_mask(rng=rng, mask=mask)
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
        step_success = _extract_success(info)
        success = success or step_success
        terminal = bool(terminated or truncated or step_success)
        reward_sum += float(reward)
        if collect_frames:
            frames.append(_render_base_env_frame(base_env))
        if collect_dataset and step + 1 > dataset_clip_initial:
            dataset_frames.append(
                _dataset_frame(
                    observation=observation,
                    action=executed_action,
                    task=task,
                    is_bad_sequence=True,
                    flip_images=cfg.dataset_flip_images,
                )
            )
        step_records.append(
            {
                "step": step + 1,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "success": bool(success),
                "step_success": bool(step_success),
                "terminal": bool(terminal),
                "policy_action": policy_action.tolist(),
                "executed_action": executed_action.tolist(),
                "noise": noise.tolist(),
                "noise_mask": mask.astype(bool).tolist(),
                "gain": gain.tolist(),
            }
        )

        if terminal:
            break

    return {
        "frames": frames,
        "dataset_frames": dataset_frames,
        "success": bool(success),
        "terminal": bool(terminal),
        "reward_sum": float(reward_sum),
        "noise_mask": mask.astype(bool).tolist() if mask is not None else None,
        "dataset_clip_initial": dataset_clip_initial,
        "dataset_frame_count": len(dataset_frames),
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


def _write_debug_video(path: Path, frames: list[np.ndarray], *, fps: int) -> str | None:
    if not frames:
        return "No frames were available for debug video."
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_video(path, frames, fps=fps)
    except Exception as exc:  # Debug video is optional; never abort data generation for it.
        error = f"{type(exc).__name__}: {exc!s}"
        print(f"[noisy-rollout] warning: failed to write debug video {path}: {error}", flush=True)
        return error
    return None


def _push_dataset(dataset: LeRobotDataset, cfg: NoisyLiberoRolloutConfig) -> dict[str, Any]:
    data_dir = dataset.root / "data"
    has_data_files = data_dir.exists() and any(data_dir.rglob("*.parquet"))
    images_dir = dataset.root / "images"
    has_image_files = images_dir.exists() and any(path.is_file() for path in images_dir.rglob("*"))

    dataset.push_to_hub(
        branch=cfg.push_branch,
        private=cfg.push_private,
        push_videos=cfg.push_videos,
        upload_large_folder=cfg.upload_large_folder,
    )

    pushed_image_files = False
    if not cfg.dataset_use_videos and has_image_files:
        # LeRobotDataset.push_to_hub() ignores images/ because regular datasets
        # are usually video-backed. If a local image directory remains, upload it
        # explicitly. In the common image-backed path, images are already embedded
        # into data/*.parquet and this directory is removed after save_episode().
        api = HfApi()
        if cfg.upload_large_folder:
            api.upload_large_folder(
                repo_id=dataset.repo_id,
                folder_path=dataset.root,
                repo_type="dataset",
                revision=cfg.push_branch,
                private=cfg.push_private,
                allow_patterns="images/**",
            )
        else:
            api.upload_folder(
                repo_id=dataset.repo_id,
                folder_path=dataset.root,
                repo_type="dataset",
                revision=cfg.push_branch,
                allow_patterns="images/**",
            )
        pushed_image_files = True

    embedded_images_in_parquet = bool(not cfg.dataset_use_videos and has_data_files)

    return {
        "pushed_images": bool(embedded_images_in_parquet or pushed_image_files),
        "embedded_images_in_parquet": embedded_images_in_parquet,
        "pushed_image_files": pushed_image_files,
        "has_data_files": has_data_files,
        "visual_storage": "videos" if cfg.dataset_use_videos else "images",
    }


def _apply_dataset_splits(dataset: LeRobotDataset, cfg: NoisyLiberoRolloutConfig) -> dict[str, Any]:
    if cfg.dataset_val_fraction < 0 or cfg.dataset_val_fraction >= 1:
        raise ValueError("`dataset_val_fraction` must be in [0, 1).")
    total_frames = int(dataset.meta.info.get("total_frames", 0))
    if total_frames <= 1 or cfg.dataset_val_fraction == 0:
        splits = {"train": f"0:{total_frames}"}
    else:
        val_count = max(1, int(round(total_frames * cfg.dataset_val_fraction)))
        val_count = min(val_count, total_frames - 1)
        train_count = total_frames - val_count
        splits = {
            "train": f"0:{train_count}",
            "validation": f"{train_count}:{total_frames}",
        }
    dataset.meta.info["splits"] = splits
    dataset.meta.info["split_unit"] = "frame"
    write_info(dataset.meta.info, dataset.root)
    return {
        "splits": splits,
        "split_unit": "frame",
        "total_frames": total_frames,
        "total_episodes": int(dataset.meta.info.get("total_episodes", 0)),
        "val_fraction": cfg.dataset_val_fraction,
    }


def _close_env_quietly(env: gym.vector.VectorEnv) -> None:
    with suppress(Exception):
        env.close()


def _camera_size_from_observation(observation: dict[str, Any]) -> tuple[int, int]:
    image = _observation_image(observation, "image", flip_hw=False)
    return int(image.shape[0]), int(image.shape[1])


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
    if cfg.dataset_clip_initial_min < 0 or cfg.dataset_clip_initial_max < 0:
        raise ValueError("`dataset_clip_initial_min/max` must be non-negative.")
    if cfg.dataset_clip_initial_min > cfg.dataset_clip_initial_max:
        raise ValueError("`dataset_clip_initial_min` must be <= `dataset_clip_initial_max`.")
    if cfg.dataset_val_fraction < 0 or cfg.dataset_val_fraction >= 1:
        raise ValueError("`dataset_val_fraction` must be in [0, 1).")
    if not cfg.save_dataset and not cfg.save_debug_video:
        raise ValueError("Nothing to write: enable --save_dataset=true or --save_debug_video=true.")

    init_logging()
    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _validate_output_paths(cfg)
    video_dir = cfg.output_dir / "debug_videos"
    if cfg.save_debug_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    action_api, envs_dict = make_action_api(cfg)
    suite, selected_task_ids = _select_collection_tasks(
        envs_dict,
        suite=cfg.suite,
        task_id=cfg.task_id,
        task_ids=cfg.task_ids,
    )
    selected_envs = [envs_dict[suite][task] for task in selected_task_ids]
    for selected_env in selected_envs:
        if selected_env.num_envs != 1:
            raise ValueError(f"This collector expects single envs, got {selected_env.num_envs}.")
    rng = np.random.default_rng(cfg.seed)

    first_env = selected_envs[0]
    first_base_env = _get_single_base_env(first_env)
    shape_observation = _reset_env_to_init_state(
        action_api=action_api,
        env=first_env,
        base_env=first_base_env,
        init_state_id=cfg.init_state_start,
        seed=cfg.seed,
    )
    height, width = _camera_size_from_observation(shape_observation)
    dataset = _make_dataset(cfg, height=height, width=width)
    task_language_translations = _load_task_language_map(cfg.task_language_map)
    original_task_languages = {
        str(task): _infer_task(envs_dict[suite][task])
        for task in selected_task_ids
    }
    task_languages = {
        str(task): _translate_task_language(
            original_task_languages[str(task)],
            suite=suite,
            task_id=task,
            translations=task_language_translations,
        )
        for task in selected_task_ids
    }

    manifest: dict[str, Any] = {
        "suite": suite,
        "task_id": selected_task_ids[0] if len(selected_task_ids) == 1 else None,
        "task_ids": selected_task_ids,
        "tasks": task_languages,
        "original_tasks": original_task_languages,
        "task_language_map": str(cfg.task_language_map) if cfg.task_language_map is not None else None,
        "policy": str(cfg.policy.pretrained_path) if cfg.policy is not None else None,
        "num_pairs": cfg.num_pairs,
        "num_pairs_per_task": cfg.num_pairs,
        "steps": cfg.steps,
        "save_dataset": cfg.save_dataset,
        "dataset_root": str(dataset.root) if dataset is not None else None,
        "dataset_repo_id": dataset.repo_id if dataset is not None else None,
        "dataset_use_videos": cfg.dataset_use_videos,
        "dataset_val_fraction": cfg.dataset_val_fraction,
        "dataset_flip_images": cfg.dataset_flip_images,
        "dataset_extra_columns": ["is_bad_sequence"] if dataset is not None else [],
        "push_to_hub": cfg.push_to_hub,
        "push_branch": cfg.push_branch,
        "push_private": cfg.push_private,
        "push_videos": cfg.push_videos,
        "upload_large_folder": cfg.upload_large_folder,
        "save_debug_video": cfg.save_debug_video,
        "video_fps": cfg.video_fps,
        "init_state_start": cfg.init_state_start,
        "init_state_stride": cfg.init_state_stride,
        "noise_std": cfg.noise_std,
        "noise_probability": cfg.noise_probability,
        "noise_dims": cfg.noise_dims,
        "randomize_noise_dims": cfg.randomize_noise_dims,
        "noise_temporal_mode": cfg.noise_temporal_mode,
        "noise_gain_min": cfg.noise_gain_min,
        "noise_gain_max": cfg.noise_gain_max,
        "dataset_clip_initial_min": cfg.dataset_clip_initial_min,
        "dataset_clip_initial_max": cfg.dataset_clip_initial_max,
        "accepted_pairs": 0,
        "skipped_successful_noisy_pairs": 0,
        "pairs": [],
        "completed": False,
        "pushed_to_hub": False,
    }

    completed = False
    try:
        for task_id in selected_task_ids:
            env = envs_dict[suite][task_id]
            base_env = _get_single_base_env(env)
            task = task_languages[str(task_id)]
            original_task = original_task_languages[str(task_id)]
            for task_pair_ix in range(cfg.num_pairs):
                pair_ix = len(manifest["pairs"])
                init_state_id = cfg.init_state_start + task_pair_ix * cfg.init_state_stride
                pair_seed = None if cfg.seed is None else int(cfg.seed) + pair_ix
                clean = None
                if cfg.save_debug_video:
                    clean = _run_rollout(
                        action_api=action_api,
                        env=env,
                        base_env=base_env,
                        task=task,
                        init_state_id=init_state_id,
                        seed=pair_seed,
                        steps=cfg.steps,
                        noisy=False,
                        collect_frames=True,
                        collect_dataset=False,
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
                    collect_frames=cfg.save_debug_video,
                    collect_dataset=dataset is not None,
                    rng=rng,
                    cfg=cfg,
                )

                prefix = f"task_{task_id:03d}_pair_{task_pair_ix:03d}_init_{init_state_id:04d}"
                pair_path = video_dir / f"{prefix}_clean_vs_noisy.mp4"
                pair_video_error = None
                if cfg.save_debug_video:
                    assert clean is not None
                    pair_video_error = _write_debug_video(
                        pair_path,
                        _pair_frames(clean["frames"], noisy["frames"]),
                        fps=cfg.video_fps,
                    )
                dataset_episode_index = None
                accepted = not noisy["success"]
                skip_reason = "noisy_rollout_succeeded" if not accepted else None
                if dataset is not None and accepted:
                    dataset_episode_index = int(dataset.meta.total_episodes)
                    for frame in noisy["dataset_frames"]:
                        dataset.add_frame(frame)
                    dataset.save_episode()
                if accepted:
                    manifest["accepted_pairs"] += 1
                else:
                    manifest["skipped_successful_noisy_pairs"] += 1

                manifest["pairs"].append(
                    {
                        "pair_index": pair_ix,
                        "task_pair_index": task_pair_ix,
                        "task_id": task_id,
                        "task": task,
                        "original_task": original_task,
                        "init_state_id": init_state_id,
                        "seed": pair_seed,
                        "accepted": accepted,
                        "skip_reason": skip_reason,
                        "dataset_episode_index": dataset_episode_index,
                        "pair_video_path": (
                            str(pair_path.relative_to(cfg.output_dir))
                            if cfg.save_debug_video and pair_video_error is None
                            else None
                        ),
                        "pair_video_error": pair_video_error,
                        "clean": (
                            {
                                "frame_count": len(clean["frames"]),
                                "success": clean["success"],
                                "terminal": clean["terminal"],
                                "reward_sum": clean["reward_sum"],
                                "noise_mask": clean["noise_mask"],
                                "dataset_clip_initial": clean["dataset_clip_initial"],
                                "dataset_frame_count": clean["dataset_frame_count"],
                                "last_step": clean["step_records"][-1] if clean["step_records"] else None,
                            }
                            if clean is not None
                            else None
                        ),
                        "noisy": {
                            "frame_count": len(noisy["frames"]),
                            "success": noisy["success"],
                            "terminal": noisy["terminal"],
                            "reward_sum": noisy["reward_sum"],
                            "noise_mask": noisy["noise_mask"],
                            "dataset_clip_initial": noisy["dataset_clip_initial"],
                            "dataset_frame_count": noisy["dataset_frame_count"],
                            "last_step": noisy["step_records"][-1] if noisy["step_records"] else None,
                        },
                    }
                )
                print(
                    "[noisy-rollout] "
                    f"task_id={task_id} pair={task_pair_ix} global_pair={pair_ix} "
                    f"init_state={init_state_id} "
                    f"clean_success={clean['success'] if clean is not None else None} "
                    f"noisy_success={noisy['success']} "
                    f"accepted={accepted} "
                    f"dataset_episode={dataset_episode_index} "
                    f"debug_video={pair_path if cfg.save_debug_video and pair_video_error is None else None}",
                    flush=True,
                )
        completed = True
    finally:
        manifest["completed"] = completed
        (cfg.output_dir / "manifest.json").write_text(json.dumps(_to_jsonable(manifest), indent=2))
        print(f"[noisy-rollout] wrote {cfg.output_dir / 'manifest.json'}", flush=True)
        if dataset is not None:
            dataset.finalize()
        for selected_env in selected_envs:
            _close_env_quietly(selected_env)

    if completed and dataset is not None:
        split_info = _apply_dataset_splits(dataset, cfg)
        manifest["dataset_split_info"] = split_info
        (cfg.output_dir / "manifest.json").write_text(json.dumps(_to_jsonable(manifest), indent=2))
        print(f"[noisy-rollout] dataset splits={split_info['splits']}", flush=True)

    if completed and dataset is not None and cfg.push_to_hub:
        print(f"[noisy-rollout] pushing dataset to Hugging Face Hub repo_id={dataset.repo_id}", flush=True)
        push_info = _push_dataset(dataset, cfg)
        manifest["pushed_to_hub"] = True
        manifest["push_info"] = push_info
        (cfg.output_dir / "manifest.json").write_text(json.dumps(_to_jsonable(manifest), indent=2))
        print(
            "[noisy-rollout] "
            f"pushed dataset to Hugging Face Hub repo_id={dataset.repo_id} "
            f"visual_storage={push_info['visual_storage']} pushed_images={push_info['pushed_images']} "
            f"embedded_images_in_parquet={push_info['embedded_images_in_parquet']} "
            f"pushed_image_files={push_info['pushed_image_files']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
