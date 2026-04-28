#!/usr/bin/env python
"""Minimal LeRobot policy inference API for external planners.

This example intentionally keeps planning outside LeRobot. It loads an env,
policy, and processors through the same factories used by `lerobot-eval`, then
exposes a small action API that a tree-search implementation can call from its
own simulator snapshot/restore loop.

Example:

```bash
uv run python examples/tree_search/policy_inference_api.py \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --steps=20 \
    --policy.device=cuda \
    --policy.use_amp=false
```
"""

import base64
import heapq
import io
import json
import logging
import math
import os
import time
import re
from collections.abc import Mapping, Sequence
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from lerobot import envs, policies  # noqa: F401 - registers config subclasses
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import write_image
from lerobot.envs import (
    close_envs,
    make_env,
    make_env_pre_post_processors,
    preprocess_observation,
)
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.types import PolicyAction
from lerobot.utils.constants import ACTION
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)
TREE_SEARCH_DIR = Path(__file__).resolve().parent


@dataclass
class PolicyInferenceConfig:
    env: envs.EnvConfig
    policy: PreTrainedConfig | None = None
    steps: int = 20
    planner: str = "baseline"
    chunk_size: int = 15
    search_num_candidates: int = 4
    search_noise_std: float = 0.05
    search_noise_mode: str = "chunk"
    mcts_simulations: int = 16
    mcts_depth: int = 2
    mcts_exploration: float = 1.4
    best_first_expansions: int = 16
    best_first_depth: int = 2
    mcts_num_candidates: int | None = None
    mcts_noise_std: float | None = None
    mcts_noise_mode: str | None = None
    suite: str | None = None
    task_id: int | None = None
    seed: int | None = 1000
    video_path: Path | None = None
    trace_dir: Path | None = None
    trace_save_step_images: bool = True
    vlm_api_url: str | None = None
    vlm_api_key_env: str | None = None
    vlm_base_url: str | None = None
    vlm_model: str | None = None
    vlm_max_tokens: int = 512
    vlm_temperature: float = 0.6
    vlm_top_p: float = 0.95
    vlm_presence_penalty: float = 0.0
    vlm_top_k: int = 20
    vlm_min_p: float = 0.0
    vlm_repetition_penalty: float = 1.0
    vlm_timeout_s: float = 30.0
    vlm_requests_per_minute: float = 10.0
    vlm_max_retries: int = 3
    vlm_retry_sleep_s: float = 6.0
    vlm_rate_limit_sleep_s: float = 60.0
    vlm_log_response_chars: int = 200
    vlm_verbose: bool = False
    search_verbose: bool = False
    vlm_observation_image_keys: str = "image2"
    vlm_include_rendered_image: bool = True
    vlm_include_robot_state: bool = True
    vlm_reference_image_dir: Path | None = TREE_SEARCH_DIR / "references"
    vlm_reference_image_paths: str | None = None
    vlm_reference_image_glob: str = "**/*"
    vlm_reference_image_max_size: int = 512
    vlm_reference_filter_by_task: bool = True
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


class LeRobotActionAPI:
    """Single-environment action-query wrapper for an already-loaded LeRobot policy.

    The method mutates policy inference caches exactly like `lerobot-eval`.
    For tree search, use this API for the action actually committed to the
    environment, or wrap hypothetical calls with a policy-state strategy in
    your planner if the policy uses action queues.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
        *,
        device: torch.device,
        use_amp: bool,
    ) -> None:
        self.policy = policy
        self.env_preprocessor = env_preprocessor
        self.env_postprocessor = env_postprocessor
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.use_amp = use_amp

    def reset(self) -> None:
        self.policy.reset()

    def _prepare_observation(
        self,
        raw_observation: Mapping[str, Any],
        *,
        env: gym.vector.VectorEnv | None = None,
        task: str | None = None,
    ) -> dict[str, Any]:
        observation = preprocess_observation(dict(raw_observation))
        observation["task"] = [task if task is not None else _infer_task(env)]

        observation = self.env_preprocessor(observation)
        return self.preprocessor(observation)

    def select_action_tensor(
        self,
        raw_observation: Mapping[str, Any],
        *,
        env: gym.vector.VectorEnv | None = None,
        task: str | None = None,
    ) -> Tensor:
        """Return one env-ready action tensor with shape `(action_dim,)`."""
        observation = self._prepare_observation(raw_observation, env=env, task=task)
        amp_context = torch.autocast(device_type=self.device.type) if self.use_amp else nullcontext()
        with torch.inference_mode(), amp_context:
            action = self.policy.select_action(observation)

        action = self.postprocessor(action)
        action_transition = self.env_postprocessor({ACTION: action})
        action = action_transition[ACTION]
        if action.ndim != 2 or action.shape[0] != 1:
            raise ValueError(f"Expected action shape `(1, action_dim)`, got {tuple(action.shape)}.")
        return action[0]

    def select_action(
        self,
        raw_observation: Mapping[str, Any],
        *,
        env: gym.vector.VectorEnv | None = None,
        task: str | None = None,
    ) -> np.ndarray:
        """Return one env-ready action array with shape `(action_dim,)`."""
        action = self.select_action_tensor(raw_observation, env=env, task=task)
        action_numpy = action.to("cpu").numpy()
        if action_numpy.ndim != 1:
            raise ValueError(f"Expected action shape `(action_dim,)`, got {action_numpy.shape}.")
        return action_numpy

    def predict_action_chunk_tensor(
        self,
        raw_observation: Mapping[str, Any],
        *,
        env: gym.vector.VectorEnv | None = None,
        task: str | None = None,
        horizon: int = 15,
    ) -> Tensor:
        """Return an env-ready action chunk with shape `(horizon, action_dim)`.

        Unlike ``select_action``, this does not use the policy action queue. This
        is the right API for search nodes, where every hypothetical state should
        get its own policy proposal.
        """
        observation = self._prepare_observation(raw_observation, env=env, task=task)
        amp_context = torch.autocast(device_type=self.device.type) if self.use_amp else nullcontext()
        with torch.inference_mode(), amp_context:
            if hasattr(self.policy, "predict_action_chunk"):
                action = self.policy.predict_action_chunk(observation)
            else:
                action = self.policy.select_action(observation).unsqueeze(1)

        if action.ndim == 2:
            action = action.unsqueeze(1)
        if action.ndim != 3 or action.shape[0] != 1:
            raise ValueError(
                f"Expected action chunk shape `(1, horizon, action_dim)`, got {tuple(action.shape)}."
            )

        action = action[:, :horizon]
        action = self.postprocessor(action)
        action_transition = self.env_postprocessor({ACTION: action})
        action = action_transition[ACTION]
        if action.ndim != 3 or action.shape[0] != 1:
            raise ValueError(
                f"Expected postprocessed chunk shape `(1, horizon, action_dim)`, got {tuple(action.shape)}."
            )
        return action[0]

    def predict_action_chunk(
        self,
        raw_observation: Mapping[str, Any],
        *,
        env: gym.vector.VectorEnv | None = None,
        task: str | None = None,
        horizon: int = 15,
    ) -> np.ndarray:
        action = self.predict_action_chunk_tensor(raw_observation, env=env, task=task, horizon=horizon)
        action_numpy = action.to("cpu").numpy()
        if action_numpy.ndim != 2:
            raise ValueError(
                f"Expected action chunk shape `(horizon, action_dim)`, got {action_numpy.shape}."
            )
        return action_numpy


def _infer_task(env: gym.vector.VectorEnv | None) -> str:
    if env is None:
        return ""

    try:
        return str(env.call("task_description")[0])
    except (AttributeError, NotImplementedError):
        try:
            return str(env.call("task")[0])
        except (AttributeError, NotImplementedError):
            return ""


def _select_env(
    envs_dict: dict[str, dict[int, gym.vector.VectorEnv]],
    *,
    suite: str | None,
    task_id: int | None,
) -> tuple[str, int, gym.vector.VectorEnv]:
    if suite is None:
        suite = next(iter(envs_dict))
    if suite not in envs_dict:
        raise ValueError(f"Unknown suite '{suite}'. Available suites: {list(envs_dict)}")

    task_envs = envs_dict[suite]
    if task_id is None:
        task_id = next(iter(task_envs))
    if task_id not in task_envs:
        raise ValueError(f"Unknown task_id '{task_id}' for suite '{suite}'. Available: {list(task_envs)}")

    return suite, task_id, task_envs[task_id]


def _extract_success(info: Mapping[str, Any]) -> bool:
    """Extract single-env task success from Gym/Gymnasium vector info."""

    def _first_bool(value: Any) -> bool:
        arr = np.asarray(value)
        if arr.ndim == 0:
            return bool(arr.item())
        return bool(arr[0])

    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, Mapping):
            successes = final_info.get("is_success")
            if successes is not None:
                return _first_bool(successes)
        elif isinstance(final_info, (list, tuple)) and final_info:
            return bool(final_info[0].get("is_success", False))

    if "is_success" in info:
        return _first_bool(info["is_success"])

    return False


def _render_single_env_frame(env: gym.vector.VectorEnv) -> np.ndarray:
    if isinstance(env, gym.vector.SyncVectorEnv):
        return env.envs[0].render()
    return env.call("render")[0]


def _infer_render_fps(env: gym.vector.VectorEnv, fallback: int) -> int:
    try:
        fps = env.unwrapped.metadata["render_fps"]
    except (AttributeError, KeyError, TypeError):
        fps = fallback
    return int(fps)


def _get_single_base_env(env: gym.vector.VectorEnv) -> gym.Env:
    if not isinstance(env, gym.vector.SyncVectorEnv):
        raise ValueError("Search planners require a single SyncVectorEnv so simulator state can be restored.")
    return env.envs[0]


def _snapshot_base_env(base_env: gym.Env) -> np.ndarray:
    if not hasattr(base_env, "snapshot"):
        raise ValueError(f"Environment {type(base_env).__name__} does not expose snapshot().")
    return base_env.snapshot()


def _restore_base_env(
    base_env: gym.Env, state: np.ndarray, *, timestep: int | None = None
) -> Mapping[str, Any]:
    if not hasattr(base_env, "restore"):
        raise ValueError(f"Environment {type(base_env).__name__} does not expose restore().")
    try:
        return base_env.restore(state, timestep=timestep)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        if timestep is not None:
            logger.debug("Environment restore() does not accept timestep; restoring simulator state only.")
        return base_env.restore(state)


def _step_base_env_no_reset(
    base_env: gym.Env, action: np.ndarray
) -> tuple[Mapping[str, Any], float, bool, bool, dict[str, Any]]:
    if hasattr(base_env, "step_no_reset"):
        return base_env.step_no_reset(action)
    observation, reward, terminated, truncated, info = base_env.step(action)
    return observation, float(reward), bool(terminated), bool(truncated), dict(info)


def _render_base_env_frame(base_env: gym.Env) -> np.ndarray:
    return base_env.render()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(val) for val in value]
    return value


def _image_array_for_vlm(image: Any, *, flip_hw: bool = False) -> np.ndarray | None:
    if image is None:
        return None
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        return None
    if image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    if flip_hw:
        image = image[::-1, ::-1]
    if np.issubdtype(image.dtype, np.floating):
        max_value = float(np.nanmax(image)) if image.size else 1.0
        if max_value <= 1.0:
            image = image * 255.0
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def _extract_vlm_observation_images(
    observation: Mapping[str, Any] | None,
    image_keys: Sequence[str],
) -> list[tuple[str, np.ndarray]]:
    if observation is None:
        return []
    pixels = observation.get("pixels")
    if not isinstance(pixels, Mapping):
        return []

    images: list[tuple[str, np.ndarray]] = []
    for key in image_keys:
        image = _image_array_for_vlm(pixels.get(key), flip_hw=True)
        if image is not None:
            images.append((f"observation.{key}", image))
    return images


_REFERENCE_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def _iter_reference_image_paths(
    *,
    image_dir: Path | None,
    image_paths: str | None,
    image_glob: str,
) -> list[Path]:
    paths: list[Path] = []
    explicit_paths = [token.strip() for token in (image_paths or "").split(",") if token.strip()]
    for token in explicit_paths:
        path = Path(token)
        if path.is_dir():
            paths.extend(
                child
                for child in sorted(path.glob(image_glob))
                if child.is_file() and child.suffix.lower() in _REFERENCE_IMAGE_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in _REFERENCE_IMAGE_SUFFIXES:
            paths.append(path)
        else:
            raise FileNotFoundError(f"Reference image path does not exist or is not an image: {path}")

    if image_dir is not None and image_dir.exists():
        paths.extend(
            child
            for child in sorted(image_dir.glob(image_glob))
            if child.is_file() and child.suffix.lower() in _REFERENCE_IMAGE_SUFFIXES
        )

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    return unique_paths


def _reference_image_matches_task(path: Path, task_id: int) -> bool:
    task_tokens = {str(task_id), f"{task_id:02d}", f"{task_id:03d}"}
    for part in [path.stem, *path.parent.parts]:
        text = part.lower()
        compact = re.sub(r"[^a-z0-9]+", "", text)
        normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
        for token in task_tokens:
            marker_forms = {
                token,
                f"task{token}",
                f"task_{token}",
                f"task-{token}",
                f"taskid{token}",
                f"task_id_{token}",
                f"task-id-{token}",
            }
            compact_marker_forms = {form.replace("_", "").replace("-", "") for form in marker_forms}
            if normalized in marker_forms or compact in compact_marker_forms:
                return True
            if normalized.startswith((f"{token}_", f"{token}-", f"task_{token}_", f"task_{token}-")):
                return True
            if normalized.startswith((f"task{token}_", f"task{token}-")):
                return True
    return False


def _load_vlm_reference_images(
    cfg: PolicyInferenceConfig,
    *,
    task_id: int | None,
) -> list[tuple[str, np.ndarray]]:
    images: list[tuple[str, np.ndarray]] = []
    paths = _iter_reference_image_paths(
        image_dir=cfg.vlm_reference_image_dir,
        image_paths=cfg.vlm_reference_image_paths,
        image_glob=cfg.vlm_reference_image_glob,
    )
    if cfg.vlm_reference_filter_by_task and task_id is not None:
        all_paths = paths
        paths = [path for path in all_paths if _reference_image_matches_task(path, task_id)]
        if all_paths and not paths:
            logger.warning(
                "Found %d reference image(s), but none matched task_id=%s. "
                "Use names like `task_%s_*.png` or set `--vlm_reference_filter_by_task=false`.",
                len(all_paths),
                task_id,
                task_id,
            )
    for image_ix, path in enumerate(paths):
        with Image.open(path) as pil_image:
            image = pil_image.convert("RGB")
            if cfg.vlm_reference_image_max_size > 0:
                max_size = int(cfg.vlm_reference_image_max_size)
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            task_label = f"task_{task_id}." if task_id is not None else ""
            label = f"reference.target.{task_label}{image_ix:02d}.{path.stem}"
            images.append((label, np.asarray(image, dtype=np.uint8)))
    if images:
        logger.info("Loaded %d VLM target reference image(s).", len(images))
    return images


def _error_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None) if response is not None else None
    if status_code is not None:
        with suppress(TypeError, ValueError):
            return int(status_code)

    match = re.search(r"Error code:\s*(\d+)", str(exc))
    return int(match.group(1)) if match else None


def _is_rate_limit_error(exc: Exception) -> bool:
    if _error_status_code(exc) == 429:
        return True
    return "rate limit" in str(exc).lower()


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if not headers:
        return None
    retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
    if retry_after is None:
        return None
    with suppress(TypeError, ValueError):
        return max(0.0, float(retry_after))
    return None


def _extract_vlm_robot_state(observation: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if observation is None:
        return None
    robot_state = observation.get("robot_state")
    if not isinstance(robot_state, Mapping):
        return None

    eef = robot_state.get("eef", {})
    gripper = robot_state.get("gripper", {})
    if not isinstance(eef, Mapping) or not isinstance(gripper, Mapping):
        return None

    return {
        "eef_position": _to_jsonable(eef.get("pos")),
        "eef_quaternion": _to_jsonable(eef.get("quat")),
        "gripper_qpos": _to_jsonable(gripper.get("qpos")),
    }


@dataclass
class ScoreResult:
    score: float
    reason: str
    prompt: str
    images: Sequence[tuple[str, np.ndarray]] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    raw_response: Any | None = None


class VLMHeuristicScorer:
    """Scores rendered states.

    If ``vlm_api_url`` is not provided this returns a success-only score
    (1.0 for environment success, otherwise 0.0). When ``vlm_model`` is
    provided, it uses the OpenAI Python client. ``vlm_base_url`` can point to an
    OpenAI-compatible provider such as NVIDIA NIM.
    """

    def __init__(self, cfg: PolicyInferenceConfig, *, task_id: int | None = None) -> None:
        self.base_url = cfg.vlm_base_url or cfg.vlm_api_url
        self.api_key_env = cfg.vlm_api_key_env or "OPENAI_API_KEY"
        self.model = cfg.vlm_model
        self.max_tokens = cfg.vlm_max_tokens
        self.temperature = cfg.vlm_temperature
        self.top_p = cfg.vlm_top_p
        self.presence_penalty = cfg.vlm_presence_penalty
        self.top_k = cfg.vlm_top_k
        self.min_p = cfg.vlm_min_p
        self.repetition_penalty = cfg.vlm_repetition_penalty
        self.timeout_s = cfg.vlm_timeout_s
        self.requests_per_minute = cfg.vlm_requests_per_minute
        self.max_retries = cfg.vlm_max_retries
        self.retry_sleep_s = cfg.vlm_retry_sleep_s
        self.rate_limit_sleep_s = cfg.vlm_rate_limit_sleep_s
        self.log_response_chars = cfg.vlm_log_response_chars
        self.verbose = cfg.vlm_verbose
        self.observation_image_keys = tuple(
            key.strip() for key in cfg.vlm_observation_image_keys.split(",") if key.strip()
        )
        self.include_rendered_image = cfg.vlm_include_rendered_image
        self.include_robot_state = cfg.vlm_include_robot_state
        self.reference_images = _load_vlm_reference_images(cfg, task_id=task_id)
        self.reference_image_labels = tuple(label for label, _ in self.reference_images)
        self._client = None
        self._last_request_monotonic: float | None = None

    def build_prompt(
        self,
        task: str,
        *,
        image_labels: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> str:
        image_description = "\n".join(
            f"{ix}. {label}" for ix, label in enumerate(image_labels, start=1)
        )
        robot_state = metadata.get("robot_state")
        robot_state_text = ""
        if robot_state is not None:
            robot_state_text = (
                "\n\nRobot proprioception, if useful for judging gripper pose:\n"
                f"{json.dumps(_to_jsonable(robot_state), indent=2)}"
            )
        return (
            f'Task: "{task}"\n\n'
            "You are scoring progress in a robot manipulation task from the supplied camera views. "
            "The images whose labels start with `reference.target.` are target-object reference images. "
            "They are not current state images; use them only to identify what the target object looks like "
            "in the current camera views. The current state views may include a rendered overview, a base "
            "camera, and a wrist camera. Judge whether the gripper is approaching, grasping, or moving the "
            "referenced target object toward the task goal. Do not assume a visible distractor object is the "
            "target unless it matches the task and target reference.\n\n"
            f"Images are supplied in this order:\n{image_description}"
            # f"{robot_state_text}\n\n"
            "Return JSON only with keys `score` and `reason`.\n"
            "Score meaning: 0.0 = no visible progress, 0.3 = robot is near or reaching the relevant object, "
            "0.5 = object is grasped or moved toward the target, 0.8 = object is near/over the target but "
            "completion is not certain."
        )

    def score(
        self,
        *,
        images: Sequence[tuple[str, np.ndarray]],
        task: str,
        success: bool,
        metadata: Mapping[str, Any],
    ) -> ScoreResult:
        if not images:
            raise ValueError("VLM scoring requires at least one image.")
        prompt = self.build_prompt(task, image_labels=[label for label, _ in images], metadata=metadata)
        if success:
            return ScoreResult(
                score=1.0,
                reason="Environment success predicate is true.",
                prompt=prompt,
                images=list(images),
                metadata=dict(metadata),
            )
        if self.model is None:
            return ScoreResult(
                score=0.0,
                reason="VLM API disabled; using success-only fallback.",
                prompt=prompt,
                images=list(images),
                metadata=dict(metadata),
            )

        client = self._get_client()
        request_content: list[dict[str, Any]] = []
        for label, image in images:
            request_content.append({"type": "text", "text": f"Image label: {label}"})
            request_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self._image_to_base64_png(image)}"
                    },
                }
            )
        request_content.append({"type": "text", "text": prompt})

        extra_body: dict[str, Any] = {
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
        }

        last_error: Exception | None = None
        last_message = ""
        raw: Any | None = None
        max_attempts = self.max_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                if self.verbose:
                    logger.info(
                        "VLM request attempt=%s/%s model=%s image_count=%s image_labels=%s metadata=%s",
                        attempt,
                        max_attempts,
                        self.model,
                        len(images),
                        [label for label, _ in images],
                        {
                            key: metadata.get(key)
                            for key in (
                                "planner",
                                "macro_step",
                                "env_step",
                                "parent_id",
                                "candidate_index",
                                "depth",
                            )
                            if key in metadata
                        },
                    )
                self._wait_for_rate_limit()
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": request_content,
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    presence_penalty=self.presence_penalty,
                    extra_body=extra_body,
                    stream=False,
                )
            except Exception as exc:
                last_error = exc
                if _is_rate_limit_error(exc) and attempt < max_attempts:
                    sleep_s = self._rate_limit_retry_sleep_s(exc)
                    logger.warning(
                        "VLM rate limit hit on attempt %s/%s; retrying in %.1fs: %s",
                        attempt,
                        max_attempts,
                        sleep_s,
                        exc,
                    )
                    time.sleep(sleep_s)
                    continue

                logger.warning("VLM request failed on attempt %s/%s: %s", attempt, max_attempts, exc)
                break

            raw = completion.model_dump() if hasattr(completion, "model_dump") else str(completion)
            last_message = completion.choices[0].message.content or ""
            response_tail = self._response_tail(last_message)
            try:
                parsed = self._parse_score_json(last_message)
            except Exception as exc:
                last_error = exc
                if attempt < max_attempts:
                    sleep_s = self._parse_retry_sleep_s()
                    logger.warning(
                        "VLM response parse failed on attempt %s/%s; response_tail=%r; retrying in %.1fs: %s",
                        attempt,
                        max_attempts,
                        response_tail,
                        sleep_s,
                        exc,
                    )
                    time.sleep(sleep_s)
                    continue

                logger.warning(
                    "VLM response parse failed on final attempt %s/%s; response_tail=%r: %s",
                    attempt,
                    max_attempts,
                    response_tail,
                    exc,
                )
                break

            score = float(parsed.get("score", 0.0))
            score = min(1.0, max(0.0, score))
            if self.verbose:
                logger.info(
                    "VLM response parsed attempt=%s/%s score=%.3f reason=%r response_tail=%r",
                    attempt,
                    max_attempts,
                    score,
                    str(parsed.get("reason", ""))[:160],
                    response_tail,
                )
            return ScoreResult(
                score=score,
                reason=str(parsed.get("reason", "")),
                prompt=prompt,
                images=list(images),
                metadata=dict(metadata),
                raw_response=raw,
            )

        response_tail = self._response_tail(last_message)
        reason = f"VLM scoring failed after {max_attempts} attempt(s): {last_error}; response_tail={response_tail!r}"
        return ScoreResult(
            score=0.0,
            reason=reason,
            prompt=prompt,
            images=list(images),
            metadata=dict(metadata),
            raw_response=raw,
        )

    def _min_request_interval_s(self) -> float:
        if self.requests_per_minute <= 0:
            return 0.0
        return 60.0 / float(self.requests_per_minute)

    def _wait_for_rate_limit(self) -> None:
        min_interval_s = self._min_request_interval_s()
        if min_interval_s <= 0:
            return
        now = time.monotonic()
        if self._last_request_monotonic is not None:
            elapsed_s = now - self._last_request_monotonic
            sleep_s = min_interval_s - elapsed_s
            if sleep_s > 0:
                if self.verbose:
                    logger.info(
                        "VLM throttle sleeping %.1fs for %.2f requests/minute limit.",
                        sleep_s,
                        self.requests_per_minute,
                    )
                time.sleep(sleep_s)
        self._last_request_monotonic = time.monotonic()

    def _parse_retry_sleep_s(self) -> float:
        return max(self.retry_sleep_s, self._min_request_interval_s())

    def _rate_limit_retry_sleep_s(self, exc: Exception) -> float:
        retry_after_s = _retry_after_seconds(exc)
        if retry_after_s is not None:
            return retry_after_s
        return max(self.rate_limit_sleep_s, self.retry_sleep_s, self._min_request_interval_s())

    def _response_tail(self, message: str) -> str:
        if not message:
            return "<empty>"
        return message[-self.log_response_chars :]

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "VLM scoring with the OpenAI client requires `openai`. "
                "Install it with `uv pip install openai` or `pip install openai`."
            ) from exc

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key environment variable: {self.api_key_env}")

        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": self.timeout_s}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    @staticmethod
    def _parse_score_json(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object from VLM, got: {type(parsed).__name__}")
        return parsed

    @staticmethod
    def _image_to_base64_png(image: np.ndarray) -> str:
        buffer = io.BytesIO()
        Image.fromarray(image.astype(np.uint8)).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class TraceRecorder:
    def __init__(self, trace_dir: Path | None, *, run_metadata: Mapping[str, Any]) -> None:
        self.trace_dir = trace_dir
        self.enabled = trace_dir is not None
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.executions: list[dict[str, Any]] = []
        self._node_ix = 0
        self._edge_ix = 0
        self._step_ix = 0
        if self.enabled:
            assert self.trace_dir is not None
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            (self.trace_dir / "images").mkdir(exist_ok=True)
            self.events_path = self.trace_dir / "events.jsonl"
            self._events_file = self.events_path.open("w")
            self.write_event("run", {"metadata": _to_jsonable(dict(run_metadata))})
        else:
            self.events_path = None
            self._events_file = None

    def close(self) -> None:
        if self._events_file is not None:
            self._events_file.close()
            self._events_file = None

    def write_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if not self.enabled or self._events_file is None:
            return
        event = {"type": event_type, "time_s": time.time(), **_to_jsonable(dict(payload))}
        self._events_file.write(json.dumps(event) + "\n")
        self._events_file.flush()

    def save_image(self, name: str, image: np.ndarray | None) -> str | None:
        if not self.enabled or image is None:
            return None
        assert self.trace_dir is not None
        path = self.trace_dir / "images" / f"{name}.png"
        write_image(image, path)
        return str(path.relative_to(self.trace_dir))

    def save_labeled_images(
        self, node_id: str, images: Sequence[tuple[str, np.ndarray]]
    ) -> list[dict[str, str]]:
        saved: list[dict[str, str]] = []
        for image_ix, (label, image) in enumerate(images):
            safe_label = "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")
            image_path = self.save_image(f"{node_id}_vlm_{image_ix:02d}_{safe_label}", image)
            if image_path is not None:
                saved.append({"label": label, "path": image_path})
        return saved

    def add_node(
        self,
        *,
        parent_id: str | None,
        depth: int,
        env_step: int,
        image: np.ndarray | None,
        score: ScoreResult,
        reward_sum: float,
        success: bool,
        terminal: bool,
        planner: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        node_id = f"n{self._node_ix:06d}"
        self._node_ix += 1
        image_path = self.save_image(f"{node_id}_state", image)
        vlm_images = self.save_labeled_images(node_id, score.images)
        node = {
            "id": node_id,
            "parent_id": parent_id,
            "depth": depth,
            "env_step": env_step,
            "planner": planner,
            "image_path": image_path,
            "prompt": score.prompt,
            "vlm_score": score.score,
            "vlm_reason": score.reason,
            "vlm_raw_response": score.raw_response,
            "vlm_images": vlm_images,
            "vlm_metadata": _to_jsonable(dict(score.metadata)),
            "reward_sum": reward_sum,
            "success": success,
            "terminal": terminal,
            "visits": 0,
            "value_sum": 0.0,
            "children": [],
        }
        if extra:
            node.update(_to_jsonable(dict(extra)))
        self.nodes.append(node)
        self.write_event("node", node)
        return node

    def add_edge(
        self,
        *,
        parent_id: str,
        child_id: str,
        candidate_index: int,
        actions: np.ndarray,
        rewards: list[float],
        step_records: list[dict[str, Any]],
        source: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        edge_id = f"e{self._edge_ix:06d}"
        self._edge_ix += 1
        edge = {
            "id": edge_id,
            "parent_id": parent_id,
            "child_id": child_id,
            "candidate_index": candidate_index,
            "source": source,
            "action_count": int(actions.shape[0]),
            "actions": _to_jsonable(actions),
            "rewards": rewards,
            "step_records": step_records,
        }
        if extra:
            edge.update(_to_jsonable(dict(extra)))
        self.edges.append(edge)
        for node in self.nodes:
            if node["id"] == parent_id:
                node["children"].append(child_id)
                break
        self.write_event("edge", edge)
        return edge

    def add_execution(self, payload: Mapping[str, Any]) -> None:
        execution = {"id": f"x{self._step_ix:06d}", **_to_jsonable(dict(payload))}
        self._step_ix += 1
        self.executions.append(execution)
        self.write_event("execution", execution)

    def write_tree(self, *, summary: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        assert self.trace_dir is not None
        tree = {
            "summary": _to_jsonable(dict(summary)),
            "nodes": self.nodes,
            "edges": self.edges,
            "executions": self.executions,
        }
        with (self.trace_dir / "tree.json").open("w") as f:
            json.dump(tree, f, indent=2)


@dataclass
class SequenceRollout:
    observation: Mapping[str, Any]
    frame: np.ndarray
    state: np.ndarray
    frames: list[np.ndarray]
    rewards: list[float]
    step_records: list[dict[str, Any]]
    action_count: int
    reward_sum: float
    terminated: bool
    truncated: bool
    success: bool


def _rollout_action_sequence(
    *,
    base_env: gym.Env,
    actions: np.ndarray,
    start_env_step: int,
    max_steps: int,
    trace: TraceRecorder | None = None,
    save_step_images: bool = False,
    collect_frames: bool = False,
    step_image_prefix: str = "step",
) -> SequenceRollout:
    rewards: list[float] = []
    step_records: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    observation: Mapping[str, Any] | None = None
    terminated = False
    truncated = False
    success = False

    max_actions = min(len(actions), max(0, max_steps - start_env_step))
    for action_index, action in enumerate(actions[:max_actions]):
        observation, reward, terminated, truncated, info = _step_base_env_no_reset(base_env, action)
        step_success = bool(info.get("is_success", False))
        success = success or step_success
        rewards.append(float(reward))

        image_path = None
        frame = None
        if collect_frames or (trace is not None and save_step_images):
            frame = _render_base_env_frame(base_env)
        if collect_frames and frame is not None:
            frames.append(frame)
        if trace is not None and save_step_images:
            image_path = trace.save_image(
                f"{step_image_prefix}_{start_env_step + action_index:05d}",
                frame,
            )
        step_records.append(
            {
                "env_step": start_env_step + action_index,
                "action": _to_jsonable(action),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "success": step_success,
                "image_path": image_path,
                "info": _to_jsonable(info),
            }
        )
        if terminated or truncated:
            break

    if observation is None:
        observation = _restore_base_env(base_env, _snapshot_base_env(base_env), timestep=start_env_step)
    frame = _render_base_env_frame(base_env)
    state = _snapshot_base_env(base_env)
    return SequenceRollout(
        observation=observation,
        frame=frame,
        state=state,
        frames=frames,
        rewards=rewards,
        step_records=step_records,
        action_count=len(rewards),
        reward_sum=float(sum(rewards)),
        terminated=bool(terminated),
        truncated=bool(truncated),
        success=success,
    )


def make_action_api(
    cfg: PolicyInferenceConfig,
) -> tuple[LeRobotActionAPI, dict[str, dict[int, gym.vector.VectorEnv]]]:
    if cfg.policy is None:
        raise ValueError("Policy config was not initialized.")

    device = get_safe_torch_device(cfg.policy.device, log=True)

    envs_dict = make_env(
        cfg.env,
        n_envs=1,
        use_async_envs=False,
        trust_remote_code=cfg.trust_remote_code,
    )

    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    return (
        LeRobotActionAPI(
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            use_amp=cfg.policy.use_amp,
        ),
        envs_dict,
    )


def _make_action_candidates(
    policy_chunk: np.ndarray,
    *,
    num_candidates: int,
    noise_std: float,
    noise_mode: str,
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    candidates = [("policy", policy_chunk.astype(np.float32, copy=True))]
    for candidate_ix in range(1, num_candidates):
        if noise_mode == "iid":
            noise = rng.normal(0.0, noise_std, size=policy_chunk.shape)
        elif noise_mode == "chunk":
            noise = rng.normal(0.0, noise_std, size=(1, policy_chunk.shape[-1]))
        elif noise_mode == "mixed":
            chunk_noise = rng.normal(0.0, noise_std, size=(1, policy_chunk.shape[-1]))
            iid_noise = rng.normal(0.0, noise_std * 0.25, size=policy_chunk.shape)
            noise = chunk_noise + iid_noise
        else:
            raise ValueError(f"Unknown noise mode '{noise_mode}'.")

        noisy = policy_chunk + noise
        noisy = np.clip(noisy, -1.0, 1.0).astype(np.float32)
        candidates.append((f"policy_{noise_mode}_noise_{candidate_ix}", noisy))
    return candidates


def _node_mean_value(node: Mapping[str, Any]) -> float:
    visits = int(node.get("visits", 0))
    if visits <= 0:
        return float(node.get("vlm_score", 0.0))
    return float(node.get("value_sum", 0.0)) / visits


def _backup_path(path: list[dict[str, Any]], value: float) -> None:
    for node in path:
        node["visits"] = int(node.get("visits", 0)) + 1
        node["value_sum"] = float(node.get("value_sum", 0.0)) + float(value)


def _select_ucb_child(
    *,
    node: Mapping[str, Any],
    child_edges: Mapping[str, dict[str, Any]],
    nodes_by_id: Mapping[str, dict[str, Any]],
    exploration: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent_visits = max(1, int(node.get("visits", 0)))
    best_score = -float("inf")
    best_edge: dict[str, Any] | None = None
    best_child: dict[str, Any] | None = None
    for child_id in node["children"]:
        edge = child_edges[child_id]
        child = nodes_by_id[child_id]
        child_visits = int(child.get("visits", 0))
        if child_visits == 0:
            score = float("inf")
        else:
            exploit = _node_mean_value(child)
            explore = exploration * math.sqrt(math.log(parent_visits + 1) / child_visits)
            score = exploit + explore
        if score > best_score:
            best_score = score
            best_edge = edge
            best_child = child
    if best_edge is None or best_child is None:
        raise ValueError("Cannot select a child from an unexpanded node.")
    return best_edge, best_child


def _score_state(
    *,
    scorer: VLMHeuristicScorer,
    image: np.ndarray,
    observation: Mapping[str, Any] | None = None,
    task: str,
    success: bool,
    metadata: Mapping[str, Any],
) -> ScoreResult:
    state_images: list[tuple[str, np.ndarray]] = []
    if scorer.include_rendered_image:
        rendered_image = _image_array_for_vlm(image)
        if rendered_image is not None:
            state_images.append(("rendered_overview", rendered_image))
    state_images.extend(_extract_vlm_observation_images(observation, scorer.observation_image_keys))
    if not state_images:
        rendered_image = _image_array_for_vlm(image)
        if rendered_image is not None:
            state_images.append(("rendered_overview", rendered_image))

    images = [*scorer.reference_images, *state_images]

    score_metadata = dict(metadata)
    if scorer.include_robot_state:
        robot_state = _extract_vlm_robot_state(observation)
        if robot_state is not None:
            score_metadata["robot_state"] = robot_state
    score_metadata["vlm_reference_image_labels"] = [label for label, _ in scorer.reference_images]
    score_metadata["vlm_state_image_labels"] = [label for label, _ in state_images]
    score_metadata["vlm_image_labels"] = [label for label, _ in images]
    return scorer.score(images=images, task=task, success=success, metadata=score_metadata)


def _expand_search_node(
    *,
    planner: str,
    node: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    child_edges: dict[str, dict[str, Any]],
    states_by_id: dict[str, np.ndarray],
    observations_by_id: dict[str, Mapping[str, Any]],
    base_env: gym.Env,
    vector_env: gym.vector.VectorEnv,
    action_api: LeRobotActionAPI,
    scorer: VLMHeuristicScorer,
    trace: TraceRecorder,
    cfg: PolicyInferenceConfig,
    rng: np.random.Generator,
    task: str,
    max_env_steps: int,
) -> list[dict[str, Any]]:
    if node["terminal"]:
        return []

    node_state = states_by_id[node["id"]]
    node_observation = observations_by_id[node["id"]]
    if cfg.search_verbose:
        logger.info(
            "%s expanding node=%s depth=%s env_step=%s terminal=%s",
            planner,
            node["id"],
            node["depth"],
            node["env_step"],
            node["terminal"],
        )
    _restore_base_env(base_env, node_state, timestep=int(node["env_step"]))
    policy_chunk = action_api.predict_action_chunk(
        node_observation,
        env=vector_env,
        task=task,
        horizon=cfg.chunk_size,
    )
    candidates = _make_action_candidates(
        policy_chunk,
        num_candidates=cfg.search_num_candidates,
        noise_std=cfg.search_noise_std,
        noise_mode=cfg.search_noise_mode,
        rng=rng,
    )
    if cfg.search_verbose:
        logger.info(
            "%s node=%s generated %s candidate action chunk(s) policy_chunk_shape=%s",
            planner,
            node["id"],
            len(candidates),
            tuple(policy_chunk.shape),
        )

    created_children: list[dict[str, Any]] = []
    for candidate_index, (source, actions) in enumerate(candidates):
        if cfg.search_verbose:
            logger.info(
                "%s rollout candidate=%s/%s parent=%s source=%s action_shape=%s",
                planner,
                candidate_index,
                len(candidates) - 1,
                node["id"],
                source,
                tuple(actions.shape),
            )
        _restore_base_env(base_env, node_state, timestep=int(node["env_step"]))
        rollout = _rollout_action_sequence(
            base_env=base_env,
            actions=actions,
            start_env_step=int(node["env_step"]),
            max_steps=min(max_env_steps, int(node["env_step"]) + cfg.chunk_size),
            trace=None,
            save_step_images=False,
        )
        score = _score_state(
            scorer=scorer,
            image=rollout.frame,
            observation=rollout.observation,
            task=task,
            success=rollout.success,
            metadata={
                "planner": planner,
                "parent_id": node["id"],
                "candidate_index": candidate_index,
                "source": source,
                "depth": int(node["depth"]) + 1,
                "reward_sum": rollout.reward_sum,
            },
        )
        if cfg.search_verbose:
            logger.info(
                "%s candidate=%s parent=%s score=%.3f reward_sum=%.3f action_count=%s "
                "terminal=%s truncated=%s success=%s reason=%r",
                planner,
                candidate_index,
                node["id"],
                score.score,
                rollout.reward_sum,
                rollout.action_count,
                rollout.terminated,
                rollout.truncated,
                rollout.success,
                score.reason[:180],
            )
        child = trace.add_node(
            parent_id=node["id"],
            depth=int(node["depth"]) + 1,
            env_step=int(node["env_step"]) + rollout.action_count,
            image=rollout.frame,
            score=score,
            reward_sum=rollout.reward_sum,
            success=rollout.success,
            terminal=rollout.terminated or rollout.truncated or rollout.success,
            planner=planner,
            extra={
                "candidate_index": candidate_index,
                "source": source,
                "action_count": rollout.action_count,
            },
        )
        edge = trace.add_edge(
            parent_id=node["id"],
            child_id=child["id"],
            candidate_index=candidate_index,
            actions=actions[: rollout.action_count],
            rewards=rollout.rewards,
            step_records=rollout.step_records,
            source=source,
            extra={
                "noise_std": cfg.search_noise_std if candidate_index else 0.0,
                "noise_mode": cfg.search_noise_mode if candidate_index else "none",
                "terminal": rollout.terminated,
                "truncated": rollout.truncated,
                "success": rollout.success,
                "value": score.score,
            },
        )
        nodes_by_id[child["id"]] = child
        child_edges[child["id"]] = edge
        states_by_id[child["id"]] = rollout.state
        observations_by_id[child["id"]] = rollout.observation
        created_children.append(child)

    _restore_base_env(base_env, node_state, timestep=int(node["env_step"]))
    return created_children


def plan_mcts_action_chunk(
    *,
    action_api: LeRobotActionAPI,
    base_env: gym.Env,
    vector_env: gym.vector.VectorEnv,
    observation: Mapping[str, Any],
    task: str,
    env_step: int,
    scorer: VLMHeuristicScorer,
    trace: TraceRecorder,
    cfg: PolicyInferenceConfig,
    rng: np.random.Generator,
    max_env_steps: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    root_state = _snapshot_base_env(base_env)
    root_frame = _render_base_env_frame(base_env)
    root_score = _score_state(
        scorer=scorer,
        image=root_frame,
        observation=observation,
        task=task,
        success=False,
        metadata={"planner": "mcts", "root": True, "env_step": env_step},
    )
    if cfg.search_verbose:
        logger.info(
            "mcts root env_step=%s score=%.3f reason=%r",
            env_step,
            root_score.score,
            root_score.reason[:180],
        )
    root = trace.add_node(
        parent_id=None,
        depth=0,
        env_step=env_step,
        image=root_frame,
        score=root_score,
        reward_sum=0.0,
        success=False,
        terminal=False,
        planner="mcts",
        extra={"root": True},
    )

    nodes_by_id = {root["id"]: root}
    child_edges: dict[str, dict[str, Any]] = {}
    states_by_id = {root["id"]: root_state}
    observations_by_id = {root["id"]: observation}

    for simulation_ix in range(cfg.mcts_simulations):
        if cfg.search_verbose:
            logger.info(
                "mcts simulation=%s/%s start root_children=%s",
                simulation_ix + 1,
                cfg.mcts_simulations,
                len(root["children"]),
            )
        node = root
        path = [node]
        while node["children"] and int(node["depth"]) < cfg.mcts_depth:
            _, child = _select_ucb_child(
                node=node,
                child_edges=child_edges,
                nodes_by_id=nodes_by_id,
                exploration=cfg.mcts_exploration,
            )
            node = child
            path.append(node)

        if node["terminal"] or int(node["depth"]) >= cfg.mcts_depth:
            value = _node_mean_value(node)
        else:
            created = _expand_search_node(
                planner="mcts",
                node=node,
                nodes_by_id=nodes_by_id,
                child_edges=child_edges,
                states_by_id=states_by_id,
                observations_by_id=observations_by_id,
                base_env=base_env,
                vector_env=vector_env,
                action_api=action_api,
                scorer=scorer,
                trace=trace,
                cfg=cfg,
                rng=rng,
                task=task,
                max_env_steps=max_env_steps,
            )
            if not created:
                value = _node_mean_value(node)
            else:
                selected = max(created, key=lambda child: float(child["vlm_score"]))
                path.append(selected)
                value = float(selected["vlm_score"])

        _backup_path(path, value)
        if cfg.search_verbose:
            logger.info(
                "mcts simulation=%s/%s leaf=%s value=%.3f path=%s",
                simulation_ix + 1,
                cfg.mcts_simulations,
                path[-1]["id"],
                value,
                [item["id"] for item in path],
            )
        trace.write_event(
            "mcts_simulation",
            {
                "simulation_ix": simulation_ix,
                "leaf_id": path[-1]["id"],
                "value": value,
                "path": [item["id"] for item in path],
            },
        )

    if not root["children"]:
        fallback = action_api.predict_action_chunk(
            observation, env=vector_env, task=task, horizon=cfg.chunk_size
        )
        return fallback, {"reason": "mcts_root_unexpanded", "root_id": root["id"]}

    best_edge, best_child = max(
        (
            (child_edges[child_id], nodes_by_id[child_id])
            for child_id in root["children"]
        ),
        key=lambda pair: (
            _node_mean_value(pair[1]),
            int(pair[1].get("visits", 0)),
            float(pair[1]["vlm_score"]),
        ),
    )
    _restore_base_env(base_env, root_state, timestep=env_step)
    if cfg.search_verbose:
        logger.info(
            "mcts selected child=%s edge=%s score=%.3f value=%.3f visits=%s",
            best_child["id"],
            best_edge["id"],
            best_child["vlm_score"],
            _node_mean_value(best_child),
            best_child["visits"],
        )
    return np.asarray(best_edge["actions"], dtype=np.float32), {
        "root_id": root["id"],
        "selected_edge_id": best_edge["id"],
        "selected_child_id": best_child["id"],
        "selected_value": _node_mean_value(best_child),
        "selected_score": best_child["vlm_score"],
        "selected_visits": best_child["visits"],
    }


def _best_descendant_root_child(
    *,
    root: Mapping[str, Any],
    nodes_by_id: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    descendants = [node for node in nodes_by_id.values() if node["id"] != root["id"]]
    if not descendants:
        return None, None

    best_descendant = max(
        descendants,
        key=lambda node: (
            bool(node.get("success", False)),
            float(node.get("vlm_score", 0.0)),
            int(node.get("visits", 0)),
            int(node.get("depth", 0)),
        ),
    )

    root_child = best_descendant
    while root_child.get("parent_id") is not None and root_child["parent_id"] != root["id"]:
        parent_id = root_child["parent_id"]
        root_child = nodes_by_id[parent_id]

    if root_child.get("parent_id") != root["id"]:
        return None, best_descendant
    return root_child, best_descendant


def plan_best_first_action_chunk(
    *,
    action_api: LeRobotActionAPI,
    base_env: gym.Env,
    vector_env: gym.vector.VectorEnv,
    observation: Mapping[str, Any],
    task: str,
    env_step: int,
    scorer: VLMHeuristicScorer,
    trace: TraceRecorder,
    cfg: PolicyInferenceConfig,
    rng: np.random.Generator,
    max_env_steps: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    root_state = _snapshot_base_env(base_env)
    root_frame = _render_base_env_frame(base_env)
    root_score = _score_state(
        scorer=scorer,
        image=root_frame,
        observation=observation,
        task=task,
        success=False,
        metadata={"planner": "best_first", "root": True, "env_step": env_step},
    )
    if cfg.search_verbose:
        logger.info(
            "best_first root env_step=%s score=%.3f reason=%r",
            env_step,
            root_score.score,
            root_score.reason[:180],
        )
    root = trace.add_node(
        parent_id=None,
        depth=0,
        env_step=env_step,
        image=root_frame,
        score=root_score,
        reward_sum=0.0,
        success=False,
        terminal=False,
        planner="best_first",
        extra={"root": True},
    )

    nodes_by_id = {root["id"]: root}
    child_edges: dict[str, dict[str, Any]] = {}
    states_by_id = {root["id"]: root_state}
    observations_by_id = {root["id"]: observation}

    frontier: list[tuple[float, int, int, str]] = []
    push_ix = 0
    heapq.heappush(frontier, (-float(root["vlm_score"]), int(root["depth"]), push_ix, root["id"]))
    push_ix += 1
    expanded: set[str] = set()

    for expansion_ix in range(cfg.best_first_expansions):
        while frontier:
            _, _, _, node_id = heapq.heappop(frontier)
            if node_id not in expanded:
                break
        else:
            break

        node = nodes_by_id[node_id]
        if node["terminal"] or int(node["depth"]) >= cfg.best_first_depth:
            if cfg.search_verbose:
                logger.info(
                    "best_first skip node=%s depth=%s terminal=%s frontier_size=%s",
                    node["id"],
                    node["depth"],
                    node["terminal"],
                    len(frontier),
                )
            continue

        expanded.add(node_id)
        node["visits"] = int(node.get("visits", 0)) + 1
        node["value_sum"] = float(node.get("value_sum", 0.0)) + float(node.get("vlm_score", 0.0))
        if cfg.search_verbose:
            logger.info(
                "best_first expansion=%s/%s node=%s depth=%s score=%.3f frontier_size=%s",
                expansion_ix + 1,
                cfg.best_first_expansions,
                node["id"],
                node["depth"],
                float(node.get("vlm_score", 0.0)),
                len(frontier),
            )

        created = _expand_search_node(
            planner="best_first",
            node=node,
            nodes_by_id=nodes_by_id,
            child_edges=child_edges,
            states_by_id=states_by_id,
            observations_by_id=observations_by_id,
            base_env=base_env,
            vector_env=vector_env,
            action_api=action_api,
            scorer=scorer,
            trace=trace,
            cfg=cfg,
            rng=rng,
            task=task,
            max_env_steps=max_env_steps,
        )

        for child in created:
            if not child["terminal"] and int(child["depth"]) < cfg.best_first_depth:
                priority = -float(child.get("vlm_score", 0.0))
                heapq.heappush(frontier, (priority, int(child["depth"]), push_ix, child["id"]))
                push_ix += 1

        root_child, best_descendant = _best_descendant_root_child(root=root, nodes_by_id=nodes_by_id)
        if cfg.search_verbose:
            logger.info(
                "best_first expansion=%s/%s created=%s frontier_size=%s best_root_child=%s "
                "best_descendant=%s best_descendant_score=%s",
                expansion_ix + 1,
                cfg.best_first_expansions,
                [child["id"] for child in created],
                len(frontier),
                root_child["id"] if root_child is not None else None,
                best_descendant["id"] if best_descendant is not None else None,
                best_descendant["vlm_score"] if best_descendant is not None else None,
            )
        trace.write_event(
            "best_first_expansion",
            {
                "expansion_ix": expansion_ix,
                "expanded_node_id": node["id"],
                "created_node_ids": [child["id"] for child in created],
                "frontier_size": len(frontier),
                "best_root_child_id": root_child["id"] if root_child is not None else None,
                "best_descendant_id": best_descendant["id"] if best_descendant is not None else None,
                "best_descendant_score": (
                    best_descendant["vlm_score"] if best_descendant is not None else None
                ),
            },
        )

        if any(child.get("success", False) for child in created):
            break

    if not root["children"]:
        fallback = action_api.predict_action_chunk(
            observation, env=vector_env, task=task, horizon=cfg.chunk_size
        )
        return fallback, {"reason": "best_first_root_unexpanded", "root_id": root["id"]}

    best_root_child, best_descendant = _best_descendant_root_child(root=root, nodes_by_id=nodes_by_id)
    if best_root_child is None:
        best_root_child = max(
            (nodes_by_id[child_id] for child_id in root["children"]),
            key=lambda child: float(child.get("vlm_score", 0.0)),
        )
        best_descendant = best_root_child

    best_edge = child_edges[best_root_child["id"]]
    _restore_base_env(base_env, root_state, timestep=env_step)
    if cfg.search_verbose:
        logger.info(
            "best_first selected root_child=%s edge=%s score=%.3f descendant=%s descendant_score=%s "
            "expanded_count=%s frontier_size=%s",
            best_root_child["id"],
            best_edge["id"],
            best_root_child["vlm_score"],
            best_descendant["id"] if best_descendant is not None else None,
            best_descendant["vlm_score"] if best_descendant is not None else None,
            len(expanded),
            len(frontier),
        )
    return np.asarray(best_edge["actions"], dtype=np.float32), {
        "root_id": root["id"],
        "selected_edge_id": best_edge["id"],
        "selected_child_id": best_root_child["id"],
        "selected_descendant_id": best_descendant["id"] if best_descendant is not None else None,
        "selected_score": best_root_child["vlm_score"],
        "selected_descendant_score": (
            best_descendant["vlm_score"] if best_descendant is not None else None
        ),
        "expanded_count": len(expanded),
        "frontier_size": len(frontier),
    }


def plan_baseline_action_chunk(
    *,
    action_api: LeRobotActionAPI,
    vector_env: gym.vector.VectorEnv,
    observation: Mapping[str, Any],
    task: str,
    cfg: PolicyInferenceConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    actions = action_api.predict_action_chunk(observation, env=vector_env, task=task, horizon=cfg.chunk_size)
    return actions, {"reason": "policy_chunk_baseline"}


def choose_action_with_external_planner(
    action_api: LeRobotActionAPI,
    observation: Mapping[str, Any],
    env: gym.vector.VectorEnv,
) -> np.ndarray:
    """Planner hook.

    Replace this function from an external package with tree search:
    1. snapshot the backend-specific simulator state,
    2. expand candidate nodes using restored simulator states,
    3. call `action_api.select_action(node_observation, env=env)` when a policy
       prior/action proposal is needed,
    4. restore the root simulator state and return the chosen root action.
    """
    return action_api.select_action(observation, env=env)


def _apply_deprecated_search_aliases(cfg: PolicyInferenceConfig) -> None:
    aliases = {
        "mcts_num_candidates": "search_num_candidates",
        "mcts_noise_std": "search_noise_std",
        "mcts_noise_mode": "search_noise_mode",
    }
    for deprecated_name, replacement_name in aliases.items():
        value = getattr(cfg, deprecated_name)
        if value is None:
            continue
        logger.warning(
            "`--%s` is deprecated because candidate generation is shared by planners. Use `--%s`.",
            deprecated_name,
            replacement_name,
        )
        setattr(cfg, replacement_name, value)


@parser.wrap()
def main(cfg: PolicyInferenceConfig) -> None:
    if cfg.planner not in {"baseline", "mcts", "best_first"}:
        raise ValueError("`planner` must be one of: baseline, mcts, best_first")
    _apply_deprecated_search_aliases(cfg)
    if cfg.chunk_size <= 0:
        raise ValueError("`chunk_size` must be positive.")
    if cfg.mcts_depth <= 0:
        raise ValueError("`mcts_depth` must be positive.")
    if cfg.mcts_simulations <= 0:
        raise ValueError("`mcts_simulations` must be positive.")
    if cfg.search_num_candidates <= 0:
        raise ValueError("`search_num_candidates` must be positive.")
    if cfg.search_noise_mode not in {"iid", "chunk", "mixed"}:
        raise ValueError("`search_noise_mode` must be one of: iid, chunk, mixed.")
    if cfg.best_first_expansions <= 0:
        raise ValueError("`best_first_expansions` must be positive.")
    if cfg.best_first_depth <= 0:
        raise ValueError("`best_first_depth` must be positive.")

    set_seed(cfg.seed)
    action_api, envs_dict = make_action_api(cfg)
    suite, task_id, env = _select_env(envs_dict, suite=cfg.suite, task_id=cfg.task_id)
    if env.num_envs != 1:
        raise ValueError(f"This example only supports a single environment, got env.num_envs={env.num_envs}.")

    base_env = _get_single_base_env(env)
    task = _infer_task(env)
    rng = np.random.default_rng(cfg.seed)
    scorer = VLMHeuristicScorer(cfg, task_id=task_id)
    trace = TraceRecorder(
        cfg.trace_dir,
        run_metadata={
            "planner": cfg.planner,
            "chunk_size": cfg.chunk_size,
            "suite": suite,
            "task_id": task_id,
            "task": task,
            "steps": cfg.steps,
            "seed": cfg.seed,
            "policy": str(cfg.policy.pretrained_path) if cfg.policy is not None else None,
            "search_num_candidates": cfg.search_num_candidates,
            "search_noise_std": cfg.search_noise_std,
            "search_noise_mode": cfg.search_noise_mode,
            "mcts_simulations": cfg.mcts_simulations,
            "mcts_depth": cfg.mcts_depth,
            "best_first_expansions": cfg.best_first_expansions,
            "best_first_depth": cfg.best_first_depth,
            "vlm_model": cfg.vlm_model,
            "vlm_base_url": cfg.vlm_base_url or cfg.vlm_api_url,
            "vlm_api_key_env": cfg.vlm_api_key_env or "OPENAI_API_KEY",
            "vlm_requests_per_minute": cfg.vlm_requests_per_minute,
            "vlm_max_retries": cfg.vlm_max_retries,
            "vlm_retry_sleep_s": cfg.vlm_retry_sleep_s,
            "vlm_rate_limit_sleep_s": cfg.vlm_rate_limit_sleep_s,
            "vlm_verbose": cfg.vlm_verbose,
            "search_verbose": cfg.search_verbose,
            "vlm_observation_image_keys": cfg.vlm_observation_image_keys,
            "vlm_include_rendered_image": cfg.vlm_include_rendered_image,
            "vlm_include_robot_state": cfg.vlm_include_robot_state,
            "vlm_reference_image_dir": str(cfg.vlm_reference_image_dir)
            if cfg.vlm_reference_image_dir is not None
            else None,
            "vlm_reference_image_paths": cfg.vlm_reference_image_paths,
            "vlm_reference_image_glob": cfg.vlm_reference_image_glob,
            "vlm_reference_filter_by_task": cfg.vlm_reference_filter_by_task,
            "vlm_reference_image_labels": list(scorer.reference_image_labels),
        },
    )

    logger.info(
        "Running planner=%s single-env inference on suite=%s task_id=%s chunk_size=%s "
        "vlm_model=%s vlm_rpm=%s vlm_max_retries=%s vlm_verbose=%s search_verbose=%s",
        cfg.planner,
        suite,
        task_id,
        cfg.chunk_size,
        cfg.vlm_model,
        cfg.vlm_requests_per_minute,
        cfg.vlm_max_retries,
        cfg.vlm_verbose,
        cfg.search_verbose,
    )

    try:
        action_api.reset()
        if cfg.seed is None:
            observation, _ = env.reset()
        else:
            observation, _ = env.reset(seed=[cfg.seed])

        video_frames: list[np.ndarray] = []
        if cfg.video_path is not None:
            video_frames.append(_render_single_env_frame(env))

        done = False
        success = False
        max_steps = cfg.steps
        if max_steps <= 0:
            max_steps = int(env.call("_max_episode_steps")[0])

        env_step = 0
        macro_step = 0
        last_reward_sum = 0.0
        baseline_parent_id: str | None = None
        while env_step < max_steps and not done:
            root_frame = _render_base_env_frame(base_env)

            baseline_root: dict[str, Any] | None = None
            if cfg.planner == "baseline":
                root_score = _score_state(
                    scorer=scorer,
                    image=root_frame,
                    observation=observation,
                    task=task,
                    success=False,
                    metadata={
                        "planner": "baseline",
                        "macro_step": macro_step,
                        "env_step": env_step,
                        "root": True,
                    },
                )
                baseline_root = trace.add_node(
                    parent_id=baseline_parent_id,
                    depth=macro_step,
                    env_step=env_step,
                    image=root_frame,
                    score=root_score,
                    reward_sum=0.0,
                    success=False,
                    terminal=False,
                    planner="baseline",
                    extra={"macro_step": macro_step, "root": True},
                )
                actions, plan_info = plan_baseline_action_chunk(
                    action_api=action_api,
                    vector_env=env,
                    observation=observation,
                    task=task,
                    cfg=cfg,
                )
            elif cfg.planner == "mcts":
                actions, plan_info = plan_mcts_action_chunk(
                    action_api=action_api,
                    base_env=base_env,
                    vector_env=env,
                    observation=observation,
                    task=task,
                    env_step=env_step,
                    scorer=scorer,
                    trace=trace,
                    cfg=cfg,
                    rng=rng,
                    max_env_steps=max_steps,
                )
            else:
                actions, plan_info = plan_best_first_action_chunk(
                    action_api=action_api,
                    base_env=base_env,
                    vector_env=env,
                    observation=observation,
                    task=task,
                    env_step=env_step,
                    scorer=scorer,
                    trace=trace,
                    cfg=cfg,
                    rng=rng,
                    max_env_steps=max_steps,
                )

            rollout = _rollout_action_sequence(
                base_env=base_env,
                actions=actions,
                start_env_step=env_step,
                max_steps=max_steps,
                trace=trace,
                save_step_images=cfg.trace_save_step_images,
                collect_frames=cfg.video_path is not None,
                step_image_prefix=f"commit_{macro_step:04d}",
            )
            if cfg.video_path is not None:
                video_frames.extend(rollout.frames)

            endpoint_score = _score_state(
                scorer=scorer,
                image=rollout.frame,
                observation=rollout.observation,
                task=task,
                success=rollout.success,
                metadata={
                    "planner": cfg.planner,
                    "macro_step": macro_step,
                    "env_step": env_step,
                    "committed": True,
                    "plan_info": plan_info,
                    "reward_sum": rollout.reward_sum,
                },
            )

            if cfg.planner == "baseline" and baseline_root is not None:
                child = trace.add_node(
                    parent_id=baseline_root["id"],
                    depth=macro_step + 1,
                    env_step=env_step + rollout.action_count,
                    image=rollout.frame,
                    score=endpoint_score,
                    reward_sum=rollout.reward_sum,
                    success=rollout.success,
                    terminal=rollout.terminated or rollout.truncated or rollout.success,
                    planner="baseline",
                    extra={"macro_step": macro_step, "committed": True},
                )
                trace.add_edge(
                    parent_id=baseline_root["id"],
                    child_id=child["id"],
                    candidate_index=0,
                    actions=actions[: rollout.action_count],
                    rewards=rollout.rewards,
                    step_records=rollout.step_records,
                    source="policy",
                    extra={"committed": True, "value": endpoint_score.score},
                )
                baseline_parent_id = child["id"]

            trace.add_execution(
                {
                    "planner": cfg.planner,
                    "macro_step": macro_step,
                    "start_env_step": env_step,
                    "end_env_step": env_step + rollout.action_count,
                    "action_count": rollout.action_count,
                    "reward_sum": rollout.reward_sum,
                    "success": rollout.success,
                    "terminated": rollout.terminated,
                    "truncated": rollout.truncated,
                    "vlm_score": endpoint_score.score,
                    "vlm_reason": endpoint_score.reason,
                    "prompt": endpoint_score.prompt,
                    "plan_info": plan_info,
                    "step_records": rollout.step_records,
                }
            )

            observation = rollout.observation
            env_step += rollout.action_count
            macro_step += 1
            last_reward_sum = rollout.reward_sum
            done = rollout.terminated or rollout.truncated or rollout.action_count == 0
            success = success or rollout.success
            logger.info(
                "macro_step=%s env_step=%s reward_sum=%s done=%s success=%s "
                "action_chunk_shape=%s vlm_score=%.3f vlm_reason=%r plan_info=%s",
                macro_step - 1,
                env_step,
                last_reward_sum,
                done,
                success,
                actions.shape,
                endpoint_score.score,
                endpoint_score.reason[:180],
                plan_info,
            )

        summary = {
            "planner": cfg.planner,
            "suite": suite,
            "task_id": task_id,
            "task": task,
            "steps": env_step,
            "macro_steps": macro_step,
            "success": success,
            "last_reward_sum": last_reward_sum,
            "trace_dir": str(cfg.trace_dir) if cfg.trace_dir is not None else None,
        }
        trace.write_tree(summary=summary)
        logger.info("Finished after %s step(s), %s macro step(s). success=%s", env_step, macro_step, success)
        if cfg.video_path is not None and video_frames:
            cfg.video_path.parent.mkdir(parents=True, exist_ok=True)
            fps = _infer_render_fps(env, fallback=cfg.env.fps)
            write_video(str(cfg.video_path), np.stack(video_frames), fps)
            logger.info("Saved rollout video to %s", cfg.video_path)
    finally:
        trace.close()
        close_envs(envs_dict)


if __name__ == "__main__":
    init_logging()
    register_third_party_plugins()
    main()
