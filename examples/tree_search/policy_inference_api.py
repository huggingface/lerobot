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

import heapq
import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
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
from reward_model import RewardBatchProcessor, load_reward_model_checkpoint, move_batch_to_device

logger = logging.getLogger(__name__)
TREE_SEARCH_DIR = Path(__file__).resolve().parent


@dataclass
class PolicyInferenceConfig:
    env: envs.EnvConfig
    policy: PreTrainedConfig | None = None
    steps: int = 20
    n_episodes: int = 1
    output_dir: Path = Path("outputs/tree_search/search_eval")
    planner: str = "baseline"
    chunk_size: int = 15
    search_num_candidates: int = 4
    search_noise_std: float = 0.05
    search_noise_mode: str = "chunk"
    search_adaptive_noise: bool = True
    search_adaptive_window: int = 4
    search_adaptive_improvement_threshold: float = 0.03
    search_adaptive_time_gain: float = 1.0
    search_adaptive_stagnation_gain: float = 1.0
    search_adaptive_oscillation_gain: float = 0.5
    search_adaptive_high_score_damping: float = 0.6
    search_noise_min_scale: float = 0.25
    search_noise_max_scale: float = 4.0
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
    max_episodes_rendered: int = 1
    trace_dir: Path | None = None
    max_traces: int = 1
    trace_save_step_images: bool = True
    search_verbose: bool = False
    reward_model_checkpoint: Path | None = None
    reward_scene_image_keys: str = "image,base_0_rgb"
    reward_wrist_image_keys: str = "image2,left_wrist_0_rgb"
    reward_use_rendered_fallback: bool = True
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


def _load_task_language_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected task language map JSON object, got {type(data).__name__}: {path}")

    translations: dict[str, str] = {}

    def add_mapping(key: object, value: object, *, prefix: str | None = None) -> None:
        if not isinstance(value, str):
            raise ValueError(f"Expected translated task language string for key={key!r} in {path}")
        text_key = str(key)
        translations[text_key] = value
        if prefix is not None:
            translations[f"{prefix}.{text_key}"] = value

    for key, value in data.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                add_mapping(nested_key, nested_value, prefix=str(key))
        else:
            add_mapping(key, value)
    return translations


def _translate_task_language(
    task: str,
    *,
    suite: str,
    task_id: int,
    translations: Mapping[str, str],
) -> str:
    for key in (
        f"{suite}.{task_id}",
        f"{suite}:{task_id}",
        f"{suite}/task_{task_id}",
        f"task_{task_id}",
        str(task_id),
        task,
    ):
        if key in translations:
            return translations[key]
    return task


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


def _iter_selected_envs(
    envs_dict: dict[str, dict[int, gym.vector.VectorEnv]],
    *,
    suite: str | None,
    task_id: int | None,
) -> list[tuple[str, int, gym.vector.VectorEnv]]:
    suite_names = [suite] if suite is not None else list(envs_dict)
    selected: list[tuple[str, int, gym.vector.VectorEnv]] = []
    for suite_name in suite_names:
        if suite_name not in envs_dict:
            raise ValueError(f"Unknown suite '{suite_name}'. Available suites: {list(envs_dict)}")
        task_envs = envs_dict[suite_name]
        task_ids = [task_id] if task_id is not None else list(task_envs)
        for selected_task_id in task_ids:
            if selected_task_id not in task_envs:
                raise ValueError(
                    f"Unknown task_id '{selected_task_id}' for suite '{suite_name}'. "
                    f"Available: {list(task_envs)}"
                )
            selected.append((suite_name, int(selected_task_id), task_envs[int(selected_task_id)]))
    return selected


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


def _image_array(image: Any, *, flip_hw: bool = False) -> np.ndarray | None:
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


def _split_keys(raw: str) -> tuple[str, ...]:
    return tuple(key.strip() for key in raw.split(",") if key.strip())


def _first_observation_image(
    observation: Mapping[str, Any] | None,
    image_keys: Sequence[str],
) -> tuple[str, np.ndarray] | None:
    if observation is None:
        return None
    pixels = observation.get("pixels")
    if not isinstance(pixels, Mapping):
        return None
    for key in image_keys:
        image = _image_array(pixels.get(key), flip_hw=True)
        if image is not None:
            return f"observation.{key}", image
    return None


def _ensure_vector(value: Any, *, dim: int) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    array = array.reshape(-1)
    if array.size != dim:
        return None
    return array.astype(np.float32, copy=True)


def _quat_xyzw_to_axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return np.zeros(3, dtype=np.float32)
    quat = quat / norm
    xyz = quat[:3]
    w = float(np.clip(quat[3], -1.0, 1.0))
    angle = 2.0 * math.acos(w)
    denom = math.sqrt(max(1e-12, 1.0 - w * w))
    if denom < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (xyz / denom * angle).astype(np.float32)


def _extract_proprioception(observation: Mapping[str, Any] | None) -> np.ndarray | None:
    if observation is None:
        return None
    for key in ("observation.state", "state"):
        state = _ensure_vector(observation.get(key), dim=8)
        if state is not None:
            return state

    robot_state = observation.get("robot_state")
    if not isinstance(robot_state, Mapping):
        return None

    eef = robot_state.get("eef", {})
    gripper = robot_state.get("gripper", {})
    if not isinstance(eef, Mapping) or not isinstance(gripper, Mapping):
        return None

    eef_pos = _ensure_vector(eef.get("pos"), dim=3)
    eef_quat = _ensure_vector(eef.get("quat"), dim=4)
    gripper_qpos = _ensure_vector(gripper.get("qpos"), dim=2)
    if eef_pos is None or eef_quat is None or gripper_qpos is None:
        return None
    return np.concatenate([eef_pos, _quat_xyzw_to_axisangle(eef_quat), gripper_qpos]).astype(np.float32)


@dataclass
class ScoreResult:
    score: float
    reason: str
    images: Sequence[tuple[str, np.ndarray]] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RewardModelScorer:
    def __init__(self, cfg: PolicyInferenceConfig, *, device: torch.device | str) -> None:
        self.scene_image_keys = _split_keys(cfg.reward_scene_image_keys)
        self.wrist_image_keys = _split_keys(cfg.reward_wrist_image_keys)
        self.use_rendered_fallback = cfg.reward_use_rendered_fallback
        self.model = None
        self.processor: RewardBatchProcessor | None = None
        self.model_cfg = None
        self.scene_temporal_window = 1
        self.device = torch.device(device)

        if cfg.reward_model_checkpoint is None:
            logger.warning("No reward model checkpoint provided; scoring will use success-only fallback.")
            return
        model, model_cfg, _ = load_reward_model_checkpoint(cfg.reward_model_checkpoint, device=self.device)
        self.model = model
        self.model_cfg = model_cfg
        self.scene_temporal_window = max(1, int(getattr(model_cfg, "scene_temporal_window", 1)))
        self.processor = RewardBatchProcessor(model_cfg)
        logger.info(
            "Loaded reward model checkpoint: %s scene_temporal_window=%s",
            cfg.reward_model_checkpoint,
            self.scene_temporal_window,
        )

    def extend_scene_history(
        self,
        scene_history: Sequence[np.ndarray] | None,
        current_scene: np.ndarray,
    ) -> list[np.ndarray]:
        history = list(scene_history or [])
        history.append(np.asarray(current_scene).copy())
        return history[-self.scene_temporal_window :]

    def prior_history_for_current(self, scene_history: Sequence[np.ndarray] | None) -> list[np.ndarray]:
        history = list(scene_history or [])
        if not history:
            return []
        return history[:-1]

    def _scene_images_for_score(
        self,
        *,
        scene_history: Sequence[np.ndarray] | None,
        current_scene: np.ndarray,
    ) -> list[np.ndarray]:
        sequence = [np.asarray(image).copy() for image in list(scene_history or [])]
        sequence.append(np.asarray(current_scene).copy())
        sequence = sequence[-self.scene_temporal_window :]
        if len(sequence) < self.scene_temporal_window:
            sequence = [sequence[0]] * (self.scene_temporal_window - len(sequence)) + sequence
        return sequence

    def current_scene_image(
        self,
        *,
        image: np.ndarray,
        observation: Mapping[str, Any] | None,
    ) -> np.ndarray:
        scene = _first_observation_image(observation, self.scene_image_keys)
        if scene is None and self.use_rendered_fallback:
            rendered = _image_array(image)
            if rendered is not None:
                scene = ("rendered_overview", rendered)
        if scene is None:
            raise ValueError(f"Could not find scene image from keys={self.scene_image_keys}.")
        return scene[1]

    def score(
        self,
        *,
        image: np.ndarray,
        observation: Mapping[str, Any] | None,
        task: str,
        success: bool,
        metadata: Mapping[str, Any],
        scene_history: Sequence[np.ndarray] | None = None,
    ) -> ScoreResult:
        scene = _first_observation_image(observation, self.scene_image_keys)
        if scene is None and self.use_rendered_fallback:
            rendered = _image_array(image)
            if rendered is not None:
                scene = ("rendered_overview", rendered)
        if scene is None:
            raise ValueError(f"Could not find scene image from keys={self.scene_image_keys}.")

        wrist = _first_observation_image(observation, self.wrist_image_keys)
        score_images = [scene]
        sample: dict[str, Any] = {
            "task": task,
            "label": 0.0,
            "scene_image": scene[1],
            "scene_images": self._scene_images_for_score(
                scene_history=scene_history,
                current_scene=scene[1],
            ),
        }
        if wrist is not None:
            sample["wrist_image"] = wrist[1]
            score_images.append(wrist)

        proprioception = _extract_proprioception(observation)
        if proprioception is not None:
            sample["proprioception"] = proprioception

        score_metadata = {
            **dict(metadata),
            "score_image_labels": [label for label, _ in score_images],
            "has_proprioception": proprioception is not None,
            "scene_temporal_window": self.scene_temporal_window,
            "scene_temporal_count": len(sample["scene_images"]),
            "scene_history_count": len(scene_history or []),
        }
        if success:
            return ScoreResult(
                score=1.0,
                reason="Environment success predicate is true.",
                images=score_images,
                metadata=score_metadata,
            )
        if self.model is None or self.processor is None:
            return ScoreResult(
                score=0.0,
                reason="Reward model disabled; using success-only fallback.",
                images=score_images,
                metadata=score_metadata,
            )

        with torch.no_grad():
            batch = move_batch_to_device(self.processor([sample]), self.device)
            value = float(
                self.model(
                    scene_pixel_values=batch["scene_pixel_values"],
                    wrist_pixel_values=batch.get("wrist_pixel_values"),
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                    proprioception=batch.get("proprioception"),
                )
                .detach()
                .cpu()[0]
            )
        return ScoreResult(
            score=min(1.0, max(0.0, value)),
            reason="Reward model prediction.",
            images=score_images,
            metadata=score_metadata,
        )


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
            image_path = self.save_image(f"{node_id}_score_{image_ix:02d}_{safe_label}", image)
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
        score_images = self.save_labeled_images(node_id, score.images)
        node = {
            "id": node_id,
            "parent_id": parent_id,
            "depth": depth,
            "env_step": env_step,
            "planner": planner,
            "image_path": image_path,
            "score": score.score,
            "score_reason": score.reason,
            "score_images": score_images,
            "score_metadata": _to_jsonable(dict(score.metadata)),
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
        step_success = _extract_success(info)
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


def _oscillation_score(values: Sequence[float]) -> float:
    if len(values) < 4:
        return 0.0
    deltas = np.diff(np.asarray(values, dtype=np.float32))
    deltas = deltas[np.abs(deltas) > 1e-6]
    if len(deltas) < 2:
        return 0.0
    sign_changes = np.sign(deltas[1:]) != np.sign(deltas[:-1])
    return float(np.mean(sign_changes))


def _adaptive_noise_schedule(
    *,
    cfg: PolicyInferenceConfig,
    env_step: int,
    max_env_steps: int,
    current_score: float,
    reward_history: Sequence[float] | None,
) -> dict[str, Any]:
    base_noise = max(0.0, float(cfg.search_noise_std))
    max_steps = max(1, int(max_env_steps))
    time_ratio = min(1.0, max(0.0, float(env_step) / float(max_steps)))
    history = [float(value) for value in list(reward_history or []) if np.isfinite(value)]
    score = float(current_score)
    values = history + [score]
    window = max(1, int(cfg.search_adaptive_window))
    recent = values[-window:]

    previous_best = max(history[-window:]) if history else score
    recent_best = max(recent) if recent else score
    recent_mean = float(np.mean(recent)) if recent else score
    improvement = float(recent_best - previous_best)
    slope = float(score - history[-1]) if history else 0.0
    threshold = max(0.0, float(cfg.search_adaptive_improvement_threshold))
    stagnation = 1.0 if len(values) >= 2 and improvement < threshold and slope <= threshold else 0.0
    oscillation = _oscillation_score(recent)
    high_score_damping = 1.0 - float(cfg.search_adaptive_high_score_damping) * min(1.0, max(0.0, score))
    high_score_damping = max(0.0, high_score_damping)

    raw_scale = (
        1.0
        + float(cfg.search_adaptive_time_gain) * time_ratio
        + float(cfg.search_adaptive_stagnation_gain) * stagnation
        + float(cfg.search_adaptive_oscillation_gain) * oscillation
    ) * high_score_damping
    min_scale = max(0.0, float(cfg.search_noise_min_scale))
    max_scale = max(min_scale, float(cfg.search_noise_max_scale))
    scale = float(np.clip(raw_scale, min_scale, max_scale))
    effective_noise = base_noise * scale if cfg.search_adaptive_noise else base_noise

    return {
        "enabled": bool(cfg.search_adaptive_noise),
        "base_noise_std": base_noise,
        "effective_noise_std": float(effective_noise),
        "noise_scale": float(scale if cfg.search_adaptive_noise else 1.0),
        "raw_noise_scale": float(raw_scale),
        "min_scale": min_scale,
        "max_scale": max_scale,
        "time_ratio": float(time_ratio),
        "env_step": int(env_step),
        "max_env_steps": int(max_env_steps),
        "current_score": score,
        "history_count": len(history),
        "window": window,
        "recent_scores": recent,
        "recent_mean": recent_mean,
        "recent_best": recent_best,
        "previous_best": previous_best,
        "improvement": improvement,
        "slope": slope,
        "improvement_threshold": threshold,
        "stagnation": float(stagnation),
        "oscillation": float(oscillation),
        "high_score_damping": float(high_score_damping),
        "time_gain": float(cfg.search_adaptive_time_gain),
        "stagnation_gain": float(cfg.search_adaptive_stagnation_gain),
        "oscillation_gain": float(cfg.search_adaptive_oscillation_gain),
    }


def _node_mean_value(node: Mapping[str, Any]) -> float:
    visits = int(node.get("visits", 0))
    if visits <= 0:
        return float(node.get("score", 0.0))
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
    scorer: RewardModelScorer,
    image: np.ndarray,
    observation: Mapping[str, Any] | None = None,
    task: str,
    success: bool,
    metadata: Mapping[str, Any],
    scene_history: Sequence[np.ndarray] | None = None,
) -> ScoreResult:
    return scorer.score(
        image=image,
        observation=observation,
        task=task,
        success=success,
        metadata=metadata,
        scene_history=scene_history,
    )


def _expand_search_node(
    *,
    planner: str,
    node: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    child_edges: dict[str, dict[str, Any]],
    states_by_id: dict[str, np.ndarray],
    observations_by_id: dict[str, Mapping[str, Any]],
    scene_histories_by_id: dict[str, list[np.ndarray]],
    base_env: gym.Env,
    vector_env: gym.vector.VectorEnv,
    action_api: LeRobotActionAPI,
    scorer: RewardModelScorer,
    trace: TraceRecorder,
    cfg: PolicyInferenceConfig,
    rng: np.random.Generator,
    task: str,
    max_env_steps: int,
    reward_history: Sequence[float] | None = None,
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
    noise_schedule = _adaptive_noise_schedule(
        cfg=cfg,
        env_step=int(node["env_step"]),
        max_env_steps=max_env_steps,
        current_score=float(node.get("score", 0.0)),
        reward_history=reward_history,
    )
    node["adaptive_noise"] = _to_jsonable(noise_schedule)
    trace.write_event(
        "adaptive_noise",
        {
            "planner": planner,
            "node_id": node["id"],
            "depth": int(node["depth"]),
            **noise_schedule,
        },
    )
    candidates = _make_action_candidates(
        policy_chunk,
        num_candidates=cfg.search_num_candidates,
        noise_std=float(noise_schedule["effective_noise_std"]),
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
                "adaptive_noise": noise_schedule,
            },
            scene_history=scene_histories_by_id.get(node["id"], []),
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
                "parent_adaptive_noise": noise_schedule,
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
                "noise_std": float(noise_schedule["effective_noise_std"]) if candidate_index else 0.0,
                "base_noise_std": float(noise_schedule["base_noise_std"]) if candidate_index else 0.0,
                "noise_scale": float(noise_schedule["noise_scale"]) if candidate_index else 0.0,
                "noise_mode": cfg.search_noise_mode if candidate_index else "none",
                "adaptive_noise": noise_schedule,
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
        if score.images:
            scene_histories_by_id[child["id"]] = scorer.extend_scene_history(
                scene_histories_by_id.get(node["id"], []),
                score.images[0][1],
            )
        else:
            scene_histories_by_id[child["id"]] = list(scene_histories_by_id.get(node["id"], []))
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
    scorer: RewardModelScorer,
    trace: TraceRecorder,
    cfg: PolicyInferenceConfig,
    rng: np.random.Generator,
    max_env_steps: int,
    scene_history: Sequence[np.ndarray] | None = None,
    reward_history: Sequence[float] | None = None,
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
        scene_history=scorer.prior_history_for_current(scene_history),
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
    scene_histories_by_id = {
        root["id"]: scorer.extend_scene_history(
            scorer.prior_history_for_current(scene_history),
            root_score.images[0][1],
        )
        if root_score.images
        else list(scene_history or [])
    }

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
                scene_histories_by_id=scene_histories_by_id,
                base_env=base_env,
                vector_env=vector_env,
                action_api=action_api,
                scorer=scorer,
                trace=trace,
                cfg=cfg,
                rng=rng,
                task=task,
                max_env_steps=max_env_steps,
                reward_history=reward_history,
            )
            if not created:
                value = _node_mean_value(node)
            else:
                selected = max(created, key=lambda child: float(child["score"]))
                path.append(selected)
                value = float(selected["score"])

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
            float(pair[1]["score"]),
        ),
    )
    _restore_base_env(base_env, root_state, timestep=env_step)
    if cfg.search_verbose:
        logger.info(
            "mcts selected child=%s edge=%s score=%.3f value=%.3f visits=%s",
            best_child["id"],
            best_edge["id"],
            best_child["score"],
            _node_mean_value(best_child),
            best_child["visits"],
        )
    return np.asarray(best_edge["actions"], dtype=np.float32), {
        "root_id": root["id"],
        "selected_edge_id": best_edge["id"],
        "selected_child_id": best_child["id"],
        "selected_value": _node_mean_value(best_child),
        "selected_score": best_child["score"],
        "selected_visits": best_child["visits"],
        "adaptive_noise": root.get("adaptive_noise"),
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
            float(node.get("score", 0.0)),
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
    scorer: RewardModelScorer,
    trace: TraceRecorder,
    cfg: PolicyInferenceConfig,
    rng: np.random.Generator,
    max_env_steps: int,
    scene_history: Sequence[np.ndarray] | None = None,
    reward_history: Sequence[float] | None = None,
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
        scene_history=scorer.prior_history_for_current(scene_history),
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
    scene_histories_by_id = {
        root["id"]: scorer.extend_scene_history(
            scorer.prior_history_for_current(scene_history),
            root_score.images[0][1],
        )
        if root_score.images
        else list(scene_history or [])
    }

    frontier: list[tuple[float, int, int, str]] = []
    push_ix = 0
    heapq.heappush(frontier, (-float(root["score"]), int(root["depth"]), push_ix, root["id"]))
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
        node["value_sum"] = float(node.get("value_sum", 0.0)) + float(node.get("score", 0.0))
        if cfg.search_verbose:
            logger.info(
                "best_first expansion=%s/%s node=%s depth=%s score=%.3f frontier_size=%s",
                expansion_ix + 1,
                cfg.best_first_expansions,
                node["id"],
                node["depth"],
                float(node.get("score", 0.0)),
                len(frontier),
            )

        created = _expand_search_node(
            planner="best_first",
            node=node,
            nodes_by_id=nodes_by_id,
            child_edges=child_edges,
            states_by_id=states_by_id,
            observations_by_id=observations_by_id,
            scene_histories_by_id=scene_histories_by_id,
            base_env=base_env,
            vector_env=vector_env,
            action_api=action_api,
            scorer=scorer,
            trace=trace,
            cfg=cfg,
            rng=rng,
            task=task,
            max_env_steps=max_env_steps,
            reward_history=reward_history,
        )

        for child in created:
            if not child["terminal"] and int(child["depth"]) < cfg.best_first_depth:
                priority = -float(child.get("score", 0.0))
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
                best_descendant["score"] if best_descendant is not None else None,
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
                    best_descendant["score"] if best_descendant is not None else None
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
            key=lambda child: float(child.get("score", 0.0)),
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
            best_root_child["score"],
            best_descendant["id"] if best_descendant is not None else None,
            best_descendant["score"] if best_descendant is not None else None,
            len(expanded),
            len(frontier),
        )
    return np.asarray(best_edge["actions"], dtype=np.float32), {
        "root_id": root["id"],
        "selected_edge_id": best_edge["id"],
        "selected_child_id": best_root_child["id"],
        "selected_descendant_id": best_descendant["id"] if best_descendant is not None else None,
        "selected_score": best_root_child["score"],
        "selected_descendant_score": (
            best_descendant["score"] if best_descendant is not None else None
        ),
        "expanded_count": len(expanded),
        "frontier_size": len(frontier),
        "adaptive_noise": root.get("adaptive_noise"),
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


def _artifact_paths_for_episode(
    cfg: PolicyInferenceConfig,
    *,
    suite: str,
    task_id: int,
    episode_ix: int,
    global_episode_ix: int,
    total_episodes: int,
) -> tuple[Path | None, Path | None]:
    safe_suite = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in suite)
    suffix = f"{safe_suite}_task{task_id:03d}_episode{episode_ix:03d}"

    trace_dir = None
    if cfg.trace_dir is not None and (cfg.max_traces < 0 or global_episode_ix < cfg.max_traces):
        trace_dir = cfg.trace_dir if total_episodes == 1 else cfg.trace_dir / suffix

    video_path = None
    if cfg.video_path is not None and (
        cfg.max_episodes_rendered < 0 or global_episode_ix < cfg.max_episodes_rendered
    ):
        video_path = (
            cfg.video_path
            if total_episodes == 1
            else cfg.video_path.with_name(f"{cfg.video_path.stem}_{suffix}{cfg.video_path.suffix or '.mp4'}")
        )
    return trace_dir, video_path


def _run_search_episode(
    *,
    cfg: PolicyInferenceConfig,
    action_api: LeRobotActionAPI,
    env: gym.vector.VectorEnv,
    suite: str,
    task_id: int,
    episode_ix: int,
    global_episode_ix: int,
    episode_seed: int | None,
    scorer: RewardModelScorer,
    trace_dir: Path | None,
    video_path: Path | None,
    task_language_translations: Mapping[str, str],
) -> dict[str, Any]:
    if env.num_envs != 1:
        raise ValueError(f"Search evaluation only supports single-env rollouts, got env.num_envs={env.num_envs}.")

    base_env = _get_single_base_env(env)
    original_task = _infer_task(env)
    task = _translate_task_language(
        original_task,
        suite=suite,
        task_id=task_id,
        translations=task_language_translations,
    )
    rng = np.random.default_rng(episode_seed)
    trace = TraceRecorder(
        trace_dir,
        run_metadata={
            "planner": cfg.planner,
            "chunk_size": cfg.chunk_size,
            "suite": suite,
            "task_id": task_id,
            "episode_ix": episode_ix,
            "global_episode_ix": global_episode_ix,
            "task": task,
            "original_task": original_task,
            "steps": cfg.steps,
            "seed": episode_seed,
            "policy": str(cfg.policy.pretrained_path) if cfg.policy is not None else None,
            "search_num_candidates": cfg.search_num_candidates,
            "search_noise_std": cfg.search_noise_std,
            "search_noise_mode": cfg.search_noise_mode,
            "search_adaptive_noise": cfg.search_adaptive_noise,
            "search_adaptive_window": cfg.search_adaptive_window,
            "search_adaptive_improvement_threshold": cfg.search_adaptive_improvement_threshold,
            "search_adaptive_time_gain": cfg.search_adaptive_time_gain,
            "search_adaptive_stagnation_gain": cfg.search_adaptive_stagnation_gain,
            "search_adaptive_oscillation_gain": cfg.search_adaptive_oscillation_gain,
            "search_adaptive_high_score_damping": cfg.search_adaptive_high_score_damping,
            "search_noise_min_scale": cfg.search_noise_min_scale,
            "search_noise_max_scale": cfg.search_noise_max_scale,
            "mcts_simulations": cfg.mcts_simulations,
            "mcts_depth": cfg.mcts_depth,
            "best_first_expansions": cfg.best_first_expansions,
            "best_first_depth": cfg.best_first_depth,
            "search_verbose": cfg.search_verbose,
            "reward_model_checkpoint": str(cfg.reward_model_checkpoint)
            if cfg.reward_model_checkpoint is not None
            else None,
            "reward_scene_image_keys": cfg.reward_scene_image_keys,
            "reward_wrist_image_keys": cfg.reward_wrist_image_keys,
            "reward_use_rendered_fallback": cfg.reward_use_rendered_fallback,
        },
    )

    start = time.time()
    try:
        action_api.reset()
        if episode_seed is None:
            observation, _ = env.reset()
        else:
            observation, _ = env.reset(seed=[episode_seed])

        video_frames: list[np.ndarray] = []
        if video_path is not None:
            video_frames.append(_render_single_env_frame(env))

        done = False
        success = False
        max_steps = cfg.steps
        if max_steps <= 0:
            max_steps = int(env.call("_max_episode_steps")[0])

        env_step = 0
        macro_step = 0
        episode_reward_sum = 0.0
        episode_max_reward = -float("inf")
        episode_plan_s = 0.0
        episode_rollout_s = 0.0
        episode_score_s = 0.0
        last_score = 0.0
        reward_score_history: list[float] = []
        baseline_parent_id: str | None = None
        committed_scene_history: list[np.ndarray] = []
        while env_step < max_steps and not done:
            root_frame = _render_base_env_frame(base_env)
            if not committed_scene_history:
                committed_scene_history = scorer.extend_scene_history(
                    [],
                    scorer.current_scene_image(image=root_frame, observation=observation),
                )

            baseline_root: dict[str, Any] | None = None
            plan_start = time.time()
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
                    scene_history=scorer.prior_history_for_current(committed_scene_history),
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
                    scene_history=committed_scene_history,
                    reward_history=reward_score_history,
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
                    scene_history=committed_scene_history,
                    reward_history=reward_score_history,
                )
            macro_plan_s = time.time() - plan_start
            episode_plan_s += macro_plan_s

            rollout_start = time.time()
            rollout = _rollout_action_sequence(
                base_env=base_env,
                actions=actions,
                start_env_step=env_step,
                max_steps=max_steps,
                trace=trace,
                save_step_images=cfg.trace_save_step_images,
                collect_frames=video_path is not None,
                step_image_prefix=f"commit_{macro_step:04d}",
            )
            macro_rollout_s = time.time() - rollout_start
            episode_rollout_s += macro_rollout_s
            if video_path is not None:
                video_frames.extend(rollout.frames)

            episode_reward_sum += rollout.reward_sum
            if rollout.rewards:
                episode_max_reward = max(episode_max_reward, max(rollout.rewards))

            score_start = time.time()
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
                scene_history=committed_scene_history,
            )
            macro_score_s = time.time() - score_start
            episode_score_s += macro_score_s
            last_score = endpoint_score.score
            reward_score_history.append(float(endpoint_score.score))
            if endpoint_score.images:
                committed_scene_history = scorer.extend_scene_history(
                    committed_scene_history,
                    endpoint_score.images[0][1],
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
                    "score": endpoint_score.score,
                    "score_reason": endpoint_score.reason,
                    "macro_plan_s": macro_plan_s,
                    "macro_rollout_s": macro_rollout_s,
                    "macro_score_s": macro_score_s,
                    "episode_plan_s": episode_plan_s,
                    "episode_rollout_s": episode_rollout_s,
                    "episode_score_s": episode_score_s,
                    "plan_info": plan_info,
                    "step_records": rollout.step_records,
                }
            )

            observation = rollout.observation
            env_step += rollout.action_count
            macro_step += 1
            done = rollout.terminated or rollout.truncated or rollout.action_count == 0
            success = success or rollout.success
            logger.info(
                "suite=%s task_id=%s episode=%s macro_step=%s env_step=%s reward_sum=%.3f "
                "done=%s success=%s score=%.3f plan_s=%.2f rollout_s=%.2f score_s=%.2f",
                suite,
                task_id,
                episode_ix,
                macro_step - 1,
                env_step,
                episode_reward_sum,
                done,
                success,
                endpoint_score.score,
                episode_plan_s,
                episode_rollout_s,
                episode_score_s,
            )

        if episode_max_reward == -float("inf"):
            episode_max_reward = 0.0

        summary = {
            "planner": cfg.planner,
            "suite": suite,
            "task_id": task_id,
            "episode_ix": episode_ix,
            "global_episode_ix": global_episode_ix,
            "task": task,
            "original_task": original_task,
            "steps": env_step,
            "macro_steps": macro_step,
            "success": success,
            "sum_reward": episode_reward_sum,
            "max_reward": episode_max_reward,
            "last_score": last_score,
            "plan_s": episode_plan_s,
            "rollout_s": episode_rollout_s,
            "score_s": episode_score_s,
            "eval_s": float(time.time() - start),
            "trace_dir": str(trace_dir) if trace_dir is not None else None,
        }
        trace.write_tree(summary=summary)
        if video_path is not None and video_frames:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            fps = _infer_render_fps(env, fallback=cfg.env.fps)
            write_video(str(video_path), np.stack(video_frames), fps)
            logger.info("Saved rollout video to %s", video_path)

        return {
            "episode_ix": episode_ix,
            "global_episode_ix": global_episode_ix,
            "suite": suite,
            "task_id": task_id,
            "task": task,
            "original_task": original_task,
            "seed": episode_seed,
            "sum_reward": float(episode_reward_sum),
            "max_reward": float(episode_max_reward),
            "success": bool(success),
            "steps": int(env_step),
            "macro_steps": int(macro_step),
            "last_score": float(last_score),
            "eval_s": float(time.time() - start),
            "plan_s": float(episode_plan_s),
            "rollout_s": float(episode_rollout_s),
            "score_s": float(episode_score_s),
            "video_path": str(video_path) if video_path is not None and video_frames else None,
            "trace_dir": str(trace_dir) if trace_dir is not None else None,
        }
    finally:
        trace.close()


def _mean(values: Sequence[Any]) -> float:
    if not values:
        return float("nan")
    return float(np.nanmean(np.asarray(values, dtype=float)))


def _aggregate_episode_records(records: list[dict[str, Any]], *, elapsed_s: float | None = None) -> dict[str, Any]:
    total_plan_s = float(sum(float(record.get("plan_s", 0.0)) for record in records))
    total_rollout_s = float(sum(float(record.get("rollout_s", 0.0)) for record in records))
    total_score_s = float(sum(float(record.get("score_s", 0.0)) for record in records))
    total_macro_steps = int(sum(int(record.get("macro_steps", 0)) for record in records))
    aggregate = {
        "avg_sum_reward": _mean([record["sum_reward"] for record in records]),
        "avg_max_reward": _mean([record["max_reward"] for record in records]),
        "pc_success": _mean([record["success"] for record in records]) * 100 if records else float("nan"),
        "n_episodes": len(records),
        "total_macro_steps": total_macro_steps,
        "total_plan_s": total_plan_s,
        "total_rollout_s": total_rollout_s,
        "total_score_s": total_score_s,
        "avg_plan_s_per_episode": total_plan_s / max(1, len(records)),
        "avg_rollout_s_per_episode": total_rollout_s / max(1, len(records)),
        "avg_score_s_per_episode": total_score_s / max(1, len(records)),
        "avg_plan_s_per_macro_step": total_plan_s / max(1, total_macro_steps),
        "avg_rollout_s_per_macro_step": total_rollout_s / max(1, total_macro_steps),
        "avg_score_s_per_macro_step": total_score_s / max(1, total_macro_steps),
        "video_paths": [record["video_path"] for record in records if record.get("video_path")],
    }
    if elapsed_s is not None:
        aggregate["eval_s"] = float(elapsed_s)
        aggregate["eval_ep_s"] = float(elapsed_s / max(1, len(records)))
        aggregate["plan_fraction"] = float(total_plan_s / elapsed_s) if elapsed_s > 0 else float("nan")
        aggregate["rollout_fraction"] = float(total_rollout_s / elapsed_s) if elapsed_s > 0 else float("nan")
        aggregate["score_fraction"] = float(total_score_s / elapsed_s) if elapsed_s > 0 else float("nan")
    return aggregate


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
    if cfg.search_adaptive_window <= 0:
        raise ValueError("`search_adaptive_window` must be positive.")
    if cfg.search_noise_min_scale < 0:
        raise ValueError("`search_noise_min_scale` must be non-negative.")
    if cfg.search_noise_max_scale < cfg.search_noise_min_scale:
        raise ValueError("`search_noise_max_scale` must be >= `search_noise_min_scale`.")
    if cfg.best_first_expansions <= 0:
        raise ValueError("`best_first_expansions` must be positive.")
    if cfg.best_first_depth <= 0:
        raise ValueError("`best_first_depth` must be positive.")
    if cfg.n_episodes <= 0:
        raise ValueError("`n_episodes` must be positive.")

    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    action_api, envs_dict = make_action_api(cfg)
    scorer = RewardModelScorer(cfg, device=action_api.device)
    task_language_translations = _load_task_language_map(cfg.task_language_map)
    selected_envs = _iter_selected_envs(envs_dict, suite=cfg.suite, task_id=cfg.task_id)
    total_episodes = len(selected_envs) * cfg.n_episodes
    logger.info(
        "Running sequential search eval planner=%s tasks=%s n_episodes=%s total_episodes=%s "
        "chunk_size=%s reward_model=%s",
        cfg.planner,
        len(selected_envs),
        cfg.n_episodes,
        total_episodes,
        cfg.chunk_size,
        cfg.reward_model_checkpoint,
    )
    if task_language_translations:
        logger.info(
            "Loaded task language translations from %s with %s keys",
            cfg.task_language_map,
            len(task_language_translations),
        )

    start_t = time.time()
    all_records: list[dict[str, Any]] = []
    per_task: list[dict[str, Any]] = []
    try:
        global_episode_ix = 0
        for suite, task_id, env in selected_envs:
            task_records: list[dict[str, Any]] = []
            for episode_ix in range(cfg.n_episodes):
                episode_seed = None if cfg.seed is None else int(cfg.seed) + global_episode_ix
                trace_dir, video_path = _artifact_paths_for_episode(
                    cfg,
                    suite=suite,
                    task_id=task_id,
                    episode_ix=episode_ix,
                    global_episode_ix=global_episode_ix,
                    total_episodes=total_episodes,
                )
                record = _run_search_episode(
                    cfg=cfg,
                    action_api=action_api,
                    env=env,
                    suite=suite,
                    task_id=task_id,
                    episode_ix=episode_ix,
                    global_episode_ix=global_episode_ix,
                    episode_seed=episode_seed,
                    scorer=scorer,
                    trace_dir=trace_dir,
                    video_path=video_path,
                    task_language_translations=task_language_translations,
                )
                task_records.append(record)
                all_records.append(record)
                global_episode_ix += 1
                running_success = _mean([item["success"] for item in all_records]) * 100
                logger.info(
                    "search eval progress episode=%s/%s suite=%s task_id=%s success=%s "
                    "running_success_rate=%.1f%%",
                    global_episode_ix,
                    total_episodes,
                    suite,
                    task_id,
                    record["success"],
                    running_success,
                )

            per_task.append(
                {
                    "task_group": suite,
                    "task_id": task_id,
                    "metrics": {
                        "sum_rewards": [record["sum_reward"] for record in task_records],
                        "max_rewards": [record["max_reward"] for record in task_records],
                        "successes": [record["success"] for record in task_records],
                        "video_paths": [
                            record["video_path"] for record in task_records if record.get("video_path")
                        ],
                        "trace_dirs": [
                            record["trace_dir"] for record in task_records if record.get("trace_dir")
                        ],
                        "per_episode": task_records,
                        "aggregated": _aggregate_episode_records(task_records),
                    },
                }
            )
    finally:
        close_envs(envs_dict)

    elapsed_s = time.time() - start_t
    per_group: dict[str, dict[str, Any]] = {}
    for suite_name in sorted({record["suite"] for record in all_records}):
        per_group[suite_name] = _aggregate_episode_records(
            [record for record in all_records if record["suite"] == suite_name]
        )
    overall = _aggregate_episode_records(all_records, elapsed_s=elapsed_s)
    info = {
        "per_task": per_task,
        "per_group": per_group,
        "overall": overall,
        "per_episode": all_records,
    }
    info_path = cfg.output_dir / "eval_info.json"
    with info_path.open("w") as f:
        json.dump(_to_jsonable(info), f, indent=2)
    overall_path = cfg.output_dir / "overall_metrics.json"
    with overall_path.open("w") as f:
        json.dump(_to_jsonable(overall), f, indent=2)

    print("Overall Aggregated Metrics:")
    print(json.dumps(_to_jsonable(overall), indent=2))
    for suite_name, suite_info in per_group.items():
        print(f"\nAggregated Metrics for {suite_name}:")
        print(json.dumps(_to_jsonable(suite_info), indent=2))
    logger.info("Wrote search eval info to %s", info_path)
    logger.info("Wrote overall metrics to %s", overall_path)


if __name__ == "__main__":
    init_logging()
    register_third_party_plugins()
    main()
