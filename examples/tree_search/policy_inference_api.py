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

import logging
import base64
import io
import json
import math
import os
import time
from collections.abc import Mapping
from contextlib import nullcontext
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


@dataclass
class PolicyInferenceConfig:
    env: envs.EnvConfig
    policy: PreTrainedConfig | None = None
    steps: int = 20
    planner: str = "baseline"
    chunk_size: int = 15
    mcts_simulations: int = 16
    mcts_depth: int = 2
    mcts_num_candidates: int = 4
    mcts_noise_std: float = 0.05
    mcts_exploration: float = 1.4
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
    vlm_temperature: float = 0.0
    vlm_top_p: float = 1.0
    vlm_timeout_s: float = 30.0
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


def _restore_base_env(base_env: gym.Env, state: np.ndarray) -> Mapping[str, Any]:
    if not hasattr(base_env, "restore"):
        raise ValueError(f"Environment {type(base_env).__name__} does not expose restore().")
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


@dataclass
class ScoreResult:
    score: float
    reason: str
    prompt: str
    raw_response: Any | None = None


class VLMHeuristicScorer:
    """Scores rendered states.

    If ``vlm_api_url`` is not provided this returns a success-only score
    (1.0 for environment success, otherwise 0.0). When ``vlm_model`` is
    provided, it uses the OpenAI Python client. ``vlm_base_url`` can point to an
    OpenAI-compatible provider such as NVIDIA NIM.
    """

    def __init__(self, cfg: PolicyInferenceConfig) -> None:
        self.base_url = cfg.vlm_base_url or cfg.vlm_api_url
        self.api_key_env = cfg.vlm_api_key_env or "OPENAI_API_KEY"
        self.model = cfg.vlm_model
        self.max_tokens = cfg.vlm_max_tokens
        self.temperature = cfg.vlm_temperature
        self.top_p = cfg.vlm_top_p
        self.timeout_s = cfg.vlm_timeout_s
        self._client = None

    def build_prompt(self, task: str) -> str:
        return (
            f'Task: "{task}"\n\n'
            "You are scoring progress in a robot manipulation task from a single rendered image. "
            "Return JSON only with keys `score` and `reason`.\n"
            "Score meaning: 0.0 = no visible progress, 0.3 = robot is near or reaching the relevant object, "
            "0.5 = object is grasped or moved toward the target, 0.8 = object is near/over the target but "
            "completion is not certain, 1.0 = task appears complete."
        )

    def score(
        self,
        *,
        image: np.ndarray,
        task: str,
        success: bool,
        metadata: Mapping[str, Any],
    ) -> ScoreResult:
        prompt = self.build_prompt(task)
        if success:
            return ScoreResult(score=1.0, reason="Environment success predicate is true.", prompt=prompt)
        if self.model is None:
            return ScoreResult(
                score=0.0, reason="VLM API disabled; using success-only fallback.", prompt=prompt
            )

        try:
            client = self._get_client()
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{self._image_to_base64_png(image)}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=False,
            )
            content = completion.choices[0].message.content or ""
            parsed = self._parse_score_json(content)
            raw = completion.model_dump() if hasattr(completion, "model_dump") else str(completion)
        except Exception as exc:
            logger.warning("VLM scoring failed: %s", exc)
            return ScoreResult(score=0.0, reason=f"VLM scoring failed: {exc}", prompt=prompt)

        score = float(parsed.get("score", 0.0))
        score = min(1.0, max(0.0, score))
        return ScoreResult(score=score, reason=str(parsed.get("reason", "")), prompt=prompt, raw_response=raw)

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
        observation = _restore_base_env(base_env, _snapshot_base_env(base_env))
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
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    candidates = [("policy", policy_chunk.astype(np.float32, copy=True))]
    for candidate_ix in range(1, num_candidates):
        noisy = policy_chunk + rng.normal(0.0, noise_std, size=policy_chunk.shape)
        noisy = np.clip(noisy, -1.0, 1.0).astype(np.float32)
        candidates.append((f"policy_noise_{candidate_ix}", noisy))
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
    task: str,
    success: bool,
    metadata: Mapping[str, Any],
) -> ScoreResult:
    return scorer.score(image=image, task=task, success=success, metadata=metadata)


def _expand_mcts_node(
    *,
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
    root_env_step: int,
    max_env_steps: int,
) -> list[dict[str, Any]]:
    if node["terminal"]:
        return []

    node_state = states_by_id[node["id"]]
    node_observation = observations_by_id[node["id"]]
    _restore_base_env(base_env, node_state)
    policy_chunk = action_api.predict_action_chunk(
        node_observation,
        env=vector_env,
        task=task,
        horizon=cfg.chunk_size,
    )
    candidates = _make_action_candidates(
        policy_chunk,
        num_candidates=cfg.mcts_num_candidates,
        noise_std=cfg.mcts_noise_std,
        rng=rng,
    )

    created_children: list[dict[str, Any]] = []
    for candidate_index, (source, actions) in enumerate(candidates):
        _restore_base_env(base_env, node_state)
        rollout = _rollout_action_sequence(
            base_env=base_env,
            actions=actions,
            start_env_step=root_env_step + int(node["depth"]) * cfg.chunk_size,
            max_steps=min(max_env_steps, root_env_step + (int(node["depth"]) + 1) * cfg.chunk_size),
            trace=None,
            save_step_images=False,
        )
        score = _score_state(
            scorer=scorer,
            image=rollout.frame,
            task=task,
            success=rollout.success,
            metadata={
                "planner": "mcts",
                "parent_id": node["id"],
                "candidate_index": candidate_index,
                "source": source,
                "depth": int(node["depth"]) + 1,
                "reward_sum": rollout.reward_sum,
            },
        )
        child = trace.add_node(
            parent_id=node["id"],
            depth=int(node["depth"]) + 1,
            env_step=root_env_step + int(node["depth"]) * cfg.chunk_size + rollout.action_count,
            image=rollout.frame,
            score=score,
            reward_sum=rollout.reward_sum,
            success=rollout.success,
            terminal=rollout.terminated or rollout.truncated or rollout.success,
            planner="mcts",
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
                "noise_std": cfg.mcts_noise_std if candidate_index else 0.0,
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

    _restore_base_env(base_env, node_state)
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
        task=task,
        success=False,
        metadata={"planner": "mcts", "root": True, "env_step": env_step},
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
            created = _expand_mcts_node(
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
                root_env_step=env_step,
                max_env_steps=max_env_steps,
            )
            if not created:
                value = _node_mean_value(node)
            else:
                selected = max(created, key=lambda child: float(child["vlm_score"]))
                path.append(selected)
                value = float(selected["vlm_score"])

        _backup_path(path, value)
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
    _restore_base_env(base_env, root_state)
    return np.asarray(best_edge["actions"], dtype=np.float32), {
        "root_id": root["id"],
        "selected_edge_id": best_edge["id"],
        "selected_child_id": best_child["id"],
        "selected_value": _node_mean_value(best_child),
        "selected_score": best_child["vlm_score"],
        "selected_visits": best_child["visits"],
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


@parser.wrap()
def main(cfg: PolicyInferenceConfig) -> None:
    if cfg.planner not in {"baseline", "mcts"}:
        raise ValueError("`planner` must be one of: baseline, mcts")
    if cfg.chunk_size <= 0:
        raise ValueError("`chunk_size` must be positive.")
    if cfg.mcts_depth <= 0:
        raise ValueError("`mcts_depth` must be positive.")
    if cfg.mcts_simulations <= 0:
        raise ValueError("`mcts_simulations` must be positive.")
    if cfg.mcts_num_candidates <= 0:
        raise ValueError("`mcts_num_candidates` must be positive.")

    set_seed(cfg.seed)
    action_api, envs_dict = make_action_api(cfg)
    suite, task_id, env = _select_env(envs_dict, suite=cfg.suite, task_id=cfg.task_id)
    if env.num_envs != 1:
        raise ValueError(f"This example only supports a single environment, got env.num_envs={env.num_envs}.")

    base_env = _get_single_base_env(env)
    task = _infer_task(env)
    rng = np.random.default_rng(cfg.seed)
    scorer = VLMHeuristicScorer(cfg)
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
            "mcts_simulations": cfg.mcts_simulations,
            "mcts_depth": cfg.mcts_depth,
            "mcts_num_candidates": cfg.mcts_num_candidates,
            "mcts_noise_std": cfg.mcts_noise_std,
            "vlm_model": cfg.vlm_model,
            "vlm_base_url": cfg.vlm_base_url or cfg.vlm_api_url,
            "vlm_api_key_env": cfg.vlm_api_key_env or "OPENAI_API_KEY",
        },
    )

    logger.info(
        "Running planner=%s single-env inference on suite=%s task_id=%s chunk_size=%s",
        cfg.planner,
        suite,
        task_id,
        cfg.chunk_size,
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
            else:
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
                "action_chunk_shape=%s vlm_score=%.3f",
                macro_step - 1,
                env_step,
                last_reward_sum,
                done,
                success,
                actions.shape,
                endpoint_score.score,
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
