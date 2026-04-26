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
from collections.abc import Mapping
from contextlib import nullcontext
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
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass
class PolicyInferenceConfig:
    env: envs.EnvConfig
    policy: PreTrainedConfig | None = None
    steps: int = 20
    suite: str | None = None
    task_id: int | None = None
    seed: int | None = 1000
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

    def select_action_tensor(
        self,
        raw_observation: Mapping[str, Any],
        *,
        env: gym.vector.VectorEnv | None = None,
        task: str | None = None,
    ) -> Tensor:
        """Return one env-ready action tensor with shape `(action_dim,)`."""
        observation = preprocess_observation(dict(raw_observation))
        observation["task"] = [task if task is not None else _infer_task(env)]

        observation = self.env_preprocessor(observation)
        observation = self.preprocessor(observation)

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


def make_action_api(cfg: PolicyInferenceConfig) -> tuple[LeRobotActionAPI, dict[str, dict[int, gym.vector.VectorEnv]]]:
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
    set_seed(cfg.seed)
    action_api, envs_dict = make_action_api(cfg)
    suite, task_id, env = _select_env(envs_dict, suite=cfg.suite, task_id=cfg.task_id)
    if env.num_envs != 1:
        raise ValueError(f"This example only supports a single environment, got env.num_envs={env.num_envs}.")

    logger.info("Running single-env policy inference demo on suite=%s task_id=%s", suite, task_id)

    try:
        action_api.reset()
        if cfg.seed is None:
            observation, _ = env.reset()
        else:
            observation, _ = env.reset(seed=[cfg.seed])

        done = False
        success = False
        max_steps = cfg.steps
        if max_steps <= 0:
            max_steps = int(env.call("_max_episode_steps")[0])

        last_step = -1
        for step in range(max_steps):
            last_step = step
            action = choose_action_with_external_planner(action_api, observation, env)
            observation, reward, terminated, truncated, info = env.step(action[None, :])
            done = bool(np.asarray(terminated)[0] or np.asarray(truncated)[0])
            success = success or _extract_success(info)
            logger.info(
                "step=%s reward=%s done=%s success=%s action_shape=%s",
                step,
                float(np.asarray(reward)[0]),
                done,
                success,
                action.shape,
            )
            if done:
                break

        logger.info("Finished after %s step(s). success=%s", last_step + 1, success)
    finally:
        close_envs(envs_dict)


if __name__ == "__main__":
    init_logging()
    register_third_party_plugins()
    main()
