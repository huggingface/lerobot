#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.rl.gym_manipulator import reset_and_build_transition, step_env_and_process_transition
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.robot_utils import precise_sleep

from .online import StepResult

if TYPE_CHECKING:
    import gymnasium as gym

    from lerobot.lerobot_types import EnvTransition, PolicyAction
    from lerobot.processor import DataProcessorPipeline, PolicyProcessorPipeline


class HILRLTEnvironment:
    """Use LeRobot's HIL robot/processors while keeping replay actions normalized."""

    def __init__(
        self,
        env: gym.Env,
        env_processor: DataProcessorPipeline,
        action_processor: DataProcessorPipeline,
        policy_preprocessor: PolicyProcessorPipeline,
        policy_postprocessor: PolicyProcessorPipeline,
        *,
        use_relative_actions: bool = False,
        fps: int | None = None,
    ) -> None:
        if use_relative_actions:
            raise NotImplementedError(
                "RLT HIL action normalization currently requires PI0 use_relative_actions=False"
            )
        self.env = env
        self.env_processor = env_processor
        self.action_processor = action_processor
        self.policy_preprocessor = policy_preprocessor
        self.policy_postprocessor = policy_postprocessor
        self.fps = fps
        if fps is not None and fps <= 0:
            raise ValueError("fps must be positive")
        self.action_normalizer = next(
            (step for step in policy_preprocessor.steps if isinstance(step, NormalizerProcessorStep)),
            None,
        )
        if self.action_normalizer is None:
            raise ValueError("PI0 preprocessor does not contain an action normalizer")
        self.transition: EnvTransition | None = None

    def reset(self) -> dict[str, torch.Tensor]:
        self.transition = reset_and_build_transition(self.env, self.env_processor, self.action_processor)
        return self.transition[TransitionKey.OBSERVATION]

    def step(self, normalized_action: torch.Tensor) -> StepResult:
        if self.transition is None:
            raise RuntimeError("call reset before step")
        start_time = time.perf_counter()
        physical_action: PolicyAction = self.policy_postprocessor.process_action(normalized_action)
        if physical_action.ndim == 2 and physical_action.shape[0] == 1:
            physical_action = physical_action[0]
        new_transition = step_env_and_process_transition(
            env=self.env,
            transition=self.transition,
            action=physical_action,
            env_processor=self.env_processor,
            action_processor=self.action_processor,
        )
        self.transition = new_transition

        info = new_transition[TransitionKey.INFO]
        complementary = new_transition[TransitionKey.COMPLEMENTARY_DATA]
        executed_physical = complementary["teleop_action"]
        normalizer_device = self.action_normalizer.device or executed_physical.device
        executed_physical = executed_physical.to(normalizer_device)
        normalized_transition = self.action_normalizer(create_transition(action=executed_physical))
        executed_normalized = normalized_transition[TransitionKey.ACTION]

        terminated = bool(new_transition.get(TransitionKey.DONE, False))
        truncated = bool(new_transition.get(TransitionKey.TRUNCATED, False))
        success = bool(info.get(TeleopEvents.SUCCESS, False))
        if self.fps is not None:
            precise_sleep(max(1 / self.fps - (time.perf_counter() - start_time), 0.0))
        return StepResult(
            observation=new_transition[TransitionKey.OBSERVATION],
            reward=float(new_transition[TransitionKey.REWARD]),
            terminated=terminated,
            truncated=truncated,
            success=success,
            executed_action=executed_normalized,
            intervened=bool(info.get(TeleopEvents.IS_INTERVENTION, False)),
            info={str(key): value for key, value in info.items()},
        )
