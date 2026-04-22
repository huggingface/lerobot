"""Base reward model config and processor step.

All reward models share:
- A common config structure (``RewardModelConfig``) with ``type`` dispatch
- A common processor step (``BaseRewardProcessorStep``) that reads a
  transition, calls the subclass's ``compute_reward()``, and writes
  reward/done back to the transition.

Subclasses only need to implement ``compute_reward()`` and optionally
``__post_init__()`` for model loading.
"""

from __future__ import annotations

import logging
import math
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep

logger = logging.getLogger(__name__)


@dataclass
class RewardModelConfig:
    """Configuration shared by all reward model types.

    ``terminate_on_success`` is NOT part of this config — it is passed to the
    step at construction time, mirroring the upstream pattern where both the
    reward classifier and the intervention step share the same flag.
    """

    type: str = "height_gripper"
    success_threshold: float = 0.5
    success_reward: float = 1.0


@dataclass
class BaseRewardProcessorStep(ProcessorStep):
    """Abstract base for reward model processor steps.

    Reads the transition's observation, calls ``compute_reward()`` (implemented
    by subclasses), and updates reward / done on the transition.

    Subclasses must implement ``compute_reward(observation) -> float``
    returning a value in [0, 1]. Values close to 1.0 are treated as success.
    """

    config: RewardModelConfig = field(default_factory=RewardModelConfig)
    terminate_on_success: bool = True

    @abstractmethod
    def compute_reward(self, observation: dict[str, Any]) -> float:
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return new_transition

        start_time = time.perf_counter()
        predicted_reward = self.compute_reward(observation)
        classifier_dt = time.perf_counter() - start_time

        reward = new_transition.get(TransitionKey.REWARD, 0.0)
        terminated = new_transition.get(TransitionKey.DONE, False)

        if math.isclose(predicted_reward, 1.0, abs_tol=1e-2):
            reward = self.config.success_reward
            if self.terminate_on_success:
                terminated = True

        new_transition[TransitionKey.REWARD] = reward
        new_transition[TransitionKey.DONE] = terminated

        info = new_transition.get(TransitionKey.INFO, {})
        info["reward_classifier_frequency"] = 1.0 / classifier_dt if classifier_dt > 0 else 0.0
        new_transition[TransitionKey.INFO] = info

        return new_transition

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
