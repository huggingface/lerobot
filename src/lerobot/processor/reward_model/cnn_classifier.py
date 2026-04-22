"""CNN reward classifier processor step for HIL-SERL.

Wraps upstream ``lerobot.policies.sac.reward_model.modeling_classifier.Classifier``
in the ``BaseRewardProcessorStep`` interface so it can be dispatched like any
other reward model via the processor pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.processor.reward_model.base import (
    BaseRewardProcessorStep,
    RewardModelConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class CNNRewardConfig(RewardModelConfig):
    type: str = "cnn"
    success_threshold: float = 0.5
    pretrained_path: str | None = None
    device: str = "cpu"


@dataclass
class CNNRewardProcessorStep(BaseRewardProcessorStep):
    """Reward step that uses an upstream ``Classifier`` for image-based success detection."""

    config: CNNRewardConfig = field(default_factory=CNNRewardConfig)

    def __post_init__(self) -> None:
        from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

        self._classifier: Classifier | None = None

        if self.config.pretrained_path is not None:
            logger.info(
                "Loading CNN reward classifier from %s onto %s",
                self.config.pretrained_path,
                self.config.device,
            )
            self._classifier = Classifier.from_pretrained(self.config.pretrained_path)
            # Stash checkpoint path so patched predict_reward can find the preprocessor.
            self._classifier.config.pretrained_path = self.config.pretrained_path
            self._classifier.to(self.config.device)
            self._classifier.eval()
            logger.info("CNN reward classifier loaded")

    def compute_reward(self, observation: dict[str, Any]) -> float:
        if self._classifier is None:
            return 0.0

        images: dict[str, torch.Tensor] = {}
        for key, value in observation.items():
            if "image" in key and isinstance(value, torch.Tensor):
                images[key] = value.to(self.config.device)

        if not images:
            logger.warning("No image keys in observation for CNN reward classifier")
            return 0.0

        with torch.inference_mode():
            reward_tensor = self._classifier.predict_reward(
                images,
                threshold=self.config.success_threshold,
            )

        reward = float(reward_tensor.item()) if reward_tensor.ndim == 0 else float(reward_tensor.flatten()[0].item())

        if reward > 0.5:
            logger.info("Success!")
        return reward
