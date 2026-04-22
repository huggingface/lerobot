"""Height + gripper reward processor step for lift tasks.

Detects success when:
1. End-effector Z position >= ``height_threshold``
2. Gripper is closed (gripper.pos < ``gripper_closed_threshold``)

Both conditions must hold simultaneously for a reward of 1.0.

``z_index`` / ``gripper_index`` must point into the flattened
``observation.state`` vector. Defaults assume the lerobot-panda 8-dim state.
Override them for gym_hil (state dim 18) or other robots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from lerobot.utils.constants import OBS_STATE

from lerobot.processor.reward_model.base import (
    BaseRewardProcessorStep,
    RewardModelConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class HeightGripperRewardConfig(RewardModelConfig):
    type: str = "height_gripper"
    height_threshold: float = 0.21
    gripper_closed_threshold: float = 0.5
    z_index: int = 2
    gripper_index: int = 7


@dataclass
class HeightGripperRewardStep(BaseRewardProcessorStep):
    """Reward step that checks EE height and gripper closure."""

    config: HeightGripperRewardConfig = field(default_factory=HeightGripperRewardConfig)

    def compute_reward(self, observation: dict[str, Any]) -> float:
        state = observation[OBS_STATE]
        flat = state.flatten()
        z = flat[self.config.z_index].item()
        gripper = flat[self.config.gripper_index].item()

        is_lifted = z >= self.config.height_threshold
        is_closed = gripper < self.config.gripper_closed_threshold
        reward = 1.0 if (is_lifted and is_closed) else 0.0

        logger.debug(
            "height_gripper: z=%.4f (thr=%.4f), gripper=%.4f (thr=%.4f) -> %.1f",
            z,
            self.config.height_threshold,
            gripper,
            self.config.gripper_closed_threshold,
            reward,
        )
        if reward > 0.5:
            logger.info("Success!")
        return reward
