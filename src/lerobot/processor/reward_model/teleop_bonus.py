"""Teleop-event-driven reward processor (no SARM, no CNN).

Reward fired purely from gamepad button events surfaced in
``info[TeleopEvents.*]``:

  * ``STAGE_ADVANCE``  → reward += ``stage_advance_bonus`` (default +1/6)
  * ``SUCCESS``        → reward += ``success_bonus``       (default +1/6)
  * Episode failure    → reward += ``failure_penalty``     (default -1/6)
    (= ``TERMINATE_EPISODE`` true AND ``SUCCESS`` false AND
    ``RERECORD_EPISODE`` false)

A full successful rollout with all 6 stage advances + success press sums
to +1.0. A failed rollout earns a one-stage-equivalent penalty.

Episode termination follows the same rule as
``InterventionActionProcessorStep``: ``done = terminate_episode OR
(terminate_on_success AND success)``. This step OVERWRITES any reward
the prior step wrote (e.g. ``float(success)`` from the intervention
step), so it must run last in the env pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.reward_model.base import RewardModelConfig
from lerobot.teleoperators.utils import TeleopEvents


@dataclass
class TeleopBonusRewardConfig(RewardModelConfig):
    type: str = "teleop_bonus"
    step_penalty: float = 0.0
    stage_advance_bonus: float = 0.16667
    success_terminal_bonus: float = 0.16667
    failure_terminal_penalty: float = -0.16667


@dataclass
@ProcessorStepRegistry.register("teleop_bonus_reward_processor")
class TeleopBonusRewardStep(ProcessorStep):
    config: TeleopBonusRewardConfig = field(default_factory=TeleopBonusRewardConfig)
    terminate_on_success: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        info = transition.get(TransitionKey.INFO, {}) or {}
        stage_advance = bool(info.get(TeleopEvents.STAGE_ADVANCE, False))
        success = bool(info.get(TeleopEvents.SUCCESS, False))
        terminate = bool(info.get(TeleopEvents.TERMINATE_EPISODE, False))
        rerecord = bool(info.get(TeleopEvents.RERECORD_EPISODE, False))
        failure = terminate and (not success) and (not rerecord)

        reward = float(self.config.step_penalty)
        if stage_advance:
            reward += float(self.config.stage_advance_bonus)
        if success:
            reward += float(self.config.success_terminal_bonus)
        if failure:
            reward += float(self.config.failure_terminal_penalty)

        new_transition = transition.copy()
        new_transition[TransitionKey.REWARD] = reward
        new_transition[TransitionKey.DONE] = terminate or (self.terminate_on_success and success)
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
