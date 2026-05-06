"""Continuous-gripper bridge ProcessorSteps for V6+ HIL-SERL.

Two steps that let SAC operate with a 5-D *continuous* action while the env
still wants {0=CLOSE, 1=STAY, 2=OPEN} as the gripper command.

  - GripperContToIntStep   : final 5th element thresholded to int before env.step
  - GripperIntToContStep   : right after teleop intervention, map int gripper to
                             continuous so the buffer stores 5-D continuous

Mapping demo/teleop class -> continuous: 0->-1, 1->0, 2->+1.
Threshold continuous -> class: <-deadband -> 0(CLOSE),  >+deadband -> 2(OPEN),
otherwise 1(STAY). Default deadband 0.33.
"""

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .core import EnvTransition, TransitionKey
from .pipeline import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("gripper_cont_to_int")
@dataclass
class GripperContToIntStep(ProcessorStep):
    """Threshold a continuous 5th action element to {0,1,2} before env.step."""

    deadband: float = 0.33
    action_dim_index: int = 4  # 5-D action: [dx, dy, dz, dyaw, gripper]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_t = transition.copy()
        action = new_t.get(TransitionKey.ACTION)
        if action is None:
            return new_t
        if not torch.is_tensor(action):
            action = torch.as_tensor(action)
        if action.shape[-1] <= self.action_dim_index:
            return new_t
        cont = action[..., self.action_dim_index]
        cls = torch.full_like(cont, 1.0)  # default STAY
        cls = torch.where(cont < -self.deadband, torch.full_like(cont, 0.0), cls)
        cls = torch.where(cont > self.deadband, torch.full_like(cont, 2.0), cls)
        new_action = action.clone()
        new_action[..., self.action_dim_index] = cls
        new_t[TransitionKey.ACTION] = new_action
        return new_t

    def get_config(self) -> dict[str, Any]:
        return {"deadband": self.deadband, "action_dim_index": self.action_dim_index}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("gripper_int_to_cont")
@dataclass
class GripperIntToContStep(ProcessorStep):
    """Map an integer-class 5th element {0,1,2} to continuous {-1,0,+1}.

    Invoked AFTER intervention so the action stored in TELEOP_ACTION_KEY (and
    eventually the replay buffer) is uniformly 5-D continuous regardless of
    whether the source was the autonomous policy or the teleop gamepad.

    Also overwrites complementary_data[TELEOP_ACTION_KEY] with the converted
    action because InterventionActionProcessorStep stashed the pre-conversion
    teleop action there, and the actor reads that key when building the
    transition that is sent to the learner's replay buffer.
    """

    action_dim_index: int = 4
    teleop_action_key: str = "teleop_action"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_t = transition.copy()
        action = new_t.get(TransitionKey.ACTION)
        if action is None:
            return new_t
        if not torch.is_tensor(action):
            action = torch.as_tensor(action)
        if action.shape[-1] <= self.action_dim_index:
            return new_t
        g = action[..., self.action_dim_index]
        # Heuristic: only remap when value is integer-valued in {0,1,2}.
        is_int_class = (g.round() == g) & (g >= -0.001) & (g <= 2.001)
        if not bool(is_int_class.all().item()):
            return new_t  # already continuous
        mapped = torch.where(g <= 0.5, torch.full_like(g, -1.0), g)  # 0 -> -1
        mapped = torch.where((g > 0.5) & (g <= 1.5), torch.full_like(g, 0.0), mapped)
        mapped = torch.where(g > 1.5, torch.full_like(g, 1.0), mapped)
        new_action = action.clone().float()
        new_action[..., self.action_dim_index] = mapped
        new_t[TransitionKey.ACTION] = new_action
        comp = new_t.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        if self.teleop_action_key in comp:
            new_comp = dict(comp)
            new_comp[self.teleop_action_key] = new_action
            new_t[TransitionKey.COMPLEMENTARY_DATA] = new_comp
        return new_t

    def get_config(self) -> dict[str, Any]:
        return {
            "action_dim_index": self.action_dim_index,
            "teleop_action_key": self.teleop_action_key,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
