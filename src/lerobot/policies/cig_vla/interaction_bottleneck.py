from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class InteractionGeometryBottleneck:
    translation_goal: Tensor
    approach_direction: Tensor
    translation_magnitude: Tensor
    rotation_goal: Tensor
    gripper_transition: Tensor
    confidence_logit: Tensor
    valid_mask: Tensor

    def normalized(self):
        return self._replace(
            approach_direction=torch.nn.functional.normalize(self.approach_direction, dim=-1)
        )

    def detached(self):
        return self._map(Tensor.detach)

    def clone(self):
        return self._map(torch.clone)

    def as_controller_tensor(self, mode="interaction_tuple"):
        if mode != "interaction_tuple":
            raise ValueError(f"Unsupported bottleneck mode: {mode}")
        return torch.cat(
            [
                self.translation_goal,
                self.approach_direction,
                self.translation_magnitude,
                self.rotation_goal,
                self.gripper_transition,
                self.confidence_logit.sigmoid(),
                self.valid_mask.to(self.translation_goal.dtype),
            ],
            dim=-1,
        )

    def with_translation_offset(self, offset):
        goal = self.translation_goal + offset
        return self._replace(
            translation_goal=goal,
            approach_direction=torch.nn.functional.normalize(goal, dim=-1),
            translation_magnitude=goal.norm(dim=-1, keepdim=True),
        )

    def with_magnitude_scale(self, scale):
        magnitude = self.translation_magnitude * scale
        return self._replace(
            translation_goal=self.approach_direction * magnitude,
            translation_magnitude=magnitude,
        )

    def with_direction(self, direction):
        direction = torch.nn.functional.normalize(direction, dim=-1)
        return self._replace(
            approach_direction=direction,
            translation_goal=direction * self.translation_magnitude,
        )

    def with_gripper_transition(self, transition):
        return self._replace(gripper_transition=transition)

    def removed(self, mask=None):
        if mask is None:
            mask = torch.ones(self.valid_mask.shape[0], dtype=torch.bool, device=self.valid_mask.device)
        mask = mask.reshape(-1, 1)
        return self._replace(valid_mask=torch.where(mask, torch.zeros_like(self.valid_mask), self.valid_mask))

    def _replace(self, **values):
        return type(self)(**(self.__dict__ | values))

    def _map(self, function):
        return type(self)(**{key: function(value) for key, value in self.__dict__.items()})
