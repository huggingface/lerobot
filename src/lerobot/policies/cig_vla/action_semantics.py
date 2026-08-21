import warnings
from typing import Protocol

import torch
from torch import Tensor


class ActionSemanticsAdapter(Protocol):
    def motion_action(self, actions: Tensor) -> Tensor: ...
    def cartesian_translation(self, actions: Tensor) -> Tensor | None: ...
    def aggregate_translation(self, actions: Tensor, prefix_steps: int) -> Tensor | None: ...
    def motion_magnitude(self, actions: Tensor, prefix_steps: int | None = None) -> Tensor: ...
    def denormalize_action(self, actions: Tensor, dataset_stats) -> Tensor | None: ...
    def safe_action(self, state: Tensor, action_dim: int, device, dtype) -> Tensor | None: ...


class LiberoOSCActionSemantics:
    action_dim = 7

    def motion_action(self, actions):
        return actions[..., :6]

    def cartesian_translation(self, actions):
        return actions[..., :3] if actions.shape[-1] == self.action_dim else None

    def aggregate_translation(self, actions, prefix_steps):
        value = self.cartesian_translation(actions)
        return None if value is None else value[:, :prefix_steps].sum(1)

    def motion_magnitude(self, actions, prefix_steps=None):
        motion = self.motion_action(actions)
        if prefix_steps is not None:
            motion = motion[:, :prefix_steps]
        return motion.flatten(1).norm(dim=-1)

    def denormalize_action(self, actions, dataset_stats):
        stats = None if dataset_stats is None else dataset_stats.get("action")
        if not stats or "mean" not in stats or "std" not in stats:
            return None
        mean = torch.as_tensor(stats["mean"], device=actions.device, dtype=actions.dtype)
        std = torch.as_tensor(stats["std"], device=actions.device, dtype=actions.dtype)
        return actions * std + mean

    def safe_action(self, state, action_dim, device, dtype):
        return None


class LiberoSafetyDeltaOSCActionSemantics(LiberoOSCActionSemantics):
    """LIBERO Panda OSC_POSE action: xyz delta, axis-angle delta, gripper command."""

    controller = "OSC_POSE"
    control_mode = "delta"

    def denormalize_actions(self, actions, action_stats):
        if action_stats is None:
            return None
        stats = action_stats.get("actions", action_stats.get("action"))
        if not stats or "mean" not in stats or "std" not in stats:
            return None
        mean = torch.as_tensor(stats["mean"], device=actions.device, dtype=actions.dtype)
        std = torch.as_tensor(stats["std"], device=actions.device, dtype=actions.dtype)
        if mean.numel() != self.action_dim or std.numel() != self.action_dim:
            raise ValueError("LIBERO-Safety action stats must have dimension 7")
        if not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError("LIBERO-Safety action std must be finite and positive")
        return actions * std + mean

    def translation_delta(self, actions):
        if actions.shape[-1] != self.action_dim:
            raise ValueError("LIBERO-Safety actions must have dimension 7")
        return actions[..., :3]

    def rotation_delta(self, actions):
        if actions.shape[-1] != self.action_dim:
            raise ValueError("LIBERO-Safety actions must have dimension 7")
        return actions[..., 3:6]

    def gripper_command(self, actions):
        if actions.shape[-1] != self.action_dim:
            raise ValueError("LIBERO-Safety actions must have dimension 7")
        return actions[..., 6]


LiberoSafetyActionSemantics = LiberoSafetyDeltaOSCActionSemantics


class UnknownActionSemantics:
    def __init__(self):
        warnings.warn("Unknown action semantics: direction and safe-action losses are disabled", stacklevel=2)

    def motion_action(self, actions):
        return actions

    def cartesian_translation(self, actions):
        return None

    def aggregate_translation(self, actions, prefix_steps):
        return None

    def motion_magnitude(self, actions, prefix_steps=None):
        if prefix_steps is not None:
            actions = actions[:, :prefix_steps]
        return actions.flatten(1).norm(dim=-1)

    def denormalize_action(self, actions, dataset_stats):
        return None

    def safe_action(self, state, action_dim, device, dtype):
        return None


def make_action_semantics(name: str):
    return (
        LiberoSafetyDeltaOSCActionSemantics()
        if name in {"libero_osc", "libero_safety"}
        else UnknownActionSemantics()
    )
