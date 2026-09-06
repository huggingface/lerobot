"""Compatibility shim: re-export the new shared action-semantics API for
older SafeDiff consumers while keeping the historical symbols available.

This module intentionally delegates to :mod:`lerobot.action_semantics`.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from lerobot.action_semantics import (
    convert_action,
    LIBERO_ACTION_CONTRACT as _LIBERO_ACTION_CONTRACT,
    LIBERO_SAFETY_ENV_ACTION_CONTRACT as _LIBERO_SAFETY_ACTION_CONTRACT,
)

# New-style shared contracts remain available via action_semantics module.
_NEW_LIBERO_ACTION_CONTRACT = _LIBERO_ACTION_CONTRACT
_NEW_LIBERO_SAFETY_ACTION_CONTRACT = _LIBERO_SAFETY_ACTION_CONTRACT


class LiberoActionContract:
    """Compatibility wrapper implementing the old LiberoActionContract API."""

    def __init__(self, name: str, fps: float, osc_output_scale: tuple[float, ...], action_dim: int = 7, gripper_index: int = 6):
        if fps <= 0:
            raise ValueError("fps must be positive, got %r" % (fps,))
        if action_dim != 7:
            raise ValueError("dimension %d" % (action_dim,))
        if len(osc_output_scale) != 6:
            raise ValueError("osc_output_scale must contain six values")
        self.name = name
        self.fps = fps
        self.osc_output_scale = tuple(float(x) for x in osc_output_scale)
        self.action_dim = action_dim
        self.gripper_index = gripper_index

    def _validate(self, action):
        import torch

        if not isinstance(action, torch.Tensor):
            raise TypeError("action must be a torch.Tensor")
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"dimension {self.action_dim}")

    def command_to_physical_delta(self, action):
        self._validate(action)
        scales = action.new_tensor(self.osc_output_scale)
        out = action.clone()
        out[..., :6] = out[..., :6] * scales
        return out

    def physical_delta_to_command(self, physical_delta):
        self._validate(physical_delta)
        scales = physical_delta.new_tensor(self.osc_output_scale)
        out = physical_delta.clone()
        out[..., :6] = out[..., :6] / scales
        return out

    def command_to_physical_velocity(self, action):
        out = self.command_to_physical_delta(action)
        out[..., :6] = out[..., :6] * self.fps
        return out

    def physical_velocity_to_command(self, physical_velocity):
        self._validate(physical_velocity)
        out = physical_velocity.clone()
        out[..., :6] = out[..., :6] / self.fps
        return self.physical_delta_to_command(out)


# For backward compatibility expose the old LiberoActionContract API expected by
# older SafeDiff tests and callers.
LIBERO_ACTION_CONTRACT = LiberoActionContract("libero", 10, (0.05, 0.05, 0.05, 0.5, 0.5, 0.5))
LIBERO_SAFETY_ACTION_CONTRACT = LiberoActionContract("libero_safety", 20, (2.0, 2.0, 2.0, 2.0, 2.0, 2.0))


def convert_libero_action(
    action: Tensor,
    source: Any,
    target: Any,
    *,
    semantics: str = "per_step",
    clip: bool = False,
) -> Tensor:
    """Deprecated compatibility wrapper around :func:`convert_action`.

    Preserves the historical signature while delegating to the new common
    action-semantics conversion code.
    """
    # Accept either the old LiberoActionContract-like objects or the new
    # ActionContract instances; convert_action expects the new ActionContract.
    from lerobot.action_semantics.contracts import ActionContract

    def _to_new(c):
        if isinstance(c, LiberoActionContract):
            # map legacy fields to the new ActionContract
            return ActionContract(
                name=c.name,
                action_dim=c.action_dim,
                fps=c.fps,
                representation="controller_command",
                controller_translation_scale=(c.osc_output_scale[0], c.osc_output_scale[1], c.osc_output_scale[2]),
                controller_rotation_scale=(c.osc_output_scale[3], c.osc_output_scale[4], c.osc_output_scale[5]),
            )
        return c

    src_new = _to_new(source)
    tgt_new = _to_new(target)
    return convert_action(action, src_new, tgt_new, semantics=semantics, clip=clip)



def compare_dataset_contracts(
    source_info: dict[str, Any],
    target_info: dict[str, Any],
    source_action: LiberoActionContract,
    target_action: LiberoActionContract,
) -> dict[str, Any]:
    """Build a machine-readable shape, time-base and action-semantics report."""

    def shape(info: dict[str, Any], *names: str) -> list[int] | None:
        for name in names:
            if name in info.get("features", {}):
                return info["features"][name].get("shape")
        return None

    def images(info: dict[str, Any]) -> list[str]:
        return sorted(
            name
            for name, feature in info.get("features", {}).items()
            if feature.get("dtype") in {"image", "video"}
        )

    source_images, target_images = images(source_info), images(target_info)
    translation_ratio = target_action.osc_output_scale[0] / source_action.osc_output_scale[0]
    rotation_ratio = target_action.osc_output_scale[3] / source_action.osc_output_scale[3]
    checks = {
        "action_shape_equal": shape(source_info, "action", "actions")
        == shape(target_info, "action", "actions"),
        "state_shape_equal": shape(source_info, "observation.state")
        == shape(target_info, "observation.state"),
        "camera_count_equal": len(source_images) == len(target_images),
        "fps_equal": source_info.get("fps") == target_info.get("fps"),
        "controller_scale_equal": source_action.osc_output_scale == target_action.osc_output_scale,
    }

    def describe(info, contract, image_names):
        return {
            "fps": info.get("fps"),
            "action_shape": shape(info, "action", "actions"),
            "state_shape": shape(info, "observation.state"),
            "image_features": image_names,
            "osc_output_scale": list(contract.osc_output_scale),
        }

    return {
        "compatible_by_shape_only": all(
            checks[key] for key in ("action_shape_equal", "state_shape_equal", "camera_count_equal")
        ),
        "semantically_interchangeable": all(checks.values()),
        "checks": checks,
        "source": describe(source_info, source_action, source_images),
        "target": describe(target_info, target_action, target_images),
        "target_to_source_controller_scale_ratio": {
            "translation": translation_ratio,
            "rotation": rotation_ratio,
        },
        "source_to_target_action_multiplier": {
            "per_step_translation": 1 / translation_ratio,
            "per_step_rotation": 1 / rotation_ratio,
            "velocity_translation": source_action.fps / target_action.fps / translation_ratio,
            "velocity_rotation": source_action.fps / target_action.fps / rotation_ratio,
            "gripper": 1.0,
        },
    }
