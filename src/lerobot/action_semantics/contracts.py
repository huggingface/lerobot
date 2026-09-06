from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

ArrayLike = object


Representation = Literal["controller_command", "physical_delta", "physical_velocity"]
RotationRep = Literal["rotvec", "axis_angle", "euler", "quaternion", "controller_rotvec"]
GripperSem = Literal[
    "identity",
    "minus1_plus1",
    "zero_one",
    "binary_open_close",
    "continuous",
]


@dataclass(frozen=True)
class ActionContract:
    name: str
    action_dim: int = 7
    fps: Optional[float] = None
    representation: Representation = "controller_command"

    translation_dim: Tuple[int, int] = (0, 3)
    rotation_dim: Tuple[int, int] = (3, 6)
    gripper_index: int = 6

    translation_unit: Literal["normalized", "m", "m/s"] = "normalized"
    rotation_unit: Literal["normalized", "rad", "rad/s"] = "normalized"
    rotation_representation: RotationRep = "rotvec"

    controller_translation_scale: Optional[Tuple[float, float, float]] = None
    controller_rotation_scale: Optional[Tuple[float, float, float]] = None

    gripper_semantics: GripperSem = "continuous"

    def __post_init__(self) -> None:
        if self.action_dim != 7:
            raise ValueError("Only 7-D action contracts are supported in this refactor")


# Predefined LIBERO contracts per spec. These are the canonical, shared
# registrations that other modules should import from `action_semantics`.

# Old-style controller-like LIBERO contract (used historically as the 'libero' contract)
LIBERO_ACTION_CONTRACT = ActionContract(
    name="libero",
    action_dim=7,
    fps=10.0,
    representation="controller_command",
    translation_unit="normalized",
    rotation_unit="normalized",
    controller_translation_scale=(0.05, 0.05, 0.05),
    controller_rotation_scale=(0.5, 0.5, 0.5),
    gripper_semantics="continuous",
)

# Env-level controller contract for LIBERO (explicit separation)
LIBERO_ENV_ACTION_CONTRACT = ActionContract(
    name="libero_env",
    action_dim=7,
    fps=10.0,
    representation="controller_command",
    controller_translation_scale=(0.05, 0.05, 0.05),
    controller_rotation_scale=(0.5, 0.5, 0.5),
)

# Dataset-level contract for LIBERO dataset (kept distinct)
LIBERO_DATASET_ACTION_CONTRACT = ActionContract(
    name="libero_dataset",
    action_dim=7,
    fps=10.0,
    representation="controller_command",
    controller_translation_scale=(0.05, 0.05, 0.05),
    controller_rotation_scale=(0.5, 0.5, 0.5),
)

# LIBERO-Safety dataset encodes per-step physical deltas
LIBERO_SAFETY_DATASET_ACTION_CONTRACT = ActionContract(
    name="libero_safety_dataset",
    action_dim=7,
    fps=20.0,
    representation="physical_delta",
    translation_unit="m",
    rotation_unit="rad",
    controller_translation_scale=None,
    controller_rotation_scale=None,
)

# LIBERO-Safety environment controller command contract
LIBERO_SAFETY_ENV_ACTION_CONTRACT = ActionContract(
    name="libero_safety_env",
    action_dim=7,
    fps=20.0,
    representation="controller_command",
    controller_translation_scale=(2.0, 2.0, 2.0),
    controller_rotation_scale=(2.0, 2.0, 2.0),
)
