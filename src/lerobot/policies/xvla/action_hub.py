# ------------------------------------------------------------------------------
# Copyright 2025 2toINF and HuggingFace Inc. (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

# =============================================================================
# Registry
# =============================================================================
ACTION_REGISTRY: dict[str, type[BaseActionSpace]] = {}


def register_action(name: str):
    """Decorator for registering a new action space."""

    def _wrap(cls):
        key = name.lower()
        if key in ACTION_REGISTRY:
            raise KeyError(f"ActionSpace '{key}' already registered -> {ACTION_REGISTRY[key]}")
        ACTION_REGISTRY[key] = cls
        cls.name = key
        return cls

    return _wrap


def build_action_space(name: str, **kwargs) -> BaseActionSpace:
    """Instantiate a registered action space by name."""
    key = name.lower()
    if key not in ACTION_REGISTRY:
        raise KeyError(f"Unknown action space '{name}'. Available: {list(ACTION_REGISTRY.keys())}")
    return ACTION_REGISTRY[key](**kwargs)


# =============================================================================
# Base class
# =============================================================================
class BaseActionSpace(nn.Module):
    """
    Abstract base class for all action-space definitions.

    Each subclass defines:
      - `dim_action`: dimension of the action vector.
      - `gripper_idx`: indices of gripper channels.
      - `compute_loss(pred, target)`: supervised loss for this space.
      - `preprocess(proprio, action, mode)`: pre-step modifications.
      - `postprocess(action)`: post-step corrections (e.g. apply sigmoid).
    """

    name: str = "base"
    dim_action: int = 0
    gripper_idx: tuple[int, ...] = ()

    def __init__(self):
        super().__init__()

    # ---------------------------------------------------------------------
    # Core supervised loss
    # ---------------------------------------------------------------------
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Alias for compute_loss."""
        return self.compute_loss(pred, target)

    # ---------------------------------------------------------------------
    # Space-level hooks
    # ---------------------------------------------------------------------
    def preprocess(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
        mode: str = "train",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Default: return unchanged."""
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Default: return unchanged."""
        return action


# =============================================================================
# Utilities
# =============================================================================
def _ensure_indices_valid(dim_action: int, idx: Iterable[int], name: str) -> None:
    bad = [i for i in idx if i < 0 or i >= dim_action]
    if bad:
        raise IndexError(f"{name} contains out-of-range indices {bad} for action dim dim_action={dim_action}")


# =============================================================================
# Implementations
# =============================================================================
@register_action("ee6d")
class EE6DActionSpace(BaseActionSpace):
    """End-effector layout with xyz, 6D rotation, and gripper channels."""

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 1.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0

    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape, "pred/target shapes must match"
        batch_size, seq_len, action_dim = pred.shape
        _ensure_indices_valid(action_dim, self.gripper_idx, "gripper_idx")

        # Gripper BCE
        g_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = sum(g_losses) / len(self.gripper_idx) * self.GRIPPER_SCALE

        # XYZ position
        pos_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1])
            + self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE

        # Rotation 6D
        rot_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1])
            + self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        return {
            "position_loss": pos_loss,
            "rotate6D_loss": rot_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """Zero-out gripper channels in proprio/action."""
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid to gripper logits."""
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action


@register_action("joint")
class JointActionSpace(BaseActionSpace):
    """Joint-space layout with joints + gripper only."""

    dim_action = 14
    gripper_idx = (6, 13)
    GRIPPER_SCALE = 0.1
    JOINTS_SCALE = 1.0

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape
        batch_size, seq_len, action_dim = pred.shape
        _ensure_indices_valid(action_dim, self.gripper_idx, "gripper_idx")

        g_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = sum(g_losses) / len(self.gripper_idx) * self.GRIPPER_SCALE

        joints_idx = tuple(i for i in range(action_dim) if i not in set(self.gripper_idx))
        joints_loss = self.mse(pred[:, :, joints_idx], target[:, :, joints_idx]) * self.JOINTS_SCALE

        return {
            "joints_loss": joints_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """Zero-out gripper channels in proprio/action."""
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid to gripper logits."""
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action


@register_action("agibot_ee6d")
class AGIBOTEE6DActionSpace(BaseActionSpace):
    """AGI-bot variant of EE6DActionSpace using MSE for all components."""

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 10.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0
    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape
        batch_size, seq_len, action_dim = pred.shape
        _ensure_indices_valid(action_dim, self.gripper_idx, "gripper_idx")

        gripper_loss = (
            self.mse(pred[:, :, self.gripper_idx], target[:, :, self.gripper_idx]) * self.GRIPPER_SCALE
        )
        pos_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1])
            + self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE
        rot_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1])
            + self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        return {
            "position_loss": pos_loss,
            "rotate6D_loss": rot_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """No preprocessing applied in AGIBOT variant."""
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """AGIBOT does not postprocess."""
        return action


@register_action("franka_joint7")
class FrankaJoint7ActionSpace(BaseActionSpace):
    """
    Franka Panda joint-space: 7 joints, with gripper.

    - Real robot action dim: 7
    - Model-facing dim: 20 (padded with zeros)
      compatible with pretrained VLA models expecting 20D.
    """

    dim_action = 20  # model dimension
    REAL_DIM = 7  # actual Franka joints

    JOINTS_SCALE = 1.0

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Pad 7 → 20 dims (zeros for the dummy channels)."""
        if x is None:
            return None
        if x.size(-1) == self.dim_action:
            return x
        if x.size(-1) != self.REAL_DIM:
            raise ValueError(
                f"Expected last dim to be {self.REAL_DIM} or {self.dim_action}, got {x.size(-1)}"
            )

        pad_shape = list(x.shape[:-1]) + [self.dim_action - self.REAL_DIM]  # 13 zeros
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=-1)

    def _trim_to_real_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Trim model output 20 → 7 dims."""
        return x[..., : self.REAL_DIM]

    def compute_loss(self, pred, target):
        """
        pred :  [B, T, 20]
        target : [B, T, 7] or [B, T, 20]

        Only compute MSE on the first 7 dims.
        """
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)

        assert pred.shape == target.shape

        joints_loss = (
            self.mse(
                pred[:, :, : self.REAL_DIM],  # use only the first 7 joints
                target[:, :, : self.REAL_DIM],
            )
            * self.JOINTS_SCALE
        )

        return {"joints_loss": joints_loss}

    def preprocess(self, proprio, action, mode="train"):
        """
        During training:
        - Pad [7] → [20]
        """
        return proprio, self._pad_to_model_dim(action)

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """
        After model prediction:
        - Trim [20] → [7] for real robot control.
        """
        return self._trim_to_real_dim(action)


@register_action("auto")
class AutoActionSpace(BaseActionSpace):
    """
    Auto-detecting action space that adapts to any action dimension.

    - Auto-detects the real action dimension from the policy feature
    - Model outputs max_dim for compatibility with pretrained models
    - Loss is computed only on the first real_dim dimensions
    - Postprocess trims output back to real_dim

    Args:
        real_dim: The actual action dimension from the dataset/policy feature
        max_dim: The model's output dimension for pretrained VLA compatibility
    """

    JOINTS_SCALE = 1.0

    def __init__(self, real_dim: int, max_dim: int):
        super().__init__()
        self.real_dim = real_dim
        self.dim_action = max_dim  # Model-facing dimension
        self.mse = nn.MSELoss()

    def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Pad real_dim → max_dim (zeros for the dummy channels)."""
        if x is None:
            return None
        if x.size(-1) == self.dim_action:
            return x
        if x.size(-1) != self.real_dim:
            # If dimension doesn't match either, pad/trim to real_dim first
            if x.size(-1) < self.real_dim:
                pad_shape = list(x.shape[:-1]) + [self.real_dim - x.size(-1)]
                pad = x.new_zeros(pad_shape)
                x = torch.cat([x, pad], dim=-1)
            else:
                x = x[..., : self.real_dim]

        pad_shape = list(x.shape[:-1]) + [self.dim_action - self.real_dim]
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=-1)

    def _trim_to_real_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Trim model output max_dim → real_dim."""
        return x[..., : self.real_dim]

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute loss only on the first real_dim dimensions.

        pred:   [B, T, max_dim] from the model
        target: [B, T, real_dim] or [B, T, max_dim]

        Loss = MSE(pred[:,:,:real_dim], target[:,:,:real_dim])
        """
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)
        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"

        # only compute loss on the real dimensions
        joints_loss = (
            self.mse(
                pred[:, :, : self.real_dim],
                target[:, :, : self.real_dim],
            )
            * self.JOINTS_SCALE
        )

        return {"joints_loss": joints_loss}

    def preprocess(self, proprio: torch.Tensor, action: torch.Tensor, mode: str = "train"):
        """
        Pad action from real_dim to max_dim for the model.
        """
        return proprio, self._pad_to_model_dim(action)

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """
        Trim model output from max_dim to real_dim for real robot control.
        """
        return self._trim_to_real_dim(action)


@register_action("so101_bimanual")
class BimanualSO101ActionSpace(BaseActionSpace):
    """
    Bimanual SO101 robot: 2 arms with 5 joints each + gripper.

    Layout (real robot):
    [left_arm (5 joints + gripper), right_arm (5 joints + gripper)]
    - Left arm:  shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    - Right arm: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

    Real action dim: 12
    Model-facing dim: 20 (extra 8 dummy dims at the end)
    """

    # Model output / training dimension (to match pretrained policy)
    dim_action = 20

    # Real robot action dimension
    REAL_DIM = 12

    # Indices of real vs dummy channels
    REAL_IDXS = tuple(range(REAL_DIM))  # 0..11
    DUMMY_IDXS = tuple(range(REAL_DIM, dim_action))  # 12..19

    # Grippers live in the real part
    gripper_idx = (5, 11)  # left_gripper at idx 5, right_gripper at idx 11
    GRIPPER_SCALE = 1.0
    JOINTS_SCALE = 1.0

    # Indices for left and right arm joints (excluding grippers)
    LEFT_ARM_JOINTS = (0, 1, 2, 3, 4)
    RIGHT_ARM_JOINTS = (6, 7, 8, 9, 10)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    # ---------- helpers ----------

    def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
        """If last dim is REAL_DIM (12), pad zeros to reach dim_action (20)."""
        if x is None:
            return None
        if x.size(-1) == self.dim_action:
            return x
        if x.size(-1) != self.REAL_DIM:
            raise ValueError(
                f"Expected last dim to be {self.REAL_DIM} or {self.dim_action}, got {x.size(-1)}"
            )
        pad_shape = list(x.shape[:-1]) + [self.dim_action - self.REAL_DIM]
        pad = x.new_zeros(pad_shape)
        return torch.cat([x, pad], dim=-1)

    def _trim_to_real_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Keep only the first REAL_DIM (12) dims for the real robot."""
        return x[..., : self.REAL_DIM]

    # ---------- loss ----------

    def compute_loss(self, pred, target):
        """
        pred:  [B, T, 20] from the model
        target: [B, T, 12] or [B, T, 20]
        We pad target → 20 and compute loss only on the real dims.
        """
        # Ensure both are [B, T, 20]
        pred = self._pad_to_model_dim(pred)
        target = self._pad_to_model_dim(target)
        assert pred.shape == target.shape

        # ---- MSE for all real dims (0–11) ----
        real_dims = 12

        joints_loss = (
            self.mse(
                pred[:, :, :real_dims],
                target[:, :, :real_dims],
            )
            * self.JOINTS_SCALE
        )

        left_arm_loss = self.mse(pred[:, :, :6], target[:, :, :6])
        right_arm_loss = self.mse(pred[:, :, 6:12], target[:, :, 6:12])

        gripper_loss = (
            self.mse(
                pred[:, :, [5, 11]],
                target[:, :, [5, 11]],
            )
            * self.GRIPPER_SCALE
        )

        return {
            "joints_loss": joints_loss,
            "gripper_loss": gripper_loss,
            "left_arm_loss": left_arm_loss,
            "right_arm_loss": right_arm_loss,
        }

    # ---------- preprocess / postprocess ----------

    def preprocess(self, proprio, action, mode="train"):
        """
        - If proprio/action are 12-dim, pad them to 20 for the model.
        - Zero-out gripper channels in proprio/action to focus learning on joints.
        """
        proprio_m = self._pad_to_model_dim(proprio.clone())
        action_m = self._pad_to_model_dim(action.clone()) if action is not None else None

        proprio_m[..., self.gripper_idx] = 0.0
        if action_m is not None:
            action_m[..., self.gripper_idx] = 0.0

        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """
        - Model outputs [*, 20]
        - Apply sigmoid to gripper logits
        - Return only the first 12 dims for the real robot:
          ["left_shoulder_pan.pos",
           "left_shoulder_lift.pos",
           "left_elbow_flex.pos",
           "left_wrist_flex.pos",
           "left_wrist_roll.pos",
           "left_gripper.pos",
           "right_shoulder_pan.pos",
           "right_shoulder_lift.pos",
           "right_elbow_flex.pos",
           "right_wrist_flex.pos",
           "right_wrist_roll.pos",
           "right_gripper.pos"]
        """
        # Ensure we at least have the real dims + grippers
        if action.size(-1) < self.REAL_DIM:
            raise ValueError(f"Expected at least {self.REAL_DIM} dims in action, got {action.size(-1)}")

        # Apply sigmoid on gripper channels in model space (indices 5 and 11)
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])

        # Return only the real 12-dim control vector for the env
        return self._trim_to_real_dim(action)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "BaseActionSpace",
    "build_action_space",
    "register_action",
    "EE6DActionSpace",
    "JointActionSpace",
    "AGIBOTEE6DActionSpace",
    "FrankaJoint7ActionSpace",
    "AutoActionSpace",
    "BimanualSO101ActionSpace",
    "ACTION_REGISTRY",
]
