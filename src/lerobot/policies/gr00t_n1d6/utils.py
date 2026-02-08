#!/usr/bin/env python

# Copyright 2025 Nvidia and The HuggingFace Inc. team. All rights reserved.
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

"""
Utility functions for Groot N1.6 policy.

This module provides helper functions for the Groot N1.6 implementation,
ported from gr00t-orig/data/utils.py and gr00t-orig/model/gr00t_n1d6/image_augmentations.py.

Core functions:
- Normalization: normalize_values_minmax, unnormalize_values_minmax, etc.
- Encoding: apply_sin_cos_encoding
- Serialization: to_json_serializable, nested_dict_to_numpy
- Config parsing: parse_modality_configs
- Image augmentations: build_image_transformations, apply_with_replay
"""

import warnings
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms

# Try to import albumentations, but make it optional
try:
    import albumentations as A  # noqa: N812

    ALBUMENTATIONS_AVAILABLE = True
    DualTransformBase = A.DualTransform
except ImportError:
    A = None
    ALBUMENTATIONS_AVAILABLE = False
    # Create a dummy base class for when albumentations is not available
    DualTransformBase = object


# =============================================================================
# Data Types (ported from gr00t-orig/data/types.py)
# =============================================================================


class ActionRepresentation(Enum):
    RELATIVE = "relative"
    DELTA = "delta"
    ABSOLUTE = "absolute"


class ActionType(Enum):
    EEF = "eef"
    NON_EEF = "non_eef"


class ActionFormat(Enum):
    DEFAULT = "default"
    XYZ_ROT6D = "xyz+rot6d"
    XYZ_ROTVEC = "xyz+rotvec"


class EmbodimentTag(Enum):
    """Embodiment tags are used to identify the robot embodiment in the data."""

    # Pretrain embodiment tags
    ROBOCASA_PANDA_OMRON = "robocasa_panda_omron"
    GR1 = "gr1"
    BEHAVIOR_R1_PRO = "behavior_r1_pro"

    # Pre-registered posttrain embodiment tags
    UNITREE_G1 = "unitree_g1"
    LIBERO_PANDA = "libero_panda"
    OXE_GOOGLE = "oxe_google"
    OXE_WIDOWX = "oxe_widowx"

    # New embodiment during post-training
    NEW_EMBODIMENT = "new_embodiment"


# =============================================================================
# Config Types (ported from gr00t-orig/data/types.py and embodiment_configs.py)
# =============================================================================


class ActionConfig:
    """Configuration for an action modality."""

    def __init__(
        self,
        rep: ActionRepresentation | str,
        type: ActionType | str,
        format: ActionFormat | str,
        state_key: str | None = None,
    ):
        if isinstance(rep, str):
            rep = ActionRepresentation[rep]
        if isinstance(type, str):
            type = ActionType[type]
        if isinstance(format, str):
            format = ActionFormat[format]

        self.rep = rep
        self.type = type
        self.format = format
        self.state_key = state_key


class ModalityConfig:
    """Configuration for a modality defining how data should be sampled and loaded."""

    def __init__(
        self,
        delta_indices: list[int],
        modality_keys: list[str],
        sin_cos_embedding_keys: list[str] | None = None,
        mean_std_embedding_keys: list[str] | None = None,
        action_configs: list[ActionConfig | dict] | None = None,
    ):
        self.delta_indices = delta_indices
        self.modality_keys = modality_keys
        self.sin_cos_embedding_keys = sin_cos_embedding_keys
        self.mean_std_embedding_keys = mean_std_embedding_keys

        # Parse action configs
        if action_configs is not None:
            assert len(action_configs) == len(modality_keys), (
                f"Number of action configs ({len(action_configs)}) must match "
                f"number of modality keys ({len(modality_keys)})"
            )
            parsed_action_configs = []
            for action_config in action_configs:
                if isinstance(action_config, dict):
                    action_config = ActionConfig(
                        rep=ActionRepresentation[action_config["rep"]],
                        type=ActionType[action_config["type"]],
                        format=ActionFormat[action_config["format"]],
                        state_key=action_config.get("state_key", None),
                    )
                parsed_action_configs.append(action_config)
            self.action_configs = parsed_action_configs
        else:
            self.action_configs = None


# =============================================================================
# Normalization Functions (ported from gr00t-orig/data/utils.py)
# =============================================================================


def apply_sin_cos_encoding(values: np.ndarray) -> np.ndarray:
    """Apply sin/cos encoding to values.

    Args:
        values: Array of shape (..., D) containing values to encode

    Returns:
        Array of shape (..., 2*D) with [sin, cos] concatenated

    Note: This DOUBLES the dimension. For example:
        Input:  [v₁, v₂, v₃] with shape (..., 3)
        Output: [sin(v₁), sin(v₂), sin(v₃), cos(v₁), cos(v₂), cos(v₃)] with shape (..., 6)
    """
    sin_values = np.sin(values)
    cos_values = np.cos(values)
    return np.concatenate([sin_values, cos_values], axis=-1)


def nested_dict_to_numpy(data):
    """
    Recursively converts bottom-level list of lists to NumPy arrays.

    Args:
        data: A nested dictionary where bottom nodes are list of lists,
              and parent nodes are strings (keys)

    Returns:
        The same dictionary structure with bottom-level lists converted to NumPy arrays
    """
    if isinstance(data, dict):
        return {key: nested_dict_to_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return np.array(data)
    else:
        return data


def normalize_values_minmax(values: np.ndarray, params: dict) -> np.ndarray:
    """
    Normalize values using min-max normalization to [-1, 1] range.

    Args:
        values: Input values to normalize, shape (T, D) or (B, T, D)
        params: Dictionary with "min" and "max" keys

    Returns:
        Normalized values in [-1, 1] range
    """
    min_vals = params["min"]
    max_vals = params["max"]
    normalized = np.zeros_like(values)

    mask = ~np.isclose(max_vals, min_vals)

    normalized[..., mask] = (values[..., mask] - min_vals[..., mask]) / (
        max_vals[..., mask] - min_vals[..., mask]
    )
    normalized[..., mask] = 2 * normalized[..., mask] - 1

    return normalized


def unnormalize_values_minmax(normalized_values: np.ndarray, params: dict) -> np.ndarray:
    """
    Min-max unnormalization from [-1, 1] range back to original range.

    Args:
        normalized_values: Normalized input values in [-1, 1] range
        params: Dictionary with "min" and "max" keys

    Returns:
        Unnormalized values in original range [min, max]
    """
    min_vals = params["min"]
    max_vals = params["max"]
    range_vals = max_vals - min_vals

    unnormalized = (np.clip(normalized_values, -1.0, 1.0) + 1.0) / 2.0 * range_vals + min_vals
    return unnormalized


def normalize_values_meanstd(values: np.ndarray, params: dict) -> np.ndarray:
    """
    Normalize values using mean-std (z-score) normalization.

    Args:
        values: Input values to normalize, shape (T, D) or (B, T, D)
        params: Dictionary with "mean" and "std" keys

    Returns:
        Normalized values using z-score normalization
    """
    mean_vals = params["mean"]
    std_vals = params["std"]

    mask = std_vals != 0
    normalized = np.zeros_like(values)

    normalized[..., mask] = (values[..., mask] - mean_vals[..., mask]) / std_vals[..., mask]
    normalized[..., ~mask] = values[..., ~mask]

    return normalized


def unnormalize_values_meanstd(normalized_values: np.ndarray, params: dict) -> np.ndarray:
    """
    Mean-std unnormalization (reverse z-score normalization).

    Args:
        normalized_values: Normalized input values (z-scores)
        params: Dictionary with "mean" and "std" keys

    Returns:
        Unnormalized values in original scale
    """
    mean_vals = params["mean"]
    std_vals = params["std"]

    mask = std_vals != 0
    unnormalized = np.zeros_like(normalized_values)

    unnormalized[..., mask] = normalized_values[..., mask] * std_vals[..., mask] + mean_vals[..., mask]
    unnormalized[..., ~mask] = normalized_values[..., ~mask]

    return unnormalized


# =============================================================================
# Serialization Functions (ported from gr00t-orig/data/utils.py)
# =============================================================================


def to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert dataclasses and numpy arrays to JSON-serializable format.

    Args:
        obj: Object to convert (can be dataclass, numpy array, dict, list, etc.)

    Returns:
        JSON-serializable representation of the object
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_json_serializable(asdict(obj))
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, Enum):
        return obj.name
    elif hasattr(obj, "__dict__"):
        # Handle custom classes like ModalityConfig, ActionConfig
        return to_json_serializable(vars(obj))
    else:
        return str(obj)


def parse_modality_configs(
    modality_configs: dict[str, dict[str, ModalityConfig | dict]],
) -> dict[str, dict[str, ModalityConfig]]:
    """Parse modality configs from dict or ModalityConfig objects."""
    parsed_modality_configs = {}
    for embodiment_tag, modality_config in modality_configs.items():
        parsed_modality_configs[embodiment_tag] = {}
        for modality, config in modality_config.items():
            if isinstance(config, dict):
                parsed_modality_configs[embodiment_tag][modality] = ModalityConfig(**config)
            else:
                parsed_modality_configs[embodiment_tag][modality] = config
    return parsed_modality_configs


# =============================================================================
# Image Augmentation Functions (ported from gr00t-orig/model/gr00t_n1d6/image_augmentations.py)
# =============================================================================


def apply_with_replay(transform, images, replay=None):
    """
    Apply albumentations transforms to multiple images with replay functionality.

    Args:
        transform: Albumentations ReplayCompose or Compose transform
        images: List of PIL Images to transform
        replay: Optional replay data for consistent transforms. If None, creates new replay.

    Returns:
        tuple: (transformed_tensors_list, replay_data)
            - transformed_tensors_list: List of transformed torch tensors (C, H, W) as uint8
            - replay_data: Replay data for consistent transforms across images (None for regular Compose)
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations is required for apply_with_replay")

    transformed_tensors = []
    current_replay = replay

    # Check if transform supports replay (ReplayCompose)
    has_replay = hasattr(transform, "replay")

    for img in images:
        if has_replay:
            if current_replay is None:
                # First image - create replay data
                augmented_image = transform(image=np.array(img))
                current_replay = augmented_image["replay"]
            else:
                # Subsequent images - use replay for consistent transforms
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    augmented_image = transform.replay(
                        image=np.array(img), saved_augmentations=current_replay
                    )
            img_array = augmented_image["image"]
        else:
            # Regular Compose transform - no replay functionality
            augmented_image = transform(image=np.array(img))
            img_array = augmented_image["image"]

        # Convert to uint8 if needed
        if img_array.dtype == np.float32:
            img_array = (img_array * 255).astype(np.uint8)
        elif img_array.dtype != np.uint8:
            raise ValueError(f"Unexpected data type: {img_array.dtype}")

        # Convert to torch tensor (C, H, W) as uint8
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        transformed_tensors.append(img_tensor)

    return transformed_tensors, current_replay


class FractionalRandomCrop(DualTransformBase):
    """Crop a random part of the input based on fractions while maintaining aspect ratio.

    Args:
        crop_fraction: Fraction of the image to crop (0.0 to 1.0). The crop will maintain
                      the original aspect ratio and be this fraction of the original area.
        p: probability of applying the transform. Default: 1.0

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, crop_fraction: float = 0.9, p: float = 1.0, always_apply: bool | None = None):
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("albumentations is required for FractionalRandomCrop")
        super().__init__(p=p, always_apply=always_apply)
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be between 0.0 and 1.0")
        self.crop_fraction = crop_fraction

    def apply(self, img: np.ndarray, crop_coords: tuple[int, int, int, int], **params) -> np.ndarray:
        x_min, y_min, x_max, y_max = crop_coords
        return img[y_min:y_max, x_min:x_max]

    def apply_to_bboxes(
        self, bboxes: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        return A.augmentations.crops.functional.crop_bboxes_by_coords(bboxes, crop_coords, params["shape"])

    def apply_to_keypoints(
        self, keypoints: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        return A.augmentations.crops.functional.crop_keypoints_by_coords(keypoints, crop_coords)

    def get_params_dependent_on_data(self, params, data) -> dict[str, tuple[int, int, int, int]]:
        image_shape = params["shape"][:2]
        height, width = image_shape

        # Calculate crop dimensions with linear scaling
        crop_height = int(height * self.crop_fraction)
        crop_width = int(width * self.crop_fraction)

        # Ensure minimum size of 1x1
        crop_height = max(1, crop_height)
        crop_width = max(1, crop_width)
        # Random position for crop
        max_y = height - crop_height
        max_x = width - crop_width

        y_min = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        x_min = np.random.randint(0, max_x + 1) if max_x > 0 else 0

        crop_coords = (x_min, y_min, x_min + crop_width, y_min + crop_height)
        return {"crop_coords": crop_coords}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("crop_fraction",)


class FractionalCenterCrop(DualTransformBase):
    """Crop the center part of the input based on fractions while maintaining aspect ratio.

    Args:
        crop_fraction: Fraction of the image to crop (0.0 to 1.0). The crop will maintain
                      the original aspect ratio and be this fraction of the original area.
        p: probability of applying the transform. Default: 1.0

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, crop_fraction: float = 0.9, p: float = 1.0, always_apply: bool | None = None):
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("albumentations is required for FractionalCenterCrop")
        super().__init__(p=p, always_apply=always_apply)
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be between 0.0 and 1.0")
        self.crop_fraction = crop_fraction

    def apply(self, img: np.ndarray, crop_coords: tuple[int, int, int, int], **params) -> np.ndarray:
        x_min, y_min, x_max, y_max = crop_coords
        return img[y_min:y_max, x_min:x_max]

    def apply_to_bboxes(
        self, bboxes: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        return A.augmentations.crops.functional.crop_bboxes_by_coords(bboxes, crop_coords, params["shape"])

    def apply_to_keypoints(
        self, keypoints: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        return A.augmentations.crops.functional.crop_keypoints_by_coords(keypoints, crop_coords)

    def get_params_dependent_on_data(self, params, data) -> dict[str, tuple[int, int, int, int]]:
        image_shape = params["shape"][:2]
        height, width = image_shape

        # Calculate crop dimensions with linear scaling
        crop_height = int(height * self.crop_fraction)
        crop_width = int(width * self.crop_fraction)

        # Ensure minimum size of 1x1
        crop_height = max(1, crop_height)
        crop_width = max(1, crop_width)

        # Center the crop
        y_min = (height - crop_height) // 2
        x_min = (width - crop_width) // 2

        crop_coords = (x_min, y_min, x_min + crop_width, y_min + crop_height)
        return {"crop_coords": crop_coords}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("crop_fraction",)


def build_image_transformations_albumentations(
    image_target_size,
    image_crop_size,
    random_rotation_angle,
    color_jitter_params,
    shortest_image_edge,
    crop_fraction,
):
    """
    Build albumentations-based image transformations.

    Args:
        image_target_size: Target size for resizing (list of [height, width])
        image_crop_size: Size for cropping (list of [height, width])
        random_rotation_angle: Maximum rotation angle in degrees (0 for no rotation)
        color_jitter_params: Dictionary with color jitter parameters
        shortest_image_edge: Shortest edge to resize to
        crop_fraction: Fraction of image to keep when cropping

    Returns:
        tuple: (train_transform, eval_transform)
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations is required for build_image_transformations_albumentations")

    fraction_to_use = image_crop_size[0] / image_target_size[0] if crop_fraction is None else crop_fraction

    max_size = image_target_size[0] if shortest_image_edge is None else shortest_image_edge

    # Training transforms (using ReplayCompose for consistent augmentation across views)
    train_transform_list = [
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
        FractionalRandomCrop(crop_fraction=fraction_to_use),
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
    ]

    if random_rotation_angle is not None and random_rotation_angle != 0:
        train_transform_list.append(A.Rotate(limit=random_rotation_angle, p=1.0))

    if color_jitter_params is not None:
        train_transform_list.append(
            A.ColorJitter(
                brightness=color_jitter_params.get("brightness", 0.0),
                contrast=color_jitter_params.get("contrast", 0.0),
                saturation=color_jitter_params.get("saturation", 0.0),
                hue=color_jitter_params.get("hue", 0.0),
                p=1.0,
            )
        )

    train_transform = A.ReplayCompose(train_transform_list, p=1.0)

    # Evaluation transforms (deterministic)
    eval_transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
            FractionalCenterCrop(crop_fraction=fraction_to_use),
            A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
        ]
    )

    return train_transform, eval_transform


class LetterBoxTransform:
    """Custom transform to pad non-square images to square by adding black bars."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        *leading_dims, c, h, w = img.shape

        if h == w:
            return img

        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if leading_dims:
            batch_size = torch.tensor(leading_dims).prod().item()
            img_reshaped = img.reshape(batch_size, c, h, w)
            padded_img = transforms.functional.pad(
                img_reshaped, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0
            )
            output_shape = leading_dims + [c, max_dim, max_dim]
            padded_img = padded_img.reshape(output_shape)
        else:
            padded_img = transforms.functional.pad(
                img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0
            )

        return padded_img


def build_image_transformations(
    image_target_size,
    image_crop_size,
    random_rotation_angle,
    color_jitter_params,
    shortest_image_edge: int = 256,
    crop_fraction: float = 0.95,
):
    """
    Build torchvision-based image transformations.

    Args:
        image_target_size: Target size for resizing (list of [height, width]).
            If None, uses shortest_image_edge.
        image_crop_size: Size for cropping (list of [height, width]).
            If None, computed from image_target_size and crop_fraction.
        random_rotation_angle: Maximum rotation angle in degrees (0 for no rotation)
        color_jitter_params: Dictionary with color jitter parameters
        shortest_image_edge: Shortest edge to resize to (used when image_target_size is None)
        crop_fraction: Fraction of image to keep when cropping (used when image_crop_size is None)

    Returns:
        tuple: (train_transform, eval_transform)
    """
    # Compute target size if not provided
    if image_target_size is None:
        image_target_size = [shortest_image_edge, shortest_image_edge]

    # Compute crop size if not provided (based on crop_fraction of target size)
    if image_crop_size is None:
        crop_size = int(image_target_size[0] * crop_fraction)
        image_crop_size = [crop_size, crop_size]

    transform_list = [
        transforms.ToImage(),
        LetterBoxTransform(),
        transforms.Resize(size=image_target_size),
        transforms.RandomCrop(size=image_crop_size),
        transforms.Resize(size=image_target_size),
    ]
    if random_rotation_angle is not None and random_rotation_angle != 0:
        transform_list.append(
            transforms.RandomRotation(degrees=[-random_rotation_angle, random_rotation_angle])
        )
    if color_jitter_params is not None:
        transform_list.append(transforms.ColorJitter(**color_jitter_params))
    train_image_transform = transforms.Compose(transform_list)

    eval_image_transform = transforms.Compose(
        [
            transforms.ToImage(),
            LetterBoxTransform(),
            transforms.Resize(size=image_target_size),
            transforms.CenterCrop(size=image_crop_size),
            transforms.Resize(size=image_target_size),
        ]
    )
    return train_image_transform, eval_image_transform


# =============================================================================
# Embodiment ID Mapping
# =============================================================================

EMBODIMENT_TAG_TO_PROJECTOR_INDEX = {
    # Pretrain embodiment ids
    "robocasa_panda_omron": 13,
    "gr1": 20,
    "behavior_r1_pro": 24,
    # Pre-registered posttrain embodiment ids
    "unitree_g1": 8,
    "libero_panda": 2,
    "oxe_google": 0,
    "oxe_widowx": 1,
    "new_embodiment": 10,
}

# =============================================================================
# Embodiment-Specific Configurations for Statistics Conversion
# =============================================================================

# SO100 Modality Metadata (defines joint group slicing indices)
SO100_MODALITY_META = {
    "state": {
        "single_arm": {"start": 0, "end": 5},
        "gripper": {"start": 5, "end": 6}
    },
    "action": {
        "single_arm": {"start": 0, "end": 5},
        "gripper": {"start": 5, "end": 6}
    },
}

# SO100 Modality Config (matching original repo format exactly)
SO100_MODALITY_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# Registry mapping embodiment tags to their full modality configs (original repo format)
MODALITY_CONFIGS = {
    "new_embodiment": SO100_MODALITY_CONFIG,
}

# Registry mapping embodiment tags to their statistics conversion configs
EMBODIMENT_STAT_CONFIGS = {
    "new_embodiment": {  # SO100
        "modality_meta": SO100_MODALITY_META,
        "modality_config": SO100_MODALITY_CONFIG,
    },
}

##### LeRobot Groot N1.6 Stats conversion process #####

def compute_relative_action_stats(
    dataset,
    embodiment_tag: str,
) -> dict[str, dict[str, list]]:
    """
    Compute per-timestep relative action statistics for joint groups that use RELATIVE representation.

    Matches the original GR00T implementation: for each position i in each episode,
    gathers a 16-action chunk using delta_indices and subtracts state[i] to produce
    a (16, joint_dim) relative chunk. All chunks are stacked → (N, 16, joint_dim)
    and stats are computed along axis=0 → (16, joint_dim) per-timestep statistics.

    Args:
        dataset: LeRobot dataset object
        embodiment_tag: Embodiment tag (e.g., "new_embodiment")

    Returns:
        Dictionary: {joint_group: {stat_type: values}}
        Stats have shape (action_horizon, joint_dim) — per-timestep statistics.

    Raises:
        ValueError: If embodiment_tag not found in EMBODIMENT_STAT_CONFIGS
    """
    if embodiment_tag not in EMBODIMENT_STAT_CONFIGS:
        raise ValueError(f"Unknown embodiment: {embodiment_tag}")

    config = EMBODIMENT_STAT_CONFIGS[embodiment_tag]
    modality_meta = config["modality_meta"]
    action_modality = config["modality_config"]["action"]
    state_modality = config["modality_config"]["state"]
    action_configs = action_modality.action_configs

    action_delta_indices = np.array(action_modality.delta_indices)  # e.g. [0,1,...,15]
    state_delta_indices = state_modality.delta_indices  # e.g. [0]

    # Find joint groups that need relative stats
    relative_joint_groups = [
        joint_group
        for joint_group, action_config in zip(action_modality.modality_keys, action_configs or [])
        if action_config.rep == ActionRepresentation.RELATIVE
    ]

    if not relative_joint_groups:
        return {}

    print(f"Computing relative action stats for joint groups: {relative_joint_groups}")
    print(f"  action_delta_indices: {action_delta_indices.tolist()}")
    print(f"  state_delta_indices: {state_delta_indices}")

    # Collect relative action chunks for each joint group
    # Each chunk has shape (action_horizon, joint_dim)
    all_relative_chunks = {jg: [] for jg in relative_joint_groups}

    # LeRobot datasets have an hf_dataset attribute (HuggingFace Dataset)
    # Convert to pandas for easier episode grouping
    if hasattr(dataset.hf_dataset, "to_pandas"):
        episode_data_dict = dataset.hf_dataset.to_pandas()
    else:
        # Already a pandas DataFrame (for testing)
        episode_data_dict = dataset.hf_dataset

    unique_episodes = episode_data_dict["episode_index"].unique()

    print(f"Processing {len(unique_episodes)} episodes...")

    for episode_idx in unique_episodes:
        # Get all frames for this episode
        episode_mask = episode_data_dict["episode_index"] == episode_idx
        episode_frames = episode_data_dict[episode_mask]

        # Extract states and actions for the episode as numpy arrays
        states_list = episode_frames["observation.state"].tolist()
        actions_list = episode_frames["action"].tolist()
        states = np.array([s.numpy() if isinstance(s, torch.Tensor) else np.array(s) for s in states_list])
        actions = np.array([a.numpy() if isinstance(a, torch.Tensor) else np.array(a) for a in actions_list])

        # Usable length: positions where a full action chunk fits
        # action_delta_indices[-1] is the last offset (e.g. 15)
        usable_length = len(episode_frames) - action_delta_indices[-1]

        for joint_group in relative_joint_groups:
            start_idx = modality_meta["action"][joint_group]["start"]
            end_idx = modality_meta["action"][joint_group]["end"]

            state_slice = states[:, start_idx:end_idx]   # (T, joint_dim)
            action_slice = actions[:, start_idx:end_idx]  # (T, joint_dim)

            for i in range(usable_length):
                # Reference state at position i (matching original: state_delta_indices[-1] + i)
                state_ind = state_delta_indices[-1] + i
                last_state = state_slice[state_ind]  # (joint_dim,)

                # Gather 16-action chunk at offsets delta_indices + i
                action_inds = action_delta_indices + i
                action_chunk = action_slice[action_inds]  # (16, joint_dim)

                # Relative chunk: each action minus the reference state
                relative_chunk = action_chunk - last_state  # (16, joint_dim)
                all_relative_chunks[joint_group].append(relative_chunk)

    # Compute per-timestep statistics
    # Stack: (N, action_horizon, joint_dim), stats along axis=0 → (action_horizon, joint_dim)
    relative_stats = {}
    for joint_group, chunks in all_relative_chunks.items():
        chunks_array = np.stack(chunks, axis=0)  # (N, 16, joint_dim)

        relative_stats[joint_group] = {
            "min": np.min(chunks_array, axis=0).tolist(),
            "max": np.max(chunks_array, axis=0).tolist(),
            "mean": np.mean(chunks_array, axis=0).tolist(),
            "std": np.std(chunks_array, axis=0).tolist(),
            "q01": np.quantile(chunks_array, 0.01, axis=0).tolist(),
            "q99": np.quantile(chunks_array, 0.99, axis=0).tolist(),
        }
        print(f"  {joint_group}: computed per-timestep stats from {len(chunks)} chunks, "
              f"shape=({chunks_array.shape[1]}, {chunks_array.shape[2]})")

    return relative_stats


def _slice_stats_by_joint_group(
    stats_dict: dict[str, Any],
    start_idx: int,
    end_idx: int,
) -> dict[str, list]:
    """
    Slice statistics dictionary by joint group indices.

    Args:
        stats_dict: Dictionary with stat_type -> values (e.g., {"min": [...], "max": [...]})
        start_idx: Start index for slicing
        end_idx: End index for slicing (exclusive)

    Returns:
        Dictionary with sliced statistics
    """
    sliced_stats = {}
    for stat_type, values in stats_dict.items():
        if isinstance(values, torch.Tensor):
            sliced_stats[stat_type] = values[start_idx:end_idx].cpu().tolist()
        elif isinstance(values, (list, np.ndarray)):
            sliced_stats[stat_type] = list(np.array(values)[start_idx:end_idx])
        else:
            # For non-sliceable values (like 'count'), keep as is
            sliced_stats[stat_type] = values
    return sliced_stats


def _get_lerobot_stats_key(modality: str) -> str:
    """
    Map modality name to LeRobot dataset stats key.

    Args:
        modality: Modality name ("state", "action", "relative_action")

    Returns:
        LeRobot dataset stats key
    """
    mapping = {
        "state": "observation.state",
        "action": "action",
        "relative_action": "relative_action",
    }
    return mapping.get(modality, modality)


def convert_lerobot_stats_to_processor_format(
    dataset_stats: dict[str, dict[str, Any]],
    embodiment_tag: str,
) -> dict[str, Any]:
    """
    Convert LeRobot dataset statistics to StateActionProcessor format.

    This function transforms flat statistics arrays into joint-group-specific statistics
    using embodiment-specific modality metadata (start/end indices).

    LeRobot format:
        {key: {stat_type: values}}
        Example: {"observation.state": {"min": [6 values], "max": [6 values], ...}}

    StateActionProcessor format:
        {embodiment_tag: {modality: {joint_group: {stat_type: values}}}}
        Example: {
            "new_embodiment": {
                "state": {
                    "single_arm": {"min": [5 values], "max": [5 values], ...},
                    "gripper": {"min": [1 value], "max": [1 value], ...}
                }
            }
        }

    Args:
        dataset_stats: Statistics dictionary in LeRobot format
        embodiment_tag: The embodiment tag (e.g., "new_embodiment" for SO100)

    Returns:
        Nested dictionary in StateActionProcessor format

    Raises:
        ValueError: If embodiment_tag not found in EMBODIMENT_STAT_CONFIGS
        KeyError: If required keys missing in dataset_stats or configs
    """
    # Check if embodiment has specific configuration
    if embodiment_tag not in EMBODIMENT_STAT_CONFIGS:
        available_embodiments = list(EMBODIMENT_STAT_CONFIGS.keys())
        raise ValueError(
            f"Embodiment '{embodiment_tag}' not found in EMBODIMENT_STAT_CONFIGS. "
            f"Available embodiments: {available_embodiments}. "
            f"Please add configuration for '{embodiment_tag}' to EMBODIMENT_STAT_CONFIGS in utils.py"
        )

    # Get embodiment configuration
    config = EMBODIMENT_STAT_CONFIGS[embodiment_tag]
    modality_meta = config["modality_meta"]
    modality_config = config["modality_config"]

    statistics = {embodiment_tag: {}}

    # Process state and action modalities
    for modality in ["state", "action"]:
        # Get LeRobot stats key
        lerobot_key = _get_lerobot_stats_key(modality)
        stats_dict = dataset_stats[lerobot_key]  # Will raise KeyError if missing

        statistics[embodiment_tag][modality] = {}

        # Get modality config and joint groups
        modality_cfg = modality_config[modality]
        joint_groups = modality_cfg.modality_keys

        # Process each joint group
        for joint_group in joint_groups:
            # Get slicing indices (will raise KeyError if missing)
            start_idx = modality_meta[modality][joint_group]["start"]
            end_idx = modality_meta[modality][joint_group]["end"]

            # Slice statistics for this joint group
            statistics[embodiment_tag][modality][joint_group] = _slice_stats_by_joint_group(
                stats_dict, start_idx, end_idx
            )

    # Process relative_action statistics if present
    action_modality = modality_config["action"]
    action_configs = action_modality.action_configs
    needs_relative_stats = any(
        cfg.rep == ActionRepresentation.RELATIVE for cfg in (action_configs or [])
    )

    if needs_relative_stats:
        if "relative_action" not in dataset_stats:
            raise ValueError(
                f"Embodiment '{embodiment_tag}' requires relative_action statistics.\n"
                f"Compute them with:\n"
                f"  from lerobot.policies.gr00t_n1d6.utils import compute_relative_action_stats\n"
                f"  rel_stats = compute_relative_action_stats(dataset, '{embodiment_tag}')\n"
                f"  dataset_stats['relative_action'] = rel_stats"
            )

        statistics[embodiment_tag]["relative_action"] = {}

        # action_configs correspond to modality_keys in order
        for joint_group, action_config in zip(action_modality.modality_keys, action_configs):
            if action_config.rep != ActionRepresentation.RELATIVE:
                continue

            if joint_group not in dataset_stats["relative_action"]:
                raise KeyError(
                    f"Missing relative_action stats for joint group '{joint_group}'. "
                    f"Available joint groups: {list(dataset_stats['relative_action'].keys())}"
                )

            # Stats are already per joint group (no slicing needed)
            relative_stats = dataset_stats["relative_action"][joint_group]
            statistics[embodiment_tag]["relative_action"][joint_group] = {
                stat_type: values if isinstance(values, list) else values
                for stat_type, values in relative_stats.items()
            }

    return statistics
