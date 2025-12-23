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
    import albumentations as A

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    A = None
    ALBUMENTATIONS_AVAILABLE = False


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

    unnormalized[..., mask] = (
        normalized_values[..., mask] * std_vals[..., mask] + mean_vals[..., mask]
    )
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
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
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


class FractionalRandomCrop:
    """Crop a random part of the input based on fractions while maintaining aspect ratio.

    Albumentations DualTransform for fractional random cropping.
    """

    def __init__(self, crop_fraction: float = 0.9, p: float = 1.0, always_apply: bool | None = None):
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("albumentations is required for FractionalRandomCrop")

        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be between 0.0 and 1.0")
        self.crop_fraction = crop_fraction
        self._base = A.DualTransform(p=p, always_apply=always_apply)

    def __call__(self, **data):
        image = data["image"]
        height, width = image.shape[:2]

        crop_height = int(height * self.crop_fraction)
        crop_width = int(width * self.crop_fraction)
        crop_height = max(1, crop_height)
        crop_width = max(1, crop_width)

        max_y = height - crop_height
        max_x = width - crop_width

        y_min = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        x_min = np.random.randint(0, max_x + 1) if max_x > 0 else 0

        cropped = image[y_min : y_min + crop_height, x_min : x_min + crop_width]
        data["image"] = cropped
        return data


class FractionalCenterCrop:
    """Crop the center part of the input based on fractions while maintaining aspect ratio."""

    def __init__(self, crop_fraction: float = 0.9, p: float = 1.0, always_apply: bool | None = None):
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("albumentations is required for FractionalCenterCrop")

        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be between 0.0 and 1.0")
        self.crop_fraction = crop_fraction

    def __call__(self, **data):
        image = data["image"]
        height, width = image.shape[:2]

        crop_height = int(height * self.crop_fraction)
        crop_width = int(width * self.crop_fraction)
        crop_height = max(1, crop_height)
        crop_width = max(1, crop_width)

        y_min = (height - crop_height) // 2
        x_min = (width - crop_width) // 2

        cropped = image[y_min : y_min + crop_height, x_min : x_min + crop_width]
        data["image"] = cropped
        return data


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

    if crop_fraction is None:
        fraction_to_use = image_crop_size[0] / image_target_size[0]
    else:
        fraction_to_use = crop_fraction

    if shortest_image_edge is None:
        max_size = image_target_size[0]
    else:
        max_size = shortest_image_edge

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
    image_target_size, image_crop_size, random_rotation_angle, color_jitter_params
):
    """
    Build torchvision-based image transformations.

    Args:
        image_target_size: Target size for resizing (list of [height, width])
        image_crop_size: Size for cropping (list of [height, width])
        random_rotation_angle: Maximum rotation angle in degrees (0 for no rotation)
        color_jitter_params: Dictionary with color jitter parameters

    Returns:
        tuple: (train_transform, eval_transform)
    """
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
