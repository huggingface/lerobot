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
Groot N1.6 processor implementation.

This module provides data processing for Groot N1.6, ported from:
- gr00t-orig/model/gr00t_n1d6/processing_gr00t_n1d6.py

Key classes:
- Gr00tN1d6Processor: Main processor class
- Gr00tN1d6DataCollator: Collation logic
- StateActionProcessor: Simplified state/action normalization

Key differences from N1.5:
- Uses vlm_content format instead of eagle_content
- Supports albumentations for image transforms
- StateActionProcessor for relative action handling
- max_action_horizon: int = 40 (vs 16 in N1.5)
"""

from __future__ import annotations

import json
import os
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature

from lerobot.policies.gr00t_n1d6.utils import (
    ALBUMENTATIONS_AVAILABLE,
    EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
    ActionRepresentation,
    EmbodimentTag,
    ModalityConfig,
    apply_sin_cos_encoding,
    apply_with_replay,
    build_image_transformations,
    build_image_transformations_albumentations,
    nested_dict_to_numpy,
    normalize_values_meanstd,
    normalize_values_minmax,
    parse_modality_configs,
    to_json_serializable,
    unnormalize_values_meanstd,
    unnormalize_values_minmax,
)

# Suppress protobuf deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf")


# =============================================================================
# StateActionProcessor (simplified from gr00t-orig/data/state_action/state_action_processor.py)
# =============================================================================


class StateActionProcessor:
    """
    Simplified processor for robot state and action data.

    Handles:
    - State normalization (min/max, mean/std, sin/cos encoding)
    - Action normalization
    - Absolute <-> Relative action representation conversion (simplified)

    Note: This is a simplified version that handles common cases.
    Complex relative action conversions (EEF, quaternions) are not fully supported.
    """

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        apply_sincos_state_encoding: bool = False,
        use_relative_action: bool = True,
    ):
        """
        Initialize unified state and action processor.

        Args:
            modality_configs: Nested dict with structure:
                {embodiment_tag: {modality: ModalityConfig}}
            statistics: Optional nested dict with structure:
                {embodiment_tag: {modality: {joint_group: {stat_type: values}}}}
            use_percentiles: Whether to use percentiles (q01/q99) instead of min/max
            clip_outliers: Whether to clip normalized values to [-1, 1]
            apply_sincos_state_encoding: Enable sin/cos encoding for states
            use_relative_action: Enable relative action processing
        """
        self.modality_configs = parse_modality_configs(modality_configs)
        self.statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action

        # Normalization parameters computed from statistics
        self.norm_params: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}

        if statistics is not None:
            self.set_statistics(statistics)

        self.train()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        override: bool = False,
    ) -> None:
        """Set dataset statistics for normalization."""
        for key in statistics:
            if key not in self.statistics or override:
                self.statistics[key] = deepcopy(statistics[key])
            else:
                print(f"Embodiment tag {key} already in statistics, skipping updating")
        self._compute_normalization_parameters()

    def _compute_normalization_parameters(self) -> None:
        """Compute and cache normalization parameters from statistics."""
        for embodiment_tag in self.statistics:
            self.norm_params[embodiment_tag] = {}

            for modality in ["state", "action"]:
                if modality not in self.statistics[embodiment_tag]:
                    continue

                self.norm_params[embodiment_tag][modality] = {}

                for joint_group, stats in self.statistics[embodiment_tag][modality].items():
                    if self.use_percentiles:
                        min_vals = np.array(stats["q01"])
                        max_vals = np.array(stats["q99"])
                    else:
                        min_vals = np.array(stats["min"])
                        max_vals = np.array(stats["max"])

                    mean_vals = np.array(stats["mean"])
                    std_vals = np.array(stats["std"])

                    # Ensure range is not zero
                    range_vals = max_vals - min_vals
                    range_vals = np.maximum(range_vals, 1e-8)

                    self.norm_params[embodiment_tag][modality][joint_group] = {
                        "min": min_vals,
                        "max": max_vals,
                        "dim": np.array(range_vals.shape[0]),
                        "mean": mean_vals,
                        "std": std_vals,
                    }

            # Override absolute action stats with relative stats where specified
            if "action" in self.modality_configs[embodiment_tag]:
                modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
                action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

                if action_configs is not None:
                    for key, action_config in zip(modality_keys, action_configs, strict=True):
                        if action_config.rep == ActionRepresentation.RELATIVE and self.use_relative_action:
                            if "relative_action" not in self.statistics[embodiment_tag]:
                                raise ValueError(
                                    f"Relative action statistics required for embodiment '{embodiment_tag}' "
                                    f"but 'relative_action' not found in statistics"
                                )
                            if key not in self.statistics[embodiment_tag]["relative_action"]:
                                raise ValueError(
                                    f"Relative action statistics required for key '{key}' "
                                    f"in embodiment '{embodiment_tag}' but not found"
                                )
                            action_dim = self.norm_params[embodiment_tag]["action"][key]["dim"]
                            self.norm_params[embodiment_tag]["action"][key] = nested_dict_to_numpy(
                                self.statistics[embodiment_tag]["relative_action"][key]
                            )
                            self.norm_params[embodiment_tag]["action"][key]["dim"] = action_dim

    def apply_state(
        self,
        state: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        """
        Apply state processing (normalization, encoding).

        Args:
            state: Dict mapping joint_group -> raw state values
            embodiment_tag: Embodiment identifier

        Returns:
            Dict mapping joint_group -> processed state values
        """
        normalized_values = {}
        state = deepcopy(state)

        # Get sin/cos embedding keys if enabled
        sin_cos_keys = None
        if self.apply_sincos_state_encoding:
            state_config = self.modality_configs[embodiment_tag].get("state")
            if state_config and hasattr(state_config, "sin_cos_embedding_keys"):
                sin_cos_keys = state_config.sin_cos_embedding_keys

        for joint_group in self.modality_configs[embodiment_tag]["state"].modality_keys:
            if joint_group not in state:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in state dict for embodiment '{embodiment_tag}'"
                )

            # Strategy 1: Sin/cos encoding
            if sin_cos_keys and joint_group in sin_cos_keys:
                normalized_values[joint_group] = apply_sin_cos_encoding(state[joint_group])

            # Strategy 2: Mean/std normalization
            elif (
                hasattr(self.modality_configs[embodiment_tag]["state"], "mean_std_embedding_keys")
                and self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
                and joint_group in self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
            ):
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized = normalize_values_meanstd(state[joint_group], params)
                normalized_values[joint_group] = normalized

            # Strategy 3: Min/max normalization
            else:
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized = normalize_values_minmax(state[joint_group], params)

                if self.clip_outliers:
                    normalized = np.clip(normalized, -1.0, 1.0)

                normalized_values[joint_group] = normalized

        return normalized_values

    def apply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Apply action processing (normalization).

        Note: Relative action conversion is simplified - only supports NON_EEF types.

        Args:
            action: Dict mapping joint_group -> raw action values
            embodiment_tag: Embodiment identifier
            state: Optional dict for relative action conversion

        Returns:
            Dict mapping joint_group -> processed action values
        """
        action = deepcopy(action)

        # Step 1: Convert absolute actions to relative (simplified)
        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
        action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

        if action_configs is not None and self.use_relative_action:
            for key, action_config in zip(modality_keys, action_configs, strict=True):
                if action_config.rep == ActionRepresentation.RELATIVE:
                    if state is None:
                        raise ValueError(f"State dict required for relative action processing of key '{key}'")

                    state_key = action_config.state_key if action_config.state_key else key
                    if state_key not in state:
                        raise KeyError(f"Reference state key '{state_key}' not found in state dict")

                    # Use last state as reference frame (simplified - just subtract)
                    reference_state = state[state_key][-1]
                    action[key] = action[key] - reference_state

        # Step 2: Normalize actions
        normalized_values = {}
        for joint_group in modality_keys:
            if joint_group not in action:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
                )

            params = self.norm_params[embodiment_tag]["action"][joint_group]
            if (
                self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys is not None
                and joint_group in self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys
            ):
                normalized = normalize_values_meanstd(action[joint_group], params)
            else:
                normalized = normalize_values_minmax(action[joint_group], params)

            if self.clip_outliers:
                normalized = np.clip(normalized, -1.0, 1.0)

            normalized_values[joint_group] = normalized

        return normalized_values

    def unapply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Reverse action processing (denormalization, relative->absolute conversion).

        Args:
            action: Dict mapping joint_group -> processed action values
            embodiment_tag: Embodiment identifier
            state: Optional dict for relative->absolute conversion

        Returns:
            Dict mapping joint_group -> raw absolute action values
        """
        # Step 1: Unnormalize actions
        unnormalized_values = {}
        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys

        for joint_group in modality_keys:
            if joint_group not in action:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
                )

            params = self.norm_params[embodiment_tag]["action"][joint_group]
            group_values = action[joint_group]

            if (
                self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys is not None
                and joint_group in self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys
            ):
                unnormalized = unnormalize_values_meanstd(group_values, params)
            else:
                unnormalized = unnormalize_values_minmax(group_values, params)

            unnormalized_values[joint_group] = unnormalized

        # Step 2: Convert relative actions to absolute (simplified)
        action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

        if action_configs is not None and self.use_relative_action:
            for key, action_config in zip(modality_keys, action_configs, strict=True):
                if action_config.rep == ActionRepresentation.RELATIVE:
                    if state is None:
                        # Skip relative->absolute conversion if no state provided
                        warnings.warn(
                            f"State dict required for relative->absolute conversion of key '{key}', "
                            "but state is None. Returning unnormalized relative actions.",
                            stacklevel=2,
                        )
                        continue

                    state_key = action_config.state_key if action_config.state_key else key

                    # Handle case where expected state key doesn't match available keys
                    # This happens when using a pretrained model with different embodiment configs
                    if state_key not in state:
                        # Try to find a matching state key or use a generic one
                        available_keys = list(state.keys())
                        if len(available_keys) == 1:
                            # Only one state key available, use it
                            fallback_key = available_keys[0]
                            warnings.warn(
                                f"Reference state key '{state_key}' not found in state dict. "
                                f"Using '{fallback_key}' instead. Available keys: {available_keys}",
                                stacklevel=2,
                            )
                            state_key = fallback_key
                        elif "state" in state:
                            # Use generic "state" key as fallback
                            warnings.warn(
                                f"Reference state key '{state_key}' not found, using generic 'state' key.",
                                stacklevel=2,
                            )
                            state_key = "state"
                        else:
                            # Skip relative->absolute conversion for this key
                            warnings.warn(
                                f"Reference state key '{state_key}' not found in state dict. "
                                f"Available keys: {available_keys}. Skipping relative->absolute conversion for '{key}'.",
                                stacklevel=2,
                            )
                            continue

                    # Handle batched and unbatched cases
                    relative_action = unnormalized_values[key]
                    reference_state = state[state_key]

                    # Get action dimension from the relative_action
                    action_dim = relative_action.shape[-1]

                    if reference_state.ndim == 2:
                        # Unbatched: (T, D) - use last timestep
                        # Truncate reference state to match action dimension
                        ref_state_slice = (
                            reference_state[-1, :action_dim]
                            if reference_state.shape[-1] >= action_dim
                            else reference_state[-1]
                        )
                        if ref_state_slice.shape[-1] < action_dim:
                            # Pad with zeros if state dimension is smaller
                            padding = np.zeros(action_dim - ref_state_slice.shape[-1])
                            ref_state_slice = np.concatenate([ref_state_slice, padding])
                        unnormalized_values[key] = relative_action + ref_state_slice
                    elif reference_state.ndim == 3:
                        # Batched: (B, T, D) - use last timestep per batch
                        ref_state_slice = (
                            reference_state[:, -1:, :action_dim]
                            if reference_state.shape[-1] >= action_dim
                            else reference_state[:, -1:]
                        )
                        if ref_state_slice.shape[-1] < action_dim:
                            # Pad with zeros if state dimension is smaller
                            padding = np.zeros(
                                (ref_state_slice.shape[0], 1, action_dim - ref_state_slice.shape[-1])
                            )
                            ref_state_slice = np.concatenate([ref_state_slice, padding], axis=-1)
                        unnormalized_values[key] = relative_action + ref_state_slice
                    elif reference_state.ndim == 1:
                        # Single state vector (D,) - use as is
                        ref_state_slice = (
                            reference_state[:action_dim]
                            if reference_state.shape[-1] >= action_dim
                            else reference_state
                        )
                        if ref_state_slice.shape[-1] < action_dim:
                            padding = np.zeros(action_dim - ref_state_slice.shape[-1])
                            ref_state_slice = np.concatenate([ref_state_slice, padding])
                        unnormalized_values[key] = relative_action + ref_state_slice

        return unnormalized_values

    def apply(
        self,
        state: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Apply both state and action processing together."""
        processed_state = self.apply_state(state, embodiment_tag)
        if action:
            processed_action = self.apply_action(action, embodiment_tag, state=state)
        else:
            assert not self.training, "Action is required in training mode"
            processed_action = {}
        return processed_state, processed_action

    def get_action_dim(self, embodiment_tag: str) -> int:
        """Get total action dimension."""
        total_dim = 0
        for joint_group in self.modality_configs[embodiment_tag]["action"].modality_keys:
            total_dim += self.norm_params[embodiment_tag]["action"][joint_group]["dim"].item()
        return total_dim


# =============================================================================
# Data Collator
# =============================================================================


def build_processor(model_name: str, transformers_loading_kwargs: dict) -> ProcessorMixin:
    """Build the Eagle processor for VLM inputs."""
    assert model_name == "nvidia/Eagle-Block2A-2B-v2", f"Processor for {model_name} not supported"
    eagle_path = os.path.join(os.path.dirname(__file__), "eagle3_model")
    return AutoProcessor.from_pretrained(eagle_path, **transformers_loading_kwargs)


class Gr00tN1d6DataCollator:
    """Data collator for Groot N1.6 that handles VLM content and tensor batching."""

    def __init__(
        self,
        model_name: str,
        model_type: Literal["eagle"] = "eagle",
        transformers_loading_kwargs: dict | None = None,
    ):
        if transformers_loading_kwargs is None:
            transformers_loading_kwargs = {}
        self.processor = build_processor(model_name, transformers_loading_kwargs)
        # Set padding side to 'left' for Flash Attention compatibility
        self.processor.tokenizer.padding_side = "left"
        self.model_type = model_type
        self.model_name = model_name

    def __call__(self, features: list[dict[str, Any]]) -> BatchFeature:
        batch = {}
        keys = list(set().union(*(elem.keys() for elem in features)))

        for key in keys:
            values = [elem[key] for elem in features if key in elem]
            if key == "vlm_content":
                # Handle vlm_content specially - extract text and images
                text_list = []
                image_inputs = []
                for v in values:
                    curr_text_list = [v["text"]]
                    text_list += curr_text_list
                    curr_image_inputs = v["images"]
                    image_inputs += curr_image_inputs

                # NOTE: some VLMs need this, others don't.
                if self.model_type == "eagle":
                    image_inputs, _ = self.processor.process_vision_info([v["conversation"] for v in values])
                vlm_inputs = self.processor(
                    text=text_list, images=image_inputs, return_tensors="pt", padding=True
                )
                for k, v in vlm_inputs.items():
                    batch[k] = v
            elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
                raise Exception("Not implemented")
            else:
                # state, state_mask, action and action_mask - stack to form batch dimension
                batch[key] = torch.from_numpy(np.stack(values))
        return BatchFeature(data={"inputs": batch})

    def __str__(self):
        return f"Gr00tN1d6DataCollator(model_name={self.model_name}, model_type={self.model_type})"


# =============================================================================
# VLAStepData (simplified from gr00t-orig/data/types.py)
# =============================================================================


@dataclass
class VLAStepData:
    """
    Represents a single step of VLA (Vision-Language-Action) data.

    This is the core data structure for processor input.
    """

    images: dict[str, list[np.ndarray]]  # view_name -> list[np.ndarray]
    states: dict[str, np.ndarray]  # state_name -> np.ndarray
    actions: dict[str, np.ndarray]  # action_name -> np.ndarray (horizon, dim)
    text: str | None = None
    embodiment: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    is_demonstration: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Main Processor
# =============================================================================


class Gr00tN1d6Processor:
    """
    Main processor for Groot N1.6.

    Handles:
    - State/action normalization via StateActionProcessor
    - Image preprocessing (crop, resize, augmentation)
    - VLM content creation for collation
    - Language formalization
    """

    data_collator_class = Gr00tN1d6DataCollator

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        image_crop_size: list[int] | None = None,
        image_target_size: list[int] | None = None,
        shortest_image_edge: int = 512,
        crop_fraction: float = 0.95,
        random_rotation_angle: int | None = None,
        color_jitter_params: dict[str, float] | None = None,
        formalize_language: bool = True,
        model_name: str = "nvidia/Eagle-Block2A-2B-v2",
        model_type: Literal["eagle"] = "eagle",
        max_state_dim: int = 29,
        max_action_dim: int = 29,
        apply_sincos_state_encoding: bool = False,
        max_action_horizon: int = 40,
        use_albumentations: bool = False,
        use_relative_action: bool = True,
        embodiment_id_mapping: dict[str, int] | None = None,
        transformers_loading_kwargs: dict | None = None,
    ):
        if transformers_loading_kwargs is None:
            transformers_loading_kwargs = {"trust_remote_code": True}

        self.modality_configs = parse_modality_configs(modality_configs)

        # Initialize StateActionProcessor for state/action normalization
        self.state_action_processor = StateActionProcessor(
            modality_configs=modality_configs,
            statistics=statistics,
            use_percentiles=use_percentiles,
            clip_outliers=clip_outliers,
            apply_sincos_state_encoding=apply_sincos_state_encoding,
            use_relative_action=use_relative_action,
        )

        # Save state action processor settings
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action

        # Save VLM settings
        self.formalize_language = formalize_language
        self.model_name = model_name
        self.model_type = model_type

        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.max_action_horizon = max_action_horizon

        # Save image processing settings
        self.image_crop_size = image_crop_size
        self.image_target_size = image_target_size
        self.random_rotation_angle = random_rotation_angle
        self.color_jitter_params = color_jitter_params
        self.processor = build_processor(model_name, transformers_loading_kwargs)
        # Set padding side to 'left' for Flash Attention compatibility
        self.processor.tokenizer.padding_side = "left"
        self.embodiment_id_mapping = embodiment_id_mapping or EMBODIMENT_TAG_TO_PROJECTOR_INDEX.copy()
        # Handle case where fine-tuning embodiment tag is not in pre-trained mapping
        for k, v in EMBODIMENT_TAG_TO_PROJECTOR_INDEX.items():
            if k not in self.embodiment_id_mapping:
                self.embodiment_id_mapping[k] = v
        self.shortest_image_edge = shortest_image_edge
        self.crop_fraction = crop_fraction

        # Choose between torchvision and albumentations transforms
        # Check if albumentations is available, fall back to torchvision if not
        if use_albumentations and not ALBUMENTATIONS_AVAILABLE:
            warnings.warn(
                "use_albumentations_transforms=True but albumentations is not installed. "
                "Falling back to torchvision transforms. Install albumentations with: "
                "pip install albumentations==1.4.18",
                UserWarning,
                stacklevel=2,
            )
            use_albumentations = False

        self.use_albumentations = use_albumentations
        if use_albumentations:
            self.train_image_transform, self.eval_image_transform = (
                build_image_transformations_albumentations(
                    image_target_size,
                    image_crop_size,
                    random_rotation_angle,
                    color_jitter_params,
                    shortest_image_edge,
                    crop_fraction,
                )
            )
        else:
            self.train_image_transform, self.eval_image_transform = build_image_transformations(
                image_target_size,
                image_crop_size,
                random_rotation_angle,
                color_jitter_params,
                shortest_image_edge,
                crop_fraction,
            )
        self._collator = self.data_collator_class(
            model_name=model_name,
            model_type=model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )
        self.training = True

    @property
    def collator(self):
        return self._collator

    def train(self):
        self.training = True
        self.state_action_processor.train()

    def eval(self):
        self.training = False
        self.state_action_processor.eval()

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        override: bool = False,
    ) -> None:
        """Set dataset statistics for normalization."""
        self.state_action_processor.set_statistics(statistics, override=override)

        # Compute action dimensions for convenience
        self.action_dim = {}
        for embodiment_tag in self.state_action_processor.statistics:
            self.action_dim[embodiment_tag] = self.state_action_processor.get_action_dim(embodiment_tag)

    def decode_action(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
    ):
        """Undo action normalization and convert relative actions to absolute."""
        # Split concatenated action into joint groups
        out_dict = {}
        start_idx = 0
        joint_groups = self.modality_configs[embodiment_tag.value]["action"].modality_keys
        action_horizon = len(self.modality_configs[embodiment_tag.value]["action"].delta_indices)
        for key in joint_groups:
            joint_dim = self.state_action_processor.norm_params[embodiment_tag.value]["action"][key][
                "dim"
            ].item()
            out_dict[key] = action[..., :action_horizon, start_idx : start_idx + joint_dim]
            start_idx += joint_dim

        # Use StateActionProcessor to unnormalize and convert to absolute
        return self.state_action_processor.unapply_action(out_dict, embodiment_tag.value, state=state)

    def _apply_vlm_processing(self, images: np.ndarray, language: str) -> dict:
        """
        Create VLM content for collation.

        Args:
            images: [T, C, H, W] numpy array
            language: Task description string

        Returns:
            vlm_content dict format for collation
        """
        # Convert images to PIL format
        pil_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in images]

        # Create conversation with images and text
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": language},
                    *[{"type": "image", "image": img} for img in pil_images],
                ],
            }
        ]

        # Apply chat template but don't process yet - let collator handle it
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)

        # Return vlm_content format for collation
        return {
            "vlm_content": {
                "text": text,
                "images": pil_images,
                "conversation": conversation,
            }
        }

    def __call__(
        self,
        messages: list[dict[str, Any]],
    ):
        """
        Process a list of messages containing VLAStepData.

        Args:
            messages: List with single message dict containing 'content' as VLAStepData

        Returns:
            Dict with processed inputs for model
        """
        assert len(messages) == 1
        content = messages[0]["content"]
        embodiment_tag = content.embodiment
        action_data = content.actions
        state_data = content.states

        # Use StateActionProcessor to handle relative conversion and normalization
        normalized_states, normalized_actions = self.state_action_processor.apply(
            state=state_data,
            action=action_data,
            embodiment_tag=embodiment_tag.value,
        )

        if normalized_actions:
            # Concatenate actions
            action_keys = self.modality_configs[embodiment_tag.value]["action"].modality_keys
            # Ensure all action arrays are 2D (t, d) before concatenation
            action_tensors = []
            for key in action_keys:
                arr = normalized_actions[key]
                arr_tensor = torch.from_numpy(arr)
                # Reshape to (t, d) if needed
                if arr_tensor.ndim == 1:
                    # (d,) -> (1, d)
                    arr_tensor = arr_tensor.unsqueeze(0)
                elif arr_tensor.ndim == 3:
                    # (1, t, d) or (t, 1, d) -> (t, d)
                    if arr_tensor.shape[0] == 1:
                        arr_tensor = arr_tensor.squeeze(0)
                    elif arr_tensor.shape[1] == 1:
                        arr_tensor = arr_tensor.squeeze(1)
                    else:
                        # (b, t, d) -> take first batch, (t, d)
                        arr_tensor = arr_tensor[0]
                action_tensors.append(arr_tensor)
            normalized_actions = torch.cat(action_tensors, dim=-1)  # (t, d)
            action_dim = normalized_actions.shape[1]
            # Pad action to max_action_dim
            normalized_actions = torch.cat(
                [
                    normalized_actions,
                    torch.zeros(
                        normalized_actions.shape[0],
                        self.max_action_dim - normalized_actions.shape[1],
                    ),
                ],
                dim=-1,
            )  # (t, max_action_dim)
            # Pad action to max_action_horizon
            action_horizon = normalized_actions.shape[0]
            normalized_actions = torch.cat(
                [
                    normalized_actions,
                    torch.zeros(
                        self.max_action_horizon - normalized_actions.shape[0],
                        self.max_action_dim,
                    ),
                ],
                dim=0,
            )  # (max_action_horizon, max_action_dim)
            # Create action mask
            action_mask = torch.ones_like(normalized_actions)
            action_mask[action_horizon:] = 0
            action_mask[:, action_dim:] = 0
        else:
            assert not self.training, "Action is required in training mode"
            normalized_actions = None
            action_mask = None

        # Concatenate states
        state_keys = self.modality_configs[embodiment_tag.value]["state"].modality_keys
        normalized_states = torch.cat(
            [torch.from_numpy(normalized_states[key]) for key in state_keys], dim=-1
        )
        normalized_states = torch.cat(
            [
                normalized_states,
                torch.zeros(normalized_states.shape[0], self.max_state_dim - normalized_states.shape[1]),
            ],
            dim=-1,
        )

        # Crop and resize images
        image_transform = self.train_image_transform if self.training else self.eval_image_transform
        # Use actual image keys from content.images if available, otherwise fall back to modality_configs
        if content.images:
            image_keys = list(content.images.keys())
        else:
            image_keys = self.modality_configs[embodiment_tag.value]["video"].modality_keys

        if self.formalize_language:
            language = content.text.lower()
            language = re.sub(r"[^\w\s]", "", language)
        else:
            language = content.text

        vlm_inputs = self._get_vlm_inputs(
            image_keys=image_keys,
            images=content.images,
            image_transform=image_transform,
            language=language,
        )

        transformed_inputs = {
            "state": normalized_states.to(torch.get_default_dtype()),
        }
        if normalized_actions is not None:
            transformed_inputs["action"] = normalized_actions.to(torch.get_default_dtype())
        # Add VLM inputs
        transformed_inputs.update(vlm_inputs)
        if action_mask is not None:
            transformed_inputs["action_mask"] = action_mask
        transformed_inputs["embodiment_id"] = self.embodiment_id_mapping[embodiment_tag.value]
        return transformed_inputs

    def _get_vlm_inputs(
        self,
        image_keys: list[str],
        images: dict[str, list],
        image_transform,
        language: str,
    ):
        """Process images and create VLM inputs."""
        temporal_stacked_images = {}

        if self.use_albumentations:
            # Use albumentations transforms with replay for consistency
            replay = None
            for view in image_keys:
                assert view in images, f"{view} not in {images}"
                transformed_images, replay = apply_with_replay(image_transform, images[view], replay)
                temporal_stacked_images[view] = torch.stack(transformed_images)  # (T, C, H, W)
        else:
            # Use torchvision transforms
            for view in image_keys:
                assert view in images, f"{view} not in {images}"
                temporal_stacked_images[view] = torch.stack(
                    [image_transform(img) for img in images[view]]
                )  # (T, C, H, W)

        for k, v in temporal_stacked_images.items():
            assert isinstance(k, str), f"{k} is not a string"
            assert isinstance(v, torch.Tensor), f"{v} is not a torch tensor"
            assert v.ndim == 4, f"{v} is not a 4D tensor"
            assert v.dtype == torch.uint8, f"{v} is not a uint8 tensor"
            assert v.shape[1] == 3, f"{v} is not a 3 channel tensor"

        stacked_images = (
            torch.stack([temporal_stacked_images[view] for view in image_keys], dim=1).flatten(0, 1).numpy()
        )  # (T*V, C, H, W), Eagle processor expects numpy array

        vlm_inputs = self._apply_vlm_processing(stacked_images, language)
        return vlm_inputs

# =============================================================================
# Factory function for LeRobot integration
# =============================================================================

from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from lerobot.configs.types import PolicyFeature
    from lerobot.processor import PolicyProcessorPipeline, ProcessorStep

from lerobot.configs.types import PipelineFeatureType  # noqa: E402
from lerobot.policies.gr00t_n1d6.configuration_gr00t_n1d6 import Gr00tN1d6Config  # noqa: E402
from lerobot.processor import (  # noqa: E402
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import (  # noqa: E402
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.core import EnvTransition, TransitionKey  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


@ProcessorStepRegistry.register(name="gr00t_n1d6_process_v1")
class Gr00tN1d6ProcessStep(ProcessorStep):
    """Processor step that uses Gr00tN1d6Processor to transform LeRobot format to model format."""

    def __init__(
        self,
        processor: Gr00tN1d6Processor | None = None,
        language_key: str = "task",
        processor_config_path: str | None = None,
    ):
        self._processor = processor
        self.language_key = language_key
        self.processor_config_path = processor_config_path
        self._pending_state: dict[str, torch.Tensor] | None = None

    @property
    def processor(self) -> Gr00tN1d6Processor:
        """Lazy initialization of processor from config if not provided."""
        if self._processor is None:
            if self.processor_config_path is None:
                raise ValueError(
                    "Processor not provided and processor_config_path not set. "
                    "Cannot create processor without configuration."
                )
            # Load policy config and create processor
            config_path = Path(self.processor_config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Policy config not found at {self.processor_config_path}")

            from lerobot.policies.gr00t_n1d6.configuration_gr00t_n1d6 import Gr00tN1d6Config

            # Load config JSON and create config object
            with open(config_path) as f:
                config_dict = json.load(f)
            # Remove 'type' field if present (not part of config schema)
            config_dict.pop("type", None)
            policy_config = Gr00tN1d6Config(**config_dict)

            # Create processor using the same logic as make_gr00t_n1d6_pre_post_processors
            from lerobot.policies.gr00t_n1d6.utils import ModalityConfig

            modality_configs = {
                policy_config.embodiment_tag: {
                    "state": ModalityConfig(
                        delta_indices=[0],
                        modality_keys=["state"],
                    ),
                    "action": ModalityConfig(
                        delta_indices=list(range(policy_config.chunk_size)),
                        modality_keys=["action"],
                    ),
                    "video": ModalityConfig(
                        delta_indices=[0],
                        modality_keys=["image"],
                    ),
                }
            }

            self._processor = Gr00tN1d6Processor(
                modality_configs=modality_configs,
                statistics=None,  # Will be set via load_state_dict
                formalize_language=policy_config.formalize_language,
                model_name=policy_config.tokenizer_assets_repo or "nvidia/Eagle-Block2A-2B-v2",
                max_state_dim=policy_config.max_state_dim,
                max_action_dim=policy_config.max_action_dim,
                max_action_horizon=policy_config.chunk_size,
                use_albumentations=policy_config.use_albumentations_transforms,
                use_relative_action=policy_config.use_relative_action,
                apply_sincos_state_encoding=policy_config.apply_sincos_state_encoding,
                embodiment_id_mapping={policy_config.embodiment_tag: 0},
                image_target_size=(
                    list(policy_config.image_target_size) if policy_config.image_target_size else [224, 224]
                ),
                image_crop_size=(
                    list(policy_config.image_crop_size) if policy_config.image_crop_size else [244, 244]
                ),
                shortest_image_edge=(
                    policy_config.shortest_image_edge if policy_config.shortest_image_edge else 256
                ),
                crop_fraction=policy_config.crop_fraction if policy_config.crop_fraction else 0.95,
                random_rotation_angle=policy_config.random_rotation_angle,
                color_jitter_params=policy_config.color_jitter_params,
            )

            # Apply pending state if any
            if self._pending_state:
                self.load_state_dict(self._pending_state)
                self._pending_state = None

        return self._processor

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Convert LeRobot transition to format expected by Gr00tN1d6 model."""
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}

        # Extract images
        img_keys = sorted([k for k in obs if k.startswith("observation.images.")])
        if not img_keys and "observation.image" in obs:
            img_keys = ["observation.image"]

        # Extract state
        state = obs.get("observation.state", None)
        if state is None:
            raise ValueError("observation.state is required")

        # Extract action (may be None for inference)
        action = transition.get(TransitionKey.ACTION, None)

        # Set processor to eval mode if no action (inference mode)
        # This is important because StateActionProcessor.apply() asserts not training when action is None
        if action is None:
            self.processor.eval()

        # Extract language
        language = comp.get(self.language_key, "")
        # Handle case where language is a list (after batch processing)
        if isinstance(language, list):
            language = language[0] if language else ""
        if not language:
            language = ""

        # Get embodiment tag from processor's mapping (use first key as default)
        embodiment_tag_mapping = self.processor.embodiment_id_mapping
        embodiment_tag_str = (
            list(embodiment_tag_mapping.keys())[0] if embodiment_tag_mapping else "new_embodiment"
        )

        # Convert images to numpy arrays (VLAStepData expects dict[str, list[np.ndarray]])
        images_dict = {}
        for img_key in img_keys:
            # Remove "observation.images." prefix to get view name
            view_name = img_key.replace("observation.images.", "").replace("observation.image", "image")
            img_tensor = obs[img_key]

            # Convert to numpy array
            if isinstance(img_tensor, torch.Tensor):
                # Handle batch dimension: (B, C, H, W) or (C, H, W)
                if img_tensor.ndim == 4:
                    # Batch dimension present - take first element
                    img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
                else:
                    # No batch dimension
                    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

                # Convert from [0, 1] float to [0, 255] uint8 if needed
                if img_np.dtype != np.uint8:
                    img_np = (img_np * 255).astype(np.uint8)

                images_dict[view_name] = [img_np]  # List of numpy arrays
            elif isinstance(img_tensor, np.ndarray):
                images_dict[view_name] = [img_tensor]
            else:
                # Assume PIL Image
                images_dict[view_name] = [np.array(img_tensor)]

        # Convert state to dict format expected by StateActionProcessor
        # Need to match modality_keys from modality_configs
        state_dict = {}
        if isinstance(state, torch.Tensor):
            state_np = state.cpu().numpy()
            # Get state modality keys from processor config
            if embodiment_tag_str in self.processor.modality_configs:
                state_keys = (
                    self.processor.modality_configs[embodiment_tag_str].get("state", {}).modality_keys
                )
                if state_keys:
                    # Split state tensor according to modality keys
                    # For now, use a single key "state" if we can't determine the split
                    state_dict[state_keys[0]] = state_np
                else:
                    state_dict["state"] = state_np
            else:
                state_dict["state"] = state_np
        else:
            state_dict = state

        # Convert action to dict format
        action_dict = None
        if action is not None:
            if isinstance(action, torch.Tensor):
                action_np = action.cpu().numpy()
                # Get action modality keys from processor config
                if embodiment_tag_str in self.processor.modality_configs:
                    action_keys = (
                        self.processor.modality_configs[embodiment_tag_str].get("action", {}).modality_keys
                    )
                    # Split action tensor according to modality keys
                    # For now, use a single key "action" if we can't determine the split
                    action_dict = {action_keys[0]: action_np} if action_keys else {"action": action_np}
                else:
                    action_dict = {"action": action_np}
            else:
                action_dict = action

        # Create VLAStepData
        # Note: VLAStepData is defined in this file, EmbodimentTag is imported at top
        try:
            embodiment_tag_enum = EmbodimentTag(embodiment_tag_str)
        except ValueError:
            # If not a valid enum value, use NEW_EMBODIMENT
            embodiment_tag_enum = EmbodimentTag.NEW_EMBODIMENT

        vla_step_data = VLAStepData(
            images=images_dict,
            text=language,
            states=state_dict,
            actions=action_dict,
            embodiment=embodiment_tag_enum,
        )

        # Call processor with message format
        messages = [{"content": vla_step_data}]
        processed = self.processor(messages)

        # Update transition with processed inputs
        transition[TransitionKey.OBSERVATION] = processed
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns the input features unchanged.

        This processor step transforms data format but doesn't change the feature schema.

        Args:
            features: A dictionary of policy features.

        Returns:
            The original dictionary of policy features.
        """
        return features

    def get_config(self) -> dict[str, Any]:
        """Returns a serializable dictionary of the processor's configuration.

        Excludes statistics since they are saved separately via state_dict().
        """
        config = {
            "language_key": self.language_key,
        }
        # Include processor_config_path if available (for loading from pretrained)
        if self.processor_config_path:
            config["processor_config_path"] = self.processor_config_path
        return config

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Returns normalization statistics as a flat state dictionary.

        This enables saving stats to safetensors files for checkpoint resume.
        """
        if not self.processor.state_action_processor.statistics:
            return {}

        flat: dict[str, torch.Tensor] = {}
        statistics = self.processor.state_action_processor.statistics

        for embodiment_tag, modalities in statistics.items():
            for modality, joint_groups in modalities.items():
                for joint_group, stats in joint_groups.items():
                    for stat_name, value in stats.items():
                        key = f"{embodiment_tag}.{modality}.{joint_group}.{stat_name}"
                        tensor = torch.as_tensor(value).cpu()
                        flat[key] = tensor
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Loads normalization statistics from a flat state dictionary.

        This enables loading stats from safetensors files during resume.
        """
        if not state:
            return

        # If processor doesn't exist yet, store state for later
        if self._processor is None:
            self._pending_state = state
            return

        # Reconstruct nested statistics dict from flat keys
        reconstructed: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}

        for flat_key, tensor in state.items():
            parts = flat_key.split(".")
            if len(parts) == 4:
                embodiment_tag, modality, joint_group, stat_name = parts

                if embodiment_tag not in reconstructed:
                    reconstructed[embodiment_tag] = {}
                if modality not in reconstructed[embodiment_tag]:
                    reconstructed[embodiment_tag][modality] = {}
                if joint_group not in reconstructed[embodiment_tag][modality]:
                    reconstructed[embodiment_tag][modality][joint_group] = {}

                reconstructed[embodiment_tag][modality][joint_group][stat_name] = tensor.tolist()

        if reconstructed:
            self.processor.state_action_processor.set_statistics(reconstructed, override=True)


def make_gr00t_n1d6_pre_post_processors(
    config: Gr00tN1d6Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create preprocessor and postprocessor for Groot N1.6 policy.

    This creates a processing pipeline that transforms LeRobot data format into
    the format expected by Gr00tN1d6 model.

    Args:
        config: Gr00tN1d6Config configuration
        dataset_stats: Optional dataset statistics for normalization

    Returns:
        Tuple of (preprocessor, postprocessor) pipelines
    """

    # Convert dataset_stats to format expected by processor
    # LeRobot dataset_stats has structure: {key: {stat_type: values}}
    # where key is like "observation.state", "action", "observation.images.right", etc.
    # StateActionProcessor expects: {embodiment_tag: {modality: {joint_group: {stat_type: values}}}}
    statistics = None
    if dataset_stats:
        statistics = {config.embodiment_tag: {}}

        # Map LeRobot keys to modality and joint_group
        for key, stats_dict in dataset_stats.items():
            # Skip image keys (not used by StateActionProcessor)
            if key.startswith("observation.images."):
                continue

            # Map keys to modality
            if key == "observation.state":
                modality = "state"
                joint_group = "state"  # Use the modality key as joint_group
            elif key == "action":
                modality = "action"
                joint_group = "action"  # Use the modality key as joint_group
            elif key == "relative_action":
                modality = "relative_action"
                joint_group = "action"  # Use "action" as joint_group for relative_action
            else:
                # Skip unknown keys
                continue

            # Initialize modality dict if needed
            if modality not in statistics[config.embodiment_tag]:
                statistics[config.embodiment_tag][modality] = {}

            # Convert stats_dict to list format
            statistics[config.embodiment_tag][modality][joint_group] = {}
            for stat_type, tensor in stats_dict.items():
                if isinstance(tensor, torch.Tensor):
                    statistics[config.embodiment_tag][modality][joint_group][stat_type] = (
                        tensor.cpu().tolist()
                    )
                else:
                    statistics[config.embodiment_tag][modality][joint_group][stat_type] = tensor

    # Create basic modality configs from config
    # This is a simplified version - in production, these should come from the pretrained model
    from lerobot.policies.gr00t_n1d6.utils import ModalityConfig

    # Create basic modality configs for the embodiment tag
    modality_configs = {
        config.embodiment_tag: {
            "state": ModalityConfig(
                delta_indices=[0],  # Single timestep
                modality_keys=["state"],  # Single state key
            ),
            "action": ModalityConfig(
                delta_indices=list(range(config.chunk_size)),  # Action horizon
                modality_keys=["action"],  # Single action key
            ),
            "video": ModalityConfig(
                delta_indices=[0],  # Single timestep
                modality_keys=["image"],  # Default image key
            ),
        }
    }

    # Create processor instance
    processor = Gr00tN1d6Processor(
        modality_configs=modality_configs,
        statistics=statistics,
        formalize_language=config.formalize_language,
        model_name=config.tokenizer_assets_repo or "nvidia/Eagle-Block2A-2B-v2",
        max_state_dim=config.max_state_dim,
        max_action_dim=config.max_action_dim,
        max_action_horizon=config.chunk_size,
        use_albumentations=config.use_albumentations_transforms,
        use_relative_action=config.use_relative_action,
        apply_sincos_state_encoding=config.apply_sincos_state_encoding,
        embodiment_id_mapping={config.embodiment_tag: 0},  # Simplified mapping
        # Add missing image transformation parameters
        image_target_size=(list(config.image_target_size) if config.image_target_size else [224, 224]),
        image_crop_size=(list(config.image_crop_size) if config.image_crop_size else [244, 244]),
        shortest_image_edge=(config.shortest_image_edge if config.shortest_image_edge else 256),
        crop_fraction=config.crop_fraction if config.crop_fraction else 0.95,
        random_rotation_angle=config.random_rotation_angle,
        color_jitter_params=config.color_jitter_params,
    )

    # Preprocessing pipeline
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        Gr00tN1d6ProcessStep(processor=processor, language_key="task"),
        DeviceProcessorStep(device=config.device),
    ]

    # Postprocessing pipeline
    # For N1.6, we need to decode actions using processor.decode_action
    # Create a simple step that slices to env action dim and moves to CPU
    output_steps: list[ProcessorStep] = [
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
