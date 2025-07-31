#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field
from typing import Any

import einops
import numpy as np
import torch
from torch import Tensor

from lerobot.configs.types import PolicyFeature
from lerobot.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE
from lerobot.processor.pipeline import EnvTransition, ProcessorStepRegistry, TransitionKey


@dataclass
class ImageProcessor:
    """Process image observations from environment format to policy format.

    Converts images from:
    - Channel-last (H, W, C) to channel-first (C, H, W)
    - uint8 [0, 255] to float32 [0, 1]
    - Adds batch dimension if needed
    - Handles both single images and dictionaries of images
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)

        if observation is None:
            return transition

        processed_obs = {}

        # Copy all observations first
        for key, value in observation.items():
            processed_obs[key] = value

        # Handle pixels key if present
        pixels = observation.get("pixels")
        if pixels is not None:
            # Remove pixels from processed_obs since we'll replace it with processed images
            processed_obs.pop("pixels", None)
            # Determine image mapping
            if isinstance(pixels, dict):
                imgs = {f"{OBS_IMAGES}.{key}": img for key, img in pixels.items()}
            else:
                imgs = {OBS_IMAGE: pixels}

            # Process each image
            for imgkey, img in imgs.items():
                processed_img = self._process_single_image(img)
                processed_obs[imgkey] = processed_img

        # Return new transition with processed observation
        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = processed_obs
        return new_transition

    def _process_single_image(self, img: np.ndarray) -> Tensor:
        """Process a single image array."""
        # Convert to tensor
        img_tensor = torch.from_numpy(img)

        # Add batch dimension if needed
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Validate image format
        _, h, w, c = img_tensor.shape
        if not (c < h and c < w):
            raise ValueError(f"Expected channel-last images, but got shape {img_tensor.shape}")

        if img_tensor.dtype != torch.uint8:
            raise ValueError(f"Expected torch.uint8 images, but got {img_tensor.dtype}")

        # Convert to channel-first format
        img_tensor = einops.rearrange(img_tensor, "b h w c -> b c h w").contiguous()

        # Convert to float32 and normalize to [0, 1]
        img_tensor = img_tensor.type(torch.float32) / 255.0

        return img_tensor

    def get_config(self) -> dict[str, Any]:
        """Return configuration for serialization."""
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return state dictionary (empty for this processor)."""
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load state dictionary (no-op for this processor)."""
        pass

    def reset(self) -> None:
        """Reset processor state (no-op for this processor)."""
        pass

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        """Transforms:
        pixels -> OBS_IMAGE,
        observation.pixels -> OBS_IMAGE,
        pixels.<cam> -> OBS_IMAGES.<cam>,
        observation.pixels.<cam> -> OBS_IMAGES.<cam>
        """
        if "pixels" in features:
            features[OBS_IMAGE] = features.pop("pixels")
        if "observation.pixels" in features:
            features[OBS_IMAGE] = features.pop("observation.pixels")

        prefixes = ("pixels.", "observation.pixels.")
        for key in list(features.keys()):
            for p in prefixes:
                if key.startswith(p):
                    suffix = key[len(p) :]
                    features[f"{OBS_IMAGES}.{suffix}"] = features.pop(key)
                    break
        return features


@dataclass
class StateProcessor:
    """Process state observations from environment format to policy format.

    Handles:
    - environment_state -> observation.environment_state
    - agent_pos -> observation.state
    - Converts numpy arrays to tensors
    - Adds batch dimension if needed
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)

        if observation is None:
            return transition

        processed_obs = dict(observation)  # Copy existing observations

        # Process environment_state
        if "environment_state" in observation:
            env_state = torch.from_numpy(observation["environment_state"]).float()
            if env_state.dim() == 1:
                env_state = env_state.unsqueeze(0)
            processed_obs[OBS_ENV_STATE] = env_state
            # Remove original key
            del processed_obs["environment_state"]

        # Process agent_pos
        if "agent_pos" in observation:
            agent_pos = torch.from_numpy(observation["agent_pos"]).float()
            if agent_pos.dim() == 1:
                agent_pos = agent_pos.unsqueeze(0)
            processed_obs[OBS_STATE] = agent_pos
            # Remove original key
            del processed_obs["agent_pos"]

        # Return new transition with processed observation
        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = processed_obs
        return new_transition

    def get_config(self) -> dict[str, Any]:
        """Return configuration for serialization."""
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return state dictionary (empty for this processor)."""
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load state dictionary (no-op for this processor)."""
        pass

    def reset(self) -> None:
        """Reset processor state (no-op for this processor)."""
        pass

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        """Transforms:
        environment_state -> OBS_ENV_STATE,
        agent_pos -> OBS_STATE,
        observation.environment_state -> OBS_ENV_STATE,
        observation.agent_pos -> OBS_STATE
        """
        pairs = (
            ("environment_state", OBS_ENV_STATE),
            ("agent_pos", OBS_STATE),
        )
        for old, new in pairs:
            if old in features:
                features[new] = features.pop(old)
            prefixed = f"observation.{old}"
            if prefixed in features:
                features[new] = features.pop(prefixed)
        return features


@dataclass
@ProcessorStepRegistry.register(name="observation_processor")
class VanillaObservationProcessor:
    """Complete observation processor that combines image and state processing.

    This processor replicates the functionality of the original preprocess_observation
    function but in a modular, composable way that fits into the pipeline architecture.
    """

    image_processor: ImageProcessor = field(default_factory=ImageProcessor)
    state_processor: StateProcessor = field(default_factory=StateProcessor)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # First process images
        transition = self.image_processor(transition)
        # Then process state
        transition = self.state_processor(transition)
        return transition

    def get_config(self) -> dict[str, Any]:
        """Return configuration for serialization."""
        return {
            "image_processor": self.image_processor.get_config(),
            "state_processor": self.state_processor.get_config(),
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return state dictionary."""
        state = {}
        state.update({f"image_processor.{k}": v for k, v in self.image_processor.state_dict().items()})
        state.update({f"state_processor.{k}": v for k, v in self.state_processor.state_dict().items()})
        return state

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load state dictionary."""
        image_state = {
            k.replace("image_processor.", ""): v for k, v in state.items() if k.startswith("image_processor.")
        }
        state_state = {
            k.replace("state_processor.", ""): v for k, v in state.items() if k.startswith("state_processor.")
        }

        self.image_processor.load_state_dict(image_state)
        self.state_processor.load_state_dict(state_state)

    def reset(self) -> None:
        """Reset processor state."""
        self.image_processor.reset()
        self.state_processor.reset()

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        features = self.image_processor.feature_contract(features)
        features = self.state_processor.feature_contract(features)
        return features
