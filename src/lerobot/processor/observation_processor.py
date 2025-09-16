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
from dataclasses import dataclass

import einops
import numpy as np
import torch
from torch import Tensor

from lerobot.configs.types import PolicyFeature
from lerobot.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE
from lerobot.processor.pipeline import ObservationProcessor, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="observation_processor")
class VanillaObservationProcessor(ObservationProcessor):
    """
    Processes environment observations into the LeRobot format by handling both images and states.

    Image processing:
        - Converts channel-last (H, W, C) images to channel-first (C, H, W)
        - Normalizes uint8 images ([0, 255]) to float32 ([0, 1])
        - Adds a batch dimension if missing
        - Supports single images and image dictionaries

    State processing:
        - Maps 'environment_state' to observation.environment_state
        - Maps 'agent_pos' to observation.state
        - Converts numpy arrays to tensors
        - Adds a batch dimension if missing
    """

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

    def _process_observation(self, observation):
        """
        Processes both image and state observations.
        """

        processed_obs = observation.copy()

        if "pixels" in processed_obs:
            pixels = processed_obs.pop("pixels")

            if isinstance(pixels, dict):
                imgs = {f"{OBS_IMAGES}.{key}": img for key, img in pixels.items()}
            else:
                imgs = {OBS_IMAGE: pixels}

            for imgkey, img in imgs.items():
                processed_obs[imgkey] = self._process_single_image(img)

        if "environment_state" in processed_obs:
            env_state_np = processed_obs.pop("environment_state")
            env_state = torch.from_numpy(env_state_np).float()
            if env_state.dim() == 1:
                env_state = env_state.unsqueeze(0)
            processed_obs[OBS_ENV_STATE] = env_state

        if "agent_pos" in processed_obs:
            agent_pos_np = processed_obs.pop("agent_pos")
            agent_pos = torch.from_numpy(agent_pos_np).float()
            if agent_pos.dim() == 1:
                agent_pos = agent_pos.unsqueeze(0)
            processed_obs[OBS_STATE] = agent_pos

        return processed_obs

    def observation(self, observation):
        return self._process_observation(observation)

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        """Transforms feature keys to a standardized contract.

        This method handles several renaming patterns:
        - Exact matches (e.g., 'pixels' -> 'OBS_IMAGE').
        - Prefixed exact matches (e.g., 'observation.pixels' -> 'OBS_IMAGE').
        - Prefix matches (e.g., 'pixels.cam1' -> 'OBS_IMAGES.cam1').
        - Prefixed prefix matches (e.g., 'observation.pixels.cam1' -> 'OBS_IMAGES.cam1').
        - environment_state -> OBS_ENV_STATE,
        - agent_pos -> OBS_STATE,
        - observation.environment_state -> OBS_ENV_STATE,
        - observation.agent_pos -> OBS_STATE
        """
        exact_pairs = {
            "pixels": OBS_IMAGE,
            "environment_state": OBS_ENV_STATE,
            "agent_pos": OBS_STATE,
        }

        prefix_pairs = {
            "pixels.": f"{OBS_IMAGES}.",
        }

        for key in list(features.keys()):
            matched_prefix = False
            for old_prefix, new_prefix in prefix_pairs.items():
                prefixed_old = f"observation.{old_prefix}"
                if key.startswith(prefixed_old):
                    suffix = key[len(prefixed_old) :]
                    features[f"{new_prefix}{suffix}"] = features.pop(key)
                    matched_prefix = True
                    break

                if key.startswith(old_prefix):
                    suffix = key[len(old_prefix) :]
                    features[f"{new_prefix}{suffix}"] = features.pop(key)
                    matched_prefix = True
                    break

            if matched_prefix:
                continue

            for old, new in exact_pairs.items():
                if key == old or key == f"observation.{old}":
                    if key in features:
                        features[new] = features.pop(key)
                        break

        return features
