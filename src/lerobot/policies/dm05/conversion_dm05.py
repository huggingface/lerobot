#!/usr/bin/env python

# Copyright 2026 Dexmal and HuggingFace Inc. team. All rights reserved.
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

"""DM05 state preprocessing, tokenization, and processor assets."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

from .constants import STATE_BINS
from .core.adapter import build_meta, get_image_keys, normalize_task_batch
from .core.tokenization import DM05Tokenization, action_to_bin_tokens
from .core.utils import DM05_STATE_BINS

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor
else:
    AutoProcessor = None


@ProcessorStepRegistry.register(name="dm05_state_bins_processor")
@dataclass
class DM05StateBinsProcessorStep(ProcessorStep):
    """Discretize normalized state before tensors move to the model device."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        state = observation.get(OBS_STATE) if isinstance(observation, dict) else None
        if state is None:
            raise ValueError("DM05 state tokenization requires observation.state.")
        state = torch.as_tensor(state)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2:
            raise ValueError(f"DM05 expects batched state [B,D], got {tuple(state.shape)}.")

        result = transition.copy()
        complementary = dict(result.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        complementary[STATE_BINS] = action_to_bin_tokens(state, DM05_STATE_BINS).tolist()
        result[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="dm05_processor_artifacts")
@dataclass
class DM05ProcessorArtifactsStep(ProcessorStep):
    """Save Gemma assets with the pipeline, including DCP-only checkpoints."""

    processor_name_or_path: str
    processor: Any = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        return transition

    def get_config(self) -> dict[str, Any]:
        return {"processor_name_or_path": self.processor_name_or_path}

    def save_artifacts(self, save_directory: Path) -> dict[str, str]:
        artifact_path = Path("dm05_processor")
        target = save_directory / artifact_path
        # Safetensors saves create this through DM05Policy first; DCP-only saves
        # serialize only the pipelines and therefore need this fallback.
        if not (target / "processor_config.json").exists():
            if self.processor is None:
                require_package("transformers", extra="dm05")
                self.processor = AutoProcessor.from_pretrained(
                    self.processor_name_or_path,
                    fix_mistral_regex=False,
                )
            self.processor.save_pretrained(target)
        return {"processor_name_or_path": artifact_path.as_posix()}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


class DM05LerobotBatchConverter:
    """Build Gemma3 inputs from a normalized, device-resident LeRobot batch."""

    def __init__(self, config: Any, processor: Any):
        self.config = config
        self._tokenization = DM05Tokenization(
            processor=processor,
            max_length=config.tokenizer_max_length,
            add_state=config.add_state,
        )

    def convert_lerobot_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Tokenize prompt, images, and optional normalized state bins."""
        if OBS_STATE not in batch:
            raise ValueError(f"DM05 requires `{OBS_STATE}` after preprocessing.")
        state = torch.as_tensor(batch[OBS_STATE])
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2:
            raise ValueError(f"DM05 expects batched state [B,D], got {tuple(state.shape)}.")

        image_keys = get_image_keys(batch, self.config.image_keys)
        if not image_keys:
            raise ValueError("DM05 requires at least one visual observation.")
        image_batches = []
        for key in image_keys:
            images = batch[key]
            if not torch.is_tensor(images):
                raise TypeError(f"DM05 expects tensor images at {key!r}, got {type(images).__name__}.")
            if images.ndim == 3:
                images = images.unsqueeze(0)
            if images.ndim != 4 or images.shape[0] != state.shape[0]:
                raise ValueError(
                    f"DM05 expects images [B,C,H,W] with B={state.shape[0]} at {key!r}, "
                    f"got {tuple(images.shape)}."
                )
            image_batches.append(images.float().div(255) if not images.is_floating_point() else images)

        batch_size = int(state.shape[0])
        state_bins = batch.get(STATE_BINS)
        if self.config.add_state and (not isinstance(state_bins, list) or len(state_bins) != batch_size):
            raise ValueError("DM05 state bins are missing or do not match the batch size.")
        tasks = normalize_task_batch(batch.get("task"), batch_size, "Execute the robot action.")
        meta = build_meta(image_keys)
        samples = [
            {
                "prompt": tasks[index],
                "images": [images[index] for images in image_batches],
                "state_bins": None if state_bins is None else state_bins[index],
                "meta_data": meta,
            }
            for index in range(batch_size)
        ]
        tokenized = self._tokenization.tokenize_robot_batch(samples)
        device = image_batches[0].device
        return {key: value.to(device) for key, value in tokenized.items()}
