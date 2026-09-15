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
from lerobot.processor import ObservationProcessorStep, ProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

from .constants import MODEL_INPUT_PREFIX, STATE_BINS
from .core.adapter import build_meta, normalize_task_batch
from .core.tokenization import DM05Tokenization, action_to_bin_tokens
from .core.utils import DM05_STATE_BINS

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor
else:
    AutoProcessor = None


@ProcessorStepRegistry.register(name="dm05_clip_normalized_processor")
@dataclass
class DM05ClipNormalizedProcessorStep(ProcessorStep):
    """Clamp quantile-normalized state and action into the range DM05 was trained on.

    Runs after `DM05ActionReferenceExtractProcessorStep` so the temporary probes are already
    consumed; clipping them would corrupt the relative-action reference offset. Whether each
    field is clipped is decided once, at pipeline construction, from `norm_clip` and the
    normalization mapping, so a reloaded pipeline cannot silently disagree with the checkpoint.
    """

    clip_state: bool = False
    clip_action: bool = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        result = transition.copy()
        if self.clip_state:
            observation = result.get(TransitionKey.OBSERVATION)
            state = observation.get(OBS_STATE) if isinstance(observation, dict) else None
            if state is not None:
                result[TransitionKey.OBSERVATION] = {
                    **observation,
                    OBS_STATE: torch.as_tensor(state).clamp(-1.0, 1.0),
                }
        if self.clip_action:
            action = result.get(TransitionKey.ACTION)
            if action is not None:
                result[TransitionKey.ACTION] = torch.as_tensor(action).clamp(-1.0, 1.0)
        return result

    def get_config(self) -> dict[str, Any]:
        """Return the serializable processor-step configuration."""
        return {"clip_state": self.clip_state, "clip_action": self.clip_action}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


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


@ProcessorStepRegistry.register(name="dm05_tokenizer_processor")
@dataclass
class DM05TokenizerProcessorStep(ObservationProcessorStep):
    """Build Gemma3 inputs from a normalized, device-resident observation.

    Runs last in the input pipeline, after `DeviceProcessorStep`, so the saved
    `policy_preprocessor.json` describes the whole path from a raw batch to model-ready
    inputs. The Gemma assets travel with the pipeline through `save_artifacts`.
    """

    processor_name_or_path: str
    tokenizer_max_length: int | None = None
    add_state: bool = True
    image_keys: list[str] | None = None
    default_task: str = "Execute the robot action."

    # Injectable so a caller can hand over an already-loaded processor; never serialized.
    processor: Any = field(default=None, repr=False)
    _tokenization: Any = field(default=None, init=False, repr=False)

    def _load_processor(self) -> Any:
        """Load the Gemma processor on first use so building a pipeline stays offline."""
        if self.processor is None:
            require_package("transformers", extra="dm05")
            self.processor = AutoProcessor.from_pretrained(
                self.processor_name_or_path,
                fix_mistral_regex=False,
            )
        return self.processor

    @property
    def tokenization(self) -> DM05Tokenization:
        """Return the lazily built DM05 chat-template tokenizer."""
        if self._tokenization is None:
            self._tokenization = DM05Tokenization(
                processor=self._load_processor(),
                max_length=self.tokenizer_max_length,
                add_state=self.add_state,
            )
        return self._tokenization

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Tokenize prompt, images, and optional normalized state bins."""
        if OBS_STATE not in observation:
            raise ValueError(f"DM05 requires `{OBS_STATE}` after preprocessing.")
        state = torch.as_tensor(observation[OBS_STATE])
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2:
            raise ValueError(f"DM05 expects batched state [B,D], got {tuple(state.shape)}.")

        # The configured keys are the whole camera set: an observation carrying an extra camera must
        # not silently change the image count or the rendered prompt.
        if not self.image_keys:
            raise ValueError(
                "DM05 has no cameras configured: declare them in input_features or set policy.image_keys."
            )
        image_keys = list(self.image_keys)
        if missing := [key for key in image_keys if key not in observation]:
            present = sorted(key for key in observation if key.startswith(OBS_IMAGES))
            raise ValueError(f"DM05 expects images at {missing}; the observation has {present}.")
        image_batches = []
        for key in image_keys:
            images = observation[key]
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
        complementary_data = self.transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        state_bins = complementary_data.get(STATE_BINS)
        if self.add_state and (not isinstance(state_bins, list) or len(state_bins) != batch_size):
            raise ValueError("DM05 state bins are missing or do not match the batch size.")
        tasks = normalize_task_batch(complementary_data.get("task"), batch_size, self.default_task)
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
        tokenized = self.tokenization.tokenize_robot_batch(samples)
        device = image_batches[0].device
        return {
            **observation,
            **{f"{MODEL_INPUT_PREFIX}{key}": value.to(device) for key, value in tokenized.items()},
        }

    def get_config(self) -> dict[str, Any]:
        """Return the serializable processor-step configuration."""
        return {
            "processor_name_or_path": self.processor_name_or_path,
            "tokenizer_max_length": self.tokenizer_max_length,
            "add_state": self.add_state,
            "image_keys": list(self.image_keys) if self.image_keys else None,
            "default_task": self.default_task,
        }

    def save_artifacts(self, save_directory: Path) -> dict[str, str]:
        """Save the Gemma assets so a checkpoint reloads its own tokenization offline."""
        artifact_path = Path("dm05_processor")
        target = save_directory / artifact_path
        if not (target / "processor_config.json").exists():
            self._load_processor().save_pretrained(target)
        return {"processor_name_or_path": artifact_path.as_posix()}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Leave policy feature metadata unchanged.

        The tokenized sequence is padded to the longest sample in the batch, so its length
        varies per batch and no honest fixed shape can be declared here.
        """
        return features
