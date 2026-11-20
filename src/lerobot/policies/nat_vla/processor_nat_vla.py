# Copyright 2026 Ahad Jawaid and The HuggingFace Inc. team. All rights reserved.
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

"""Processors of the NAT VLA policy: the prompt token ids and image inputs of the VLM are built here, so the
saved pipeline holds everything the model reads."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    TransitionKey,
    load_pretrained_policy_processors,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

from .configuration_nat_vla import NATVLAConfig, downloaded

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor
else:
    AutoProcessor = None

IMAGE_INPUTS = ("pixel_values", "pixel_attention_mask", "spatial_shapes")


def make_nat_vla_pre_post_processors(
    config: NATVLAConfig,
    dataset_stats: dict[str, dict[str, Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """The preprocessor renames the observations, adds the batch dimension, moves the batch to the device,
    normalizes it and builds the prompt token ids and image inputs. The postprocessor unnormalizes the actions
    and moves them to the CPU."""
    steps = make_default_policy_processor_steps(config, dataset_stats)
    processor_folder = downloaded(config.vlm, ["vlm/*"], config.vlm_revision) / "vlm"
    vlm_config = config.vlm_config or vlm_config_of(processor_folder)
    tokenizer_step = NATVLATokenizerProcessorStep(
        processor_dir=str(processor_folder),
        text_vocabulary_size=vlm_config["text_config"]["vocab_size"],
        **prompt_settings(config),
    )
    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.to_device,
        steps.normalize,
        tokenizer_step,
        DeviceProcessorStep(device=steps.to_device.device),
    ]
    return make_policy_processor_pipelines(
        input_steps=input_steps, output_steps=[steps.unnormalize, steps.to_cpu]
    )


def make_nat_vla_pre_post_processors_from_pretrained(
    config: NATVLAConfig,
    pretrained_path: str | Path,
    *,
    dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    dataset_meta: Any = None,
    preprocessor_overrides: dict[str, Any] | None = None,
    **load_options: Any,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Loads a checkpoint's processors with the prompt settings of the active config over the saved ones, so a
    fine-tune can change the cameras, setup and control mode. The dataset statistics arrive as normalizer
    overrides; explicit overrides win."""
    overrides = dict(preprocessor_overrides or {})
    tokenizer_step = "nat_vla_tokenizer_processor"
    overrides[tokenizer_step] = prompt_settings(config) | overrides.get(tokenizer_step, {})
    return load_pretrained_policy_processors(
        pretrained_path, preprocessor_overrides=overrides, **load_options
    )


def prompt_settings(config: NATVLAConfig) -> dict[str, Any]:
    return {
        "camera_keys": list(config.image_features),
        "setup": config.setup,
        "control_mode": config.control_mode,
        "state_bins": config.state_bins,
    }


@dataclass
@ProcessorStepRegistry.register(name="nat_vla_tokenizer_processor")
class NATVLATokenizerProcessorStep(ComplementaryDataProcessorStep):
    """Builds each sample's VLM inputs: the prompt token ids (the camera images, the task, the setup, the binned
    state and the control mode, ending with <action_output> and <action_start>), left-padded so the policy
    appends the action tokens, and the processed images. The VLM processor is saved with the pipeline as
    `vlm/`."""

    processor_dir: str
    text_vocabulary_size: int
    camera_keys: list[str]
    setup: str = ""
    control_mode: str = ""
    state_bins: int = 256

    _processor: Any = field(default=None, init=False, repr=False)
    _vocabulary: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        require_package("transformers", extra="nat_vla")
        self._processor = AutoProcessor.from_pretrained(self.processor_dir)
        self._vocabulary = Vocabulary(self.text_vocabulary_size, self.state_bins, action_tokens=0)

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        observation = self.transition[TransitionKey.OBSERVATION] or {}
        tasks = complementary_data["task"]
        cameras = [(observation[key] * 255).round().clamp(0, 255).to(torch.uint8) for key in self.camera_keys]
        frames = [camera[sample] for sample in range(len(tasks)) for camera in cameras]
        images = self._processor.image_processor(frames, return_row_col_info=True, return_tensors="pt")
        image_prompts = [
            self._processor.replace_image_token(images, index, use_image_special_tokens=True)
            for index in range(len(frames))
        ]

        states = observation.get(OBS_STATE, [None] * len(tasks))
        prompts = [
            self.prompt_tokens(
                task, image_prompts[sample * len(cameras) : (sample + 1) * len(cameras)], states[sample]
            )
            for sample, task in enumerate(tasks)
        ]
        input_ids, attention_mask = left_padded_token_ids(prompts, self._processor.tokenizer.pad_token_id)

        complementary_data["input_ids"], complementary_data["attention_mask"] = input_ids, attention_mask
        complementary_data |= {name: images[name] for name in IMAGE_INPUTS}
        return complementary_data

    def prompt_tokens(self, task: str, image_prompts: list[str], state: Tensor | None) -> list[int]:
        vocabulary, tokenizer = self._vocabulary, self._processor.tokenizer
        images = "".join(f"Image {number}{prompt}" for number, prompt in enumerate(image_prompts, 1))

        prompt_parts: list[str | int] = [
            f"{tokenizer.bos_token}{images}<|im_start|>user\nThe task is to {task}. The setup is "
        ]
        if self.setup:
            prompt_parts += [vocabulary.setup_start, self.setup, vocabulary.setup_end]

        if state is not None:
            state_tokens = (state_bin_indices(state, self.state_bins) + vocabulary.first_state_bin).tolist()
            prompt_parts += [
                ". The current state of the robot is ",
                vocabulary.state_start,
                *state_tokens,
                vocabulary.state_end,
            ]

        prompt_parts.append(". The expected control mode is ")
        if self.control_mode:
            prompt_parts += [vocabulary.control_start, self.control_mode, vocabulary.control_end]

        prompt_parts += [
            ". Given these, what action should the robot take to complete the task?"
            f"{tokenizer.eos_token}\n<|im_start|>assistant\n",
            vocabulary.action_output,
            vocabulary.action_start,
        ]

        encoded_parts = (
            [part] if isinstance(part, int) else tokenizer.encode(part, add_special_tokens=False)
            for part in prompt_parts
        )
        return [token for part in encoded_parts for token in part]

    def get_config(self) -> dict[str, Any]:
        return {
            "processor_dir": self.processor_dir,
            "text_vocabulary_size": self.text_vocabulary_size,
            "camera_keys": self.camera_keys,
            "setup": self.setup,
            "control_mode": self.control_mode,
            "state_bins": self.state_bins,
        }

    def save_artifacts(self, save_directory: Path) -> dict[str, str]:
        """Save the VLM processor with the pipeline as `vlm/`."""
        self._processor.save_pretrained(save_directory / "vlm")
        return {"processor_dir": "vlm"}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


class Vocabulary:
    """Token ids of the VLM's text vocabulary followed by the added ones: prompt markers, state bins and NAT
    action tokens."""

    def __init__(self, text_vocabulary_size: int, state_bins: int, action_tokens: int) -> None:
        start = text_vocabulary_size
        self.setup_start, self.setup_end, self.control_start, self.control_end = range(start, start + 4)
        self.state_start, self.state_end, self.first_state_bin = range(start + 4, start + 7)

        self.action_output = self.first_state_bin + state_bins
        self.action_start, self.action_end, self.first_action_token = range(
            self.action_output + 1, self.action_output + 4
        )

        self.size = self.first_action_token + action_tokens


def vlm_config_of(processor_folder: Path) -> dict[str, Any]:
    """The VLM's architecture config in processor_folder, with the tokenizer's pad id."""
    vlm_config = json.loads((processor_folder / "config.json").read_text())
    pad_token_id = AutoProcessor.from_pretrained(processor_folder).tokenizer.pad_token_id
    vlm_config["text_config"]["pad_token_id"] = pad_token_id
    return vlm_config


def left_padded_token_ids(sequences: list[list[int]], pad_token_id: int) -> tuple[Tensor, Tensor]:
    """Token id sequences padded on the left with pad_token_id, and their attention mask."""
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    positions = torch.arange(int(lengths.max()))
    mask = positions >= len(positions) - lengths[:, None]

    ids = torch.full(mask.shape, pad_token_id, dtype=torch.long)
    ids[mask] = torch.tensor([token for sequence in sequences for token in sequence])
    return ids, mask.long()


def state_bin_indices(state: Tensor, bins: int) -> Tensor:
    """State in [-1, 1] as bin indices."""
    state = torch.nan_to_num(state.float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)
    return torch.round((state + 1) / 2 * (bins - 1)).long()
