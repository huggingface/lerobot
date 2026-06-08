#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import require_package

from .configuration_cosmos3 import Cosmos3Config

COSMOS3_VIDEO = "cosmos3_video"
COSMOS3_ACTION_CONDITION = "cosmos3_action_condition"
COSMOS3_ACTION_CONDITION_MASK = "cosmos3_action_condition_mask"
COSMOS3_ACTION_DOMAIN_ID = "cosmos3_action_domain_id"
COSMOS3_CONDITIONING_FPS = "cosmos3_conditioning_fps"
COSMOS3_RAW_ACTION_DIM = "cosmos3_raw_action_dim"
COSMOS3_CLEAN_ACTION = "cosmos3_clean_action"
COSMOS3_PROMPT = "cosmos3_prompt"
COSMOS3_FORMATTED_PROMPT = "cosmos3_formatted_prompt"
COSMOS3_COND_INPUT_IDS = "cosmos3_cond_input_ids"
COSMOS3_UNCOND_INPUT_IDS = "cosmos3_uncond_input_ids"
COSMOS3_TRAINING_SIGMA = "cosmos3_training_sigma"

COSMOS3_VISION_START_TOKEN = "<|vision_start|>"  # nosec B105
COSMOS3_VISION_END_TOKEN = "<|vision_end|>"  # nosec B105

_VIEWPOINT_TEMPLATES = {
    "concat_view": "This video contains concatenated views from multiple camera perspectives.",
    "ego_view": "This video is captured from a first-person perspective looking at the scene.",
    "third_person_view": "This video is captured from a third-person perspective looking towards the agent from the front.",
    "wrist_view": "This video is captured from a wrist-mounted camera.",
}


def _as_batched_image_tensor(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    elif image.ndim != 4:
        raise ValueError(f"Expected image tensor with 3 or 4 dims, got shape={tuple(image.shape)}")

    if image.shape[-1] in {1, 3, 4}:
        image = image.permute(0, 3, 1, 2)
    elif image.shape[1] not in {1, 3, 4}:
        raise ValueError(f"Could not infer image channel dimension from shape={tuple(image.shape)}")
    if image.shape[1] == 4:
        image = image[:, :3]
    if image.shape[1] == 1:
        image = image.expand(-1, 3, -1, -1)
    return image.contiguous()


def _as_batched_image_sequence_tensor(
    image: torch.Tensor, *, sequence_length: int | None = None
) -> torch.Tensor:
    if image.ndim == 5:
        if image.shape[2] in {1, 3, 4}:
            image = image
        elif image.shape[-1] in {1, 3, 4}:
            image = image.permute(0, 1, 4, 2, 3)
        else:
            raise ValueError(f"Could not infer image channel dimension from shape={tuple(image.shape)}")
        if image.shape[2] == 4:
            image = image[:, :, :3]
        if image.shape[2] == 1:
            image = image.expand(-1, -1, 3, -1, -1)
        return image.contiguous()

    if image.ndim == 4 and sequence_length is not None and image.shape[0] == sequence_length:
        if image.shape[1] in {1, 3, 4}:
            image = image.unsqueeze(0)
        elif image.shape[-1] in {1, 3, 4}:
            image = image.permute(0, 3, 1, 2).unsqueeze(0)
        else:
            raise ValueError(f"Could not infer image channel dimension from shape={tuple(image.shape)}")
        if image.shape[2] == 4:
            image = image[:, :, :3]
        if image.shape[2] == 1:
            image = image.expand(-1, -1, 3, -1, -1)
        return image.contiguous()

    image = _as_batched_image_tensor(image)
    return image.unsqueeze(1)


def _to_uint8_nthwc(image: torch.Tensor, *, sequence_length: int | None = None) -> np.ndarray:
    image = _as_batched_image_sequence_tensor(image, sequence_length=sequence_length)
    if image.dtype.is_floating_point:
        image = image.clamp(0.0, 1.0).mul(255.0)
    image = image.round().clamp(0, 255).to(torch.uint8)
    return image.permute(0, 1, 3, 4, 2).cpu().numpy()


def resize_with_pad_uint8(images: np.ndarray, height: int, width: int) -> np.ndarray:
    """Match RoboLab's openpi-client resize_with_pad behavior for NHWC uint8 images."""
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int) -> np.ndarray:
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.array(image)

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=Image.BILINEAR)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    return np.array(zero_image)


def compose_cosmos3_views(
    views: list[torch.Tensor],
    *,
    primary_height: int,
    composed_height: int,
    composed_width: int,
    sequence_length: int | None = None,
) -> torch.Tensor:
    """Tile camera views onto a single uint8 canvas.

    ``views[0]`` is the primary view rendered full-width across the top
    ``primary_height`` rows; the remaining views are resized and concatenated
    left-to-right across the bottom ``composed_height - primary_height`` rows.
    The layout generalizes to any number of views (RoboLab's wrist-on-top,
    two-shoulder-bottom layout is the three-view default).

    Returns ``[N, composed_height, composed_width, 3]`` for single-frame inputs
    or ``[N, T, composed_height, composed_width, 3]`` for sequences.
    """
    if not views:
        raise ValueError("compose_cosmos3_views requires at least one view.")

    def _resize(view: torch.Tensor, height: int, width: int) -> np.ndarray:
        return resize_with_pad_uint8(_to_uint8_nthwc(view, sequence_length=sequence_length), height, width)

    if len(views) == 1:
        composed = _resize(views[0], composed_height, composed_width)
    else:
        primary = _resize(views[0], primary_height, composed_width)
        bottom_views = views[1:]
        bottom_height = composed_height - primary_height
        tile_width = composed_width // len(bottom_views)
        tiles = [_resize(view, bottom_height, tile_width) for view in bottom_views]
        bottom = np.concatenate(tiles, axis=-2)
        if bottom.shape[-2] < composed_width:
            pad_width = composed_width - bottom.shape[-2]
            pad_spec = [(0, 0)] * (bottom.ndim - 2) + [(0, pad_width), (0, 0)]
            bottom = np.pad(bottom, pad_spec)
        composed = np.concatenate((primary, bottom), axis=-3)

    composed_tensor = torch.from_numpy(composed)
    if composed_tensor.shape[1] == 1:  # squeeze the singleton time axis for single frames
        return composed_tensor[:, 0]
    return composed_tensor


def _normalise_prompt_list(prompts: Any, batch_size: int) -> list[str]:
    if isinstance(prompts, str):
        return [prompts] * batch_size
    if isinstance(prompts, tuple):
        prompts = list(prompts)
    if isinstance(prompts, list) and len(prompts) == batch_size and all(isinstance(p, str) for p in prompts):
        return prompts
    raise ValueError(f"Expected a prompt string or list[str] with batch_size={batch_size}, got {prompts!r}")


def _append_sentence(text: str, addition: str) -> str:
    if not addition:
        return text
    if not text:
        return addition
    separator = " " if text.rstrip().endswith(".") else ". "
    return text.rstrip() + separator + addition


def classify_cosmos3_action_size(
    source_height: int,
    source_width: int,
    *,
    resolution_tier: int,
) -> tuple[int, int, int, int]:
    # Reuse Cosmos3's trained resolution buckets straight from Diffusers as the
    # single source of truth. This is a deliberate exception to the "no Diffusers
    # internals" rule for one model-defined constant; the diffusers commit is
    # pinned in pyproject. Imported lazily to keep diffusers an optional extra.
    require_package("diffusers", extra="cosmos3")
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import _ACTION_RESOLUTION_BINS

    resolution_key = str(resolution_tier)
    if resolution_key not in _ACTION_RESOLUTION_BINS:
        raise ValueError(
            f"Unsupported action resolution_tier={resolution_tier!r}; "
            f"expected one of {sorted(int(k) for k in _ACTION_RESOLUTION_BINS)}."
        )

    aspect_ratio = source_height / source_width
    ratios = _ACTION_RESOLUTION_BINS[resolution_key]
    target_height, target_width = min(
        ratios.values(),
        key=lambda size: abs((size[0] / size[1]) - aspect_ratio),
    )
    scale = min(target_width / source_width, target_height / source_height, 1.0)
    content_height = max(1, int(scale * source_height + 0.5))
    content_width = max(1, int(scale * source_width + 0.5))
    return target_height, target_width, content_height, content_width


def format_cosmos3_action_prompt(
    prompt: str,
    *,
    viewpoint: str,
    additional_view_description: str,
    num_frames: int,
    height: int,
    width: int,
    fps: float,
) -> str:
    caption = prompt.rstrip()
    viewpoint_text = _VIEWPOINT_TEMPLATES.get(viewpoint)
    if viewpoint_text is not None:
        additional_view_description = additional_view_description.rstrip()
        if additional_view_description:
            viewpoint_text = _append_sentence(viewpoint_text, additional_view_description)
        caption = _append_sentence(caption, viewpoint_text)

    duration = int(num_frames / fps) if fps > 0 else 0
    caption = _append_sentence(caption, f"The video is {duration:.1f} seconds long and is of {fps:.0f} FPS.")
    caption = _append_sentence(caption, f"This video is of {height}x{width} resolution.")
    return caption


def add_cosmos3_special_tokens(tokenizer: Any) -> dict[str, int]:
    existing_special_tokens: list[str] = []
    for value in tokenizer.special_tokens_map.values():
        if isinstance(value, str):
            existing_special_tokens.append(value)
        elif isinstance(value, list):
            existing_special_tokens.extend(value)

    tokens_to_add = []
    if COSMOS3_VISION_START_TOKEN not in existing_special_tokens:
        tokens_to_add.append(COSMOS3_VISION_START_TOKEN)
    if COSMOS3_VISION_END_TOKEN not in existing_special_tokens:
        tokens_to_add.append(COSMOS3_VISION_END_TOKEN)
    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add)

    return {
        "start_of_generation": tokenizer.convert_tokens_to_ids(COSMOS3_VISION_START_TOKEN),
        "end_of_generation": tokenizer.convert_tokens_to_ids(COSMOS3_VISION_END_TOKEN),
        "eos_token_id": tokenizer.eos_token_id,
    }


@dataclass
@ProcessorStepRegistry.register(name="cosmos3_pack_inputs")
class Cosmos3PackInputsStep(ComplementaryDataProcessorStep):
    """Pack raw multi-view observations into Cosmos3 model inputs.

    Generic over the data source: any ordered ``image_keys`` are composed into a
    single conditioning video, and the proprioceptive state is validated/padded
    against ``max_state_dim``. RoboLab's DROID cameras are only the default config.
    """

    image_keys: list[str]
    num_views: int
    image_height: int
    chunk_size: int
    raw_action_dim: int
    max_action_dim: int
    max_state_dim: int
    use_state: bool
    history_length: int
    invert_gripper: bool
    domain_id: int
    conditioning_fps: float
    resolution_tier: int
    viewpoint: str
    additional_view_description: str
    prompt_key: str = "task"
    composed_image_height: int = 540
    composed_image_width: int = 640

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required for Cosmos3PackInputsStep.")

        views = []
        for image_key in self.image_keys:
            view = observation.get(image_key)
            if view is None:
                raise ValueError(
                    f"Cosmos3 expected image observation {image_key!r} (image_keys={self.image_keys})."
                )
            views.append(view)
        if len(views) > self.num_views:
            raise ValueError(f"Got {len(views)} image views but num_views={self.num_views}.")
        while len(views) < self.num_views:  # pad missing views with a blank frame
            views.append(torch.zeros_like(views[0]))

        composed = compose_cosmos3_views(
            views,
            primary_height=self.image_height,
            composed_height=self.composed_image_height,
            composed_width=self.composed_image_width,
            sequence_length=self.chunk_size + 1,
        )
        if composed.shape[-3:-1] != (self.composed_image_height, self.composed_image_width):
            raise ValueError(
                "Unexpected composed Cosmos3 image shape: "
                f"{tuple(composed.shape[-3:-1])}, expected {(self.composed_image_height, self.composed_image_width)}."
            )

        batch_size = composed.shape[0]
        if composed.ndim == 4:
            video = torch.zeros(
                batch_size,
                3,
                self.chunk_size + 1,
                self.composed_image_height,
                self.composed_image_width,
                dtype=torch.uint8,
            )
            video[:, :, 0] = composed.permute(0, 3, 1, 2)
        elif composed.ndim == 5:
            if composed.shape[1] < self.chunk_size + 1:
                pad = composed[:, -1:].expand(-1, self.chunk_size + 1 - composed.shape[1], -1, -1, -1)
                composed = torch.cat([composed, pad], dim=1)
            composed = composed[:, : self.chunk_size + 1]
            video = composed.permute(0, 4, 1, 2, 3).contiguous()
        else:
            raise ValueError(f"Unexpected composed Cosmos3 image rank: {composed.ndim}")

        action_len = self.chunk_size + int(self.use_state)
        action_condition = torch.zeros(batch_size, action_len, self.raw_action_dim, dtype=torch.float32)
        action_condition_mask = torch.zeros(batch_size, action_len, 1, dtype=torch.float32)
        if self.use_state:
            state = observation.get(OBS_STATE)
            if state is None:
                raise ValueError(f"{OBS_STATE} is required when Cosmos3 use_state=True.")
            if state.ndim == 3:
                state = state[:, 0]
            if state.ndim == 2 and state.shape[0] != batch_size and state.shape[0] == self.chunk_size + 1:
                state = state[:1]
            if state.ndim == 1:
                state = state.unsqueeze(0)
            state = state.to(dtype=torch.float32)
            if state.shape[0] != batch_size:
                raise ValueError("Batch size mismatch between Cosmos3 images and state.")
            state_dim = state.shape[-1]
            if state_dim > self.max_state_dim:
                raise ValueError(
                    f"Cosmos3 state width {state_dim} exceeds max_state_dim={self.max_state_dim}."
                )
            if state_dim < self.raw_action_dim:
                state = F.pad(state, (0, self.raw_action_dim - state_dim))
            else:
                state = state[:, : self.raw_action_dim].clone()
            if self.invert_gripper:
                state[:, -1] = 1.0 - state[:, -1]
            action_condition[:, 0] = state
            action_condition_mask[:, 0, 0] = 1.0

        action = self.transition.get(TransitionKey.ACTION)
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(
                    f"Cosmos3 training action must have shape [B,T,D], got {tuple(action.shape)}."
                )
            if action.shape[0] != batch_size:
                raise ValueError("Batch size mismatch between Cosmos3 images and actions.")
            action = action[:, : self.chunk_size, : self.raw_action_dim].to(dtype=torch.float32).clone()
            if self.invert_gripper:
                action[:, :, -1] = 1.0 - action[:, :, -1]
            clean_action = torch.zeros(batch_size, action_len, self.max_action_dim, dtype=torch.float32)
            clean_action[:, :, : self.raw_action_dim] = action_condition
            future_start = int(self.use_state)
            clean_action[:, future_start : future_start + action.shape[1], : self.raw_action_dim] = action
            complementary_data[COSMOS3_CLEAN_ACTION] = clean_action

        prompts = _normalise_prompt_list(complementary_data.get(self.prompt_key), batch_size)
        target_height, target_width, _content_height, _content_width = classify_cosmos3_action_size(
            self.composed_image_height,
            self.composed_image_width,
            resolution_tier=self.resolution_tier,
        )
        formatted_prompts = [
            format_cosmos3_action_prompt(
                prompt,
                viewpoint=self.viewpoint,
                additional_view_description=self.additional_view_description,
                num_frames=self.chunk_size + 1,
                height=target_height,
                width=target_width,
                fps=self.conditioning_fps,
            )
            for prompt in prompts
        ]
        complementary_data[COSMOS3_PROMPT] = prompts
        complementary_data[COSMOS3_FORMATTED_PROMPT] = formatted_prompts
        complementary_data[COSMOS3_VIDEO] = video
        complementary_data[COSMOS3_ACTION_CONDITION] = action_condition
        complementary_data[COSMOS3_ACTION_CONDITION_MASK] = action_condition_mask
        complementary_data[COSMOS3_ACTION_DOMAIN_ID] = torch.full(
            (batch_size,), self.domain_id, dtype=torch.long
        )
        complementary_data[COSMOS3_CONDITIONING_FPS] = torch.full(
            (batch_size,), self.conditioning_fps, dtype=torch.float32
        )
        complementary_data[COSMOS3_RAW_ACTION_DIM] = torch.full(
            (batch_size,), self.raw_action_dim, dtype=torch.long
        )
        return complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "image_keys": self.image_keys,
            "num_views": self.num_views,
            "image_height": self.image_height,
            "chunk_size": self.chunk_size,
            "raw_action_dim": self.raw_action_dim,
            "max_action_dim": self.max_action_dim,
            "max_state_dim": self.max_state_dim,
            "use_state": self.use_state,
            "history_length": self.history_length,
            "invert_gripper": self.invert_gripper,
            "domain_id": self.domain_id,
            "conditioning_fps": self.conditioning_fps,
            "resolution_tier": self.resolution_tier,
            "viewpoint": self.viewpoint,
            "additional_view_description": self.additional_view_description,
            "prompt_key": self.prompt_key,
            "composed_image_height": self.composed_image_height,
            "composed_image_width": self.composed_image_width,
        }


@dataclass
@ProcessorStepRegistry.register(name="cosmos3_qwen_prompt_tokenizer")
class Cosmos3QwenPromptTokenizerStep(ComplementaryDataProcessorStep):
    processor_name: str
    local_files_only: bool = True

    _tokenizer: Any | None = field(default=None, init=False, repr=False)
    _special_tokens: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        require_package("transformers", extra="cosmos3")
        try:
            from transformers import Qwen3VLProcessor

            processor = Qwen3VLProcessor.from_pretrained(
                self.processor_name,
                local_files_only=self.local_files_only,
            )
            tokenizer = processor.tokenizer
        except (OSError, ValueError, ImportError):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.processor_name,
                local_files_only=self.local_files_only,
            )

        self._tokenizer = tokenizer
        self._special_tokens = add_cosmos3_special_tokens(tokenizer)

    def _tokenize(self, text: str) -> torch.Tensor:
        input_ids = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=False,
            return_dict=False,
        )
        return torch.tensor(input_ids, dtype=torch.long)

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        prompts = complementary_data.get(COSMOS3_FORMATTED_PROMPT)
        if prompts is None:
            raise ValueError(f"{COSMOS3_FORMATTED_PROMPT} is required for Cosmos3 prompt tokenization.")
        if isinstance(prompts, str):
            prompts = [prompts]

        cond_ids = [self._tokenize(prompt) for prompt in prompts]
        uncond_ids = [self._tokenize("") for _ in prompts]
        complementary_data[COSMOS3_COND_INPUT_IDS] = cond_ids
        complementary_data[COSMOS3_UNCOND_INPUT_IDS] = uncond_ids
        return complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "processor_name": self.processor_name,
            "local_files_only": self.local_files_only,
        }


def make_cosmos3_pre_post_processors(
    config: Cosmos3Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    config.validate_features()

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Cosmos3PackInputsStep(
            image_keys=config.image_keys,
            num_views=config.num_views,
            image_height=config.image_height,
            composed_image_height=config.composed_image_height,
            composed_image_width=config.composed_image_width,
            chunk_size=config.chunk_size,
            raw_action_dim=config.raw_action_dim,
            max_action_dim=config.max_action_dim,
            max_state_dim=config.max_state_dim,
            use_state=config.use_state,
            history_length=config.history_length,
            invert_gripper=config.invert_gripper,
            domain_id=config.domain_id,
            conditioning_fps=config.conditioning_fps,
            resolution_tier=config.resolution_tier,
            viewpoint=config.viewpoint,
            additional_view_description=config.additional_view_description,
            prompt_key=config.prompt_key,
        ),
    ]
    if config.text_processor_name_or_path is not None:
        input_steps.append(
            Cosmos3QwenPromptTokenizerStep(
                processor_name=config.text_processor_name_or_path,
                local_files_only=config.local_files_only,
            )
        )
    input_steps.append(DeviceProcessorStep(device=config.device))

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
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
