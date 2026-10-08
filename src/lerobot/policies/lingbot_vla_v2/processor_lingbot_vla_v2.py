# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""LingBot-VLA 2.0 policy processor.

Standard steps (rename, batch, relative actions, normalization, tokenization, device) plus
the LingBot ones: the slot mapping onto the 55-D canonical state/action layout, the Qwen3-VL
image processor and the chat template. Normalization and relative actions run on the raw
features, before the slot mapping. The postprocessor maps canonical actions back to the raw
dims, unnormalizes, restores absolute actions and moves to CPU.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torchvision.transforms.v2.functional import resize as tv_resize

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    PolicyAction,
    PolicyActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    load_pretrained_policy_processors,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import _transformers_available

from .configuration_lingbot_vla_v2 import LingbotVLAV2Config

if _transformers_available:
    from transformers import AutoImageProcessor, AutoTokenizer
else:  # pragma: no cover - transformers is an optional dependency
    AutoImageProcessor = None
    AutoTokenizer = None

DEFAULT_TASK = "Execute the robot action."

# Registry names of the custom steps (override keys for from_pretrained).
SLOT_MAPPING_STEP = "lingbot_vla_v2_slot_mapping"
INVERSE_SLOT_MAPPING_STEP = "lingbot_vla_v2_inverse_slot_mapping"
IMAGE_STEP = "lingbot_vla_v2_image"
CHAT_TEMPLATE_STEP = "lingbot_vla_v2_chat_template"


def _canonical_mask(
    spans: dict[str, list[list[int]]], canonical_joints: dict[str, int], width: int
) -> torch.Tensor:
    """(width,) bool mask of the canonical dims filled by ``spans``."""
    mask = torch.zeros(width, dtype=torch.bool)
    offset = 0
    for joint, dim in canonical_joints.items():
        real = sum(end - start for start, end in spans.get(joint, []))
        mask[offset : offset + real] = True
        offset += dim
    return mask


def _to_canonical(
    raw: torch.Tensor, spans: dict[str, list[list[int]]], canonical_joints: dict[str, int], width: int
) -> torch.Tensor:
    """Gather the raw spans of each canonical joint into a zero-padded (..., width) fp32 vector."""
    out = raw.new_zeros(*raw.shape[:-1], width, dtype=torch.float32)
    offset = 0
    for joint, dim in canonical_joints.items():
        position = offset
        for start, end in spans.get(joint, []):
            out[..., position : position + end - start] = raw[..., start:end]
            position += end - start
        offset += dim
    return out


# ---------------------------------------------------------------------------
# Custom processor steps
# ---------------------------------------------------------------------------


@dataclass
@ProcessorStepRegistry.register(name=SLOT_MAPPING_STEP)
class LingbotVLAV2SlotMappingProcessorStep(ProcessorStep):
    """Map raw ``observation.state`` / ``action`` onto the canonical layout.

    Each canonical joint (``canonical_joints`` order) gets the concatenation of its raw
    ``[start, end]`` spans, zero-padded to the joint's width and then to
    ``max_state_dim`` / ``max_action_dim``. Emits the validity masks the model uses
    (``state_joint_mask``, ``action_joint_mask``, ``joint_mask``).
    """

    state_spans: dict[str, list[list[int]]] = field(default_factory=dict)
    action_spans: dict[str, list[list[int]]] = field(default_factory=dict)
    canonical_joints: dict[str, int] = field(default_factory=dict)
    chunk_size: int = 50
    max_state_dim: int = 55
    max_action_dim: int = 55

    def __call__(self, transition):
        transition = transition.copy()
        new_obs = dict(transition[TransitionKey.OBSERVATION])
        state = new_obs[OBS_STATE]
        batch_size = state.shape[0]

        new_obs[OBS_STATE] = _to_canonical(state, self.state_spans, self.canonical_joints, self.max_state_dim)
        state_mask = _canonical_mask(self.state_spans, self.canonical_joints, self.max_state_dim)
        action_mask = _canonical_mask(self.action_spans, self.canonical_joints, self.max_action_dim)
        new_obs["state_joint_mask"] = state_mask.to(state.device).expand(batch_size, -1)
        new_obs["action_joint_mask"] = action_mask.to(state.device).expand(batch_size, -1)

        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            action = _to_canonical(action, self.action_spans, self.canonical_joints, self.max_action_dim)
            transition[TransitionKey.ACTION] = action
            new_obs["joint_mask"] = action_mask.to(action.device).expand(batch_size, action.shape[-2], -1)
        transition[TransitionKey.OBSERVATION] = new_obs
        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "state_spans": self.state_spans,
            "action_spans": self.action_spans,
            "canonical_joints": self.canonical_joints,
            "chunk_size": self.chunk_size,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
            type=FeatureType.STATE, shape=(self.max_state_dim,)
        )
        features[PipelineFeatureType.ACTION][ACTION] = PolicyFeature(
            type=FeatureType.ACTION, shape=(self.chunk_size, self.max_action_dim)
        )
        return features


@dataclass
@ProcessorStepRegistry.register(name=INVERSE_SLOT_MAPPING_STEP)
class LingbotVLAV2InverseSlotMappingProcessorStep(PolicyActionProcessorStep):
    """Write predicted canonical actions back to the raw action dims each slot reads."""

    action_spans: dict[str, list[list[int]]] = field(default_factory=dict)
    canonical_joints: dict[str, int] = field(default_factory=dict)

    def action(self, action: PolicyAction) -> PolicyAction:
        raw_dim = max(end for spans in self.action_spans.values() for _, end in spans)
        # fp32 whatever the model dtype: envs and numpy expect float32 actions.
        raw = action.new_zeros(*action.shape[:-1], raw_dim, dtype=torch.float32)
        offset = 0
        for joint, dim in self.canonical_joints.items():
            position = offset
            for start, end in self.action_spans.get(joint, []):
                raw[..., start:end] = action[..., position : position + end - start]
                position += end - start
            offset += dim
        return raw

    def get_config(self) -> dict[str, Any]:
        return {"action_spans": self.action_spans, "canonical_joints": self.canonical_joints}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name=IMAGE_STEP)
class LingbotVLAV2ImageProcessorStep(ProcessorStep):
    """Run the standard Qwen3-VL image processor over the canonical cameras.

    Each camera is resized to ``resize_imgs_with_padding`` and scaled to [0, 255],
    then all frames go through the HF image processor in one call, which returns
    ``pixel_values`` plus the ``image_grid_thw`` patch grid. Missing views are
    filled with -1 and ``img_masks=False``.
    """

    tokenizer_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    cameras: list = field(default_factory=list)
    resize_imgs_with_padding: tuple = (224, 224)
    # Cap the Qwen3-VL image processor's dynamic-resolution token budget. Left
    # uncapped, a native 1080x1920 frame explodes to ~8k vision tokens -> an
    # O(N^2) eager-attention tensor that OOMs and does not match the checkpoint's
    # training resolution. Qwen3-VL uses 16px patches + 2x2 merge (=1024 px/token),
    # so 1,048,576 px ~= 1024 tokens.
    image_max_pixels: int = 262144
    image_min_pixels: int = 131072

    _image_processor: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not _transformers_available:
            raise ImportError(
                "transformers is required for LingbotVLAV2ImageProcessorStep. "
                "Install it with `pip install 'lerobot[lingbot_vla2]'`."
            )
        self._image_processor = AutoImageProcessor.from_pretrained(
            self.tokenizer_path,
            max_pixels=self.image_max_pixels,
            min_pixels=self.image_min_pixels,
        )

    def __call__(self, transition):
        transition = transition.copy()
        new_obs = dict(transition[TransitionKey.OBSERVATION])

        keys = [f"{OBS_IMAGES}.{cam}" for cam in self.cameras]
        present = [key for key in keys if key in new_obs]
        if not present:
            raise ValueError(f"None of the configured camera keys are present in the observation: {keys}")
        # LeRobot images are float in [0, 1]; the HF processor expects [0, 255].
        frames = torch.stack(
            [
                tv_resize(new_obs[key], list(self.resize_imgs_with_padding), antialias=True) * 255.0
                for key in present
            ],
            dim=1,
        )  # (B, n_present, C, H, W)
        batch_size, n_present = frames.shape[:2]
        processed = self._image_processor(list(frames.flatten(0, 1)))
        pixels = processed["pixel_values"].unflatten(0, (batch_size, n_present, -1))
        grid = processed["image_grid_thw"].view(batch_size, n_present, 3)

        # Missing views are filled with -1 pixels and masked out.
        index = {key: i for i, key in enumerate(present)}
        images, grids = [], []
        for key in keys:
            i = index.get(key)
            images.append(pixels[:, i] if i is not None else torch.full_like(pixels[:, 0], -1.0))
            grids.append(grid[:, i if i is not None else 0])
        new_obs["images"] = torch.stack(images, dim=1)
        new_obs["img_masks"] = torch.tensor([key in index for key in keys]).expand(batch_size, -1)
        new_obs["image_grid_thw"] = torch.stack(grids, dim=1)
        transition[TransitionKey.OBSERVATION] = new_obs
        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "tokenizer_path": self.tokenizer_path,
            "cameras": self.cameras,
            "resize_imgs_with_padding": list(self.resize_imgs_with_padding),
            # Must round-trip: these cap the Qwen3-VL vision-token budget and change
            # the effective input resolution. Omitting them silently reset the reload
            # to defaults and mismatched the checkpoint's training resolution.
            "image_max_pixels": self.image_max_pixels,
            "image_min_pixels": self.image_min_pixels,
        }

    def save_artifacts(self, save_directory: Path) -> dict[str, str]:
        """Save the image processor so the step reloads without a hub lookup."""
        artifact_path = Path("image_processor")
        self._image_processor.save_pretrained(save_directory / artifact_path)
        return {"tokenizer_path": artifact_path.as_posix()}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name=CHAT_TEMPLATE_STEP)
class LingbotVLAV2ChatTemplateProcessorStep(ProcessorStep):
    """Wrap the task in the Qwen3 chat template for the standard ``TokenizerProcessorStep``."""

    tokenizer_name: str | None = None
    task_key: str = "task"

    _tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not _transformers_available:
            raise ImportError(
                "transformers is required for LingbotVLAV2ChatTemplateProcessorStep. "
                "Install it with `pip install 'lerobot[lingbot_vla2]'`."
            )
        if self.tokenizer_name is None:
            raise ValueError("LingbotVLAV2ChatTemplateProcessorStep requires a tokenizer_name.")
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def __call__(self, transition):
        transition = transition.copy()
        batch_size = transition[TransitionKey.OBSERVATION][OBS_STATE].shape[0]
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        task = complementary.get(self.task_key, DEFAULT_TASK)
        tasks = [task] * batch_size if isinstance(task, str) else task
        complementary[self.task_key] = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=False
            )
            for t in tasks
        ]
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def get_config(self) -> dict[str, Any]:
        return {"task_key": self.task_key, "tokenizer_name": self.tokenizer_name}

    def save_artifacts(self, save_directory: Path) -> dict[str, str]:
        """Save the tokenizer so the step reloads without a hub lookup."""
        artifact_path = Path("chat_template_tokenizer")
        self._tokenizer.save_pretrained(save_directory / artifact_path)
        return {"tokenizer_name": artifact_path.as_posix()}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def make_lingbot_vla_v2_pre_post_processors(
    config: LingbotVLAV2Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the LingBot-VLA 2.0 pre- and post-processing pipelines.

    Normalization and relative actions run on the raw features, before the slot mapping.
    """
    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=config.relative_exclude_joints,
        action_names=config.action_feature_names,
    )
    steps = make_default_policy_processor_steps(config, dataset_stats)
    cameras = [key.removeprefix(f"{OBS_IMAGES}.") for key in config.image_features]

    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        relative_step,
        steps.normalize,
        LingbotVLAV2SlotMappingProcessorStep(
            **_slot_spans(config),
            canonical_joints=config.canonical_joints,
            chunk_size=config.chunk_size,
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
        ),
        LingbotVLAV2ImageProcessorStep(
            tokenizer_path=config.tokenizer_path,
            cameras=cameras,
            resize_imgs_with_padding=tuple(config.resize_imgs_with_padding),
            image_max_pixels=config.image_max_pixels,
            image_min_pixels=config.image_min_pixels,
        ),
        LingbotVLAV2ChatTemplateProcessorStep(tokenizer_name=config.tokenizer_path),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_path,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        steps.to_device,
    ]
    output_steps: list[ProcessorStep] = [
        LingbotVLAV2InverseSlotMappingProcessorStep(
            action_spans=config.slot_spans(ACTION), canonical_joints=config.canonical_joints
        ),
        steps.unnormalize,
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)


def _slot_spans(config: LingbotVLAV2Config) -> dict[str, dict[str, list[list[int]]]]:
    return {"state_spans": config.slot_spans(OBS_STATE), "action_spans": config.slot_spans(ACTION)}


def make_lingbot_vla_v2_pre_post_processors_from_pretrained(
    config: LingbotVLAV2Config,
    pretrained_path: str,
    *,
    revision: str | None = None,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
    dataset_meta: Any | None = None,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
    preprocessor_config_filename: str = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    postprocessor_config_filename: str = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Load the saved pipelines with the slot mapping of the active config.

    A fine-tune maps a new robot onto the canonical slots, so its config's spans replace the
    checkpoint's. Explicit overrides still win.
    """
    # Fine-tuning stats arrive as the caller's normalizer overrides (lerobot-train injects them).
    del dataset_stats, dataset_meta
    spans = _slot_spans(config)
    preprocessor_overrides = {SLOT_MAPPING_STEP: spans, **(preprocessor_overrides or {})}
    postprocessor_overrides = {
        INVERSE_SLOT_MAPPING_STEP: {"action_spans": spans["action_spans"]},
        **(postprocessor_overrides or {}),
    }
    return load_pretrained_policy_processors(
        pretrained_path,
        revision=revision,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
        preprocessor_config_filename=preprocessor_config_filename,
        postprocessor_config_filename=postprocessor_config_filename,
    )
