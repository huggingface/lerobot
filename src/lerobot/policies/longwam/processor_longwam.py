# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from torch.nn import functional
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, RobotObservation, TransitionKey
from lerobot.processor import (
    ActionProcessorStep,
    ComplementaryDataProcessorStep,
    ObservationProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    load_pretrained_policy_processors,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_longwam import LongWAMConfig

FeatureMap = dict[PipelineFeatureType, dict[str, PolicyFeature]]


@ProcessorStepRegistry.register(name="longwam_observations")
@dataclass
class LongWAMObservations(ObservationProcessorStep):
    """Prepare camera pixels and clamp normalized proprioception."""

    camera_keys: tuple[str, ...]
    camera_size: tuple[int, int]
    image_dtype: str = "float32"

    def observation(self, observation: RobotObservation) -> RobotObservation:
        for key in self.camera_keys:
            image = observation[key]
            if image.ndim not in (4, 5) or image.shape[-3] != 3:
                raise ValueError(f"{key} must be [B,3,H,W] or [B,T,3,H,W]")
            if not image.is_floating_point():
                raise ValueError("Camera images must be floating point in [0,1]")
            shape = image.shape[:-3]
            height, width = self.camera_size
            # Share resize/crop geometry without changing released inference pixels.
            input_height, input_width = image.shape[-2:]
            scale = max(width / input_width, height / input_height)
            resized_height, resized_width = round(input_height * scale), round(input_width * scale)
            top, left = (resized_height - height) // 2, (resized_width - width) // 2
            if image.ndim == 4:
                # Preserve released inference pixels, before the pipeline's device transfer.
                frames = []
                for frame in image:
                    pil = to_pil_image(frame.mul(255).round().to(torch.uint8))
                    pil = pil.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
                    frames.append(pil_to_tensor(pil.crop((left, top, left + width, top + height))))
                observation[key] = (
                    torch.stack(frames).to(getattr(torch, self.image_dtype)).mul(2 / 255).sub(1)
                )
                continue
            image = image.reshape(-1, *image.shape[-3:]).float()
            image = functional.interpolate(
                image,
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            image = image[..., top : top + height, left : left + width]
            observation[key] = image.reshape(*shape, 3, height, width).mul(2).sub(1)
        observation["observation.state"] = observation["observation.state"].clamp(-5, 5)
        return observation

    def get_config(self) -> dict[str, Any]:
        return {
            "camera_keys": self.camera_keys,
            "camera_size": self.camera_size,
            "image_dtype": self.image_dtype,
        }

    def transform_features(self, features: FeatureMap) -> FeatureMap:
        features = {kind: dict(values) for kind, values in features.items()}
        for key in self.camera_keys:
            feature = features[PipelineFeatureType.OBSERVATION][key]
            features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(
                type=feature.type, shape=(3, *self.camera_size)
            )
        return features


@ProcessorStepRegistry.register(name="longwam_prompt")
@dataclass
class LongWAMPrompt(ComplementaryDataProcessorStep):
    """Format task instructions before tokenization."""

    prompt_template: str

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        tasks = complementary_data["task"]
        tasks = [tasks] if isinstance(tasks, str) else tasks
        complementary_data["task"] = [
            self.prompt_template.format(task=" ".join(task.split())) for task in tasks
        ]
        return complementary_data

    def get_config(self) -> dict[str, Any]:
        return {"prompt_template": self.prompt_template}

    def transform_features(self, features: FeatureMap) -> FeatureMap:
        return features


@ProcessorStepRegistry.register(name="longwam_action_clamp")
class LongWAMActionClamp(ActionProcessorStep):
    """Clamp normalized training actions; inference has no action input."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if transition.get(TransitionKey.ACTION) is None:
            return transition
        return super().__call__(transition)

    def action(self, action: torch.Tensor) -> torch.Tensor:
        return action.clamp(-5, 5)

    def transform_features(self, features: FeatureMap) -> FeatureMap:
        return features


@ProcessorStepRegistry.register(name="longwam_action_toggle")
@dataclass
class LongWAMActionToggle(ActionProcessorStep):
    dimensions: list[int]

    def action(self, action: torch.Tensor) -> torch.Tensor:
        result = action.clone()
        result[..., self.dimensions] = torch.sign(1 - 2 * result[..., self.dimensions])
        return result

    def get_config(self) -> dict[str, Any]:
        return {"dimensions": self.dimensions}

    def transform_features(self, features: FeatureMap) -> FeatureMap:
        return features


def make_longwam_pre_post_processors(
    config: LongWAMConfig, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    steps = make_default_policy_processor_steps(config, dataset_stats)
    pre = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.normalize,
        LongWAMObservations(
            config.camera_keys,
            config.camera_size,
            str(config.dtype or torch.float32).removeprefix("torch."),
        ),
        LongWAMActionClamp(),
        LongWAMPrompt(config.prompt_template),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding="max_length",
        ),
        steps.to_device,
    ]
    post: list[ProcessorStep] = [steps.unnormalize]
    if config.toggle_action_dimensions:
        post.append(LongWAMActionToggle(config.toggle_action_dimensions))
    post.append(steps.to_cpu)
    return make_policy_processor_pipelines(input_steps=pre, output_steps=post)


def make_longwam_pre_post_processors_from_pretrained(
    config: LongWAMConfig,
    pretrained_path: str,
    *,
    revision: str | None = None,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
    preprocessor_config_filename: str = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    postprocessor_config_filename: str = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
    **_: Any,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Keep saved normalization while applying the active action-toggle configuration."""
    pre, post = load_pretrained_policy_processors(
        pretrained_path,
        revision=revision,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
        preprocessor_config_filename=preprocessor_config_filename,
        postprocessor_config_filename=postprocessor_config_filename,
    )
    # Explicit processor overrides retain precedence over the policy configuration.
    if any(
        "dimensions" in (postprocessor_overrides or {}).get(key, {})
        for key in ("longwam_action_toggle", "LongWAMActionToggle")
    ):
        return pre, post
    toggles = [step for step in post.steps if isinstance(step, LongWAMActionToggle)]
    if len(toggles) > 1:
        raise ValueError("LongWAM postprocessor must contain at most one action toggle")
    dimensions = list(config.toggle_action_dimensions)
    if toggles:
        toggles[0].dimensions = dimensions
    elif dimensions:
        unnormalizers = [
            i for i, step in enumerate(post.steps) if isinstance(step, UnnormalizerProcessorStep)
        ]
        if len(unnormalizers) != 1:
            raise ValueError("LongWAM action toggle requires exactly one action unnormalizer")
        index = unnormalizers[0] + 1
        post.steps = [*post.steps[:index], LongWAMActionToggle(dimensions), *post.steps[index:]]
    return pre, post
