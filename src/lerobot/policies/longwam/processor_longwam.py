# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from torch.nn import functional
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    ActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_longwam import LongWAMConfig

FeatureMap = dict[PipelineFeatureType, dict[str, PolicyFeature]]


@ProcessorStepRegistry.register(name="longwam_inputs")
@dataclass
class LongWAMInputs(ProcessorStep):
    camera_keys: tuple[str, ...]
    camera_size: tuple[int, int]
    prompt_template: str
    image_dtype: str = "float32"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        raw_observation = transition[TransitionKey.OBSERVATION]
        complementary_data = transition[TransitionKey.COMPLEMENTARY_DATA]
        if raw_observation is None or complementary_data is None:
            raise ValueError("Long-WAM requires observations and a task")
        observation = dict(raw_observation)
        for key in self.camera_keys:
            image = observation[key]
            if image.ndim not in (4, 5) or image.shape[-3] != 3:
                raise ValueError(f"{key} must be [B,3,H,W] or [B,T,3,H,W]")
            if not image.is_floating_point():
                raise ValueError("Camera images must be floating point in [0,1]")
            shape = image.shape[:-3]
            height, width = self.camera_size
            if image.ndim == 4:
                # Preserve released inference pixels, before the pipeline's device transfer.
                frames = []
                for frame in image:
                    pil = to_pil_image(frame.mul(255).round().to(torch.uint8))
                    scale = max(width / pil.width, height / pil.height)
                    pil = pil.resize(
                        (round(pil.width * scale), round(pil.height * scale)), Image.Resampling.BILINEAR
                    )
                    left, top = (pil.width - width) // 2, (pil.height - height) // 2
                    frames.append(pil_to_tensor(pil.crop((left, top, left + width, top + height))))
                observation[key] = (
                    torch.stack(frames).to(getattr(torch, self.image_dtype)).mul(2 / 255).sub(1)
                )
                continue
            image = image.reshape(-1, *image.shape[-3:]).float()
            image = functional.interpolate(
                image,
                size=self.camera_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            observation[key] = image.reshape(*shape, 3, height, width).mul(2).sub(1)
        observation["observation.state"] = observation["observation.state"].clamp(-5, 5)
        transition[TransitionKey.OBSERVATION] = observation
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            transition[TransitionKey.ACTION] = action.clamp(-5, 5)
        complementary = dict(complementary_data)
        tasks = complementary["task"]
        tasks = [tasks] if isinstance(tasks, str) else tasks
        complementary["task"] = [self.prompt_template.format(task=" ".join(task.split())) for task in tasks]
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "camera_keys": self.camera_keys,
            "camera_size": self.camera_size,
            "prompt_template": self.prompt_template,
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
        LongWAMInputs(
            config.camera_keys,
            config.camera_size,
            config.prompt_template,
            str(config.dtype or torch.float32).removeprefix("torch."),
        ),
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
