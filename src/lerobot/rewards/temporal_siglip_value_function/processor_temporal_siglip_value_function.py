"""Processor for past-only temporal SigLIP2 value inputs."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import FeatureType
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
)
from lerobot.rewards.distributional_value_function.processor_distributional_value_function import (
    IMAGE_MASK_SUFFIX,
    DistributionalVFPrepareTaskPromptStep,
    resize_with_pad_torch,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_temporal_siglip_value_function import TemporalSiglipVFConfig


@ProcessorStepRegistry.register(name="temporal_siglip_vf_image_processor")
@dataclass
class TemporalSiglipImageProcessorStep(ProcessorStep):
    image_resolution: tuple[int, int]
    image_keys: tuple[str, ...]
    history_steps: int

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION, {}))
        batch_size = self._batch_size(observation)
        for key in self.image_keys:
            if key not in observation:
                height, width = self.image_resolution
                observation[key] = torch.full((batch_size, self.history_steps, 3, height, width), -1.0)
                observation[key + IMAGE_MASK_SUFFIX] = torch.zeros(
                    batch_size, self.history_steps, dtype=torch.bool
                )
                continue

            image = observation[key].float()
            if image.ndim == 4:
                image = image[:, None]
            if image.ndim != 5 or image.shape[2] != 3:
                raise ValueError(f"Expected {key} as [B,T,3,H,W], got {tuple(image.shape)}")
            batch_size, history = image.shape[:2]
            image = image.flatten(0, 1).permute(0, 2, 3, 1)
            image = image * 2.0 - 1.0
            if image.shape[1:3] != self.image_resolution:
                image = resize_with_pad_torch(image, *self.image_resolution)
            observation[key] = image.permute(0, 3, 1, 2).unflatten(0, (batch_size, history))
            observation[key + IMAGE_MASK_SUFFIX] = torch.ones(
                batch_size, history, dtype=torch.bool, device=image.device
            )
        transition[TransitionKey.OBSERVATION] = observation
        return transition

    @staticmethod
    def _batch_size(observation: dict[str, Any]) -> int:
        for value in observation.values():
            if isinstance(value, Tensor) and value.ndim >= 2:
                return value.shape[0]
        return 1

    def transform_features(self, features):
        return features

    def get_config(self):
        return {
            "image_resolution": self.image_resolution,
            "image_keys": self.image_keys,
            "history_steps": self.history_steps,
        }


def make_temporal_siglip_vf_pre_post_processors(
    config: TemporalSiglipVFConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    image_keys = tuple(
        key for key, feature in config.input_features.items() if feature.type == FeatureType.VISUAL
    )
    preprocessor = PolicyProcessorPipeline(
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            ),
            TemporalSiglipImageProcessorStep(
                image_resolution=config.image_resolution,
                image_keys=image_keys,
                history_steps=config.history_steps,
            ),
            DistributionalVFPrepareTaskPromptStep(),
            TokenizerProcessorStep(
                tokenizer_name=config.siglip_path,
                max_length=config.tokenizer_max_length,
                padding_side="right",
                padding="max_length",
            ),
            DeviceProcessorStep(device=config.device or "cpu"),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline(
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
    )
    return preprocessor, postprocessor
