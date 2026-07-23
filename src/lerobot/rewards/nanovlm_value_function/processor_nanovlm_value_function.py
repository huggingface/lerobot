"""Processor for the nanoVLM value-function experiment."""

from typing import Any

import torch

from lerobot.configs import FeatureType
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
)
from lerobot.rewards.distributional_value_function.processor_distributional_value_function import (
    DistributionalVFImagePreprocessorStep,
    DistributionalVFPrepareTaskPromptStep,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_nanovlm_value_function import NanoVLMVFConfig


def make_nanovlm_vf_pre_post_processors(
    config: NanoVLMVFConfig,
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
            DistributionalVFImagePreprocessorStep(
                image_resolution=config.image_resolution,
                image_keys=image_keys,
            ),
            DistributionalVFPrepareTaskPromptStep(),
            TokenizerProcessorStep(
                tokenizer_name=config.tokenizer_path,
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
