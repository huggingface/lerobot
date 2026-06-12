"""Pre/post-processor pipelines for the vllm policy.

The remote server performs normalization/denormalization, so the policy-side pipeline is
intentionally identity w.r.t. values: it only renames observation keys and handles device
placement. The pipeline still includes a ``device_processor`` and a
``rename_observations_processor`` step because ``lerobot-eval`` passes overrides keyed on
those step names (and validates that every override was applied).
"""

from __future__ import annotations

from typing import Any

import torch

from lerobot.processor import (
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_vllm import VllmConfig


def make_vllm_pre_post_processors(
    config: VllmConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        DeviceProcessorStep(device=config.device),
    ]
    output_steps = [
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
