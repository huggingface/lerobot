from typing import Any

import torch

from lerobot.lerobot_types import PolicyAction
from lerobot.processor import (
    PolicyProcessorPipeline,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_cig_vla import CIGVLAConfig


def make_cig_vla_pre_post_processors(
    config: CIGVLAConfig, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    steps = make_default_policy_processor_steps(config, dataset_stats)
    # Default transition steps operate by feature type and preserve auxiliary geometry keys.
    return make_policy_processor_pipelines(
        input_steps=[steps.rename_observations, steps.add_batch_dim, steps.to_device, steps.normalize],
        output_steps=[steps.unnormalize, steps.to_cpu],
    )
