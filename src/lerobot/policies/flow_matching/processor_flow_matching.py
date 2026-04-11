from typing import Any
import torch

from lerobot.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def make_flow_matching_pre_post_processors(
    config: FlowMatchingConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for flow matching policy.
    """
    
    # 1. Pre-processor
    preprocessor = PolicyProcessorPipeline(name=POLICY_PREPROCESSOR_DEFAULT_NAME)
    
    # Add rename step if needed (usually empty rename mapping works as a base)
    preprocessor.add_step(RenameObservationsProcessorStep({}))
    
    # Add normalization
    if dataset_stats is not None:
        norm_step = NormalizerProcessorStep(
            stats=dataset_stats,
            normalize_min_max=True # typical flow matching uses [-1, 1] range standard
        )
        preprocessor.add_step(norm_step)
        
    preprocessor.add_step(AddBatchDimensionProcessorStep())
    if torch.cuda.is_available():
        preprocessor.add_step(DeviceProcessorStep("cuda"))
        
    # 2. Post-processor
    postprocessor = PolicyProcessorPipeline(
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    
    if dataset_stats is not None:
        post_norm_step = UnnormalizerProcessorStep(
            stats=dataset_stats,
            normalize_min_max=True
        )
        postprocessor.add_step(post_norm_step)
        
    return preprocessor, postprocessor
