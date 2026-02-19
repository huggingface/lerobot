#!/usr/bin/env python

# Created by Indraneel on 01/13/25

from dataclasses import dataclass
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .converters import to_tensor
from .core import EnvAction, EnvTransition, PolicyAction
from .pipeline import ActionProcessorStep, ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("degrees2radians_action_processor")
@dataclass
class Degrees2RadiansActionProcessorStep(ActionProcessorStep):
    """
    Converts an action in degrees to radians.
    """

    squeeze_batch_dim: bool = True

    def action(self, action: PolicyAction) -> PolicyAction:
        if not isinstance(action, PolicyAction):
            raise TypeError(
                f"Expected PolicyAction or None, got {type(action).__name__}. "
                "Use appropriate processor for non-tensor actions."
            )

        action_rad = torch.deg2rad(action)

        return action_rad

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features