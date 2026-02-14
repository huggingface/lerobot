#!/usr/bin/env python

# Created by Indraneel on 01/17/25

from dataclasses import dataclass
import numpy as np
from typing import Any

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import ObservationProcessorStep

from .converters import to_tensor
from .core import EnvAction, EnvTransition, PolicyAction
from .pipeline import ActionProcessorStep, ProcessorStep, ProcessorStepRegistry

@ProcessorStepRegistry.register("radians2degrees_observation_processor")
@dataclass
class Radians2DegreesObservationProcessor(ObservationProcessorStep):
    """
    Converts joint angle positions from radians to degrees
    """
    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        for key, val in observation.items():
            if key == "agent_pos" and isinstance(val, dict):
                observation[key] = self.observation(val)
            elif not isinstance(val, dict):
                observation[key] = np.rad2deg(val)
        return observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features