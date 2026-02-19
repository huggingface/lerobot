

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    EnvTransition,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    TransitionKey,
)
from lerobot.utils.rotation import Rotation



@ProcessorStepRegistry.register("add_processor_observations_to_state")
@dataclass
class AddProcessorObservationsToState(ObservationProcessorStep):
    """
    Env processors generates new observations, this should be added to 
    observation.state for further processing/ database upload
    """


    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:

        # Process other observations
        other_observations = []
        other_observations_keys = []
        for k,v in observation.items():
            if not isinstance(v, torch.Tensor):
                other_observations.append(v)
                other_observations_keys.append(k)
        for other_key in other_observations_keys:
            observation.pop(other_key)
        other_observations = torch.tensor(other_observations).unsqueeze(0)

        observation["observation.state"] = torch.cat([observation["observation.state"], other_observations],dim=1)
        return observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features