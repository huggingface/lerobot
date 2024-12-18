import abc
from dataclasses import dataclass, field
from enum import Enum
from pprint import pformat

import draccus

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.optim.optimizers import OptimizerConfig


# Note: We subclass str so that serialization is straightforward
# https://stackoverflow.com/questions/24481852/serialising-an-enum-member-to-json
class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"


class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"


@dataclass
class PolicyFeature:
    key: str
    type: FeatureType
    shape: list | tuple
    normalization_mode: NormalizationMode


@dataclass
class PretrainedConfig(draccus.ChoiceRegistry, abc.ABC):
    """
    Base configuration class for policy models.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        input_shapes: A dictionary defining the shapes of the input data for the policy.
        output_shapes: A dictionary defining the shapes of the output data for the policy.
        input_normalization_modes: A dictionary with key representing the modality and the value specifies the
            normalization mode to apply.
        output_normalization_modes: Similar dictionary as `input_normalization_modes`, but to unnormalize to
            the original scale.
    """

    n_obs_steps: int = 1

    robot_state_feature: PolicyFeature | None = None
    env_state_feature: PolicyFeature | None = None
    action_feature: PolicyFeature | None = None
    image_features: list[PolicyFeature] | None = None

    normalization_mapping: dict[FeatureType, NormalizationMode] = field(
        default_factory=lambda: {
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.VISUAL: NormalizationMode.MEAN_STD,
            FeatureType.ENV: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
        }
    )

    optimizer_preset: OptimizerConfig | None = None

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractproperty
    def observation_delta_indices(self) -> list | None:
        raise NotImplementedError

    @abc.abstractproperty
    def action_delta_indices(self) -> list | None:
        raise NotImplementedError

    @abc.abstractproperty
    def reward_delta_indices(self) -> list | None:
        raise NotImplementedError

    @property
    def input_features(self) -> list[PolicyFeature]:
        input_features = []
        for ft in [self.robot_state_feature, self.env_state_feature, *self.image_features]:
            if ft is not None:
                input_features.append(ft)

        return input_features

    @property
    def output_features(self) -> list[PolicyFeature]:
        return [self.action_feature]

    def parse_features_from_dataset(self, ds_meta: LeRobotDatasetMetadata):
        # TODO(aliberts): Implement PolicyFeature in LeRobotDataset and remove the need for this
        robot_state_features = []
        env_state_features = []
        action_features = []
        image_features = []

        for key in ds_meta.features:
            if key in ds_meta.camera_keys:
                shape = ds_meta.features[key]["shape"]
                names = ds_meta.features[key]["names"]
                if len(shape) != 3:
                    raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")
                if names[2] == "channel":  # (h, w, c) -> (c, h, w)
                    shape = (shape[2], shape[0], shape[1])
                image_features.append(
                    PolicyFeature(
                        key=key,
                        type=FeatureType.VISUAL,
                        shape=shape,
                        normalization_mode=self.normalization_mapping[FeatureType.VISUAL],
                    )
                )
            elif key == "observation.environment_state":
                env_state_features.append(
                    PolicyFeature(
                        key=key,
                        type=FeatureType.ENV,
                        shape=ds_meta.features[key]["shape"],
                        normalization_mode=self.normalization_mapping[FeatureType.ENV],
                    )
                )
            elif key.startswith("observation"):
                robot_state_features.append(
                    PolicyFeature(
                        key=key,
                        type=FeatureType.STATE,
                        shape=ds_meta.features[key]["shape"],
                        normalization_mode=self.normalization_mapping[FeatureType.STATE],
                    )
                )
            elif key == "action":
                action_features.append(
                    PolicyFeature(
                        key=key,
                        type=FeatureType.ACTION,
                        shape=ds_meta.features[key]["shape"],
                        normalization_mode=self.normalization_mapping[FeatureType.ACTION],
                    )
                )

        if len(robot_state_features) > 1:
            raise ValueError(
                "Found multiple features for the robot's state. Please select only one or concatenate them."
                f"Robot state features found:\n{pformat(robot_state_features)}"
            )

        if len(env_state_features) > 1:
            raise ValueError(
                "Found multiple features for the env's state. Please select only one or concatenate them."
                f"Env state features found:\n{pformat(env_state_features)}"
            )

        if len(action_features) > 1:
            raise ValueError(
                "Found multiple features for the action. Please select only one or concatenate them."
                f"Action features found:\n{pformat(action_features)}"
            )

        self.robot_state_feature = robot_state_features[0] if len(robot_state_features) == 1 else None
        self.env_state_feature = env_state_features[0] if len(env_state_features) == 1 else None
        self.action_feature = action_features[0] if len(action_features) == 1 else None
        self.image_features = image_features
