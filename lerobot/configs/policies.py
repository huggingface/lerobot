import abc
import os
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Type, TypeVar

import draccus
import gymnasium as gym
from huggingface_hub import ModelHubMixin, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import flatten_dict, get_nested_item
from lerobot.common.envs.configs import EnvConfig
from lerobot.common.optim.optimizers import OptimizerConfig
from lerobot.common.optim.schedulers import LRSchedulerConfig
from lerobot.configs.types import FeatureType, NormalizationMode

# Generic variable that is either ModelHubMixin or a subclass thereof
T = TypeVar("T", bound="PretrainedConfig")


@dataclass
class PolicyFeature:
    key: str
    type: FeatureType
    shape: list | tuple
    normalization_mode: NormalizationMode


@dataclass
class PretrainedConfig(draccus.ChoiceRegistry, ModelHubMixin, abc.ABC):
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

    type: str | None = ""

    n_obs_steps: int = 1
    normalization_mapping: dict[str, NormalizationMode] = field(default_factory=dict)

    def __post_init__(self):
        self.type = self.get_choice_name(self.__class__)
        self.pretrained_path = None

    @abc.abstractproperty
    def observation_delta_indices(self) -> list | None:
        raise NotImplementedError

    @abc.abstractproperty
    def action_delta_indices(self) -> list | None:
        raise NotImplementedError

    @abc.abstractproperty
    def reward_delta_indices(self) -> list | None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self) -> LRSchedulerConfig:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
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

    def _save_pretrained(self, save_directory: Path) -> None:
        to_save = copy(self)
        del to_save.path
        with open(save_directory / "config.json", "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: Type[T],
        pretrained_model_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **model_kwargs,
    ) -> T:
        model_id = str(pretrained_model_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if "config.json" in os.listdir(model_id):
                config_file = os.path.join(model_id, "config.json")
            else:
                print(f"config.json not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename="config.json",
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                print(f"config.json not found on the HuggingFace Hub: {str(e)}")

        # HACK: this is very ugly, ideally we'd like to be able to do that natively with draccus
        # something like --policy.path (in addition to --policy.type)
        cli_overrides = model_kwargs.pop("cli_overrides", [])
        instance = draccus.parse(cls, config_file, args=[])
        return draccus.parse(instance.__class__, config_file, args=cli_overrides)

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
                # Backward compatibility for "channel" which is an error introduced in LeRobotDataset v2.0 for ported datasets.
                if names[2] in ["channel", "channels"]:  # (h, w, c) -> (c, h, w)
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

    def parse_features_from_env(self, env: gym.Env, env_cfg: EnvConfig):
        robot_state_features = []
        env_state_features = []
        action_features = []
        image_features = []

        flat_dict = flatten_dict(env_cfg.feature_types)

        for key, _type in flat_dict.items():
            env_ft = (
                env.action_space
                if _type is FeatureType.ACTION
                else get_nested_item(env.observation_space, key)
            )
            shape = env_ft.shape[1:]
            if _type is FeatureType.VISUAL:
                h, w, c = shape
                if not c < h and c < w:
                    raise ValueError(
                        f"Expect channel last images for visual feature {key} of {env_cfg.type} env, but instead got {shape=}"
                    )
                shape = (c, h, w)

            feature = PolicyFeature(
                key=key,
                type=_type,
                shape=shape,
                normalization_mode=self.normalization_mapping[_type],
            )
            if _type is FeatureType.VISUAL:
                image_features.append(feature)
            elif _type is FeatureType.STATE:
                robot_state_features.append(feature)
            elif _type is FeatureType.ENV:
                env_state_features.append(feature)
            elif _type is FeatureType.ACTION:
                action_features.append(feature)

        # TODO(aliberts, rcadene): remove this hardcoding of keys and just use the nested keys as is
        # (need to also refactor preprocess_observation and externalize normalization from policies)
        for ft in image_features:
            if len(ft.key.split("/")) > 1:
                ft.key = f"observation.images.{ft.key.split('/')[-1]}"
            elif len(ft.key.split("/")) == 1:
                image_features[0].key = "observation.image"

        if len(robot_state_features) == 1:
            robot_state_features[0].key = "observation.state"

        if len(env_state_features) == 1:
            env_state_features[0].key = "observation.environment_state"

        self.robot_state_feature = robot_state_features[0] if len(robot_state_features) == 1 else None
        self.env_state_feature = env_state_features[0] if len(env_state_features) == 1 else None
        self.action_feature = action_features[0] if len(action_features) == 1 else None
        self.image_features = image_features
