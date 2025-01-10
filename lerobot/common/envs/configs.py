import abc
from dataclasses import dataclass, field

import draccus

from lerobot.configs.types import FeatureType


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    n_envs: int | None = None
    task: str | None = None
    fps: int = 30
    feature_types: dict = field(default_factory=dict)

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractproperty
    def gym_kwargs(self) -> dict:
        raise NotImplementedError()


@EnvConfig.register_subclass("aloha")
@dataclass
class AlohaEnv(EnvConfig):
    task: str = "AlohaInsertion-v0"
    fps: int = 50
    episode_length: int = 400
    feature_types: dict = field(
        default_factory=lambda: {
            "agent_pos": FeatureType.STATE,
            "pixels": {
                "top": FeatureType.VISUAL,
            },
            "action": FeatureType.ACTION,
        }
    )
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("pusht")
@dataclass
class PushtEnv(EnvConfig):
    task: str = "PushT-v0"
    fps: int = 10
    episode_length: int = 300
    feature_types: dict = field(
        default_factory=lambda: {
            "agent_pos": FeatureType.STATE,
            "pixels": FeatureType.VISUAL,
            "action": FeatureType.ACTION,
        }
    )
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384

    def __post_init__(self):
        if self.obs_type == "environment_state_agent_pos":
            self.feature_types = {
                "agent_pos": FeatureType.STATE,
                "environment_state": FeatureType.ENV,
                "action": FeatureType.ACTION,
            }

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
        }


@EnvConfig.register_subclass("xarm")
@dataclass
class XarmEnv(EnvConfig):
    task: str = "XarmLift-v0"
    fps: int = 15
    episode_length: int = 200
    feature_types: dict = field(
        default_factory=lambda: {
            "agent_pos": FeatureType.STATE,
            "pixels": FeatureType.VISUAL,
            "action": FeatureType.ACTION,
        }
    )
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
        }
