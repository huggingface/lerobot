from dataclasses import dataclass, field

import draccus

from lerobot.configs.types import FeatureType


@dataclass
class EnvConfig(draccus.ChoiceRegistry):
    n_envs: int | None = None
    task: str | None = None
    fps: int = 30
    feature_types: dict = field(default_factory=dict)

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@EnvConfig.register_subclass("real_world")
@dataclass
class RealEnv(EnvConfig):
    pass


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
    gym: dict = field(
        default_factory=lambda: {
            "obs_type": "pixels_agent_pos",
            "render_mode": "rgb_array",
        }
    )


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
    gym: dict = field(
        default_factory=lambda: {
            "obs_type": "pixels_agent_pos",
            "render_mode": "rgb_array",
            "visualization_width": 384,
            "visualization_height": 384,
        }
    )


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
    gym: dict = field(
        default_factory=lambda: {
            "obs_type": "pixels_agent_pos",
            "render_mode": "rgb_array",
            "visualization_width": 384,
            "visualization_height": 384,
        }
    )
