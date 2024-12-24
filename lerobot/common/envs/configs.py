from dataclasses import dataclass, field

import draccus


@dataclass
class GymConfig:
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"


@dataclass
class EnvConfig(draccus.ChoiceRegistry):
    n_envs: int | None = None
    task: str | None = None
    state_dim: int = 18
    action_dim: int = 18
    fps: int = 30

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
    state_dim: int = 14
    action_dim: int = 14
    fps: int = 50
    episode_length: int = 400
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
    state_dim: int = 2
    action_dim: int = 2
    image_size: int = 96
    fps: int = 10
    episode_length: int = 300
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
    state_dim: int = 4
    action_dim: int = 4
    image_size: int = 84
    fps: int = 15
    episode_length: int = 200
    gym: dict = field(
        default_factory=lambda: {
            "obs_type": "pixels_agent_pos",
            "render_mode": "rgb_array",
            "visualization_width": 384,
            "visualization_height": 384,
        }
    )
