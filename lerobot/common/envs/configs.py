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
