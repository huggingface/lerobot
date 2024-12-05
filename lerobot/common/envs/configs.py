from dataclasses import dataclass

import draccus


@dataclass
class EnvConfig(draccus.ChoiceRegistry):
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
