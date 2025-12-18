from dataclasses import dataclass, field

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.envs.configs import EnvConfig, GripperConfig, HILSerlProcessorConfig
from lerobot.robots import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass(kw_only=True)
class ACFQLTrainRLServerPipelineConfig(TrainRLServerPipelineConfig):
    online_dataset: DatasetConfig | None = None
    save_offline_replay_buffer_on_checkpoint: bool = True
    save_replay_buffer_on_checkpoint: bool = True


@dataclass
class ACFQLGripperConfig(GripperConfig):
    min_bound_gripper_pos: float = 0.0
    max_bound_gripper_pos: float = 2.0
    neutral_action: float = 1.0


@dataclass
class ACFQLHILSerlProcessorConfig(HILSerlProcessorConfig):
    gripper: ACFQLGripperConfig = field(default_factory=ACFQLGripperConfig)


@EnvConfig.register_subclass(name="gym_manipulator_acfql")
@dataclass
class HILSerlRobotEnvConfig(EnvConfig):
    """Configuration for the HILSerlRobotEnv environment."""

    robot: RobotConfig | None = None
    teleop: TeleoperatorConfig | None = None
    processor: ACFQLHILSerlProcessorConfig = field(default_factory=ACFQLHILSerlProcessorConfig)

    name: str = "real_robot"

    @property
    def gym_kwargs(self) -> dict:
        return {}
