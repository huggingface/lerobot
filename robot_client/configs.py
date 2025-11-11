from collections.abc import Callable
from dataclasses import dataclass, field

import torch

# Значения по умолчанию
DEFAULT_FPS = 30

# Регистр функций агрегации
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


def get_aggregate_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if name not in AGGREGATE_FUNCTIONS:
        available = list(AGGREGATE_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]


@dataclass
class RobotClientConfig:
    # Политика
    policy_type: str
    pretrained_name_or_path: str
    actions_per_chunk: int

    # Сеть
    server_address: str = "localhost:8080"
    policy_device: str = "cpu"

    # Поведение
    chunk_size_threshold: float = 0.5
    fps: int = DEFAULT_FPS
    task: str = ""
    aggregate_fn_name: str = "weighted_average"
    debug_visualize_queue_size: bool = False

    # DummyRobot параметры
    robot_id: str = "dummy"
    num_joints: int = 4
    cameras: dict[str, tuple[int, int, int]] = field(default_factory=lambda: {"front": (240, 320, 3)})
    extra_actions: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.server_address:
            raise ValueError("server_address cannot be empty")
        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")
        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")
        if not self.policy_device:
            raise ValueError("policy_device cannot be empty")
        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")
        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)

    @property
    def environment_dt(self) -> float:
        return 1 / self.fps
