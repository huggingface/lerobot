import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler(Path(f"logs/{name}_{int(time.time())}.log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


Action = torch.Tensor
RawObservation = dict[str, Any]
Observation = dict[str, torch.Tensor]


@dataclass
class TimedData:
    timestamp: float
    timestep: int

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep


@dataclass
class TimedAction(TimedData):
    action: Action

    def get_action(self):
        return self.action


@dataclass
class TimedObservation(TimedData):
    observation: RawObservation
    must_go: bool = False

    def get_observation(self):
        return self.observation


@dataclass
class FPSTracker:
    target_fps: float
    first_timestamp: float | None = None
    total_obs_count: int = 0

    def calculate_fps_metrics(self, current_timestamp: float) -> dict[str, float]:
        self.total_obs_count += 1
        if self.first_timestamp is None:
            self.first_timestamp = current_timestamp
        total_duration = current_timestamp - self.first_timestamp
        avg_fps = (self.total_obs_count - 1) / total_duration if total_duration > 1e-6 else 0.0
        return {"avg_fps": avg_fps, "target_fps": self.target_fps}

    def reset(self):
        self.first_timestamp = None
        self.total_obs_count = 0


@dataclass
class RemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, dict]
    actions_per_chunk: int
    device: str = "cpu"
    rename_map: dict[str, str] = field(default_factory=dict)
