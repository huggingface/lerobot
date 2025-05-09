import logging
import logging.handlers
import os
import time
from typing import Any

import torch


def setup_logging(prefix: str, info_bracket: str):
    """Sets up logging"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Delete any existing prefix_* log files
    for old_log_file in os.listdir("logs"):
        if old_log_file.startswith(prefix) and old_log_file.endswith(".log"):
            try:
                os.remove(os.path.join("logs", old_log_file))
                print(f"Deleted old log file: {old_log_file}")
            except Exception as e:
                print(f"Failed to delete old log file {old_log_file}: {e}")

    # Set up logging with both console and file output
    logger = logging.getLogger(prefix)
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            f"%(asctime)s.%(msecs)03d [{info_bracket}] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(console_handler)

    # File handler - creates a new log file for each run
    file_handler = logging.handlers.RotatingFileHandler(
        f"logs/policy_server_{int(time.time())}.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    file_handler.setFormatter(
        logging.Formatter(
            f"%(asctime)s.%(msecs)03d [{info_bracket}] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    return logger


class TimedData:
    def __init__(self, timestamp: float, data: Any, timestep: int):
        """Initialize a TimedData object.

        Args:
            timestamp: Unix timestamp relative to data's creation.
            data: The actual data to wrap a timestamp around.
            timestep: The timestep of the data.
        """
        self.timestamp = timestamp
        self.data = data
        self.timestep = timestep

    def get_data(self):
        return self.data

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep


class TimedAction(TimedData):
    def __init__(self, timestamp: float, action: torch.Tensor, timestep: int):
        super().__init__(timestamp=timestamp, data=action, timestep=timestep)

    def get_action(self):
        return self.get_data()


class TimedObservation(TimedData):
    def __init__(
        self, timestamp: float, observation: dict[str, torch.Tensor], timestep: int, transfer_state: int = 0
    ):
        super().__init__(timestamp=timestamp, data=observation, timestep=timestep)
        self.transfer_state = transfer_state

    def get_observation(self):
        return self.get_data()


class TinyPolicyConfig:
    def __init__(
        self,
        policy_type: str = "act",
        pretrained_name_or_path: str = "fracapuano/act_so100_test",
        device: str = "cpu",
    ):
        self.policy_type = policy_type
        self.pretrained_name_or_path = pretrained_name_or_path
        self.device = device
