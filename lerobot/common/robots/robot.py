# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from pathlib import Path
from typing import Any

import draccus

from lerobot.common.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from lerobot.common.motors import MotorCalibration

from .config import RobotConfig


# TODO(aliberts): action/obs typing such as Generic[ObsType, ActType] similar to gym.Env ?
# https://github.com/Farama-Foundation/Gymnasium/blob/3287c869f9a48d99454306b0d4b4ec537f0f35e3/gymnasium/core.py#L23
class Robot(abc.ABC):
    """The main LeRobot class for implementing robots."""

    # Set these in ALL subclasses
    config_class: RobotConfig
    name: str

    def __init__(self, config: RobotConfig):
        self.robot_type = self.name
        self.id = config.id
        self.calibration_dir = (
            config.calibration_dir if config.calibration_dir else HF_LEROBOT_CALIBRATION / ROBOTS / self.name
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_fpath = self.calibration_dir / f"{self.id}.json"
        self.calibration: dict[str, MotorCalibration] = {}
        if self.calibration_fpath.is_file():
            self._load_calibration()

    def __str__(self) -> str:
        return f"{self.id} {self.__class__.__name__}"

    # TODO(aliberts): create a proper Feature class for this that links with datasets
    @property
    @abc.abstractmethod
    def observation_features(self) -> dict:
        pass

    @property
    @abc.abstractmethod
    def action_features(self) -> dict:
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        pass

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:
        """Connects to the robot."""
        pass

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        pass

    @abc.abstractmethod
    def calibrate(self) -> None:
        """Calibrates the robot."""
        pass

    def _load_calibration(self, fpath: Path | None = None) -> None:
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as f, draccus.config_type("json"):
            self.calibration = draccus.load(dict[str, MotorCalibration], f)

    def _save_calibration(self, fpath: Path | None = None) -> None:
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, "w") as f, draccus.config_type("json"):
            draccus.dump(self.calibration, f, indent=4)

    @abc.abstractmethod
    def configure(self) -> None:
        pass

    @abc.abstractmethod
    def get_observation(self) -> dict[str, Any]:
        """Gets observation from the robot."""
        pass

    @abc.abstractmethod
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Sends actions to the robot."""
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnects from the robot."""
        pass
